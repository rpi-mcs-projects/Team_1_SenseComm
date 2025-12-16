#!/usr/bin/env python3
"""
Interactive angle‑scan script for TinyRad R‑fiducial tag RCS measurements.

This utility coordinates the following tasks:

1. **Prompt for angle:** Before each measurement, the script asks you to
   confirm or type the physical azimuth angle at which the tag (or radar)
   has been rotated. This allows you to build an RCS vs angle pattern.

2. **Capture data via `measure_rcs.py`:** For each angle, the script
   spawns a subprocess invoking `measure_rcs.py`, which must reside in
   the same directory. You specify the number of frames, FFT size, mode
   (`baseline`, `off`, `on`), and optionally a path to a baseline CSV.
   If a baseline is given and the mode is not `baseline`, the call to
   `measure_rcs.py` produces both a magnitude CSV (range vs mag_db) and
   a relative CSV (range vs relative_rcs_db). The relative CSV is
   required for meaningful RCS difference measurements.

3. **Extract tag amplitude:** After the capture, the script reads the
   output CSV (magnitude or relative) and extracts the peak value within
   a specified range gate around the tag distance. This yields a single
   scalar amplitude (in dB or difference in dB) per angle.

4. **Plot:** Once all angles have been scanned, the script pops up a
   polar plot showing the measured amplitude vs angle. This is a
   “true” RCS polar plot (angles correspond to physical rotation),
   unlike the synthetic polar used by `live_polar_rcs.py`.

**Usage Examples**
------------------

First, capture a baseline (environment only) once:

    python rcs_angle_scan.py --mode baseline --frames 200 \
        --baseline baseline.csv --angle-start 0 --angle-stop 0 --angle-step 1 \
        --tag-range 1.8 --gate 0.4

For baseline, you can set `angle-stop` equal to `angle-start` to perform
just one measurement. The script will prompt you to place the radar
in its baseline configuration and press ENTER.

Then, to measure the tag OFF pattern:

    python3 rcs_angle_scan.py --mode off --frames 200 \
        --baseline baseline.csv --angle-start 0 --angle-stop 180 \
        --angle-step 5 --tag-range 1.8 --gate 0.4

And the tag ON pattern:

    python rcs_angle_scan.py --mode on --frames 200 \
        --baseline baseline.csv --angle-start 0 --angle-stop 180 \
        --angle-step 5 --tag-range 1.8 --gate 0.4

During execution, the script will ask you to set the rotation to the
required angle and press ENTER. If you type `q` (lowercase) and press
ENTER, the scan will terminate early.

After scanning all angles, the script writes a summary CSV with
`angle_deg` and `peak_db` (peak relative amplitude) and displays a
polar plot for immediate visual inspection.

**Note:** This script requires `measure_rcs.py` and a compatible
TinyRad radar environment. It uses `subprocess` to call `measure_rcs.py`
and reads its output files. The script itself contains no direct
TinyRad calls; all hardware communication is delegated to
`measure_rcs.py`.

"""

import argparse
import csv
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ScanResult:
    angle_deg: float
    peak_db: float


def run_measure_rcs(mode: str,
                    frames: int,
                    nfft: int,
                    out_csv: str,
                    baseline_csv: str = None) -> Tuple[str, str]:
    """Run measure_rcs.py as a subprocess and return output file paths.

    Parameters
    ----------
    mode : str
        Measurement mode passed to measure_rcs.py (baseline/off/on).
    frames : int
        Number of frames to average.
    nfft : int
        FFT length for range processing.
    out_csv : str
        Filename (with .csv extension) for the magnitude CSV.
    baseline_csv : str or None
        Path to a baseline CSV; if provided and mode != 'baseline', it will be
        passed as --ref and a *_rel.csv will be produced.

    Returns
    -------
    mag_csv : str
        Path to the magnitude CSV.
    rel_csv : str or None
        Path to the relative CSV if baseline is provided and mode != baseline,
        otherwise None.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    measure_py = os.path.join(script_dir, "measure_rcs.py")
    if not os.path.isfile(measure_py):
        raise RuntimeError(f"measure_rcs.py not found in {script_dir}")

    cmd = [sys.executable, measure_py,
           "--mode", mode,
           "--frames", str(frames),
           "--nfft", str(nfft),
           "--out", out_csv]
    if baseline_csv and mode != "baseline":
        cmd.extend(["--ref", baseline_csv])

    print(f"\n[rcs_angle_scan] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running measure_rcs.py: {e}")

    mag_csv = out_csv
    rel_csv = None
    if baseline_csv and mode != "baseline":
        root, ext = os.path.splitext(out_csv)
        rel_csv = root + "_rel" + ext
        if not os.path.isfile(rel_csv):
            print(f"[WARN] Expected relative CSV {rel_csv} not found.")
            rel_csv = None

    return mag_csv, rel_csv


def load_peak_from_csv(csv_path: str,
                        column: str,
                        target_range: float,
                        gate: float) -> float:
    """Load CSV (produced by measure_rcs.py) and return peak value in gate.

    Parameters
    ----------
    csv_path : str
        Path to CSV. Must have columns 'range_m' and the specified column.
    column : str
        Column to read ('mag_db' or 'relative_rcs_db').
    target_range : float
        Approximate range to the tag (meters). Used to place the gate.
    gate : float
        Width of the range gate (meters) within which the peak is sought.

    Returns
    -------
    float
        Maximum value (dB) of the selected column within the gate. Returns
        NaN if gate does not overlap any range bins.
    """
    ranges = []
    values = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if column not in reader.fieldnames:
            raise RuntimeError(f"Column '{column}' not found in {csv_path}.")
        for row in reader:
            try:
                r = float(row['range_m'])
                v = float(row[column])
                ranges.append(r)
                values.append(v)
            except (ValueError, KeyError):
                continue
    if not ranges:
        return float('nan')
    rng = np.asarray(ranges)
    vals = np.asarray(values)
    gate_min = target_range - gate / 2.0
    gate_max = target_range + gate / 2.0
    mask = (rng >= gate_min) & (rng <= gate_max)
    if not mask.any():
        return float('nan')
    return float(np.max(vals[mask]))


def run_angle_scan(mode: str,
                   frames: int,
                   nfft: int,
                   baseline_csv: str,
                   angle_start: float,
                   angle_stop: float,
                   angle_step: float,
                   target_range: float,
                   gate: float) -> List[ScanResult]:
    """Perform an angle scan by repeatedly calling measure_rcs.py.

    Parameters
    ----------
    mode : str
        Measurement mode: 'baseline', 'off', or 'on'.
    frames : int
        Number of frames per measurement.
    nfft : int
        FFT length (power of two) used for range processing.
    baseline_csv : str
        Path to a baseline CSV. Required for modes other than 'baseline'.
    angle_start : float
        Start angle (degrees).
    angle_stop : float
        Stop angle (degrees) inclusive; scan stops at or just above this angle.
    angle_step : float
        Step size (degrees) between measurements.
    target_range : float
        Approximate distance to the tag (meters).
    gate : float
        Width of range gate for peak extraction (meters).

    Returns
    -------
    List[ScanResult]
        Results containing angle and peak amplitude for each measured angle.
    """
    # Validate baseline when necessary
    if mode != 'baseline' and not baseline_csv:
        raise RuntimeError("A baseline CSV must be provided for modes 'off' and 'on'.")
    if mode == 'baseline' and baseline_csv:
        # Ensure baseline is not used to subtract baseline from baseline measurement
        print("[rcs_angle_scan] Baseline CSV is ignored in baseline mode.")
        baseline_csv = None

    # Build list of angles
    if angle_step <= 0:
        raise ValueError("angle_step must be positive")
    angles = []
    ang = angle_start
    # Always include the final angle by checking slightly beyond
    while ang <= angle_stop + 1e-6:
        angles.append(round(ang, 5))
        ang += angle_step
    results: List[ScanResult] = []

    # Column to read from CSV depending on mode and baseline availability
    col = 'mag_db' if (baseline_csv is None or mode == 'baseline') else 'relative_rcs_db'

    print("\n[rcs_angle_scan] Starting scan...")
    print(f"Mode: {mode}, frames: {frames}, nfft: {nfft}, column: {col}")
    print(f"Angles: {angles}")
    # Suggest environment state
    if mode == 'baseline':
        print("Please remove the tag or turn it off completely for baseline capture.")
    elif mode == 'off':
        print("Ensure the tag is present but switches are OFF.")
    else:
        print("Ensure the tag is present and switches are ON.")

    for angle in angles:
        prompt = f"Rotate to {angle}° and press ENTER (or type 'q' to quit): "
        val = input(prompt).strip().lower()
        if val == 'q':
            print("Scan aborted by user.")
            break
        # Generate temporary filenames to avoid collisions
        out_csv = f"angle_scan_{mode}_{angle:.1f}.csv"
        mag_csv, rel_csv = run_measure_rcs(mode, frames, nfft, out_csv, baseline_csv)
        # Determine which file and column to extract
        file_to_read = mag_csv if col == 'mag_db' else rel_csv
        if file_to_read is None:
            print(f"[ERROR] Expected {col} output not found for angle {angle}.")
            results.append(ScanResult(angle, float('nan')))
            continue
        peak_db = load_peak_from_csv(file_to_read, col, target_range, gate)
        print(f"Angle {angle}°: Peak {col} = {peak_db:.2f} dB")
        results.append(ScanResult(angle, peak_db))

    return results


def plot_results(results: List[ScanResult], title: str) -> None:
    """Plot the scan results on a polar plot and show it."""
    # Extract angles and peaks
    angles = [math.radians(res.angle_deg) for res in results]
    peaks = [res.peak_db for res in results]
    # Convert peaks (in dB) to linear amplitude and normalise for plotting
    # If any peaks are NaN, treat them as very low
    peaks_lin = np.array([0.0 if np.isnan(p) else 10**(p / 20.0) for p in peaks])
    if peaks_lin.max() > 0:
        peaks_norm = peaks_lin / peaks_lin.max()
    else:
        peaks_norm = peaks_lin
    # Wrap around if first angle != last angle to close the plot
    if angles and angles[0] != angles[-1]:
        angles.append(angles[0])
        peaks_norm = np.append(peaks_norm, peaks_norm[0])
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='polar')
    ax.plot(angles, peaks_norm, marker='o')
    ax.set_rmax(1.1)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(title)
    plt.show()


def save_results(results: List[ScanResult], filename: str) -> None:
    """Save the scan results to a CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['angle_deg', 'peak_db'])
        for res in results:
            writer.writerow([f"{res.angle_deg:.4f}", f"{res.peak_db:.4f}"])
    print(f"Results saved to {filename}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive RCS vs angle scanner for TinyRad experiments.")
    parser.add_argument('--mode', choices=['baseline', 'off', 'on'], required=True,
                        help="Measurement mode. 'baseline' for environment only, 'off' for tag off, 'on' for tag on.")
    parser.add_argument('--frames', type=int, default=200,
                        help="Number of frames to average per angle (default: 200)")
    parser.add_argument('--nfft', type=int, default=4096,
                        help="FFT length for range processing (power of two). Default: 4096")
    parser.add_argument('--baseline', type=str, default=None,
                        help="Path to baseline CSV (needed for 'off' and 'on' modes)")
    parser.add_argument('--angle-start', type=float, default=0.0,
                        help="Start angle in degrees (default: 0)")
    parser.add_argument('--angle-stop', type=float, default=0.0,
                        help="Stop angle in degrees (default: 0)")
    parser.add_argument('--angle-step', type=float, default=5.0,
                        help="Angle increment in degrees (default: 5)")
    parser.add_argument('--tag-range', type=float, required=True,
                        help="Approximate distance to the tag in meters for gating")
    parser.add_argument('--gate', type=float, default=0.5,
                        help="Range gate width in meters for peak extraction (default: 0.5 m)")
    parser.add_argument('--out', type=str, default='angle_scan_results.csv',
                        help="Filename for the summary CSV (default: angle_scan_results.csv)")
    parser.add_argument('--title', type=str, default=None,
                        help="Custom title for the polar plot")
    return parser.parse_args()


def main():
    args = parse_args()
    title = args.title or f"RCS Angle Scan ({args.mode})"
    results = run_angle_scan(
        mode=args.mode,
        frames=args.frames,
        nfft=args.nfft,
        baseline_csv=args.baseline,
        angle_start=args.angle_start,
        angle_stop=args.angle_stop,
        angle_step=args.angle_step,
        target_range=args.tag_range,
        gate=args.gate
    )
    if results:
        # Save results to CSV
        save_results(results, args.out)
        # Plot
        plot_results(results, title)
    else:
        print("No results captured. Exiting.")


if __name__ == '__main__':
    main()