#!/usr/bin/env python3
"""
Enhanced live FMCW processing for TinyRad R‑fiducial tag experiments.

This script builds upon the basic ``live_polar_rcs.py`` example and adds a
number of useful signal processing features for TinyRad users:

* **Range profiling & RCS estimation** – the script computes the magnitude
  range spectrum in decibels and optionally subtracts a baseline
  environment measurement.  A simple RCS approximation is also reported
  by adding a ``40·log10(R)`` term to account for range losses in the
  radar equation【104724948150232†L169-L199】.
* **Range–Doppler processing** – when multiple chirps per frame are
  configured via ``--nloops``, a 2‑D FFT is performed across the fast
  time (range) and slow time (Doppler) axes, producing a range–velocity
  map similar to the example in ``AN24_06.py``.
* **Angle‑of‑arrival (AoA) estimation** – when sequential transmit
  antennas are enabled (``--enable-angle``), the script forms a virtual
  antenna array from the two TX illuminations and three RX channels.  A
  beamforming FFT across this virtual array yields a coarse estimate of
  the scatterer azimuth, which is plotted on a polar axis.  This is
  analogous to the digital beamforming example in ``AN24_07.py``.
* **Live line‑of‑sight tracking** – the strongest AoA peak within a
  specified range gate is identified each update and the azimuth angle
  is printed for situational awareness.  The polar plot visualises
  amplitude versus bearing, so the user can deduce where the tag is
  located.

Usage example:

    # baseline capture (no tag) for 200 frames
    python3 enhanced_live_rcs.py --mode baseline --frames 200 --out baseline.csv

    # live operation with relative RCS subtraction, Doppler and AoA
    python3 enhanced_live_rcs.py --mode on --ref baseline.csv \
        --nloops 16 --enable-doppler --enable-angle \
        --gate-range 1.8 --gate-width 0.5

The script requires the TinyRad Python API and matplotlib (for plotting).
It should reside in the ``Software/v6/Python`` folder so that
``Class/TinyRad`` can be imported without modification.
"""

import argparse
import csv
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Extend sys.path so we can import the TinyRad driver when run from the
# ``Software/v6/Python`` directory or any subfolder.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import Class.TinyRad as TinyRad
except ImportError as exc:
    print(
        "Error importing TinyRad library. Ensure this script resides in "
        "the TinyRad Software/v6/Python folder and that Class/TinyRad exists."
    )
    print(f"Original error: {exc}")
    sys.exit(1)


def configure_radar(nloops: int, enable_angle: bool) -> Tuple[object, dict]:
    """Configure the TinyRad board for FMCW measurements.

    Parameters
    ----------
    nloops : int
        Number of chirps per frame (slow‑time dimension) used for
        Doppler processing.  Must be at least 1.
    enable_angle : bool
        When True, sequential transmit antennas are enabled (TX1 and
        TX2) to form a virtual array for direction of arrival (DoA)
        estimation.  When False, only a single TX is used.

    Returns
    -------
    Brd : TinyRad.TinyRad
        Configured TinyRad board object.
    dCfg : dict
        Dictionary with the applied measurement parameters.
    """
    if nloops < 1:
        raise ValueError("nloops must be ≥ 1")
    Brd = TinyRad.TinyRad("Usb")
    Brd.BrdRst()
    Brd.RfRxEna()
    # Always use TX2 with full power; the TX sequence is set below
    Brd.RfTxEna(2, 100)
    # Build measurement configuration
    seq = [1, 2] if enable_angle else [1]
    dCfg = {
        "fStrt": 24.00e9,
        "fStop": 24.25e9,
        "TRampUp": 128e-6,
        # A slightly longer period is used when multiple loops are required
        "Perd": 0.25e-3,
        "N": 128,
        "Seq": seq,
        "CycSiz": 4,
        # FrmSiz is irrelevant here but must be ≥ nloops
        "FrmSiz": max(100, nloops),
        "FrmMeasSiz": nloops,
    }
    Brd.RfMeas(dCfg)
    return Brd, dCfg


def load_reference(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a reference range profile (baseline) from CSV.

    The CSV is expected to have two columns: range_m and mag_db.  Lines
    starting with '#' are ignored.  Returns two arrays: range axis and
    magnitude in dB.
    """
    rngs, mags = [], []
    with open(path, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if row[0] == "range_m":
                continue
            try:
                rngs.append(float(row[0]))
                mags.append(float(row[1]))
            except (ValueError, IndexError):
                continue
    return np.array(rngs), np.array(mags)


def compute_cross_section_db(mag_db: np.ndarray, rng_axis: np.ndarray) -> np.ndarray:
    """Approximate radar cross‑section from amplitude.

    A simple approximation for the radar cross‑section (in dBsm) can be
    obtained by adding 40·log10(R) to the measured amplitude in dB.  This
    compensates for the R‑4 dependence of the received power in the radar
    range equation【104724948150232†L169-L199】.  A constant term containing
    transmit power and antenna gains is ignored because it is fixed for a
    given experimental setup.  When the measured profile has already had
    a baseline subtracted (i.e. it represents a difference in returned
    power), this function yields a relative RCS.

    Parameters
    ----------
    mag_db : np.ndarray
        Magnitude of the range profile in decibels.
    rng_axis : np.ndarray
        Corresponding range axis in metres.

    Returns
    -------
    np.ndarray
        Cross‑section estimate in dBsm for each range bin.
    """
    # Avoid log10 of zero range
    rng = np.maximum(rng_axis, 1e-6)
    return mag_db + 40.0 * np.log10(rng)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Enhanced live FMCW processing with RCS, Doppler and AoA. "
            "Press Ctrl+C to stop."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "off", "on"],
        default="baseline",
        help="Label for the measurement. Only used for plot titles.",
    )
    parser.add_argument(
        "--ref",
        type=str,
        help="CSV file containing a baseline range profile to subtract (optional).",
    )
    parser.add_argument(
        "--nloops",
        type=int,
        default=1,
        help=(
            "Number of chirps per frame for Doppler processing (slow‑time dimension). "
            "Use 1 to disable Doppler."
        ),
    )
    parser.add_argument(
        "--nfft",
        type=int,
        default=4096,
        help="FFT length for range processing (power of two).",
    )
    parser.add_argument(
        "--nfft-doppler",
        type=int,
        default=256,
        help="FFT length for Doppler processing (power of two).",
    )
    parser.add_argument(
        "--nfft-angle",
        type=int,
        default=256,
        help="FFT length for angle‑of‑arrival processing (virtual array).",
    )
    parser.add_argument(
        "--enable-doppler",
        action="store_true",
        help="Enable range–Doppler processing and display.",
    )
    parser.add_argument(
        "--enable-angle",
        action="store_true",
        help="Enable angle‑of‑arrival estimation using sequential TX antennas.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=50,
        help="Number of frames to average per update. Each frame calls BrdGetData().",
    )
    parser.add_argument(
        "--gate-range",
        type=float,
        default=None,
        help="Approximate range to the tag (metres) for AoA and RCS extraction.",
    )
    parser.add_argument(
        "--gate-width",
        type=float,
        default=0.5,
        help="Width of the range gate (metres) used for AoA and cross‑section.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional pause in seconds between updates.",
    )
    args = parser.parse_args()

    # Validate options
    if args.enable_angle and args.nloops != 1:
        # When AoA is enabled, sequential TX is used with only one chirp per update
        print(
            "[Warn] Angle estimation currently supports only nloops=1. "
            "Setting nloops to 1."
        )
        args.nloops = 1

    # Configure radar
    Brd, dCfg = configure_radar(args.nloops, args.enable_angle)
    fs = Brd.Get("fs")
    N = int(Brd.Get("N"))
    NrChn = int(Brd.Get("NrChn"))
    # Chirp slope
    slope = (dCfg["fStop"] - dCfg["fStrt"]) / dCfg["TRampUp"]
    # Range axis for one‑sided FFT
    rng_axis = np.arange(args.nfft // 2) / float(args.nfft) * fs * 299792458.0 / (2.0 * slope)

    # Reference baseline
    ref_rng: Optional[np.ndarray] = None
    ref_mag: Optional[np.ndarray] = None
    if args.ref:
        if not os.path.isfile(args.ref):
            print(f"[Warn] Reference file {args.ref} not found – ignoring.")
        else:
            rr, mm = load_reference(args.ref)
            if rr.size > 0:
                ref_rng, ref_mag = rr, mm
                print(f"Loaded baseline from {args.ref}")
            else:
                print(f"[Warn] Reference file {args.ref} appears empty – ignoring.")

    # Create matplotlib figures
    plt.ion()
    fig = plt.figure(figsize=(8, 6))
    # Main range profile subplot
    ax_r = fig.add_subplot(221)
    line_r, = ax_r.plot([], [], lw=1)
    ax_r.set_xlabel("Range [m]")
    ax_r.set_ylabel("Amplitude [dB]")
    ax_r.grid(True)
    ax_r.set_title(f"Range profile ({args.mode})")

    # Doppler subplot
    if args.enable_doppler and args.nloops > 1:
        ax_d = fig.add_subplot(222)
        im_d = None  # to hold the image handle
        ax_d.set_xlabel("Velocity [m/s]")
        ax_d.set_ylabel("Range [m]")
        ax_d.set_title("Range–Doppler map")
    else:
        ax_d = None

    # Angle/AoA polar subplot
    if args.enable_angle:
        ax_p = fig.add_subplot(223, projection="polar")
        line_p, = ax_p.plot([], [], lw=2)
        ax_p.set_theta_zero_location("N")
        ax_p.set_theta_direction(-1)
        ax_p.set_title("AoA / RCS polar")
    else:
        ax_p = None

    # Frame accumulation buffers for Doppler
    if args.enable_doppler and args.nloops > 1:
        # Pre‑allocate a buffer for one update: shape (N, nloops)
        doppler_buf = np.zeros((N, args.nloops), dtype=complex)
        win_doppler_fast = np.hanning(N).reshape(N, 1)
        win_doppler_slow = np.hanning(args.nloops).reshape(1, args.nloops)
        win2d = win_doppler_fast * win_doppler_slow

        # Velocity axis (Doppler) – approximate using PRF = 1/Perd
        fc = (dCfg["fStrt"] + dCfg["fStop"]) / 2.0
        v_freq = np.arange(-args.nfft_doppler // 2, args.nfft_doppler // 2) / args.nfft_doppler * (1.0 / dCfg["Perd"])
        v_axis = v_freq * 299792458.0 / (2.0 * fc)
    else:
        doppler_buf = None
        v_axis = None

    # Window for range processing (all channels)
    win_range = np.hanning(N).reshape(N, 1)
    scale_win = float(np.sum(win_range))

    # Angle processing settings
    if args.enable_angle:
        # Create window for virtual array (across channels) for beamforming
        n_virtual = 2 * NrChn - 1
        win_ant = np.hanning(n_virtual)
        # Precompute angle axis in radians for the polar plot
        # arcsin argument: k / (NFFT_angle/2)
        angle_bins = np.arange(-args.nfft_angle // 2, args.nfft_angle // 2) / float(args.nfft_angle)
        angle_rad = np.arcsin(2.0 * angle_bins)
        # Mask invalid values (outside [-1,1]) by clipping
        angle_rad = np.clip(angle_rad, -np.pi / 2, np.pi / 2)
    else:
        n_virtual = 0
        win_ant = None
        angle_rad = None

    # Determine gate indices once when gate range is provided
    gate_idx = None
    if args.gate_range is not None:
        gate_min = args.gate_range - args.gate_width / 2.0
        gate_max = args.gate_range + args.gate_width / 2.0
        gate_idx = np.where((rng_axis >= gate_min) & (rng_axis <= gate_max))[0]
        if gate_idx.size == 0:
            print("[Warn] Range gate does not overlap any bins; AoA and RCS extraction disabled.")
            gate_idx = None

    try:
        while True:
            # Acquire the specified number of frames and accumulate
            mag_sum = np.zeros(args.nfft // 2, dtype=float)
            # Reset doppler buffer
            if doppler_buf is not None:
                doppler_buf[:] = 0.0
            for _ in range(args.frames):
                Data = Brd.BrdGetData()
                # For angle processing, Data length includes two TX sequences
                # and one loop only.  For Doppler, Data contains nloops chirps
                # but only one TX.
                if args.enable_angle:
                    # Data shape: TinyRad AN24_07‑style sequential TX: 2*N rows
                    # Row 0     : frame counter
                    # Rows 1..N‑1   : TX1 samples (N‑1 useful fast‑time samples)
                    # Rows N+1..2*N‑1 : TX2 samples (N‑1 useful fast‑time samples)
                    # Use the same slicing pattern as AN24_07 to ensure both halves
                    # have identical fast‑time length.
                    tx1 = Data[1:N, :]
                    tx2 = Data[N+1:, :]
                    # Construct virtual array: first all channels from TX1, then
                    # all but the first channel from TX2 to remove the overlapping
                    # virtual element.
                    virt = np.concatenate((tx1, tx2[:, 1:]), axis=1)
                    # Apply a range window matched to the number of fast‑time samples
                    n_fast = virt.shape[0]
                    win_aoa = np.hanning(n_fast).reshape(n_fast, 1)
                    scale_aoa = float(np.sum(win_aoa))
                    # Range FFT on the virtual array
                    rp = 2.0 * np.fft.fft(virt * win_aoa, n=args.nfft, axis=0) / scale_aoa * Brd.FuSca
                    rp = rp[: args.nfft // 2, :]
                    # Average magnitude across channels for baseline subtraction
                    mag_ch = np.abs(rp).mean(axis=1)
                    mag_sum += mag_ch
                    # For AoA: accumulate virtual array at gate bins
                    if gate_idx is not None:
                        # Sum over gate bins to get a single complex value per channel
                        # shape of rp_gate: (len(gate_idx), n_virtual)
                        rp_gate = rp[gate_idx, :]
                        # Average across range gate to reduce noise
                        rp_gate_mean = rp_gate.mean(axis=0)
                        # Apply antenna window
                        rp_gate_win = rp_gate_mean * win_ant
                        # Beamforming FFT across virtual array
                        angle_spec = np.fft.fftshift(
                            np.fft.fft(rp_gate_win, n=args.nfft_angle)
                        )
                        # Convert to magnitude
                        angle_mag = np.abs(angle_spec)
                        # Normalise for plotting
                        if angle_mag.max() > 0:
                            angle_norm = angle_mag / angle_mag.max()
                        else:
                            angle_norm = angle_mag
                        line_p.set_data(angle_rad, angle_norm)
                        # Estimate line‑of‑sight angle
                        max_idx = int(np.argmax(angle_norm))
                        los_deg = float(np.degrees(angle_rad[max_idx]))
                        print(f"LoS ≈ {los_deg:+.1f}°")
                        # Draw polar
                        ax_p.set_rlim(0, 1.1)
                        ax_p.figure.canvas.draw()
                        ax_p.figure.canvas.flush_events()
                elif args.enable_doppler and args.nloops > 1:
                    # Doppler mode (single TX, multiple chirps)
                    # Reshape fast time samples for the chosen channel (RX0)
                    data_raw = Data[1:, 0]
                    if data_raw.size != N * args.nloops:
                        # Unexpected shape, skip this frame
                        continue
                    data_mat = np.reshape(data_raw, (N, args.nloops), order="F")
                    doppler_buf += data_mat
                else:
                    # Simple single‑channel range profile
                    data_ch = Data[1:, 0].reshape(N, 1)
                    rp = 2.0 * np.fft.fft(data_ch * win_range, n=args.nfft, axis=0) / scale_win * Brd.FuSca
                    rp = rp[: args.nfft // 2, 0]
                    mag_ch = np.abs(rp)
                    mag_sum += mag_ch

            # Compute average magnitude and convert to dB
            mag_mean = mag_sum / float(args.frames)
            mag_db = 20.0 * np.log10(mag_mean + 1e-12)
            # Subtract baseline if provided
            if ref_rng is not None and ref_mag is not None:
                # Interpolate baseline onto our range axis
                ref_interp = np.interp(rng_axis, ref_rng, ref_mag, left=ref_mag[0], right=ref_mag[-1])
                diff_db = mag_db - ref_interp
                mag_for_plot = diff_db
            else:
                mag_for_plot = mag_db
            # Plot range profile
            line_r.set_data(rng_axis, mag_for_plot)
            ax_r.set_xlim(0.0, rng_axis[-1])
            # Also compute cross‑section estimate and optionally display on secondary y‑axis
            cs_db = compute_cross_section_db(mag_for_plot, rng_axis)
            # Only plot cross‑section when gate is set
            if gate_idx is not None:
                rng_gate = rng_axis[gate_idx]
                cs_gate = cs_db[gate_idx]
                # Update polar amplitude if AoA not enabled
                if ax_p is not None and not args.enable_angle:
                    # Create evenly spaced angles around 0–2π
                    theta = np.linspace(0.0, 2.0 * np.pi, len(rng_gate), endpoint=False)
                    amp = cs_gate - cs_gate.min()
                    amp_norm = amp / (amp.max() + 1e-12)
                    line_p.set_data(theta, amp_norm)
                    ax_p.set_rlim(0, 1.1)
            # Draw range plot
            ax_r.figure.canvas.draw()
            ax_r.figure.canvas.flush_events()

            # Compute and display range–Doppler map if enabled
            if doppler_buf is not None:
                # Apply 2D window and compute 2D FFT
                rd = 2.0 * np.fft.fft(
                    doppler_buf * win_doppler_fast * win_doppler_slow,
                    n=args.nfft,
                    axis=0,
                ) / scale_win * Brd.FuSca
                rd = rd[: args.nfft // 2, :]
                rd2 = np.fft.fftshift(
                    np.fft.fft(rd, n=args.nfft_doppler, axis=1), axes=1
                )
                rd_mag = 20.0 * np.log10(np.abs(rd2) + 1e-12)
                # Plot as image
                if im_d is None:
                    im_d = ax_d.imshow(
                        rd_mag.T,
                        aspect="auto",
                        origin="lower",
                        extent=[rng_axis[0], rng_axis[-1], v_axis[0], v_axis[-1]],
                        cmap="viridis",
                    )
                    ax_d.figure.colorbar(im_d, ax=ax_d, label="Mag [dB]")
                else:
                    im_d.set_data(rd_mag.T)
                    im_d.set_extent([rng_axis[0], rng_axis[-1], v_axis[0], v_axis[-1]])
                ax_d.figure.canvas.draw()
                ax_d.figure.canvas.flush_events()

            if args.sleep > 0.0:
                time.sleep(args.sleep)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Cleaning up...")
    finally:
        plt.ioff()
        plt.show(block=False)
        del Brd


if __name__ == "__main__":
    main()