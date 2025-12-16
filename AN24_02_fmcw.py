# AN24_02 -- FMCW Basics  (Enhanced)
#
# Upgrades:
#   - Compute and use range axis in *meters*
#   - Proper axis labels
#   - Smoothed range spectrum
#   - Automatic peak detection + labels
#   - CSV logging of range profiles (throttled)

import sys, os
import csv
from datetime import datetime

import numpy as np
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg

# --------------------------------------------------------------------------
# Path setup so "Class.TinyRad" can be found whether you run from Appnotes
# or from the Python root folder.
# --------------------------------------------------------------------------
this_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(this_dir)
sys.path.append(os.path.abspath(os.path.join(this_dir, "..")))
import Class.TinyRad as TinyRad  # noqa: E402

# --------------------------------------------------------------------------
# Display / processing configuration
# --------------------------------------------------------------------------
Disp_FrmNr = 1
Disp_TimSig = 1
Disp_RP = 1

c0 = 3e8  # speed of light [m/s]

NUM_FRAMES = 1000          # how many frames to grab in this run
NFFT = 2 ** 14             # FFT size for range processing

SMOOTH_WIN = 7             # moving-average window (samples) for smoothing
NUM_LABELED_PEAKS = 3      # max number of peaks to label per channel
MIN_PEAK_SEP_M = 0.5       # min separation between labeled peaks [m]
PEAK_THRESHOLD_DB = -110   # don't label peaks below this level

LOG_CSV = False                         # turn CSV logging on/off
LOG_EVERY_N_FRAMES = 10                # log one frame out of this many
MAX_LOG_FRAMES = 200                   # safety cap to keep file size sane
CSV_FILENAME = f"AN24_02_range_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


# --------------------------------------------------------------------------
# Simple 1D moving-average smoother
# --------------------------------------------------------------------------
def smooth_1d(x, win):
    if win is None or win <= 1:
        return x
    win = int(win)
    if win < 2:
        return x
    kernel = np.ones(win, dtype=float) / float(win)
    return np.convolve(x, kernel, mode="same")


# --------------------------------------------------------------------------
# Setup plotting windows
# --------------------------------------------------------------------------
if Disp_TimSig > 0:
    WinTim = pg.GraphicsLayoutWidget(title="Time signals", show=True)
    WinTim.setBackground((255, 255, 255))
    WinTim.resize(1000, 600)

    PltTim = WinTim.addPlot(title="TimSig", col=0, row=0)
    PltTim.showGrid(x=True, y=True)
    PltTim.setLabel("bottom", "Sample index")
    PltTim.setLabel("left", "Amplitude", units="a.u.")

if Disp_RP > 0:
    WinRP = pg.GraphicsLayoutWidget(title="Range Profile", show=True)
    WinRP.setBackground((255, 255, 255))
    WinRP.resize(1000, 600)

    PltRP = WinRP.addPlot(title="Range", col=0, row=0)
    PltRP.showGrid(x=True, y=True)
    PltRP.setLabel("bottom", "Range", units="m")
    PltRP.setLabel("left", "Amplitude", units="dB")


# --------------------------------------------------------------------------
# Setup Connection
# --------------------------------------------------------------------------
Brd = TinyRad.TinyRad('Usb')
Brd.BrdRst()

# --------------------------------------------------------------------------
# Software Version
# --------------------------------------------------------------------------
Brd.BrdDispSwVers()

# --------------------------------------------------------------------------
# Configure Receiver / Transmitter
# --------------------------------------------------------------------------
Brd.RfRxEna()
TxPwr = 100
Brd.RfTxEna(2, TxPwr)  # antenna index 2, full power

# --------------------------------------------------------------------------
# Configure Measurements
#   Cfg.Perd: time between measurements
#   Cfg.N:    number of samples per ramp
# --------------------------------------------------------------------------
dCfg = dict()
dCfg['fStrt'] = 24.00e9
dCfg['fStop'] = 24.25e9
dCfg['TRampUp'] = 128e-6
dCfg['Perd'] = 0.2e-3
dCfg['N'] = 128
dCfg['Seq'] = [1]
dCfg['CycSiz'] = 4
dCfg['FrmSiz'] = 100
dCfg['FrmMeasSiz'] = 1

Brd.RfMeas(dCfg)

# --------------------------------------------------------------------------
# Read actual configuration from board
# --------------------------------------------------------------------------
N = int(Brd.Get('N'))
NrChn = int(Brd.Get('NrChn'))
fs = Brd.Get('fs')

print("Samples per ramp N      :", N)
print("Number of RX channels   :", NrChn)
print("Sampling rate fs [Hz]   :", fs)

# --------------------------------------------------------------------------
# Check data rate
# --------------------------------------------------------------------------
DataRate = 16 * NrChn * N * dCfg['FrmMeasSiz'] / (dCfg['FrmSiz'] * dCfg['Perd'])
print('DataRate: ', (DataRate / 1e6), ' MBit/s')

# --------------------------------------------------------------------------
# Configure signal processing: window, range axis in meters
# --------------------------------------------------------------------------
Win = Brd.hanning(N - 1, NrChn)
ScaWin = np.sum(Win[:, 0])

# chirp slope
kf = (dCfg['fStop'] - dCfg['fStrt']) / dCfg['TRampUp']

# range axis in meters (one-sided FFT)
vRange = np.arange(NFFT // 2) / NFFT * fs * c0 / (2 * kf)
dR = vRange[1] - vRange[0]
Rmax = vRange[-1]
print(f"Range resolution dR  ≈ {dR:.3f} m")
print(f"Max unambiguous Rmax ≈ {Rmax:.1f} m")

# --------------------------------------------------------------------------
# Generate curves for plots
# --------------------------------------------------------------------------
if Disp_TimSig:
    n = np.arange(int(N))
    CurveTim = []
    for IdxChn in range(NrChn):
        CurveTim.append(PltTim.plot(pen=pg.intColor(IdxChn, hues=NrChn)))

if Disp_RP:
    CurveRP = []
    PeakScatter = []
    PeakText = []  # list of lists of TextItems per channel

    for IdxChn in range(NrChn):
        # main range profile curve
        CurveRP.append(PltRP.plot(pen=pg.intColor(IdxChn, hues=NrChn)))

        # scatter plot for peak markers
        sp = pg.ScatterPlotItem(
            size=8,
            pen=None,
            brush=pg.intColor(IdxChn, hues=NrChn)
        )
        PltRP.addItem(sp)
        PeakScatter.append(sp)

        # text labels for peaks
        PeakText.append([])

    # Optional: set a sensible x-range for lab work (e.g., 0–60 m)
    PltRP.setXRange(0, min(60, Rmax), padding=0.02)

# --------------------------------------------------------------------------
# CSV logging setup
# --------------------------------------------------------------------------
csv_file = None
csv_writer = None
logged_frames = 0

if LOG_CSV:
    csv_file = open(CSV_FILENAME, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_idx", "channel", "range_m", "magnitude_dB"])
    print(f"CSV logging enabled -> {CSV_FILENAME}")
else:
    print("CSV logging disabled.")


# --------------------------------------------------------------------------
# Helper: detect and select strongest peaks
# --------------------------------------------------------------------------
min_peak_sep_bins = max(1, int(MIN_PEAK_SEP_M / dR))


def find_strong_peaks(mag_db):
    """Return indices of strongest separated peaks in a 1D dB array."""
    mag_db = np.asarray(mag_db)

    # local maxima (simple 3-point check)
    mask = (mag_db[1:-1] > mag_db[:-2]) & (mag_db[1:-1] > mag_db[2:])
    cand_idx = np.where(mask)[0] + 1

    # threshold
    cand_idx = cand_idx[mag_db[cand_idx] > PEAK_THRESHOLD_DB]
    if cand_idx.size == 0:
        return []

    # sort by amplitude, descending
    cand_idx = cand_idx[np.argsort(mag_db[cand_idx])[::-1]]

    selected = []
    for idx in cand_idx:
        if all(abs(idx - s) >= min_peak_sep_bins for s in selected):
            selected.append(int(idx))
        if len(selected) >= NUM_LABELED_PEAKS:
            break
    return selected


# --------------------------------------------------------------------------
# Main acquisition / display loop
# --------------------------------------------------------------------------
try:
    for frame_idx in range(NUM_FRAMES):

        Data = Brd.BrdGetData()

        if Disp_FrmNr > 0:
            FrmCntr = Data[0, :]
            print("FrmCntr:", FrmCntr)

        # ---------------- Time-domain plotting ----------------
        if Disp_TimSig > 0:
            for IdxChn in range(NrChn):
                CurveTim[IdxChn].setData(n[1:], Data[1:, IdxChn])

        # ---------------- Range profile processing ------------ 
        if Disp_RP > 0:
            # FFT over fast-time -> beat frequency -> range
            RP = 2 * np.fft.fft(Data[1:, :] * Win, n=NFFT, axis=0) / ScaWin * Brd.FuSca
            RP = RP[:NFFT // 2, :]

            do_log_this_frame = (
                LOG_CSV
                and (frame_idx % LOG_EVERY_N_FRAMES == 0)
                and (logged_frames < MAX_LOG_FRAMES)
            )

            for IdxChn in range(NrChn):
                # magnitude, smoothing, dB
                mag_lin = np.abs(RP[:, IdxChn])
                mag_lin_s = smooth_1d(mag_lin, SMOOTH_WIN)
                mag_db = 20 * np.log10(mag_lin_s + 1e-12)

                # update curve
                CurveRP[IdxChn].setData(vRange, mag_db)

                # ---- peak detection + labeling ----
                peak_idx = find_strong_peaks(mag_db)

                # update scatter markers
                peak_ranges = vRange[peak_idx] if len(peak_idx) else []
                peak_mags = mag_db[peak_idx] if len(peak_idx) else []
                PeakScatter[IdxChn].setData(peak_ranges, peak_mags)

                # remove old text labels
                for txt in PeakText[IdxChn]:
                    PltRP.removeItem(txt)
                PeakText[IdxChn].clear()

                # add new text labels
                for r, m in zip(peak_ranges, peak_mags):
                    txt = pg.TextItem(text=f"{r:.1f} m", anchor=(0.5, 1.2), color=(0, 0, 0))
                    txt.setPos(float(r), float(m))
                    PltRP.addItem(txt)
                    PeakText[IdxChn].append(txt)

                # ---- CSV logging (smoothed magnitude) ----
                if do_log_this_frame and csv_writer is not None:
                    for r, m in zip(vRange, mag_db):
                        csv_writer.writerow([frame_idx, IdxChn, float(r), float(m)])

            if do_log_this_frame:
                logged_frames += 1

        # allow Qt to update windows
        if Disp_TimSig > 0 or Disp_RP > 0:
            QtGui.QGuiApplication.processEvents()

finally:
    # Clean up nicely
    if csv_file is not None:
        csv_file.close()
        print("CSV file closed.")
    del Brd
