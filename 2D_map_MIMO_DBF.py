#!/usr/bin/env python3
# 2D_map_MIMO_DBF.py - TinyRad 24 GHz MIMO DBF with live range-angle map
# - 2-Tx MIMO: dCfg['Seq'] = [1, 2]
# - Virtual 7-element array (2*NrChn - 1)
# - Runs until Ctrl+C
# - Shows:
#     * Time-domain signals (virtual channels)
#     * Range profiles
#     * Range–angle (cross-range) heatmap with multiple peaks

import sys, os
sys.path.append("../")
import Class.TinyRad as TinyRad
import numpy as np
from pyqtgraph.Qt import QtGui, QtWidgets
import pyqtgraph as pg

# ---------------------- Config flags ----------------------
Disp_FrmNr  = 1      # print frame counter
Disp_TimSig = 1      # show time-domain signals
Disp_RP     = 1      # show range profiles   <<< turned ON
Disp_JOpt   = 1      # show range-angle heatmap

# -----------------------------------------------------------------------------
# Default parameters for the interactive range–angle display
RMin_default = 0.5       # initial minimum range [m]
RMax_default = 5.0       # initial maximum range [m]

# Scaling configuration for the range–angle heatmap
# Relative mode works in dB with a fixed dynamic range around the peak.
REL_DYN_RANGE = 25.0     # dynamic range for relative dB scaling [dB]

# GUI widget placeholders (populated when Disp_JOpt is enabled)
_spnRMin     = None
_spnRMax     = None
_chkInvert   = None
_chkAbsScale = None

# Peak detection config
PEAK_THRESH_dB = -6  # dB below max
MAX_PEAKS      = 20  # max number of scatter points

# Physical constant
c0 = 1 / np.sqrt(8.85e-12 * 4 * np.pi * 1e-7)

# ---------------------- Plot setup ------------------------
if Disp_TimSig > 0:
    WinTim = pg.GraphicsLayoutWidget(title="Time Signals (Virtual Channels)", show=True)
    WinTim.setBackground((255, 255, 255))
    WinTim.resize(1000, 600)

    PltTim = WinTim.addPlot(title="Time Signals", col=0, row=0)
    PltTim.showGrid(x=True, y=True)

if Disp_RP > 0:
    WinRP = pg.GraphicsLayoutWidget(title="Range Profiles (Virtual Channels)", show=True)
    WinRP.setBackground((255, 255, 255))
    WinRP.resize(1000, 600)

    PltRP = WinRP.addPlot(title="Range", col=0, row=0)
    PltRP.showGrid(x=True, y=True)

if Disp_JOpt:
    # Interactive Range–Angle (Cross Range) window with controls
    CrossWin = QtWidgets.QWidget()
    CrossWin.setWindowTitle("Range–Angle (Cross Range) Plot")
    vlayout = QtWidgets.QVBoxLayout(CrossWin)

    # PlotItem used as the view for the ImageView so we can control axes
    View = pg.PlotItem()
    View.setLabel('left', 'R', units='m')
    View.setLabel('bottom', 'Angle', units='deg')

    # x‑axis ticks in degrees, positioned at u = sin(theta)
    angles_deg = np.arange(-60, 61, 15)
    u_ticks = [(np.sin(np.deg2rad(a)), f"{a}") for a in angles_deg]
    bottom_axis = View.getAxis('bottom')
    bottom_axis.setTicks([u_ticks])

    # ImageView for the heatmap
    Img = pg.ImageView(view=View)
    Img.ui.roiBtn.hide()
    Img.ui.menuBtn.hide()
    Img.getHistogramWidget().gradient.loadPreset('flame')
    vlayout.addWidget(Img)

    # --- Control panel ---------------------------------------------------
    ctrlWidget = QtWidgets.QWidget()
    ctrlLayout = QtWidgets.QHBoxLayout(ctrlWidget)

    # RMin control
    _spnRMin = QtWidgets.QDoubleSpinBox()
    _spnRMin.setRange(0.0, 20.0)
    _spnRMin.setSingleStep(0.1)
    _spnRMin.setDecimals(2)
    _spnRMin.setValue(RMin_default)
    _spnRMin.setPrefix("R min: ")
    _spnRMin.setSuffix(" m")
    ctrlLayout.addWidget(_spnRMin)

    # RMax control
    _spnRMax = QtWidgets.QDoubleSpinBox()
    _spnRMax.setRange(0.0, 20.0)
    _spnRMax.setSingleStep(0.1)
    _spnRMax.setDecimals(2)
    _spnRMax.setValue(RMax_default)
    _spnRMax.setPrefix("R max: ")
    _spnRMax.setSuffix(" m")
    ctrlLayout.addWidget(_spnRMax)

    # Invert‑axis checkbox
    # Unchecked  → R increases from bottom to top (conventional)
    # Checked    → flip the R axis (R increases from top to bottom)
    _chkInvert = QtWidgets.QCheckBox("Invert R axis")
    _chkInvert.setChecked(False)
    ctrlLayout.addWidget(_chkInvert)

    # Toggle between relative dB map and absolute power (W) map
    _chkAbsScale = QtWidgets.QCheckBox("Absolute power (W)")
    _chkAbsScale.setChecked(False)
    ctrlLayout.addWidget(_chkAbsScale)

    ctrlLayout.addStretch(1)
    vlayout.addWidget(ctrlWidget)

    # Scatter plot for peaks
    peakScatter = pg.ScatterPlotItem(size=10, pen='w', brush='w')
    View.addItem(peakScatter)

    CrossWin.resize(900, 700)
    CrossWin.show()

# ---------------------- Board setup -----------------------
Brd = TinyRad.TinyRad('Usb')   # use 'RadServe','127.0.0.1' if needed

Brd.BrdRst()
Brd.BrdDispSwVers()

Brd.RfRxEna()
TxPwr = 100
Brd.RfTxEna(2, TxPwr)          # enable Tx chain; sequence below sets Tx1,Tx2

# Measurement configuration (MIMO, Tx1 & Tx2)
dCfg = dict()
dCfg['fStrt']      = 24.00e9
dCfg['fStop']      = 24.25e9
dCfg['TRampUp']    = 256e-6
dCfg['Perd']       = 0.4e-3
dCfg['N']          = 256
dCfg['Seq']        = [1, 2]    # <<< MIMO: Tx1 & Tx2 (AN24_07 style)
dCfg['CycSiz']     = 4
dCfg['FrmSiz']     = 128
dCfg['FrmMeasSiz'] = 1

Brd.RfMeas(dCfg)

# ---------------------- Read config -----------------------
NrChn = int(Brd.Get('NrChn'))   # physical Rx channels (4)
N     = int(Brd.Get('N'))       # samples per chirp
fs    = Brd.Get('fs')
kf    = Brd.Get('kf')

# ---------------------- Signal processing setup ----------
# Range processing
NFFT = 2**16

# Virtual array size (2*NrChn - 1 = 7)
NrVirtChn = 2 * NrChn - 1

# Window over fast-time (N-1) x NrVirtChn
Win2D  = Brd.hanning(N-1, NrVirtChn)
ScaWin = np.sum(Win2D[:, 0])

vRange = np.arange(NFFT) / NFFT * fs * c0 / (2 * kf)

# Range window
RMin = 0.5
RMax = 5
RMinIdx = np.argmin(np.abs(vRange - RMin))
RMaxIdx = np.argmin(np.abs(vRange - RMax))
vRangeExt = vRange[RMinIdx:RMaxIdx]

# Angular (DBF) dimension
NFFTAnt   = 256
WinAnt2D  = Brd.hanning(NrVirtChn, len(vRangeExt))
ScaWinAnt = np.sum(WinAnt2D[:, 0])
WinAnt2D  = WinAnt2D.transpose()

# Plot curves
if Disp_TimSig:
    n = np.arange(int(N))
    CurveTim = []
    for idx in range(NrVirtChn):
        CurveTim.append(PltTim.plot(pen=pg.intColor(idx, hues=NrVirtChn)))

if Disp_RP:
    CurveRP = []
    for idx in range(NrVirtChn):
        CurveRP.append(PltRP.plot(pen=pg.intColor(idx, hues=NrVirtChn)))

# ---------------------- Main measurement loop ------------
try:
    Cycles = 0
    while True:
        # Data layout (AN24_07 style):
        # row 0: frame counter(s)
        # rows 1..N:   Tx1 data (N-1 samples used)
        # rows N+1..:  Tx2 data (N-1 samples used)
        Data = Brd.BrdGetData()

        if Disp_FrmNr > 0:
            FrmCntr = Data[0, :]
            print("FrmCntr:", FrmCntr)

        # Build virtual array: [Tx1*Rx1..Rx4, Tx2*Rx2..Rx4]  → 7 channels
        # Drop overlapping element (Tx2,Rx1)
        DataV = np.concatenate((Data[1:N, :], Data[N+1:, 1:]), axis=1)

        # Time-domain plot (virtual channels)
        if Disp_TimSig:
            for idx in range(NrVirtChn):
                CurveTim[idx].setData(n[1:], DataV[:, idx])

        # --- Range window from GUI controls (for both range profiles and
        #     range–angle map).  If the controls are not available, fall back
        #     to the default limits.
        try:
            RMin_val = float(_spnRMin.value())
            RMax_val = float(_spnRMax.value())
            if RMax_val <= RMin_val:
                RMax_val = RMin_val + 0.1
        except Exception:
            RMin_val = RMin_default
            RMax_val = RMax_default

        RMinIdx = int(np.argmin(np.abs(vRange - RMin_val)))
        RMaxIdx = int(np.argmin(np.abs(vRange - RMax_val)))
        if RMaxIdx <= RMinIdx:
            RMaxIdx = RMinIdx + 1
        vRangeExt = vRange[RMinIdx:RMaxIdx]

        # Range FFT (per virtual channel)
        RP = 2 * np.fft.fft(DataV * Win2D, n=NFFT, axis=0) / ScaWin * Brd.FuSca
        RP = RP[RMinIdx:RMaxIdx, :]

        if Disp_RP:
            for idx in range(NrVirtChn):
                CurveRP[idx].setData(
                    vRangeExt,
                    20 * np.log10(np.abs(RP[:, idx]) + 1e-12)
                )

        if Disp_JOpt:
            # DBF across virtual antenna dimension.  Use a dynamic antenna
            # window whose range dimension matches the current vRangeExt.
            WinAnt2D_dyn = Brd.hanning(NrVirtChn, len(vRangeExt)).transpose()
            ScaWinAnt_dyn = np.sum(WinAnt2D_dyn[:, 0])

            JOpt = np.fft.fftshift(
                np.fft.fft(RP * WinAnt2D_dyn, NFFTAnt, axis=1) / ScaWinAnt_dyn,
                axes=1
            )

            # Convert to dB (for relative mode and peak detection)
            JdB = 20 * np.log10(np.abs(JOpt) + 1e-12)
            try:
                abs_mode = _chkAbsScale.isChecked()  # True → plot absolute power (W)
            except Exception:
                abs_mode = False

            if abs_mode:
                # Absolute POWER map (Watts, up to a constant factor).
                # We stay in linear |JOpt|**2 and let ImageView auto‑scale
                # the colour bar each frame.
                Jdisp = np.abs(JOpt) ** 2
                levels = None  # autoLevels in setImage
            else:
                # Relative (per‑frame) 0 dB normalisation in the dB domain.
                JMax = np.max(JdB)
                Jdisp = JdB - JMax
                Jdisp[Jdisp < -REL_DYN_RANGE] = -REL_DYN_RANGE
                levels = (-REL_DYN_RANGE, 0.0)

            # --- multiple‑peak detection ---
            # Always work in a relative dB domain so PEAK_THRESH_dB is
            # interpreted as "dB below the frame peak" regardless of what
            # is displayed.
            Jrel = JdB - np.max(JdB)
            mask = Jrel > PEAK_THRESH_dB
            r_idx, u_idx = np.where(mask)

            if r_idx.size > 0:
                # limit to top‑K strongest bins
                if r_idx.size > MAX_PEAKS:
                    flat_vals = Jrel[mask].ravel()
                    top_idx = np.argsort(flat_vals)[-MAX_PEAKS:]
                    r_idx = r_idx[top_idx]
                    u_idx = u_idx[top_idx]

                # Map indices → (u, R)
                u_vals = -1.0 + (u_idx * 2.0 / NFFTAnt)
                r_vals = vRangeExt[r_idx]
                peakScatter.setData(u_vals, r_vals)
            else:
                peakScatter.setData([], [])

            # Update image (range–angle map) with optional inverted R axis
            try:
                invert_y = _chkInvert.isChecked()
            except Exception:
                invert_y = False

            # Map range bins uniformly over [RMin_val, RMax_val]
            nR = max(1, vRangeExt.shape[0])
            pos_y = RMin_val
            scale_y = (RMax_val - RMin_val) / nR

            if levels is None:
                # Absolute power → let ImageView choose levels from the data
                Img.setImage(
                    Jdisp.T,
                    pos=[-1, pos_y],
                    scale=[2.0 / NFFTAnt, scale_y],
                    autoLevels=True,
                )
            else:
                # Relative dB → fixed dynamic range [‑REL_DYN_RANGE, 0]
                Img.setImage(
                    Jdisp.T,
                    pos=[-1, pos_y],
                    scale=[2.0 / NFFTAnt, scale_y],
                    levels=levels,
                )

            # Set visible y‑range and (optionally) invert the axis so the
            # checkbox immediately affects the orientation of the "Range" axis
            # labels.
            vb = View.getViewBox()
            vb.setYRange(RMin_val, RMax_val, padding=0.0)
            vb.invertY(bool(invert_y))
            View.setRange(xRange=(-1.0, 1.0))
            View.setAspectLocked(False)

        QtGui.QGuiApplication.processEvents()
        Cycles += 1

except KeyboardInterrupt:
    print("Measurement stopped by user (Ctrl+C).")

finally:
    del Brd
