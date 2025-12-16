# Calculate digital beamforming with one TX antenna

# (1) Connect to Radarbook2 with ADF24 Frontend
# (2) Enable Supply
# (3) Configure RX
# (4) Configure TX
# (5) Start Measurements
# (6) Configure signal processing
# (7) Calculate DBF algorithm

import sys, os
sys.path.append("../")
import  Class.TinyRad as TinyRad
import  time as time
import  numpy as np
from    pyqtgraph.Qt import QtGui, QtCore
import  pyqtgraph as pg

# Configure script
Disp_FrmNr = 1
Disp_TimSig = 1     # display time signals
Disp_RP = 0      # display range profile
Disp_JOpt = 1      # display cost function for DBF

c0 = 1/np.sqrt(8.85e-12*4*np.pi*1e-7)

if Disp_TimSig > 0:  

    WinTim = pg.GraphicsLayoutWidget(title="Time signals", show=True)
    WinTim.setBackground((255, 255, 255))
    WinTim.resize(1000,600)

    PltTim = WinTim.addPlot(title="TimSig", col=0, row=0)
    PltTim.showGrid(x=True, y=True)

if Disp_RP > 0:
    WinRP = pg.GraphicsLayoutWidget(title="Range Profile", show=True)
    WinRP.setBackground((255, 255, 255))
    WinRP.resize(1000,600)

    PltRP = WinRP.addPlot(title="Range", col=0, row=0)
    PltRP.showGrid(x=True, y=True)

if Disp_JOpt:
    if Disp_TimSig == 0 and Disp_RP == 0:
        WinJOpt = pg.GraphicsLayoutWidget(title="Cross Range Plot", show=False)

    View = pg.PlotItem(title='Cross Range Plot')
    View.setLabel('left', 'R', units='m')
    View.setLabel('bottom', 'Angle', units='deg')   # label in degrees

    # --- add this: tick labels in degrees, positioned at u = sin(theta) ---
    angles_deg = np.arange(-60, 61, 15)  # choose the angles you care about
    u_ticks = [(np.sin(np.deg2rad(a)), f"{a}") for a in angles_deg]

    bottom_axis = View.getAxis('bottom')
    bottom_axis.setTicks([u_ticks])

    Img = pg.ImageView(view=View)
    Img.show()
    Img.ui.roiBtn.hide()
    Img.ui.menuBtn.hide()
    #Img.ui.histogram.hide()
    Img.getHistogramWidget().gradient.loadPreset('flame')
    
    # --- add this: a scatter plot for peaks ---
    peakScatter = pg.ScatterPlotItem(size=10, pen='w', brush='w')
    View.addItem(peakScatter)

#--------------------------------------------------------------------------
# Setup Connection
#--------------------------------------------------------------------------
Brd = TinyRad.TinyRad('Usb')

Brd.BrdRst()

#--------------------------------------------------------------------------
# Software Version
#--------------------------------------------------------------------------
Brd.BrdDispSwVers()

#--------------------------------------------------------------------------
# Configure Receiver
#--------------------------------------------------------------------------
Brd.RfRxEna()
TxPwr =   100

#--------------------------------------------------------------------------
# Configure Transmitter (Antenna 0 - 2, Pwr 0 - 100)
#--------------------------------------------------------------------------
Brd.RfTxEna(2, TxPwr)

#--------------------------------------------------------------------------
# Configure Measurements
# Cfg.Perd: time between measuremnets: must be greater than 1 us*N + 10us
# Cfg.N: number of samples collected: 1e66 * TRampUp = N; if N is smaller
# only the first part of the ramp is sampled; if N is larger than the 
# 
#--------------------------------------------------------------------------
dCfg = dict()
dCfg['fStrt'] = 24.00e9
dCfg['fStop'] = 24.25e9
dCfg['TRampUp'] = 256e-6 
dCfg['Perd'] = 0.4e-3
dCfg['N'] = 256
dCfg['Seq'] = [1]
dCfg['CycSiz'] = 4
dCfg['FrmSiz'] = 100
dCfg['FrmMeasSiz'] = 1

Brd.RfMeas(dCfg)

#--------------------------------------------------------------------------
# Read actual configuration
#--------------------------------------------------------------------------
NrChn = int(Brd.Get('NrChn'))
N = int(Brd.Get('N'))
fs = Brd.Get('fs')

#--------------------------------------------------------------------------
# Configure Signal Processing
#--------------------------------------------------------------------------
# Processing of range profile
Win2D = Brd.hanning(N-1,NrChn)
ScaWin = sum(Win2D[:,0])
NFFT = 2**8
kf = (dCfg['fStop'] - dCfg['fStrt'])/dCfg['TRampUp']
vRange = np.arange(NFFT//2)/NFFT*fs*c0/(2*kf)

RMin = 1
RMax = 20
RMinIdx = np.argmin(np.abs(vRange - RMin))
RMaxIdx = np.argmin(np.abs(vRange - RMax))
vRangeExt = vRange[RMinIdx:RMaxIdx]

# Window function for receive channels
NFFTAnt = 256
WinAnt2D = Brd.hanning(NrChn, len(vRangeExt))
ScaWinAnt = np.sum(WinAnt2D[:,0])
WinAnt2D = WinAnt2D.transpose()
vAngDeg  = np.arcsin(2*np.arange(-NFFTAnt//2, NFFTAnt//2)/NFFTAnt)/np.pi*180

#--------------------------------------------------------------------------
# Generate Curves for plots
#--------------------------------------------------------------------------
if Disp_TimSig:
    n = np.arange(int(N))
    CurveTim = []
    for IdxChn in np.arange(NrChn):
        CurveTim.append(PltTim.plot(pen=pg.intColor(IdxChn, hues=NrChn)))

if Disp_RP:
    CurveRP = []
    for IdxChn in np.arange(NrChn):
        CurveRP.append(PltRP.plot(pen=pg.intColor(IdxChn, hues=NrChn)))   



#--------------------------------------------------------------------------
# Measure and calculate DBF
#--------------------------------------------------------------------------
try:
    Cycles = 0
    while True:

        Data = Brd.BrdGetData()

        if Disp_FrmNr > 0:
            FrmCntr     =   Data[0,:]
            print("FrmCntr: ", FrmCntr)
    
        # Remove Framenumber from processing
        Data = Data[1:,:]
        
        if Disp_TimSig > 0:      
            # Display time domain signals
            for IdxChn in np.arange(NrChn):
                CurveTim[IdxChn].setData(n[1:],Data[:,IdxChn])

        # Calculate range profiles and display them
        RP = 2*np.fft.fft(Data[:,:]*Win2D, n=NFFT, axis=0)/ScaWin*Brd.FuSca
        RP = RP[RMinIdx:RMaxIdx,:]
        
        if Disp_RP> 0:
            for IdxChn in np.arange(NrChn):
                CurveRP[IdxChn].setData(vRangeExt,20*np.log10(np.abs(RP[:,IdxChn])))

        
        if Disp_JOpt > 0:
            JOpt = np.fft.fftshift(np.fft.fft(RP*WinAnt2D, NFFTAnt, axis=1)/ScaWinAnt, axes=1)
            
            JdB = 20*np.log10(np.abs(JOpt))
        JMax = np.max(JdB)
        JNorm = JdB - JMax
        JNorm[JNorm < -25] = -25

        # # --- simple peak detection: strongest point ---
        # peak_r_idx, peak_u_idx = np.unravel_index(np.argmax(JNorm), JNorm.shape)

        # # convert indices → (u, R) coordinates
        # u_val = -1.0 + (peak_u_idx * 2.0 / NFFTAnt)      # same mapping as setImage()
        # r_val = vRangeExt[peak_r_idx]

        # # update the scatter item
        # peakScatter.setData([u_val], [r_val])

        thresh = -6  # dB below max
        mask = JNorm > thresh
        r_idx, u_idx = np.where(mask)

        # Optional: limit number of points
        K = 20
        if r_idx.size > K:
            flat_vals = JNorm[mask].ravel()
            topK_idx = np.argsort(flat_vals)[-K:]
            r_idx = r_idx[topK_idx]
            u_idx = u_idx[topK_idx]

        u_vals = -1.0 + (u_idx * 2.0 / NFFTAnt)
        r_vals = vRangeExt[r_idx]
        peakScatter.setData(u_vals, r_vals)


        Img.setImage(JNorm.T, pos=[-1, RMin], scale=[2.0/NFFTAnt, (RMax - RMin)/vRangeExt.shape[0]])
        View.setAspectLocked(False)
        

        pg.QtGui.QGuiApplication.processEvents()

        Cycles += 1

except KeyboardInterrupt:
    print("Measurement stopped by user")

finally:
    del Brd