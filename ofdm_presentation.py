import numpy as np
import matplotlib.pyplot as plt
from ofdm_common import (
    qam_mapper,
    get_qam_constellation,
    apply_awgn
)

def generate_gold_code(length):
    """
    Generates a Gold Code sequence of length 31.
    (Polynomials: x^5 + x^2 + 1 and x^5 + x^4 + x^3 + x^2 + 1)
    """     
    # LFSR A: x^5 + x^2 + 1
    reg_a = np.ones(5, dtype=int)
    # LFSR B: x^5 + x^4 + x^3 + x^2 + 1
    reg_b = np.ones(5, dtype=int)
    
    code = []
    for _ in range(length):
        out_a = reg_a[-1]
        out_b = reg_b[-1]
        code.append(out_a ^ out_b)
        
        # Feedback
        fb_a = reg_a[2] ^ reg_a[4]
        fb_b = reg_b[0] ^ reg_b[1] ^ reg_b[2] ^ reg_b[4]
        
        # Shift
        reg_a = np.roll(reg_a, 1)
        reg_a[0] = fb_a
        reg_b = np.roll(reg_b, 1)
        reg_b[0] = fb_b
        
    # Map 0/1 to -1/+1
    return 2 * np.array(code) - 1

def generate_ofdm_signal(tx_syms_grid, N, cp_len):
    """Generates Time-Domain OFDM signal with CP"""
    # IFFT
    tx_time = np.fft.ifft(tx_syms_grid, axis=1) * np.sqrt(N)
    
    # Add Cyclic Prefix
    tx_cp = np.hstack([tx_time[:, -cp_len:], tx_time])
    
    # Serialize
    return tx_cp.flatten()

def plot_spectrum(signal, fs, title, color='b', label=None):
    """Helper to plot PSD"""
    f = np.fft.fftfreq(len(signal), d=1/fs)
    Pxx = 10 * np.log10(np.abs(np.fft.fft(signal))**2 / len(signal))
    
    # Shift to center 0 Hz
    f_shifted = np.fft.fftshift(f)
    Pxx_shifted = np.fft.fftshift(Pxx)
    
    plt.plot(f_shifted/1e6, Pxx_shifted, color=color, label=label, alpha=0.7)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Frequency (MHz)', fontsize=12)
    plt.ylabel('EIRP (dB)', fontsize=12)
    plt.grid(True, ls='--', alpha=0.5)

def simulation_presentation_steps():
    # --- Parameters ---
    N = 64                  # Subcarriers
    cp_len = 16             # Cyclic Prefix
    mod_order = 4           # QPSK
    num_symbols = 100       # Number of OFDM symbols
    fs = 20e6               # 20 MHz Bandwidth
    
    # --- User Requested Changes ---
    # 1. Direct Path (Clutter) -> Tree/Human (Weak Reflection)
    # Amplitude < 1
    h_d_amp = 0.2 
    
    # 2. Tag Reflectivity -> Stronger than Clutter
    # Amplitude < 1, but > h_d
    tag_mod_index = 0.5
    
    code_length = 31
    gold_code = generate_gold_code(code_length)
    f_m = 2e6               # Tag shift frequency (2 MHz)
    
    # --- Step 1: Generate Direct Path Signal (Clutter Only) ---
    print("--- Step 1: Generating Direct Path Signal (Clutter) ---")
    
    # Generate Bits & Symbols
    constellation = get_qam_constellation(mod_order)
    bits_per_symbol = int(np.log2(mod_order))
    total_bits = num_symbols * N * bits_per_symbol
    tx_bits = np.random.randint(0, 2, total_bits)
    tx_syms = qam_mapper(tx_bits, mod_order, constellation)
    tx_syms_grid = tx_syms.reshape(num_symbols, N)
    
    # OFDM Modulation
    tx_signal = generate_ofdm_signal(tx_syms_grid, N, cp_len)
    
    # Channel (Direct Path / Clutter)
    h_d = h_d_amp * np.abs(1 + 0.5j) # Normalize phase, set amp
    rx_direct = tx_signal * h_d
    
    # Add Noise
    snr_db = 20 # High SNR to see constellations clearly
    rx_direct_noisy = apply_awgn(rx_direct, snr_db)
    
    # --- Step 2: Generate Tag Signal (R-Fiducial) ---
    print("--- Step 2: Generating Tag Signal ---")
    
    # Tag Modulation: c(t) * cos(2*pi*f_m*t)
    t = np.arange(len(tx_signal)) / fs
    
    # Repeat Gold Code to match signal length
    code_repeated = np.tile(gold_code, int(np.ceil(len(tx_signal)/len(gold_code))))[:len(tx_signal)]
    
    # Apply Modulation
    tag_signal = tx_signal * tag_mod_index * code_repeated * np.cos(2 * np.pi * f_m * t)
    
    # Combine: Direct + Tag + Noise
    rx_total = rx_direct + tag_signal
    rx_total_noisy = apply_awgn(rx_total, snr_db)
    
    # --- Calculate EIRP (Integrated Power) ---
    eirp_clutter = 10 * np.log10(np.mean(np.abs(rx_direct_noisy)**2))
    eirp_total = 10 * np.log10(np.mean(np.abs(rx_total_noisy)**2))
    print(f"--- EIRP Measurements ---")
    print(f"Clutter EIRP: {eirp_clutter:.2f} dB")
    print(f"Total EIRP:   {eirp_total:.2f} dB")
    print(f"Difference:   {eirp_total - eirp_clutter:.2f} dB")
    
    # --- Plotting ---
    plt.figure(figsize=(20, 10))
    
    # Row 1: Spectra
    plt.subplot(2, 4, 1)
    plot_spectrum(rx_direct_noisy, fs, "1. Spectrum: Clutter Only", color='blue')
    
    plt.subplot(2, 4, 2)
    plot_spectrum(rx_total_noisy, fs, "2. Spectrum: Clutter + Tag", color='green')
    
    plt.subplot(2, 4, 3)
    plot_spectrum(rx_direct_noisy, fs, "3. Comparison (Zoomed)", color='blue', label='Clutter')
    plot_spectrum(rx_total_noisy, fs, "3. Comparison (Zoomed)", color='red', label='Total')
    plt.legend()
    plt.xlim([-5, 5])
    
    plt.subplot(2, 4, 4)
    residual = rx_total_noisy - rx_direct_noisy
    plot_spectrum(residual, fs, "4. Residual (Tag Signal)", color='purple')

    # Row 2: Constellations
    # We need to extract symbols first (simple FFT)
    def get_symbols(rx_time, N, cp_len):
        # Reshape to symbols
        sym_len = N + cp_len
        num = len(rx_time) // sym_len
        rx_reshaped = rx_time[:num*sym_len].reshape(num, sym_len)
        # Remove CP
        rx_no_cp = rx_reshaped[:, cp_len:]
        # FFT
        rx_freq = np.fft.fft(rx_no_cp, axis=1) / np.sqrt(N)
        return rx_freq.flatten()

    rx_syms_direct = get_symbols(rx_direct_noisy, N, cp_len)
    rx_syms_total = get_symbols(rx_total_noisy, N, cp_len)
    rx_syms_tag = get_symbols(tag_signal, N, cp_len)
    
    plt.subplot(2, 4, 5)
    plt.scatter(np.real(rx_syms_direct), np.imag(rx_syms_direct), alpha=0.5, s=5)
    plt.title("5. Constellation: Clutter Only")
    plt.grid(True)
    plt.axis('equal')
    
    plt.subplot(2, 4, 6)
    plt.scatter(np.real(rx_syms_total), np.imag(rx_syms_total), alpha=0.5, s=5, color='green')
    plt.title("6. Constellation: Clutter + Tag")
    plt.grid(True)
    plt.axis('equal')
    
    plt.subplot(2, 4, 7)
    plt.scatter(np.real(rx_syms_tag), np.imag(rx_syms_tag), alpha=0.5, s=5, color='purple')
    plt.title("7. Constellation: Tag Only")
    plt.grid(True)
    plt.axis('equal')
    
    # --- Step 5: The "Fix" (Correlation) ---
    # The Tag Signal is: Tag = OFDM_Signal * Gold_Code * Carrier
    # If we just correlate 'Tag' with 'Gold_Code', we get 'OFDM_Signal', which looks like noise!
    # We must "Wipe" the OFDM signal first.
    
    # 1. Estimate the Tag Component by removing the OFDM Signal
    # We use the Direct Path signal as our reference for the OFDM signal
    # Tag_Est = Residual / Direct_Path
    #         = (OFDM * Code * Carrier) / (OFDM * H_d)
    #         = (Code * Carrier) / H_d
    
    # Avoid division by zero
    safe_direct = rx_direct_noisy.copy()
    safe_direct[np.abs(safe_direct) < 1e-6] = 1e-6
    
    tag_extracted_time = residual / safe_direct
    
    # 2. Demodulate the Tag Frequency Shift (f_m)
    t = np.arange(len(residual)) / fs
    # Multiply by cos(2*pi*f_m*t) to bring it back to baseband
    demod_time = tag_extracted_time * np.cos(2 * np.pi * f_m * t)
    
    # 3. Correlate with Gold Code
    lags = np.arange(100)
    correlations = []
    
    for lag in lags:
        window = demod_time[lag : lag + code_length]
        if len(window) < code_length: break
        
        # Correlate
        score = np.abs(np.sum(window * gold_code))
        correlations.append(score)
        
    plt.subplot(2, 4, 8)
    plt.plot(lags, correlations, 'r-o', linewidth=2)
    plt.title("8. The Fix: Correlation Peak")
    plt.xlabel("Lag Index (Samples)")
    plt.ylabel("Correlation Score")
    plt.grid(True)
    
    plt.tight_layout()
    # plt.show()
    
    # --- Step 6: Time Domain Data Visualization (1 0 1 0...) ---
    print("\n--- Step 6: Generating Time Domain Data Stream ---")
    
    # Define Data Bits
    data_bits = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    print(f"Transmitting Bits: {data_bits}")
    
    # Switching every 1ms
    # We need to stretch the simulation so each bit lasts 1ms.
    # fs = 20 MHz. 1ms = 20,000 samples.
    # CRITICAL FIX: The bit duration MUST be an exact multiple of the code_length (31).
    # Otherwise, the code alignment breaks at the bit boundary, and subsequent bits are lost.
    
    target_samples = int(1e-3 * fs) # 20,000
    reps_per_bit = int(round(target_samples / code_length))
    samples_per_bit = reps_per_bit * code_length
    
    print(f"Adjusted Bit Duration: {samples_per_bit/fs*1000:.4f} ms ({reps_per_bit} code repetitions)")
    
    # Create Data Sequence (Modulate Gold Code)
    data_sequence = []
    for bit in data_bits:
        # Create the pattern for this bit
        # +Code or -Code
        code_sign = 1 if bit == 1 else -1
        current_code = code_sign * gold_code
        
        # Repeat to fill the bit duration exactly
        bit_seq = np.tile(current_code, reps_per_bit)
        data_sequence.extend(bit_seq)
        
    data_sequence = np.array(data_sequence)
    
    # Re-generate OFDM signal for this duration
    num_samples_data = len(data_sequence)
    # Ensure we have enough OFDM symbols
    sym_len = N + cp_len
    num_ofdm_syms_needed = int(np.ceil(num_samples_data / sym_len)) + 1
    
    # Generate new OFDM signal
    total_bits_new = num_ofdm_syms_needed * N * bits_per_symbol
    tx_bits_new = np.random.randint(0, 2, total_bits_new)
    tx_syms_new = qam_mapper(tx_bits_new, mod_order, constellation)
    tx_syms_grid_new = tx_syms_new.reshape(num_ofdm_syms_needed, N)
    tx_signal_new = generate_ofdm_signal(tx_syms_grid_new, N, cp_len)
    
    # Trim to match data length exactly
    tx_signal_new = tx_signal_new[:num_samples_data]
    
    # Apply Tag Modulation
    t_new = np.arange(num_samples_data) / fs
    tag_signal_new = tx_signal_new * tag_mod_index * data_sequence * np.cos(2 * np.pi * f_m * t_new)
    
    # Channel & Noise
    rx_direct_new = tx_signal_new * h_d
    rx_total_new = rx_direct_new + tag_signal_new
    rx_total_new_noisy = apply_awgn(rx_total_new, snr_db)
    rx_direct_new_noisy = apply_awgn(rx_direct_new, snr_db)
    
    # --- Receiver Processing (Wipe & Correlate) ---
    
    # 1. Wipe
    safe_direct_new = rx_direct_new_noisy.copy()
    safe_direct_new[np.abs(safe_direct_new) < 1e-6] = 1e-6
    residual_new = rx_total_new_noisy - rx_direct_new_noisy 
    tag_extracted_new = residual_new / safe_direct_new
    
    # 2. Demodulate
    demod_new = tag_extracted_new * np.cos(2 * np.pi * f_m * t_new)
    
    # 3. Sliding Correlation
    # Since we have many repetitions, we will get many peaks per bit!
    # This is actually good, it shows the continuous "1 1 1" or "-1 -1 -1" nature.
    
    # We can just correlate every 'code_length' samples to save time
    # Instead of sliding by 1, let's slide by code_length (Jump)
    # This is how a real receiver would read the stream once locked
    
    jump_step = code_length
    num_jumps = len(demod_new) // jump_step
    
    corr_output = []
    time_axis = []
    
    for i in range(num_jumps):
        start_idx = i * jump_step
        window = demod_new[start_idx : start_idx + code_length]
        score = np.sum(window * gold_code)
        corr_output.append(score)
        # Time for this point (center of window)
        time_axis.append( (start_idx + code_length/2) / fs * 1000 ) # in ms
        
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, corr_output, 'b.-', linewidth=0.5, markersize=2)
    
    plt.title("Step 6: Time Domain Data Recovery (1ms per Bit)", fontsize=16, fontweight='bold')
    plt.xlabel("Time (ms)", fontsize=14)
    plt.ylabel("Correlation Score", fontsize=14)
    plt.grid(True)
    
    # Annotate Bits
    # We place a label at the center of each bit period
    for i, bit in enumerate(data_bits):
        # Center time of the bit
        t_center = (i + 0.5) * 1.0 # 1ms per bit
        
        # Find rough y-value for label
        # It should be positive for 1, negative for 0
        y_pos = 10 if bit == 1 else -10 # Arbitrary height for label
        
        label = f"Bit {bit}"
        plt.text(t_center, y_pos, label, 
                 ha='center', fontsize=12, fontweight='bold', color='red',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Draw vertical lines for bit boundaries
        plt.axvline(x=i * 1.0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()
    print("Time Domain Simulation Complete.")

if __name__ == "__main__":
    simulation_presentation_steps()

#eirp comparison: total vs direct
#contellation diagram: total vs direct
#change reflectivity of direct path --> tree or human  --> make the amplitude of direct path less than 1
#change tag reflectivity --> make the amplitude of tag less than 1 (more than direct path, because direct path is a random object)

#take reflectivity of tag/baseline = 1.618
