import numpy as np

def get_qam_constellation(mod_order):
    """
    Generates a QAM constellation with Gray coding and unit average power.
    
    Args:
        mod_order (int): Modulation order (e.g., 2, 4, 16).
        
    Returns:
        np.array: Complex constellation points.
    """
    if mod_order == 2: # BPSK
        constellation = np.array([-1, 1], dtype=complex)
    elif mod_order == 4: # QPSK
        constellation = np.array([-1-1j, -1+1j, 1+1j, 1-1j], dtype=complex) 
    elif mod_order == 16: # 16-QAM
        # Generate 16-QAM points
        real = np.array([-3, -1, 1, 3])
        imag = np.array([-3, -1, 1, 3])
        constellation = []
        for r in real:
            for i in imag:
                constellation.append(r + 1j*i)
        constellation = np.array(constellation, dtype=complex)
    else:
        raise ValueError("Unsupported modulation order")
    
    # Normalize energy to 1
    avg_power = np.mean(np.abs(constellation)**2)
    constellation /= np.sqrt(avg_power)
    
    return constellation

def qam_mapper(bits, mod_order, constellation):
    """
    Maps bits to QAM symbols.
    
    Args:
        bits (np.array): Input binary bits (0s and 1s).
        mod_order (int): Modulation order.
        constellation (np.array): QAM constellation points.
        
    Returns:
        np.array: Mapped complex symbols.
    """
    bits_per_symbol = int(np.log2(mod_order))
    # Reshape bits into symbols
    num_symbols = len(bits) // bits_per_symbol
    
    # Efficient bit to integer conversion
    powers = 2**np.arange(bits_per_symbol)[::-1]
    indices = np.reshape(bits, (num_symbols, bits_per_symbol)).dot(powers)
    
    return constellation[indices]

def apply_awgn(signal, snr_db):
    """
    Adds AWGN to a signal for a given SNR.
    
    Args:
        signal (np.array): Input complex signal.
        snr_db (float): SNR in dB.
        
    Returns:
        np.array: Signal with added noise.
    """
    sig_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db / 10.0)
    noise_power = sig_power / snr_linear
    
    # Generate complex Gaussian noise
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    
    return signal + noise
