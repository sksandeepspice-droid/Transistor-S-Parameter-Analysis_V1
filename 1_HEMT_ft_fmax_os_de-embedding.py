import numpy as np
import skrf as rf
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import LogLocator, AutoMinorLocator
from skrf.calibration import OpenShort

# --- User settings ---
# --- Read S-parameters files ---

FET_device_file = 'device.s2p'   # Touchstone file of FET with CPW
FET_open_file = 'open.s2p'     # Touchstone file
FET_short_file = 'short.s2p'   # Touchstone file
# FET_thru_file = 'R2706-S1_RFG5V2_ONW_thru.s2p'         # Touchstone file
eps = 1e-12                                             # small floor to avoid log10(0)


def load_networks(device_file, open_file, short_file):
    fet_cpw = rf.Network(device_file)
    open_dummy = rf.Network(open_file)
    short_dummy = rf.Network(short_file)
    return fet_cpw, open_dummy, short_dummy


def deembed_open_short(fet_cpw, open_dummy, short_dummy):
    dm = OpenShort(dummy_open=open_dummy, dummy_short=short_dummy)
    return dm.deembed(fet_cpw)


def save_deembedded_network(network, device_file, suffix='_OSDeemb'):
    deembedded_stem = os.path.splitext(device_file)[0] + suffix
    network.write_touchstone(deembedded_stem)
    print(f"Saved de-embedded S2P: {deembedded_stem}.s2p")
    return deembedded_stem


def set_auto_title(ax, title_text, min_size=8, max_size=14, y=0.995, width_fraction=0.95):
    fig = ax.figure
    ax.set_title('')
    title_obj = fig.suptitle(title_text, fontsize=max_size, y=y)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    available_width = fig.bbox.width * width_fraction

    font_size = float(max_size)
    while font_size > min_size:
        title_width = title_obj.get_window_extent(renderer=renderer).width
        if title_width <= available_width:
            break
        font_size -= 0.5
        title_obj.set_fontsize(font_size)
        fig.canvas.draw()

    return title_obj


# --- Function to analyze S-parameters, calculate ft and fmax, and plot results ---
def analyze_sparameters(network, title_prefix):
    """
    Performs S-parameter analysis on a given skrf Network object,
    calculates ft and fmax, and plots the results.

    Args:
        network (skrf.Network): The network object to analyze.
        title_prefix (str): A prefix for the plot titles and filenames.
    """
    freq = network.f                # frequency array in Hz
    # S-parameters: network.s shape = (n_freqs, 2, 2)
    s11 = network.s[:, 0, 0]
    s12 = network.s[:, 0, 1]
    s21 = network.s[:, 1, 0]
    s22 = network.s[:, 1, 1]

    # --- Convert S -> H parameters and compute |h21| in dB ---
    h_para = network.h              # returns (n_freqs, 2, 2) array of h-params
    h21 = h_para[:, 1, 0]
    abs_h21 = np.abs(h21)
    mag_h21 = 20.0 * np.log10(np.maximum(abs_h21, eps))

    # --- John Rollett stability factor K ---
    k_num = 1.0 - (np.abs(s11) ** 2) - (np.abs(s22) ** 2) + (np.abs(s11 * s22 - s12 * s21) ** 2)
    k_den = 2.0 * np.abs(s12 * s21)
    # avoid division by zero
    k_den_safe = np.where(k_den == 0, eps, k_den)
    k = k_num / k_den_safe

    # --- Mason's Unilateral Gain U ---
    # u_num = |(s21/s12) - 1|^2
    # u_den = 2*k*|s21/s12| - 2*Re(s21/s12)
    ratio = np.zeros_like(s21, dtype=complex)
    # avoid division by zero for s12
    s12_safe = np.where(np.abs(s12) < eps, eps + 0j, s12)
    ratio = s21 / s12_safe

    u_num = (np.abs(ratio - 1.0) ** 2)
    u_den = 2.0 * k * np.abs(ratio) - 2.0 * np.real(ratio)
    # avoid zeros or negative denominators that would blow up; keep safe floor
    u_den_safe = np.where(np.abs(u_den) < eps, eps, u_den)
    u = u_num / u_den_safe
    u_db = 10.0 * np.log10(np.maximum(u, eps))

    # --- Maximum Stable Gain (MSG) and Maximum Available Gain (MAG) ---
    msg = np.abs(ratio)

    # compute mag only where k >= 1 (to avoid sqrt of negative); else set nan
    mag = np.full_like(msg, np.nan)
    k_ge1 = k >= 1.0
    # compute using safe sqrt for k^2 - 1 (should be >= 0 here)
    sqrt_term = np.zeros_like(k)
    sqrt_term[k_ge1] = np.sqrt(k[k_ge1] ** 2 - 1.0)
    mag[k_ge1] = (np.abs(ratio[k_ge1])) * (k[k_ge1] - sqrt_term[k_ge1])

    # --- Choose MSG or MAG depending on k (same logic as MATLAB) ---
    msg_mag = np.where(k < 1.0, msg, mag)
    # if any entries are nan or <=0, clip to eps so log10 is defined
    msg_mag_db = 10.0 * np.log10(np.maximum(np.real(msg_mag), eps))

    # --- Fit a straight line to mag_h21 and find the x-intercept ---
    # Use a logarithmic scale for frequency for the fit
    log_freq = np.log10(freq)

    # Filter data for fitting within the specified frequency range (0.5 GHz to 3 GHz)
    freq_min_h21 = 0.5e9  # 0.5 GHz
    freq_max_h21 = 20.0e9  # 3 GHz
    freq_for_fit_h21 = freq[(freq >= freq_min_h21) & (freq <= freq_max_h21)]
    mag_h21_for_fit = mag_h21[(freq >= freq_min_h21) & (freq <= freq_max_h21)]
    log_freq_for_fit_h21 = np.log10(freq_for_fit_h21)

    # Fit a first-degree polynomial (straight line) to log_freq and mag_h21
    # polyfit returns [slope, intercept]
    slope_h21, intercept_h21 = np.polyfit(log_freq_for_fit_h21, mag_h21_for_fit, 1)

    # The equation of the line is y = slope * x + intercept
    # 0 = slope * x + intercept
    # x = -intercept / slope
    # Convert x from log10(freq) back to freq
    ft = 10**(-intercept_h21 / slope_h21)

    # Print the result for h21
    print(f"ft (|h21|) for {title_prefix}: {ft:.2e} Hz")

    # --- Fit a straight line to msg_mag_db and find the x-intercept ---
    # Filter msg_mag_db for fitting within the specified frequency range (8 GHz to 20 GHz)
    freq_min_msg_mag = 20.0e9  # 8 GHz
    freq_max_msg_mag = 40.0e9  # 20 GHz
    freq_for_fit_msg_mag = freq[(freq >= freq_min_msg_mag) & (freq <= freq_max_msg_mag)]
    msg_mag_db_for_fit = msg_mag_db[(freq >= freq_min_msg_mag) & (freq <= freq_max_msg_mag)]
    log_freq_for_fit_msg_mag = np.log10(freq_for_fit_msg_mag)

    # Fit a first-degree polynomial (straight line) to log_freq and msg_mag_db
    slope_msg_mag, intercept_msg_mag = np.polyfit(log_freq_for_fit_msg_mag, msg_mag_db_for_fit, 1)

    # Calculate the x-intercept for msg_mag_db
    fmax = 10**(-intercept_msg_mag / slope_msg_mag)

    # Print the result for msg_mag_db
    print(f"fmax (MSG/MAG) for {title_prefix}: {fmax:.2e} Hz")

    # --- Calculate y-values of the fitted line for |h21| for plotting ---
    fitted_line_h21_y = slope_h21 * log_freq + intercept_h21

    # --- Calculate y-values of the fitted line for MSG/MAG for plotting ---
    fitted_line_msg_mag_y = slope_msg_mag * log_freq + intercept_msg_mag

    # --- Plot: |h21|, U, MSG/MAG ---
    plt.figure(figsize=(5, 6))
    plt.semilogx(freq, mag_h21, linewidth=3, label='|h21| (dB)')
    plt.semilogx(freq, u_db, linewidth=3, label='U (dB)')
    plt.semilogx(freq, msg_mag_db, linewidth=3, label='MSG/MAG (dB)')
    plt.semilogx(freq, fitted_line_h21_y, '--', color='red', label='Fitted |h21| Line (0.5-3 GHz)')  # Add the fitted line for |h21|
    plt.semilogx(freq, fitted_line_msg_mag_y, '--', color='purple', label='Fitted MSG/MAG Line (8-20 GHz)')  # Add the fitted line for MSG/MAG with updated label
    plt.legend()
    ax_gain = plt.gca()
    ax_gain.minorticks_on()
    ax_gain.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax_gain.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax_gain.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax_gain.grid(True, which='major', linestyle='--', linewidth=0.6)
    ax_gain.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.7)
    set_auto_title(ax_gain, f'{title_prefix} - Gain')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.xlim([1e8, 1e11])
    plt.ylim([0, 50])
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    plt.savefig(f'{title_prefix}_FtFmax.jpg', dpi=300)
    plt.show()

    # --- Plot: stability factor k ---
    plt.figure(figsize=(5, 6))
    plt.semilogx(freq, k, linewidth=2, label='k')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_k = plt.gca()
    set_auto_title(ax_k, f'{title_prefix} - Stability Factor')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Stability (k)')
    plt.xlim([1e8, 1e11])
    plt.ylim([0, 5])
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    plt.savefig(f'{title_prefix}_K.jpg', dpi=300)
    plt.show()

def main():
    fet_cpw, open_dummy, short_dummy = load_networks(FET_device_file, FET_open_file, FET_short_file)
    fet_os = deembed_open_short(fet_cpw, open_dummy, short_dummy)
    save_deembedded_network(fet_os, FET_device_file, suffix='_OSDeemb')

    # Analyze and plot for FET_CPW (before de-embedding)
    analyze_sparameters(fet_cpw, FET_device_file + '_CPW')

    # Analyze and plot for FET_OS (after Open-Short de-embedding)
    analyze_sparameters(fet_os, FET_device_file + '_OSDeemb')


if __name__ == '__main__':
    main()

