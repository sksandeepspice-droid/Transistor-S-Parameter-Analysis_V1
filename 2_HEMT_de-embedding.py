"""
Combined program: SS2_SS3_SS4
"""

import numpy as np
import skrf as rf
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import matplotlib.ticker as ticker
from matplotlib.widgets import Slider
import csv
import json
import os
import atexit

# ============================================================================
# Slider Values Persistence
# ============================================================================

SLIDER_VALUES_FILE = 'slider_values.json'

# Dictionary to store all slider objects for saving on exit
slider_dict = {}

def load_slider_values():
    """Load slider values from JSON file if it exists."""
    if os.path.exists(SLIDER_VALUES_FILE):
        try:
            with open(SLIDER_VALUES_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_slider_values():
    """Save current slider values to JSON file."""
    values_to_save = {}
    for name, slider in slider_dict.items():
        values_to_save[name] = slider.val
    with open(SLIDER_VALUES_FILE, 'w') as f:
        json.dump(values_to_save, f, indent=2)
    print(f"Slider values saved to {SLIDER_VALUES_FILE}")

# Load previous slider values
loaded_values = load_slider_values()

# Register save function to run on exit
atexit.register(save_slider_values)

# ============================================================================
# FILE CONFIGURATION
# ============================================================================
# Define all S-parameter files here for easy access
filename_ss2 = 'deviceVg-5VD0.s2p'       # PART 1: SS2 - Pad Capacitance Extraction
filename_ss3 = 'deviceVg0VD0.s2p'        # PART 2: SS3 - Z-Parameter Extraction
filename_ss4 = 'deviceVg-2.75VD28.s2p'   # PART 3: SS4 - Intrinsic FET Parameter Extraction

# ============================================================================
# PART 1: SS2 - Pad Capacitance Extraction
# ============================================================================

# --- User settings ---
filename = filename_ss2
eps = 1e-15

# --- Read S-parameters and convert to Y-parameters ---
ntw = rf.Network(filename)
freq = ntw.f  # Hz
freq_pts = len(freq)

y_para = ntw.y
y11 = y_para[:, 0, 0]
y12 = y_para[:, 0, 1]
y21 = y_para[:, 1, 0]
y22 = y_para[:, 1, 1]

# --- Equations for pad cap extraction ---
eq1 = np.imag(y12)
eq2 = np.imag(y11 + y12)
eq3 = np.imag(y22 + y12)

# --- Fit range and SS2 loop for different pad_fit ranges ---
padcap_fitrange = 40  # same as MATLAB

while True:
    f_fit = freq[:padcap_fitrange]

    # --------- Fit eq1 (Y12) ----------
    def fun1(m):
        return m * f_fit - eq1[:padcap_fitrange]

    res1 = least_squares(fun1, 1e-12)
    m12 = res1.x[0]
    eq1_est = m12 * f_fit

    Cb = -(3 * m12) / (2 * np.pi)

    # --------- Fit eq2 (Y11 + Y12) ----------
    def fun2(m):
        return m * f_fit - eq2[:padcap_fitrange]

    res2 = least_squares(fun2, 1e-12)
    m1112 = res2.x[0]
    eq2_est = m1112 * f_fit

    Cpg = (m1112 / (2 * np.pi)) - (Cb / 3)

    # --------- Fit eq3 (Y22 + Y12) ----------
    def fun3(m):
        return m * f_fit - eq3[:padcap_fitrange]

    res3 = least_squares(fun3, 1e-12)
    m2212 = res3.x[0]
    eq3_est = m2212 * f_fit

    Cpd = (m2212 / (2 * np.pi)) - (Cb / 3)

    # --- Print results from SS2 ---
    print("=" * 60)
    print("PART 1: Pad Capacitance Extraction (SS2)")
    print("=" * 60)
    print(f"Cb  = {Cb:.2e} F")
    print(f"Cpg = {Cpg:.2e} F")
    print(f"Cpd = {Cpd:.2e} F")

    # --- Plot results from SS2 ---
    plt.figure(figsize=(10, 8))
    plt.plot(f_fit, eq1[:padcap_fitrange], 'go', label='Y12 data')
    plt.plot(f_fit, eq1_est, 'g-', linewidth=2, label='Y12 fit')

    plt.plot(f_fit, eq2[:padcap_fitrange], 'rs', label='Y11+Y12 data')
    plt.plot(f_fit, eq2_est, 'r-', linewidth=2, label='Y11+Y12 fit')

    plt.plot(f_fit, eq3[:padcap_fitrange], 'b^', label='Y22+Y12 data')
    plt.plot(f_fit, eq3_est, 'b-', linewidth=2, label='Y22+Y12 fit')

    plt.grid(True, which="both", linestyle="--", linewidth=0.7)
    plt.legend()
    plt.title(filename)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Im(Y) (S)")
    plt.xlim([1e8, f_fit[-1]])
    plt.tight_layout()
    plt.savefig("3_Cpg_Cpd.jpg", dpi=300)
    plt.show()
    
    # --- Interactive slider to adjust padcap_fitrange ---
    # Use a taller canvas and weighted rows so the saved figure is less vertically compressed.
    fig_ss2 = plt.figure(figsize=(10, 9))
    gs_ss2 = fig_ss2.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.35)
    ax1_ss2 = fig_ss2.add_subplot(gs_ss2[0])
    ax2_ss2 = fig_ss2.add_subplot(gs_ss2[1])
    
    # Initial plot setup
    line1_eq1, = ax1_ss2.plot(f_fit, eq1[:padcap_fitrange], 'go', label='Y12 data')
    line1_eq1_est, = ax1_ss2.plot(f_fit, eq1_est, 'g-', linewidth=2, label='Y12 fit')
    line1_eq2, = ax1_ss2.plot(f_fit, eq2[:padcap_fitrange], 'rs', label='Y11+Y12 data')
    line1_eq2_est, = ax1_ss2.plot(f_fit, eq2_est, 'r-', linewidth=2, label='Y11+Y12 fit')
    line1_eq3, = ax1_ss2.plot(f_fit, eq3[:padcap_fitrange], 'b^', label='Y22+Y12 data')
    line1_eq3_est, = ax1_ss2.plot(f_fit, eq3_est, 'b-', linewidth=2, label='Y22+Y12 fit')
    
    ax1_ss2.grid(True, which="both", linestyle="--", linewidth=0.7)
    ax1_ss2.legend()
    ax1_ss2.set_title(filename)
    ax1_ss2.set_xlabel("Frequency (Hz)")
    ax1_ss2.set_ylabel("Im(Y) (S)")
    ax1_ss2.set_xlim([1e8, f_fit[-1]])
    
    # Summary text box in second subplot
    ax2_ss2.axis('off')
    summary_text = f"Cb  = {Cb:.2e} F\nCpg = {Cpg:.2e} F\nCpd = {Cpd:.2e} F\npadcap_fitrange = {padcap_fitrange}"
    text_box = ax2_ss2.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12, 
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), family='monospace')
    
    # Adjust layout to make room for sliders while preserving plot height
    plt.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.18)
    
    # Create slider axes
    ax_fitrange = plt.axes([0.12, 0.08, 0.83, 0.03])
    
    # Define slider for padcap_fitrange
    max_fitrange = min(freq_pts // 2, 200)  # Limit to reasonable range
    padcap_fitrange_init = loaded_values.get('padcap_fitrange', padcap_fitrange)
    slider_fitrange = Slider(ax_fitrange, 'padcap_fitrange', 1, max_fitrange, 
                             valinit=padcap_fitrange_init, valstep=1, color='orange')
    slider_dict['padcap_fitrange'] = slider_fitrange
    
    # Update function for slider
    def update_ss2_fitrange(val):
        new_fitrange = int(slider_fitrange.val)
        new_f_fit = freq[:new_fitrange]
        
        # Recalculate fits with new range
        def fun1_upd(m):
            return m * new_f_fit - eq1[:new_fitrange]
        def fun2_upd(m):
            return m * new_f_fit - eq2[:new_fitrange]
        def fun3_upd(m):
            return m * new_f_fit - eq3[:new_fitrange]
        
        res1_upd = least_squares(fun1_upd, 1e-12)
        m12_upd = res1_upd.x[0]
        eq1_est_upd = m12_upd * new_f_fit
        Cb_upd = -(3 * m12_upd) / (2 * np.pi)
        
        res2_upd = least_squares(fun2_upd, 1e-12)
        m1112_upd = res2_upd.x[0]
        eq2_est_upd = m1112_upd * new_f_fit
        Cpg_upd = (m1112_upd / (2 * np.pi)) - (Cb_upd / 3)
        
        res3_upd = least_squares(fun3_upd, 1e-12)
        m2212_upd = res3_upd.x[0]
        eq3_est_upd = m2212_upd * new_f_fit
        Cpd_upd = (m2212_upd / (2 * np.pi)) - (Cb_upd / 3)
        
        # Update plot data
        line1_eq1.set_data(new_f_fit, eq1[:new_fitrange])
        line1_eq1_est.set_data(new_f_fit, eq1_est_upd)
        line1_eq2.set_data(new_f_fit, eq2[:new_fitrange])
        line1_eq2_est.set_data(new_f_fit, eq2_est_upd)
        line1_eq3.set_data(new_f_fit, eq3[:new_fitrange])
        line1_eq3_est.set_data(new_f_fit, eq3_est_upd)
        
        # Update axis limits
        ax1_ss2.set_xlim([1e8, new_f_fit[-1]])
        
        # Auto-scale y-axis based on data range
        all_y_data = np.concatenate([
            eq1[:new_fitrange], eq1_est_upd,
            eq2[:new_fitrange], eq2_est_upd,
            eq3[:new_fitrange], eq3_est_upd
        ])
        y_min = np.min(all_y_data)
        y_max = np.max(all_y_data)
        y_range = y_max - y_min
        padding = 0.1 * y_range if y_range > 0 else 1
        ax1_ss2.set_ylim([y_min - padding, y_max + padding])
        
        # Update summary text
        summary_text_upd = f"Cb  = {Cb_upd:.2e} F\nCpg = {Cpg_upd:.2e} F\nCpd = {Cpd_upd:.2e} F\npadcap_fitrange = {new_fitrange}"
        text_box.set_text(summary_text_upd)
        
        fig_ss2.canvas.draw_idle()
        fig_ss2.savefig('3_Cpg_Cpd.jpg', dpi=100, bbox_inches='tight')
    
    # Connect slider to update function
    slider_fitrange.on_changed(update_ss2_fitrange)
    
    # Call update function once to sync plot with loaded slider values
    update_ss2_fitrange(None)
    
    plt.savefig('3_Cpg_Cpd.jpg', dpi=100, bbox_inches='tight')
    plt.show()
    
    # Capture final slider value and recalculate parameters
    padcap_fitrange = int(slider_fitrange.val)
    f_fit = freq[:padcap_fitrange]
    
    # Recalculate final fits
    def fun1(m):
        return m * f_fit - eq1[:padcap_fitrange]
    res1 = least_squares(fun1, 1e-12)
    m12 = res1.x[0]
    eq1_est = m12 * f_fit
    Cb = -(3 * m12) / (2 * np.pi)
    
    def fun2(m):
        return m * f_fit - eq2[:padcap_fitrange]
    res2 = least_squares(fun2, 1e-12)
    m1112 = res2.x[0]
    eq2_est = m1112 * f_fit
    Cpg = (m1112 / (2 * np.pi)) - (Cb / 3)
    
    def fun3(m):
        return m * f_fit - eq3[:padcap_fitrange]
    res3 = least_squares(fun3, 1e-12)
    m2212 = res3.x[0]
    eq3_est = m2212 * f_fit
    Cpd = (m2212 / (2 * np.pi)) - (Cb / 3)
    
    print(f"\nFinal SS2 Results with padcap_fitrange = {padcap_fitrange}:")
    print(f"Cb  = {Cb:.2e} F")
    print(f"Cpg = {Cpg:.2e} F")
    print(f"Cpd = {Cpd:.2e} F\n")
    
    break


# ============================================================================
# PART 2: SS3 - Z-Parameter Extraction
# ============================================================================

print("\n" + "=" * 60)
print("PART 2: Z-Parameter Extraction (SS3)")
print("=" * 60)

# --- Load S-parameter file ---
filename = filename_ss3
ntw = rf.Network(filename)

s11 = ntw.s[:, 0, 0]
s12 = ntw.s[:, 0, 1]
s21 = ntw.s[:, 1, 0]
s22 = ntw.s[:, 1, 1]

z0 = ntw.z0[0]
freq = ntw.f
freq_pts = len(freq)

# Convert to Z-parameters
z_params = ntw.z
z11 = z_params[:, 0, 0]
z12 = z_params[:, 0, 1]
z21 = z_params[:, 1, 0]
z22 = z_params[:, 1, 1]

# --- Extraction of Rch, Rd+Rs, Ld+Ls, Cds ---
f_exc = 1042 # excitation frequency point to exclude
freq_pts_l = freq_pts - f_exc

fm = freq[freq_pts_l // 2]
wm = 2 * np.pi * fm

Z22_fl = np.mean(z22[0:5])
Z22_fh = np.mean(z22[freq_pts_l-5:freq_pts_l])

# Store calculated values for table display
table_data = []

Rch = np.real(Z22_fl - Z22_fh)
Rch_calc = Rch
Rch = 0.2  # override
table_data.append(['Rch_calculated', f'{Rch_calc:.2e}'])
table_data.append(['Rch_Override', f'{Rch:.2e}'])

Rds = np.real(Z22_fh)
Rds_calc = Rds
Rds = 1.4  # override
table_data.append(['Rds_calculated', f'{Rds_calc:.2e}'])
table_data.append(['Rds_Override', f'{Rds:.2e}'])

Z22_fm = z22[freq_pts_l // 2]
Cds = (1 / (wm * Rch)) * np.sqrt((Rch / np.real(Z22_fm - Z22_fh)) - 1)
Cds_calc = Cds
Cds = 1e-12  # override if complex
table_data.append(['Cds_calculated', f'{Cds_calc:.2e}'])
table_data.append(['Cds_Override', f'{Cds:.2e}'])

fh = freq[freq_pts_l]
wh = 2 * np.pi * fh
Lds = (1 / wh) * np.imag(Z22_fh - (Rch / (1 + 1j * wh * Cds * Rch)))
Lds_calc = Lds
Lds = 100e-12  # override
table_data.append(['Lds_calculated', f'{Lds_calc:.2e}'])
table_data.append(['Lds_Override', f'{Lds:.2e}'])

Z22_mod = Rds + (1j * 2 * np.pi * freq * Lds) + Rch / (1 + 1j * 2 * np.pi * freq * Cds * Rch)

# Plot Z22 with interactive sliders
from matplotlib.widgets import Slider

fig = plt.figure(figsize=(10, 8))
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

# Initial plot setup
line1_meas, = ax1.plot(freq, np.real(z22), '*', label='measured')
line1_mod, = ax1.plot(freq, np.real(Z22_mod), '-r', linewidth=2, label='model')
ax1.set_ylabel('Real z22')
ax1.set_xlim([1e8, freq[freq_pts_l]])
ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.2e10))
ax1.legend()
ax1.grid(True, which='both')
ax1.set_title(filename)

line2_meas, = ax2.plot(freq, np.imag(z22), '+', label='measured')
line2_mod, = ax2.plot(freq, np.imag(Z22_mod), '-r', linewidth=2, label='model')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Imag z22')
ax2.set_xlim([1e8, freq[freq_pts_l]])
ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.2e10))
ax2.legend()
ax2.grid(True, which='both')


def autoscale_z22_axes(freq_limit_idx, z22_model_data):
    """Auto-scale Z22 real/imag plots using visible measured and modeled data."""
    idx = max(6, min(len(freq) - 1, int(freq_limit_idx)))
    freq_visible = freq[:idx + 1]

    real_visible = np.concatenate([
        np.real(z22[:idx + 1]),
        np.real(z22_model_data[:idx + 1])
    ])
    imag_visible = np.concatenate([
        np.imag(z22[:idx + 1]),
        np.imag(z22_model_data[:idx + 1])
    ])

    def add_padding(y_data):
        y_min = np.min(y_data)
        y_max = np.max(y_data)
        y_span = y_max - y_min
        pad = 0.1 * y_span if y_span > 0 else max(1e-6, 0.05 * max(abs(y_max), 1.0))
        return y_min - pad, y_max + pad

    x_min = max(1e8, freq_visible[0])
    x_max = freq_visible[-1]
    ax1.set_xlim([x_min, x_max])
    ax2.set_xlim([x_min, x_max])
    ax1.set_ylim(add_padding(real_visible))
    ax2.set_ylim(add_padding(imag_visible))

# Adjust layout to make room for sliders
plt.subplots_adjust(left=0.25, bottom=0.35)

# Create slider axes
ax_Rch = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_Rds = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_Cds = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_Lds = plt.axes([0.25, 0.10, 0.65, 0.03])
ax_f_exc = plt.axes([0.25, 0.05, 0.65, 0.03])

# Define slider ranges (adjust ranges as needed)
slider_Rch = Slider(ax_Rch, 'Rch (Ω)', 0.01, 1.0, valinit=loaded_values.get('Rch', Rch), valstep=0.01)
slider_Rds = Slider(ax_Rds, 'Rds (Ω)', 0.1, 5.0, valinit=loaded_values.get('Rds', Rds), valstep=0.1)
slider_Cds = Slider(ax_Cds, 'Cds (F)', 0.1e-12, 10e-12, valinit=loaded_values.get('Cds', Cds), valstep=0.1e-12)
slider_Lds = Slider(ax_Lds, 'Lds (H)', 10e-12, 200e-12, valinit=loaded_values.get('Lds', Lds), valstep=5e-12)
slider_f_exc = Slider(ax_f_exc, 'f_exc', 100, 1500, valinit=loaded_values.get('f_exc', f_exc), valstep=1)

slider_dict['Rch'] = slider_Rch
slider_dict['Rds'] = slider_Rds
slider_dict['Cds'] = slider_Cds
slider_dict['Lds'] = slider_Lds
slider_dict['f_exc'] = slider_f_exc

# Update function for sliders
def update(val):
    f_exc_val = int(slider_f_exc.val)
    freq_pts_l_val = max(6, min(freq_pts - 1, freq_pts - f_exc_val))
    
    # Recalculate dependent values based on f_exc
    fm_val = freq[freq_pts_l_val // 2]
    wm_val = 2 * np.pi * fm_val
    
    Z22_fh_val = np.mean(z22[freq_pts_l_val-5:freq_pts_l_val])
    Z22_fm_val = z22[freq_pts_l_val // 2]
    
    fh_val = freq[freq_pts_l_val]
    wh_val = 2 * np.pi * fh_val
    
    # Recalculate Rch, Rds, Cds, Lds based on new f_exc
    Z22_fl_val = np.mean(z22[0:5])
    Rch_measured = np.real(Z22_fl_val - Z22_fh_val)
    Rds_measured = np.real(Z22_fh_val)
    Cds_measured = (1 / (wm_val * Rch_measured)) * np.sqrt((Rch_measured / np.real(Z22_fm_val - Z22_fh_val)) - 1)
    Lds_measured = (1 / wh_val) * np.imag(Z22_fh_val - (Rch_measured / (1 + 1j * wh_val * Cds_measured * Rch_measured)))
    
    Rch_val = slider_Rch.val
    Rds_val = slider_Rds.val
    Cds_val = slider_Cds.val
    Lds_val = slider_Lds.val
    
    # Recalculate Z22_mod with new values
    Z22_mod_updated = Rds_val + (1j * 2 * np.pi * freq * Lds_val) + Rch_val / (1 + 1j * 2 * np.pi * freq * Cds_val * Rch_val)
    
    # Update plots
    line1_mod.set_ydata(np.real(Z22_mod_updated))
    line2_mod.set_ydata(np.imag(Z22_mod_updated))
    
    # Auto-scale y-axes and visible x-range from current data
    autoscale_z22_axes(freq_pts_l_val, Z22_mod_updated)
    
    fig.canvas.draw_idle()
    
    # Save updated plot
    fig.savefig('4_Z22.jpg', dpi=100, bbox_inches='tight')

# Connect sliders to update function
slider_Rch.on_changed(update)
slider_Rds.on_changed(update)
slider_Cds.on_changed(update)
slider_Lds.on_changed(update)
slider_f_exc.on_changed(update)

# Call update function once to sync plot with loaded slider values
update(None)

plt.savefig('4_Z22.jpg')
plt.show()

# --- Rs and Ls extraction ---
Z12_fh = np.mean(z12[freq_pts_l-5:freq_pts_l])
Rs = np.real(Z12_fh)
Rs_calc = Rs
Rs = 0.08  # override
table_data.append(['Rs_calculated', f'{Rs_calc:.2e}'])
table_data.append(['Rs_Override', f'{Rs:.2e}'])

Rd = Rds - Rs
Ls = (1 / wh) * np.imag(Z12_fh - ((0.5 * Rch) / (1 + 1j * wh * Cds * Rch)))
Ls_calc = Ls
Ls = 1e-12  # override
table_data.append(['Ls_calculated', f'{Ls_calc:.2e}'])
table_data.append(['Ls_override', f'{Ls:.2e}'])


Z12_mod = Rs + (1j * 2 * np.pi * freq * Ls) + (Rch / (2 * (1 + 1j * 2 * np.pi * freq * Cds * Rch)))

# Plot Z12 with interactive sliders
fig = plt.figure(figsize=(10, 8))
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

# Initial plot setup
line1_meas, = ax1.plot(freq, np.real(z12), '*', label='measured')
line1_mod, = ax1.plot(freq, np.real(Z12_mod), '-r', linewidth=2, label='model')
ax1.set_ylabel('Real z12')
ax1.set_xlim([1e8, freq[freq_pts_l]])
ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.2e10))
ax1.legend()
ax1.grid(True, which='both')
ax1.set_title(filename)

line2_meas, = ax2.plot(freq, np.imag(z12), '*', label='measured')
line2_mod, = ax2.plot(freq, np.imag(Z12_mod), '-r', linewidth=2, label='model')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Imag z12')
ax2.set_xlim([1e8, freq[freq_pts_l]])
ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.2e10))
ax2.legend()
ax2.grid(True, which='both')


def autoscale_z12_axes(freq_limit_idx, z12_model_data):
    """Auto-scale Z12 real/imag plots from visible measured and modeled data."""
    idx = max(6, min(len(freq) - 1, int(freq_limit_idx)))
    freq_visible = freq[:idx + 1]

    real_visible = np.concatenate([
        np.real(z12[:idx + 1]),
        np.real(z12_model_data[:idx + 1])
    ])
    imag_visible = np.concatenate([
        np.imag(z12[:idx + 1]),
        np.imag(z12_model_data[:idx + 1])
    ])

    def add_padding(y_data):
        y_min = np.min(y_data)
        y_max = np.max(y_data)
        y_span = y_max - y_min
        pad = 0.1 * y_span if y_span > 0 else max(1e-6, 0.05 * max(abs(y_max), 1.0))
        return y_min - pad, y_max + pad

    x_min = max(1e8, freq_visible[0])
    x_max = freq_visible[-1]
    ax1.set_xlim([x_min, x_max])
    ax2.set_xlim([x_min, x_max])
    ax1.set_ylim(add_padding(real_visible))
    ax2.set_ylim(add_padding(imag_visible))

# Adjust layout to make room for sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

# Create slider axes
ax_Rs = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_Ls = plt.axes([0.25, 0.10, 0.65, 0.03])

# Define slider ranges
slider_Rs = Slider(ax_Rs, 'Rs (Ω)', 0.01, 0.5, valinit=loaded_values.get('Rs', Rs), valstep=0.01)
slider_Ls = Slider(ax_Ls, 'Ls (H)', 0.1e-12, 5e-12, valinit=loaded_values.get('Ls', Ls), valstep=0.1e-12)

slider_dict['Rs'] = slider_Rs
slider_dict['Ls'] = slider_Ls

# Update function for sliders
def update_z12(val):
    Rs_val = slider_Rs.val
    Ls_val = slider_Ls.val
    
    # Recalculate Z12_mod with new values
    Z12_mod_updated = Rs_val + (1j * 2 * np.pi * freq * Ls_val) + (Rch / (2 * (1 + 1j * 2 * np.pi * freq * Cds * Rch)))
    
    # Update plots
    line1_mod.set_ydata(np.real(Z12_mod_updated))
    line2_mod.set_ydata(np.imag(Z12_mod_updated))

    # Auto-scale y-axes for current measured + modeled data
    autoscale_z12_axes(freq_pts_l, Z12_mod_updated)
    
    fig.canvas.draw_idle()
    
    # Save updated plot
    fig.savefig('5_Z12.jpg', dpi=100, bbox_inches='tight')

# Connect sliders to update function
slider_Rs.on_changed(update_z12)
slider_Ls.on_changed(update_z12)

# Call update function once to sync plot with loaded slider values
update_z12(None)

plt.savefig('5_Z12.jpg')
plt.show()

# --- Gate side extraction (Z11) ---
Ld = Lds - Ls
Ld_calc = Ld
table_data.append(['Ld_calculated', f'{Ld_calc:.2e}'])
wl = 2 * np.pi * freq[0]

Z11_fl = np.mean(z11[0:5])
Cg = (1 / wl) * np.imag((Z11_fl - (Rch / (3 * (1 + 1j * wl * Cds * Rch)))) ** -1)
Cg_calc = Cg
Cg = 10e-12  # override
table_data.append(['Cg_calculated', f'{Cg_calc:.2e}'])
table_data.append(['Cg_override', f'{Cg:.2e}'])

Rdy = (1 / (wl * Cg) ** 2) * np.real((Z11_fl - (Rch / (3 * (1 + 1j * wl * Cds * Rch)))) ** -1)
Rdy_calc = Rdy
Rdy = 3e3  # override
table_data.append(['Rdy_calculated', f'{Rdy_calc:.2e}'])
table_data.append(['Rdy_override', f'{Rdy:.2e}'])

Z11_fh = np.mean(z11[freq_pts_l-5:freq_pts_l])
Rg = np.real(Z11_fh) - Rs
Rg_calc = Rg
Rg = 0.5  # override
table_data.append(['Rg_calculated', f'{Rg_calc:.2e}'])
table_data.append(['Rg_override', f'{Rg:.2e}'])

Lg = -Ls + (1 / wh) * np.imag(Z11_fh - (Rch / (3 * (1 + 1j * wh * Cds * Rch))) - (Rdy / (1 + 1j * wh * Cg * Rdy)))
Lg_calc = Lg
Lg = 100e-12  # override
table_data.append(['Lg_calculated', f'{Lg_calc:.2e}'])
table_data.append(['Lg_override', f'{Lg:.2e}'])

# Print extracted parameters in tabular form
print("\n" + "="*50)
print("EXTRACTED PARAMETERS (Calculated vs Override)")
print("="*50)
print(f"{'Parameter':<20} {'Value':<20}")
print("-"*50)
for param, value in table_data:
    print(f"{param:<20} {value:<20}")
print("="*50 + "\n")

Z11_mod = Rg + Rs + (1j * 2 * np.pi * freq * (Lg + Ls)) + (Rch / (3 * (1 + 1j * 2 * np.pi * freq * Cds * Rch))) + (Rdy / (1 + 1j * 2 * np.pi * freq * Cg * Rdy))

# Plot Z11 with interactive sliders
fig = plt.figure(figsize=(10, 8))
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

# Initial plot setup
line1_meas, = ax1.plot(freq, np.real(z11), '*', label='measured')
line1_mod, = ax1.plot(freq, np.real(Z11_mod), '-r', linewidth=2, label='model')
ax1.set_ylabel('Real z11')
ax1.set_xlim([1e8, freq[freq_pts_l]])
ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.2e10))
ax1.legend()
ax1.grid(True, which='both')
ax1.set_title(filename)

line2_meas, = ax2.plot(freq, np.imag(z11), '*', label='measured')
line2_mod, = ax2.plot(freq, np.imag(Z11_mod), '-r', linewidth=2, label='model')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Imag z11')
ax2.set_xlim([1e8, freq[freq_pts_l]])
ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.2e10))
ax2.legend()
ax2.grid(True, which='both')


def autoscale_z11_axes(freq_limit_idx, z11_model_data):
    """Auto-scale Z11 real/imag plots from visible measured and modeled data."""
    idx = max(6, min(len(freq) - 1, int(freq_limit_idx)))
    freq_visible = freq[:idx + 1]

    real_visible = np.concatenate([
        np.real(z11[:idx + 1]),
        np.real(z11_model_data[:idx + 1])
    ])
    imag_visible = np.concatenate([
        np.imag(z11[:idx + 1]),
        np.imag(z11_model_data[:idx + 1])
    ])

    def add_padding(y_data):
        y_min = np.min(y_data)
        y_max = np.max(y_data)
        y_span = y_max - y_min
        pad = 0.1 * y_span if y_span > 0 else max(1e-6, 0.05 * max(abs(y_max), 1.0))
        return y_min - pad, y_max + pad

    x_min = max(1e8, freq_visible[0])
    x_max = freq_visible[-1]
    ax1.set_xlim([x_min, x_max])
    ax2.set_xlim([x_min, x_max])
    ax1.set_ylim(add_padding(real_visible))
    ax2.set_ylim(add_padding(imag_visible))

# Adjust layout to make room for sliders
plt.subplots_adjust(left=0.25, bottom=0.40)

# Create slider axes
ax_Rg = plt.axes([0.25, 0.30, 0.65, 0.03])
ax_Lg = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_Rdy = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_Cg = plt.axes([0.25, 0.15, 0.65, 0.03])

# Define slider ranges
slider_Rg = Slider(ax_Rg, 'Rg (Ω)', 0.1, 2.0, valinit=loaded_values.get('Rg', Rg), valstep=0.1)
slider_Lg = Slider(ax_Lg, 'Lg (H)', 10e-12, 500e-12, valinit=loaded_values.get('Lg', Lg), valstep=10e-12)
slider_Rdy = Slider(ax_Rdy, 'Rdy (Ω)', 500, 10000, valinit=loaded_values.get('Rdy', Rdy), valstep=500)
slider_Cg = Slider(ax_Cg, 'Cg (F)', 5e-12, 50e-12, valinit=loaded_values.get('Cg', Cg), valstep=1e-12)

slider_dict['Rg'] = slider_Rg
slider_dict['Lg'] = slider_Lg
slider_dict['Rdy'] = slider_Rdy
slider_dict['Cg'] = slider_Cg

# Update function for sliders
def update_z11(val):
    Rg_val = slider_Rg.val
    Lg_val = slider_Lg.val
    Rdy_val = slider_Rdy.val
    Cg_val = slider_Cg.val
    
    # Recalculate Z11_mod with new values
    Z11_mod_updated = Rg_val + Rs + (1j * 2 * np.pi * freq * (Lg_val + Ls)) + (Rch / (3 * (1 + 1j * 2 * np.pi * freq * Cds * Rch))) + (Rdy_val / (1 + 1j * 2 * np.pi * freq * Cg_val * Rdy_val))
    
    # Update plots
    line1_mod.set_ydata(np.real(Z11_mod_updated))
    line2_mod.set_ydata(np.imag(Z11_mod_updated))

    # Auto-scale y-axes for current measured + modeled data
    autoscale_z11_axes(freq_pts_l, Z11_mod_updated)
    
    fig.canvas.draw_idle()
    
    # Save updated plot
    fig.savefig('6_Z11.jpg', dpi=100, bbox_inches='tight')

# Connect sliders to update function
slider_Rg.on_changed(update_z11)
slider_Lg.on_changed(update_z11)
slider_Rdy.on_changed(update_z11)
slider_Cg.on_changed(update_z11)

# Call update function once to sync plot with loaded slider values
update_z11(None)

plt.savefig('6_Z11.jpg')
plt.show()

# --- Capture final slider values after user interaction ---
# Update parameters with final slider values from all plots
Rch = slider_Rch.val  # From Z22 plot
Rds = slider_Rds.val  # From Z22 plot
Cds = slider_Cds.val  # From Z22 plot
Lds = slider_Lds.val  # From Z22 plot
Rs = slider_Rs.val    # From Z12 plot
Ls = slider_Ls.val    # From Z12 plot
Rg = slider_Rg.val    # From Z11 plot
Lg = slider_Lg.val    # From Z11 plot
Rdy = slider_Rdy.val  # From Z11 plot
Cg = slider_Cg.val    # From Z11 plot

# Recalculate Rd and Ld based on updated values
Rd = Rds - Rs
Ld = Lds - Ls

# --- Save extracted parameters with updated slider values ---
# Organized by type: Capacitances, Inductances, Resistors
parameters = {
    # Capacitances
    'Cb': Cb,
    'Cpg': Cpg,
    'Cpd': Cpd,
    'Cds': Cds,
    'Cg': Cg,
    # Inductances
    'Lds': Lds,
    'Ls': Ls,
    'Ld': Ld,
    'Lg': Lg,
    # Resistors
    'Rch': Rch,
    'Rds': Rds,
    'Rs': Rs,
    'Rd': Rd,
    'Rg': Rg,
    'Rdy': Rdy
}

print("\n" + "="*50)
print("FINAL UPDATED PARAMETERS (From Sliders)")
print("="*50)
print(f"{'Parameter':<20} {'Value':<20}")
print("-"*50)
for param, value in parameters.items():
    print(f"{param:<20} {value:.2e}")
print("="*50 + "\n")

def save_parameters_to_csv(filename_label, parameter_dict):
    with open('Data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"------{filename_label}-----"])
        for key, val in parameter_dict.items():
            writer.writerow([key, f"{val:.2e}"])

save_parameters_to_csv(filename, parameters)

print("\n" + "=" * 60)
print("Combined program completed successfully!")
print("=" * 60)


# ============================================================================
# PART 3: SS4 - Intrinsic FET Parameter Extraction
# ============================================================================

print("\n" + "=" * 60)
print("PART 3: Intrinsic FET Parameter Extraction (SS4)")
print("=" * 60)

# --- Load S-parameters for intrinsic extraction ---
ntw_ss4 = rf.Network(filename_ss4)
freq_ss4 = ntw_ss4.f  # frequency array in Hz
freq_pts_ss4 = len(freq_ss4)
wf_ss4 = 2 * np.pi * freq_ss4

freq_pts_l_ss4 = 200

# --- Load parameters from data.csv for intrinsic extraction ---
print("\nLoading extrinsic parameters from Data.csv for intrinsic extraction...")
Cpg_ext = parameters.get('Cpg', 4.2e-12) if parameters else 4.25e-12
Cpd_ext = parameters.get('Cpd', 0.35e-12) if parameters else 0.36e-12
Rg_ext = parameters.get('Rg', 0.5) if parameters else 0.5
Rd_ext = parameters.get('Rd', 1.5) if parameters else 1.4
Rs_ext = parameters.get('Rs', 0.1) if parameters else 0.1
Lg_ext = parameters.get('Lg', 100e-12) if parameters else 100e-12
Ld_ext = parameters.get('Ld', 100e-12) if parameters else 9.72e-11
Ls_ext = parameters.get('Ls', 1e-12) if parameters else 2.8e-12

print(f"Cpg: {Cpg_ext:.2e}, Cpd: {Cpd_ext:.2e}")
print(f"Rg: {Rg_ext:.2e}, Rd: {Rd_ext:.2e}, Rs: {Rs_ext:.2e}")
print(f"Lg: {Lg_ext:.2e}, Ld: {Ld_ext:.2e}, Ls: {Ls_ext:.2e}\n")

# --- Extract Y-parameters for intrinsic analysis ---
y_para_ss4 = ntw_ss4.y
y11_ss4 = y_para_ss4[:, 0, 0].copy()
y12_ss4 = y_para_ss4[:, 0, 1].copy()
y21_ss4 = y_para_ss4[:, 1, 0].copy()
y22_ss4 = y_para_ss4[:, 1, 1].copy()

# --- Subtract parasitic pad capacitances ---
y11_ss4 -= 1j * wf_ss4 * Cpg_ext
y22_ss4 -= 1j * wf_ss4 * Cpd_ext

# --- Y -> Z conversion ---
from scipy.linalg import inv

z11_ss4 = np.zeros(freq_pts_ss4, dtype=complex)
z12_ss4 = np.zeros(freq_pts_ss4, dtype=complex)
z21_ss4 = np.zeros(freq_pts_ss4, dtype=complex)
z22_ss4 = np.zeros(freq_pts_ss4, dtype=complex)

for ii in range(freq_pts_ss4):
    y_mat = np.array([[y11_ss4[ii], y12_ss4[ii]],
                      [y21_ss4[ii], y22_ss4[ii]]])
    z_mat = inv(y_mat)
    z11_ss4[ii] = z_mat[0,0]
    z12_ss4[ii] = z_mat[0,1]
    z21_ss4[ii] = z_mat[1,0]
    z22_ss4[ii] = z_mat[1,1]

# --- Subtract parasitic resistances and inductances ---
z11_ss4 -= Rs_ext + 1j*wf_ss4*Ls_ext + Rg_ext + 1j*wf_ss4*Lg_ext
z12_ss4 -= Rs_ext + 1j*wf_ss4*Ls_ext
z21_ss4 -= Rs_ext + 1j*wf_ss4*Ls_ext
z22_ss4 -= Rs_ext + 1j*wf_ss4*Ls_ext + Rd_ext + 1j*wf_ss4*Ld_ext

# --- Z -> Y conversion ---
for ii in range(freq_pts_ss4):
    z_mat = np.array([[z11_ss4[ii], z12_ss4[ii]],
                      [z21_ss4[ii], z22_ss4[ii]]])
    y_mat = inv(z_mat)
    y11_ss4[ii] = y_mat[0,0]
    y12_ss4[ii] = y_mat[0,1]
    y21_ss4[ii] = y_mat[1,0]
    y22_ss4[ii] = y_mat[1,1]

# --- Intrinsic DUT: Y -> S conversion ---
def y2s_ss4(y, z0=50):
    I = np.eye(2)
    z0_mat = z0 * I
    return (z0_mat @ y - I) @ inv(z0_mat @ y + I)

s11int_ss4 = np.zeros(freq_pts_ss4, dtype=complex)
s12int_ss4 = np.zeros(freq_pts_ss4, dtype=complex)
s21int_ss4 = np.zeros(freq_pts_ss4, dtype=complex)
s22int_ss4 = np.zeros(freq_pts_ss4, dtype=complex)

for ii in range(freq_pts_ss4):
    y_mat = np.array([[y11_ss4[ii], y12_ss4[ii]],
                      [y21_ss4[ii], y22_ss4[ii]]])
    s_mat = y2s_ss4(y_mat)
    s11int_ss4[ii] = s_mat[0,0]
    s12int_ss4[ii] = s_mat[0,1]
    s21int_ss4[ii] = s_mat[1,0]
    s22int_ss4[ii] = s_mat[1,1]

# --- Extract Ggd, Ggs ---
Ggdgs_ss4 = np.real(np.mean(y11_ss4[:5]))
print(f"Ggdgs_calculated: {Ggdgs_ss4:.2e}")
Ggd_ss4 = -np.real(np.mean(y12_ss4[:5]))
print(f"Ggd_calculated: {Ggd_ss4:.2e}")
Ggd_ss4 = 1e-5
print(f"Ggd_Override: {Ggd_ss4:.2e}")

Ggs_ss4 = Ggdgs_ss4 - Ggd_ss4
print(f"Ggs_calculated: {Ggs_ss4:.2e}")
Ggs_ss4 = 4.67e-4
print(f"Ggs_Override: {Ggs_ss4:.2e}")

# --- Extract Cgd, Rgd with interactive Ggd slider ---
Cgd_ss4 = -(np.imag(y12_ss4)/wf_ss4) * (1 + ((np.real(y12_ss4)+Ggd_ss4)/np.imag(y12_ss4))**2)
Rgd_ss4 = (np.real(y12_ss4)+Ggd_ss4) / (np.imag(y12_ss4) * Cgd_ss4 * wf_ss4)

# Plot Cgd and Rgd with interactive Ggd slider
fig_cgd_ss4 = plt.figure(figsize=(10, 8))
ax_cgd_plot_ss4 = plt.subplot(2, 1, 1)
ax_rgd_plot_ss4 = plt.subplot(2, 1, 2)

# Initial plot setup
line_cgd_ss4, = ax_cgd_plot_ss4.plot(freq_ss4, Cgd_ss4, '*', label='Cgd')
ax_cgd_plot_ss4.set_ylabel('Cgd (F)')
ax_cgd_plot_ss4.set_xlabel('Frequency (Hz)')
ax_cgd_plot_ss4.set_xlim([1e8, freq_ss4[freq_pts_l_ss4]])
ax_cgd_plot_ss4.set_ylim(0, 4e-13)
ax_cgd_plot_ss4.yaxis.set_major_locator(ticker.MultipleLocator(1e-13))
ax_cgd_plot_ss4.xaxis.set_major_locator(ticker.MultipleLocator(0.2e10))
ax_cgd_plot_ss4.grid(True, which='both')
ax_cgd_plot_ss4.legend()
ax_cgd_plot_ss4.set_title(f'Cgd and Rgd (Ggd = {Ggd_ss4:.2e} S)')

line_rgd_ss4, = ax_rgd_plot_ss4.plot(freq_ss4, Rgd_ss4, 'o', label='Rgd')
ax_rgd_plot_ss4.set_ylabel('Rgd (Ohm)')
ax_rgd_plot_ss4.set_xlabel('Frequency (Hz)')
ax_rgd_plot_ss4.set_xlim([1e8, freq_ss4[freq_pts_l_ss4]])
ax_rgd_plot_ss4.set_ylim(-20, 20)
ax_rgd_plot_ss4.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax_rgd_plot_ss4.xaxis.set_major_locator(ticker.MultipleLocator(0.2e10))
ax_rgd_plot_ss4.grid(True, which='both')
ax_rgd_plot_ss4.legend()

# Adjust layout for sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

# Create slider axes
ax_ggd_slider_ss4 = plt.axes([0.25, 0.12, 0.65, 0.03])
ax_freq_slider_ss4 = plt.axes([0.25, 0.08, 0.65, 0.03])

# Define sliders for Ggd and freq_pts_l_ss4
slider_ggd_ss4 = Slider(ax_ggd_slider_ss4, 'Ggd (S)', 1e-8, 5e-7, valinit=loaded_values.get('Ggd_ss4', Ggd_ss4), valstep=1e-7, color='red')
slider_freq_pts_l_ss4 = Slider(ax_freq_slider_ss4, 'freq_pts_l_ss4', 50, freq_pts_ss4-10, valinit=loaded_values.get('freq_pts_l_ss4', freq_pts_l_ss4), valstep=10)

slider_dict['Ggd_ss4'] = slider_ggd_ss4
slider_dict['freq_pts_l_ss4'] = slider_freq_pts_l_ss4

# Update function for sliders
def update_ggd_ss4(val):
    ggd_val = slider_ggd_ss4.val
    freq_pts_l_val = int(slider_freq_pts_l_ss4.val)
    cgd_updated = -(np.imag(y12_ss4)/wf_ss4) * (1 + ((np.real(y12_ss4)+ggd_val)/np.imag(y12_ss4))**2)
    rgd_updated = (np.real(y12_ss4)+ggd_val) / (np.imag(y12_ss4) * cgd_updated * wf_ss4)
    line_cgd_ss4.set_ydata(cgd_updated)
    line_rgd_ss4.set_ydata(rgd_updated)
    # Update axis limits based on freq_pts_l_ss4
    if freq_pts_l_val < freq_pts_ss4:
        ax_cgd_plot_ss4.set_xlim([1e8, freq_ss4[freq_pts_l_val]])
        ax_rgd_plot_ss4.set_xlim([1e8, freq_ss4[freq_pts_l_val]])
    ax_cgd_plot_ss4.set_title(f'Cgd and Rgd (Ggd = {ggd_val:.2e} S, freq_pts_l = {freq_pts_l_val})')
    fig_cgd_ss4.canvas.draw_idle()
    fig_cgd_ss4.savefig('7_Cgd_Rgd.jpg', dpi=100, bbox_inches='tight')

def update_freq_ss4(val):
    update_ggd_ss4(val)
    # Also update xlim for Cgs-Rgs, gm-Tgm, and gds-Cds plots
    freq_pts_l_val = int(slider_freq_pts_l_ss4.val)
    if freq_pts_l_val < freq_pts_ss4:
        ax_cgs_plot_ss4.set_xlim([1e8, freq_ss4[freq_pts_l_val]])
        ax_rgs_plot_ss4.set_xlim([1e8, freq_ss4[freq_pts_l_val]])
        fig_cgs_ss4.canvas.draw_idle()
        try:
            ax_gm_ss4.set_xlim([1e8, freq_ss4[freq_pts_l_val]])
            ax_tgm_ss4.set_xlim([1e8, freq_ss4[freq_pts_l_val]])
            fig_gm_ss4.canvas.draw_idle()
        except:
            pass
        try:
            ax_gds_ss4.set_xlim([1e8, freq_ss4[freq_pts_l_val]])
            ax_cds_ss4.set_xlim([1e8, freq_ss4[freq_pts_l_val]])
            fig_gds_ss4.canvas.draw_idle()
        except:
            pass

slider_ggd_ss4.on_changed(update_ggd_ss4)
slider_freq_pts_l_ss4.on_changed(update_freq_ss4)

# Call update function once to sync plot with loaded slider values
update_ggd_ss4(None)

fig_cgd_ss4.suptitle(filename_ss4, fontsize=14, fontweight='bold')
plt.savefig('7_Cgd_Rgd.jpg', dpi=100, bbox_inches='tight')
plt.show()

# --- Save Cgd, Rgd at 2 GHz and Ggd into Data.csv ---
Ggd_ss4 = slider_ggd_ss4.val
Cgd_ss4 = -(np.imag(y12_ss4)/wf_ss4) * (1 + ((np.real(y12_ss4)+Ggd_ss4)/np.imag(y12_ss4))**2)
Rgd_ss4 = (np.real(y12_ss4)+Ggd_ss4) / (np.imag(y12_ss4) * Cgd_ss4 * wf_ss4)

target_freq_hz = 2e9
idx_2ghz = np.argmin(np.abs(freq_ss4 - target_freq_hz))
freq_2ghz_actual = freq_ss4[idx_2ghz]

parameters['Ggd'] = Ggd_ss4
parameters['Cgd'] = Cgd_ss4[idx_2ghz]
parameters['Rgd'] = Rgd_ss4[idx_2ghz]
save_parameters_to_csv(filename, parameters)

print(f"Saved SS4 values at {freq_2ghz_actual/1e9:.3f} GHz in Data.csv")
print(f"Ggd = {parameters['Ggd']:.2e}, Cgd = {parameters['Cgd']:.2e}, Rgd = {parameters['Rgd']:.2e}")

# --- Extract Cgs, Rgs with interactive Ggs slider ---
Cgs_ss4 = ((np.imag(y11_ss4)+np.imag(y12_ss4))/wf_ss4) * (1 + ((np.real(y11_ss4)+np.real(y12_ss4)-Ggs_ss4)**2)/((np.imag(y11_ss4)+np.imag(y12_ss4))**2))
Rgs_ss4 = (np.real(y11_ss4)+np.real(y12_ss4)-Ggs_ss4) / (wf_ss4 * Cgs_ss4 * (np.imag(y11_ss4)+np.imag(y12_ss4)))

fig_cgs_ss4 = plt.figure(figsize=(10, 8))
ax_cgs_plot_ss4 = plt.subplot(2, 1, 1)
ax_rgs_plot_ss4 = plt.subplot(2, 1, 2)

line_cgs_ss4, = ax_cgs_plot_ss4.plot(freq_ss4, Cgs_ss4, '*', label='Cgs')
ax_cgs_plot_ss4.set_ylabel('Cgs (F)')
ax_cgs_plot_ss4.set_xlabel('Frequency (Hz)')
ax_cgs_plot_ss4.set_xlim([1e8, freq_ss4[int(slider_freq_pts_l_ss4.val)]])
ax_cgs_plot_ss4.set_ylim(0, 10e-12)
ax_cgs_plot_ss4.yaxis.set_major_locator(ticker.MultipleLocator(2e-12))
ax_cgs_plot_ss4.xaxis.set_major_locator(ticker.MultipleLocator(0.2e10))
ax_cgs_plot_ss4.grid(True, which='both')
ax_cgs_plot_ss4.legend()
ax_cgs_plot_ss4.set_title(f'Cgs and Rgs (Ggs = {Ggs_ss4:.2e} S)')

line_rgs_ss4, = ax_rgs_plot_ss4.plot(freq_ss4, Rgs_ss4, 'o', label='Rgs')
ax_rgs_plot_ss4.set_ylabel('Rgs (Ohm)')
ax_rgs_plot_ss4.set_xlabel('Frequency (Hz)')
ax_rgs_plot_ss4.set_xlim([1e8, freq_ss4[int(slider_freq_pts_l_ss4.val)]])
ax_rgs_plot_ss4.set_ylim(-2, 6)
ax_rgs_plot_ss4.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax_rgs_plot_ss4.xaxis.set_major_locator(ticker.MultipleLocator(0.2e10))
ax_rgs_plot_ss4.grid(True, which='both')
ax_rgs_plot_ss4.legend()

plt.subplots_adjust(left=0.25, bottom=0.20)
ax_ggs_slider_ss4 = plt.axes([0.25, 0.10, 0.65, 0.05])
slider_ggs_ss4 = Slider(ax_ggs_slider_ss4, 'Ggs (S)', 1e-6, 10e-5, valinit=loaded_values.get('Ggs_ss4', Ggs_ss4), valstep=1e-5, color='orange')

slider_dict['Ggs_ss4'] = slider_ggs_ss4

def update_ggs_ss4(val):
    ggs_val = slider_ggs_ss4.val
    cgs_updated = ((np.imag(y11_ss4)+np.imag(y12_ss4))/wf_ss4) * (1 + ((np.real(y11_ss4)+np.real(y12_ss4)-ggs_val)**2)/((np.imag(y11_ss4)+np.imag(y12_ss4))**2))
    rgs_updated = (np.real(y11_ss4)+np.real(y12_ss4)-ggs_val) / (wf_ss4 * cgs_updated * (np.imag(y11_ss4)+np.imag(y12_ss4)))
    line_cgs_ss4.set_ydata(cgs_updated)
    line_rgs_ss4.set_ydata(rgs_updated)
    ax_cgs_plot_ss4.set_title(f'Cgs and Rgs (Ggs = {ggs_val:.2e} S)')
    fig_cgs_ss4.canvas.draw_idle()
    fig_cgs_ss4.savefig('8_Cgs_Rgs.jpg', dpi=100, bbox_inches='tight')

slider_ggs_ss4.on_changed(update_ggs_ss4)

# Call update function once to sync plot with loaded slider values
update_ggs_ss4(None)

fig_cgs_ss4.suptitle(filename_ss4, fontsize=14, fontweight='bold')
plt.savefig('8_Cgs_Rgs.jpg', dpi=100, bbox_inches='tight')
plt.show()

# --- Save Cgs, Rgs at 2 GHz and Ggs into Data.csv ---
Ggs_ss4 = slider_ggs_ss4.val
Cgs_ss4 = ((np.imag(y11_ss4)+np.imag(y12_ss4))/wf_ss4) * (1 + ((np.real(y11_ss4)+np.real(y12_ss4)-Ggs_ss4)**2)/((np.imag(y11_ss4)+np.imag(y12_ss4))**2))
Rgs_ss4 = (np.real(y11_ss4)+np.real(y12_ss4)-Ggs_ss4) / (wf_ss4 * Cgs_ss4 * (np.imag(y11_ss4)+np.imag(y12_ss4)))

parameters['Ggs'] = Ggs_ss4
parameters['Cgs'] = Cgs_ss4[idx_2ghz]
parameters['Rgs'] = Rgs_ss4[idx_2ghz]
save_parameters_to_csv(filename, parameters)

print(f"Saved SS4 values at {freq_2ghz_actual/1e9:.3f} GHz in Data.csv")
print(f"Ggs = {parameters['Ggs']:.2e}, Cgs = {parameters['Cgs']:.2e}, Rgs = {parameters['Rgs']:.2e}")

# --- Compute gm and Tgm ---
gm_ss4 = np.sqrt((np.real(y21_ss4-y12_ss4)**2 + np.imag(y21_ss4-y12_ss4)**2) * (1 + (wf_ss4**2)*(Cgs_ss4**2)*(Rgs_ss4**2)))
Tgm_ss4 = np.arctan(np.imag((y21_ss4-y12_ss4)*(1 + 1j*wf_ss4*Cgs_ss4*Rgs_ss4)) / np.real((y21_ss4-y12_ss4)*(1 + 1j*wf_ss4*Cgs_ss4*Rgs_ss4))) / wf_ss4

fig_gm_ss4, axs = plt.subplots(2, 1, figsize=(10, 8))
ax_gm_ss4 = axs[0]
ax_tgm_ss4 = axs[1]

ax_gm_ss4.plot(freq_ss4, gm_ss4, '*')
ax_gm_ss4.set_ylabel('gm (S)')
ax_gm_ss4.set_xlabel('Frequency (Hz)')
ax_gm_ss4.set_xlim([1e8, freq_ss4[int(slider_freq_pts_l_ss4.val)]])
ax_gm_ss4.set_ylim(0, 1)
ax_gm_ss4.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax_gm_ss4.xaxis.set_major_locator(ticker.MultipleLocator(0.2e10))
ax_gm_ss4.grid(True, which='both')

ax_tgm_ss4.plot(freq_ss4, Tgm_ss4, 'o')
ax_tgm_ss4.set_ylabel('Tgm (s)')
ax_tgm_ss4.set_xlabel('Frequency (Hz)')
ax_tgm_ss4.set_xlim([1e8, freq_ss4[int(slider_freq_pts_l_ss4.val)]])
ax_tgm_ss4.set_ylim(-3e-12, 9e-12)
ax_tgm_ss4.yaxis.set_major_locator(ticker.MultipleLocator(1e-12))
ax_tgm_ss4.xaxis.set_major_locator(ticker.MultipleLocator(0.2e10))
ax_tgm_ss4.grid(True, which='both')

fig_gm_ss4.suptitle(filename_ss4, fontsize=14, fontweight='bold')
fig_gm_ss4.tight_layout()
plt.savefig('9_gm_Tgm.jpg')
plt.show()

# --- Save gm, Tgm at 2 GHz into Data.csv ---
parameters['gm'] = gm_ss4[idx_2ghz]
parameters['Tgm'] = Tgm_ss4[idx_2ghz]
save_parameters_to_csv(filename, parameters)

print(f"Saved SS4 values at {freq_2ghz_actual/1e9:.3f} GHz in Data.csv")
print(f"gm = {parameters['gm']:.2e}, Tgm = {parameters['Tgm']:.2e}")

# --- Compute gds and Cds_intrinsic ---
gds_ss4 = np.real(y22_ss4+y12_ss4)
Cds_intrinsic = np.imag(y22_ss4+y12_ss4)/wf_ss4

fig_gds_ss4, axs = plt.subplots(2, 1, figsize=(10, 8))
ax_gds_ss4 = axs[0]
ax_cds_ss4 = axs[1]

ax_gds_ss4.plot(freq_ss4, gds_ss4, '*')
ax_gds_ss4.set_ylabel('gds (S)')
ax_gds_ss4.set_xlabel('Frequency (Hz)')
ax_gds_ss4.set_xlim([1e8, freq_ss4[int(slider_freq_pts_l_ss4.val)]])
ax_gds_ss4.set_ylim(-0.01, 0.05)
ax_gds_ss4.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
ax_gds_ss4.xaxis.set_major_locator(ticker.MultipleLocator(0.2e10))
ax_gds_ss4.grid(True, which='both')

ax_cds_ss4.plot(freq_ss4, Cds_intrinsic, 'o')
ax_cds_ss4.set_ylabel('Cds (F)')
ax_cds_ss4.set_xlabel('Frequency (Hz)')
ax_cds_ss4.set_xlim([1e8, freq_ss4[int(slider_freq_pts_l_ss4.val)]])
ax_cds_ss4.set_ylim(3e-13, 11e-13)
ax_cds_ss4.yaxis.set_major_locator(ticker.MultipleLocator(1e-13))
ax_cds_ss4.xaxis.set_major_locator(ticker.MultipleLocator(0.2e10))
ax_cds_ss4.grid(True, which='both')

fig_gds_ss4.suptitle(filename_ss4, fontsize=14, fontweight='bold')
fig_gds_ss4.tight_layout()
plt.savefig('10_gds_Cds.jpg')
plt.show()

# --- Save gds, intrinsic Cds at 2 GHz into Data.csv ---
parameters['gds'] = gds_ss4[idx_2ghz]
parameters['Cds_intrinsic'] = Cds_intrinsic[idx_2ghz]
save_parameters_to_csv(filename, parameters)

print(f"Saved SS4 values at {freq_2ghz_actual/1e9:.3f} GHz in Data.csv")
print(f"gds = {parameters['gds']:.2e}, Cds_intrinsic = {parameters['Cds_intrinsic']:.2e}")

print("\n" + "=" * 60)
print("PART 3 (SS4) completed successfully!")
print("=" * 60)
