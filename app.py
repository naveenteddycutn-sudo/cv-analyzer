import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.interpolate import interp1d

# --- Helper Functions ---
def normalize_unit(u):
    u = u.strip().replace("µ", "u").lower()
    if u in ["a"]: return "A"
    if u in ["ma"]: return "mA"
    if u in ["ua"]: return "uA"
    return "A"

def factor_to_A(unit):
    return {"A": 1.0, "mA": 1e-3, "uA": 1e-6}[normalize_unit(unit)]

def factor_from_A(unit):
    return {"A": 1.0, "mA": 1e3, "uA": 1e6}[normalize_unit(unit)]

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Page Config ---
st.set_page_config(page_title="Electrochem Dashboard", layout="wide")
st.title("🔋 Advanced CV & GCD Analysis Dashboard")

# --- Global Sidebar Setup (For CV) ---
st.sidebar.header("CV Global Settings")
uploaded_file = st.sidebar.file_uploader("Upload Raw CV (CSV)", type=["csv"], key="cv_file")
mass = st.sidebar.number_input("CV Active Mass (grams)", min_value=0.0001, value=0.1000, format="%.4f")
data_unit = st.sidebar.selectbox("CV Current Unit in CSV", ["A", "mA", "uA"], index=1)
plot_unit = st.sidebar.selectbox("CV Plot Unit", ["A", "mA", "uA"], index=1)

# --- Interactive Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "1. CV & Capacitance", 
    "2. Peak Extraction (b-value)", 
    "3. Origin Export (Dunn)", 
    "4. GCD Analysis (New)"
])

# ==========================================
# CV DATA PROCESSING (Tabs 1, 2, 3)
# ==========================================
if uploaded_file:
    uploaded_file.seek(0)
    data = pd.read_csv(uploaded_file).apply(pd.to_numeric, errors='coerce')
    to_A = factor_to_A(data_unit)
    A_to_plot = factor_from_A(plot_unit)
    
    potential_cols = data.columns[::2]
    current_cols = data.columns[1::2]
    
    valid_data = []
    for i, (pot_col, cur_col) in enumerate(zip(potential_cols, current_cols)):
        if "Unnamed" in str(cur_col): continue
        clean_data = pd.DataFrame({"V": data[pot_col], "I": data[cur_col]}).dropna()
        if clean_data.empty: continue
        
        sr = float((i + 1) * 10) 
        valid_data.append({
            "scan_rate": sr,
            "voltage": clean_data["V"].astype(float).values,
            "current_A": clean_data["I"].astype(float).values * to_A
        })

    # --- TAB 1: BASIC CV ---
    with tab1:
        st.subheader("Cyclic Voltammetry Curves")
        fig, ax = plt.subplots(figsize=(8, 6))
        results = []
        cv_export_data = pd.DataFrame()
        
        for item in valid_data:
            v = item["voltage"]
            i_A = item["current_A"]
            sr = item["scan_rate"]
            
            try:
                area = np.trapezoid(i_A, v)
            except AttributeError:
                area = np.trapz(i_A, v)
                
            V1, V2 = v.min(), v.max()
            spec_cap = np.abs(area) / (mass * sr * 1e-3 * (V2 - V1))
            
            results.append({"Scan Rate (mV/s)": sr, "Capacitance (F/g)": spec_cap})
            ax.plot(v, i_A * A_to_plot, label=f"{sr} mV/s")
            
            cv_export_data[f"V_{int(sr)}mVs"] = pd.Series(v)
            cv_export_data[f"I_{int(sr)}mVs"] = pd.Series(i_A * A_to_plot)
            
        ax.set_xlabel("Potential (V)")
        ax.set_ylabel(f"Specific Current ({plot_unit}/g)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)
        
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Download Capacitance Results", convert_df(pd.DataFrame(results)), "cv_capacitance.csv", "text/csv")
        with c2:
            st.download_button("⬇️ Download CV Plot Data (Origin)", convert_df(cv_export_data), "cv_plot_data.csv", "text/csv")

    # --- TAB 2: PEAK EXTRACTION ---
    with tab2:
        st.subheader("Peak Extraction & Reaction Kinetics")
        c1, c2 = st.columns(2)
        anodic_v = c1.number_input("Anodic Peak Potential (V)", value=0.50, step=0.05)
        cathodic_v = c2.number_input("Cathodic Peak Potential (V)", value=-0.50, step=0.05)
        
        scan_rates, anodic_i, cathodic_i = [], [], []
        for item in valid_data:
            v_arr = item["voltage"]
            i_arr_plot = item["current_A"] * A_to_plot 
            sr = item["scan_rate"]
            scan_rates.append(sr)
            
            idx_a = (np.abs(v_arr - anodic_v)).argmin()
            anodic_i.append(i_arr_plot[idx_a])
            
            idx_c = (np.abs(v_arr - cathodic_v)).argmin()
            cathodic_i.append(i_arr_plot[idx_c])
            
        sr_arr, a_i_arr, c_i_arr = np.array(scan_rates), np.array(anodic_i), np.array(cathodic_i)
        
        log_sr = np.log(sr_arr)
        fit_a_b = linregress(log_sr, np.log(np.abs(a_i_arr)))
        fit_c_b = linregress(log_sr, np.log(np.abs(c_i_arr)))
        
        cb1, cb2 = st.columns(2)
        with cb1:
            fig1, ax1 = plt.subplots(figsize=(5,4))
            ax1.scatter(log_sr, np.log(np.abs(a_i_arr)), color='black')
            ax1.plot(log_sr, fit_a_b.slope * log_sr + fit_a_b.intercept, 'r-', label=f"b = {fit_a_b.slope:.3f}")
            ax1.set_title("Anodic Peak Power Law")
            ax1.legend()
            st.pyplot(fig1)
            
        with cb2:
            fig2, ax2 = plt.subplots(figsize=(5,4))
            ax2.scatter(log_sr, np.log(np.abs(c_i_arr)), color='black')
            ax2.plot(log_sr, fit_c_b.slope * log_sr + fit_c_b.intercept, 'b-', label=f"b = {fit_c_b.slope:.3f}")
            ax2.set_title("Cathodic Peak Power Law")
            ax2.legend()
            st.pyplot(fig2)

        sqrt_v = np.sqrt(sr_arr)
        fit_a_k = linregress(sqrt_v, a_i_arr / sqrt_v)
        fit_c_k = linregress(sqrt_v, c_i_arr / sqrt_v)
        
        st.session_state['sr_arr'] = sr_arr
        st.session_state['k_anodic'] = (fit_a_k.slope, fit_a_k.intercept)
        st.session_state['k_cathodic'] = (fit_c_k.slope, fit_c_k.intercept)

    # --- TAB 3: DUNN EXPORT ---
    with tab3:
        st.subheader("Bar Graphs & Origin Export")
        if 'sr_arr' in st.session_state:
            mode = st.radio("Select Peak for Bar Graph:", ["Anodic", "Cathodic"], horizontal=True)
            v = st.session_state['sr_arr']
            k1, k2 = st.session_state['k_anodic'] if mode == "Anodic" else st.session_state['k_cathodic']
                
            i_cap = k1 * v
            i_diff = k2 * np.sqrt(v)
            cap_pct = (np.abs(i_cap) / (np.abs(i_cap) + np.abs(i_diff))) * 100.0
            diff_pct = (np.abs(i_diff) / (np.abs(i_cap) + np.abs(i_diff))) * 100.0
            
            fig5, ax5 = plt.subplots(figsize=(8,5))
            ax5.bar(v, cap_pct, width=3.0, color='black', label="Capacitive")
            ax5.bar(v, diff_pct, width=3.0, bottom=cap_pct, color='red', label="Diffusion")
            ax5.set_ylim(0, 100)
            ax5.set_xlabel("Scan rate (mV/s)")
            ax5.set_ylabel("Contribution (%)")
            ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig5)

        st.markdown("---")
        st.subheader("Full Sweep Processed Data (Origin Export)")
        if len(valid_data) > 1:
            v_min = max([np.min(item['voltage']) for item in valid_data])
            v_max = min([np.max(item['voltage']) for item in valid_data])
            
            v_grid_anodic = np.linspace(v_min, v_max, 500)
            v_grid_cathodic = np.linspace(v_max, v_min, 500)
            
            scan_rates_list, anodic_currents_interp, cathodic_currents_interp = [], [], []
            for item in valid_data:
                v_raw, i_raw = item['voltage'], item['current_A'] * A_to_plot
                dv = np.diff(v_raw)
                split_idx = np.where(np.sign(dv[:-1]) != np.sign(dv[1:]))[0]
                
                if len(split_idx) >= 1:
                    idx_turn = split_idx[0] + 1
                    if v_raw[1] > v_raw[0]: 
                        v_an, i_an = v_raw[:idx_turn], i_raw[:idx_turn]
                        v_cat, i_cat = v_raw[idx_turn:], i_raw[idx_turn:]
                    else:
                        v_cat, i_cat = v_raw[:idx_turn], i_raw[:idx_turn]
                        v_an, i_an = v_raw[idx_turn:], i_raw[idx_turn:]
                        
                    v_an, unique_an = np.unique(v_an, return_index=True)
                    v_cat_sorted, unique_cat = np.unique(v_cat, return_index=True)
                    
                    interp_an = interp1d(v_an, i_an[unique_an], bounds_error=False, fill_value="extrapolate")
                    interp_cat = interp1d(v_cat_sorted, i_cat[unique_cat], bounds_error=False, fill_value="extrapolate")
                    
                    anodic_currents_interp.append(interp_an(v_grid_anodic))
                    cathodic_currents_interp.append(interp_cat(v_grid_cathodic))
                    scan_rates_list.append(item['scan_rate'])
            
            sr_array = np.array(scan_rates_list)
            sqrt_sr = np.sqrt(sr_array)
            
            k1_anodic, k2_anodic = np.zeros_like(v_grid_anodic), np.zeros_like(v_grid_anodic)
            k1_cathodic, k2_cathodic = np.zeros_like(v_grid_cathodic), np.zeros_like(v_grid_cathodic)
            
            for idx in range(len(v_grid_anodic)):
                i_an = np.array([curve[idx] for curve in anodic_currents_interp])
                res_an = linregress(sqrt_sr, i_an / sqrt_sr)
                k1_anodic[idx], k2_anodic[idx] = res_an.slope, res_an.intercept
                
                i_cat = np.array([curve[idx] for curve in cathodic_currents_interp])
                res_cat = linregress(sqrt_sr, i_cat / sqrt_sr)
                k1_cathodic[idx], k2_cathodic[idx] = res_cat.slope, res_cat.intercept

            selected_sr = st.selectbox("Select Scan Rate for Origin Export:", sr_array)
            if selected_sr:
                df_anode = pd.DataFrame({
                    "Voltage (V)": v_grid_anodic,
                    "Total_Current": k1_anodic * selected_sr + k2_anodic * np.sqrt(selected_sr),
                    "Capacitive_Current": k1_anodic * selected_sr,
                    "Diffusion_Current": k2_anodic * np.sqrt(selected_sr)
                })
                df_cathode = pd.DataFrame({
                    "Voltage (V)": v_grid_cathodic,
                    "Total_Current": k1_cathodic * selected_sr + k2_cathodic * np.sqrt(selected_sr),
                    "Capacitive_Current": k1_cathodic * selected_sr,
                    "Diffusion_Current": k2_cathodic * np.sqrt(selected_sr)
                })
                cx1, cx2 = st.columns(2)
                with cx1: st.download_button(f"⬇️ Export Anode CV ({int(selected_sr)} mV/s)", convert_df(df_anode), f"anode_{int(selected_sr)}mVs.csv", "text/csv")
                with cx2: st.download_button(f"⬇️ Export Cathode CV ({int(selected_sr)} mV/s)", convert_df(df_cathode), f"cathode_{int(selected_sr)}mVs.csv", "text/csv")

else:
    with tab1: st.info("👈 Please upload a CV CSV file in the sidebar to begin.")

# ==========================================
# TAB 4: GCD ANALYSIS (Isolated Input)
# ==========================================
with tab4:
    st.subheader("Galvanostatic Charge-Discharge (GCD) Analysis")
    st.write("Upload a dedicated GCD file to calculate specific capacitance and energy densities.")
    
    col_g1, col_g2 = st.columns([1, 2])
    
    with col_g1:
        gcd_file = st.file_uploader("Upload GCD Data (CSV)", type=["csv"], key="gcd_file")
        gcd_current = st.number_input("Applied Current (mA)", min_value=0.0001, value=1.0000, format="%.4f")
        gcd_mass = st.number_input("Active Mass (grams)", min_value=0.0001, value=0.1000, format="%.4f")
        
    with col_g2:
        if gcd_file:
            # Assume Time is col 0, Voltage is col 1
            gcd_data = pd.read_csv(gcd_file).apply(pd.to_numeric, errors='coerce').dropna()
            
            if gcd_data.shape[1] >= 2:
                t = gcd_data.iloc[:, 0].values
                v = gcd_data.iloc[:, 1].values
                
                # Auto-detect discharge segment (max voltage to min voltage after max)
                idx_max = np.argmax(v)
                idx_min = idx_max + np.argmin(v[idx_max:])
                
                v_max, v_min = v[idx_max], v[idx_min]
                delta_v = v_max - v_min
                delta_t = t[idx_min] - t[idx_max]
                
                if delta_t > 0 and delta_v > 0:
                    current_A = gcd_current * 1e-3
                    
                    # Core Equations
                    c_s = (current_A * delta_t) / (gcd_mass * delta_v)
                    e_d = (c_s * (delta_v ** 2)) / (2 * 3.6)
                    p_d = (e_d * 3600) / delta_t
                    
                    # Plot
                    fig_gcd, ax_gcd = plt.subplots(figsize=(8, 6))
                    ax_gcd.plot(t, v, 'k-', linewidth=2, label="GCD Curve")
                    ax_gcd.axvspan(t[idx_max], t[idx_min], color='red', alpha=0.2, label=f"Discharge: {delta_t:.1f} s")
                    ax_gcd.set_xlabel("Time (s)")
                    ax_gcd.set_ylabel("Potential (V)")
                    ax_gcd.set_title(f"GCD Profile at {gcd_current} mA")
                    ax_gcd.legend()
                    st.pyplot(fig_gcd)
                    
                    # Results Table
                    gcd_results = pd.DataFrame([{
                        "Current (mA)": gcd_current,
                        "ΔV (V)": delta_v,
                        "Δt Discharge (s)": delta_t,
                        "Capacitance (F/g)": c_s,
                        "Energy Density (Wh/kg)": e_d,
                        "Power Density (W/kg)": p_d
                    }])
                    
                    st.dataframe(gcd_results, use_container_width=True)
                    
                    c_g1, c_g2 = st.columns(2)
                    with c_g1: st.download_button("⬇️ Download GCD Results", convert_df(gcd_results), "gcd_results.csv", "text/csv")
                    with c_g2: st.download_button("⬇️ Download GCD Plot Data", convert_df(gcd_data), "gcd_raw_data.csv", "text/csv")
                    
                    # Mathematical Breakdown
                    with st.expander("Show Step-by-Step Calculations"):
                        st.latex(r"C_s = \frac{I \cdot \Delta t}{m \cdot \Delta V}")
                        st.write(f"**Specific Capacitance ($C_s$):** ({current_A} A $\\times$ {delta_t:.2f} s) / ({gcd_mass} g $\\times$ {delta_v:.3f} V) = **{c_s:.2f} F/g**")
                        
                        st.latex(r"E = \frac{C_s \cdot \Delta V^2}{2 \cdot 3.6}")
                        st.write(f"**Energy Density ($E$):** ({c_s:.2f} $\\times$ {delta_v:.3f}$^2$) / 7.2 = **{e_d:.2f} Wh/kg**")
                        
                        st.latex(r"P = \frac{E \cdot 3600}{\Delta t}")
                        st.write(f"**Power Density ($P$):** ({e_d:.2f} $\\times$ 3600) / {delta_t:.2f} s = **{p_d:.2f} W/kg**")
                        
                else:
                    st.warning("Could not automatically detect a valid discharge curve. Ensure your CSV has Time in Column 1 and Voltage in Column 2.")
            else:
                st.error("CSV must contain at least two columns: Time and Voltage.")
