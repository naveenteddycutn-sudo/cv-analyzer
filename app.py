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
st.title("🔋 Advanced CV & Dunn Method Analyzer (Data Export Mode)")

# --- Global Sidebar Setup ---
st.sidebar.header("1. Upload & Global Settings")
uploaded_file = st.sidebar.file_uploader("Upload Raw CV (CSV)", type=["csv"])
mass = st.sidebar.number_input("Active Mass (grams)", min_value=0.0001, value=0.1000, format="%.4f")
data_unit = st.sidebar.selectbox("Current Unit in CSV", ["A", "mA", "uA"], index=1)
plot_unit = st.sidebar.selectbox("Plot Unit", ["A", "mA", "uA"], index=1)

# --- Interactive Tabs ---
tab1, tab2, tab3 = st.tabs(["1. CV & Capacitance", "2. Peak Extraction (b-value & k1/k2)", "3. Bar Graphs & Origin Export"])

if uploaded_file:
    # Read and clean the data globally
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
        
        sr = float((i + 1) * 10) # Assumes 10, 20, 30... mV/s based on column index
        valid_data.append({
            "scan_rate": sr,
            "voltage": clean_data["V"].astype(float).values,
            "current_A": clean_data["I"].astype(float).values * to_A
        })

    # ==========================================
    # TAB 1: BASIC CV & CAPACITANCE
    # ==========================================
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
            
            # Store CV data for bulk export
            cv_export_data[f"V_{int(sr)}mVs"] = pd.Series(v)
            cv_export_data[f"I_{int(sr)}mVs"] = pd.Series(i_A * A_to_plot)
            
        ax.set_xlabel("Potential (V)")
        ax.set_ylabel(f"Specific Current ({plot_unit}/g)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)
        
        # Datasets & Downloads
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Download Capacitance Results", convert_df(pd.DataFrame(results)), "capacitance_results.csv", "text/csv")
        with c2:
            st.download_button("⬇️ Download Raw CV Plot Data (Origin)", convert_df(cv_export_data), "cv_raw_plot_data.csv", "text/csv")

    # ==========================================
    # TAB 2: PEAK EXTRACTION
    # ==========================================
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
        
        # --- 1. Power Law (b-value) ---
        st.markdown("### 1. Power Law Analysis ($i = a v^b$)")
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
            
            df_pl_a = pd.DataFrame({"Scan_Rate": sr_arr, "log_v": log_sr, "log_i_anodic": np.log(np.abs(a_i_arr))})
            st.download_button("⬇️ Export Anodic Power Law Data", convert_df(df_pl_a), "power_law_anodic.csv", "text/csv")
            
        with cb2:
            fig2, ax2 = plt.subplots(figsize=(5,4))
            ax2.scatter(log_sr, np.log(np.abs(c_i_arr)), color='black')
            ax2.plot(log_sr, fit_c_b.slope * log_sr + fit_c_b.intercept, 'b-', label=f"b = {fit_c_b.slope:.3f}")
            ax2.set_title("Cathodic Peak Power Law")
            ax2.legend()
            st.pyplot(fig2)
            
            df_pl_c = pd.DataFrame({"Scan_Rate": sr_arr, "log_v": log_sr, "log_i_cathodic": np.log(np.abs(c_i_arr))})
            st.download_button("⬇️ Export Cathodic Power Law Data", convert_df(df_pl_c), "power_law_cathodic.csv", "text/csv")

        # --- 2. Dunn Method Parameters (k1/k2) ---
        st.markdown("---")
        st.markdown("### 2. Dunn Method Peak Fits ($i/v^{1/2}$ vs $v^{1/2}$)")
        sqrt_v = np.sqrt(sr_arr)
        y_a = a_i_arr / sqrt_v
        y_c = c_i_arr / sqrt_v
        
        fit_a_k = linregress(sqrt_v, y_a)
        fit_c_k = linregress(sqrt_v, y_c)
        
        ck1, ck2 = st.columns(2)
        with ck1:
            fig3, ax3 = plt.subplots(figsize=(5,4))
            ax3.scatter(sqrt_v, y_a, color='black')
            ax3.plot(sqrt_v, fit_a_k.slope * sqrt_v + fit_a_k.intercept, 'r-', label=f"k1={fit_a_k.slope:.2e}\nk2={fit_a_k.intercept:.2e}")
            ax3.set_title("Anodic Fit")
            ax3.legend()
            st.pyplot(fig3)
            
            df_k_a = pd.DataFrame({"Scan_Rate": sr_arr, "sqrt_v": sqrt_v, "i_div_sqrt_v": y_a})
            st.download_button("⬇️ Export Anodic k1/k2 Data", convert_df(df_k_a), "k1_k2_anodic.csv", "text/csv")
            
        with ck2:
            fig4, ax4 = plt.subplots(figsize=(5,4))
            ax4.scatter(sqrt_v, y_c, color='black')
            ax4.plot(sqrt_v, fit_c_k.slope * sqrt_v + fit_c_k.intercept, 'b-', label=f"k1={fit_c_k.slope:.2e}\nk2={fit_c_k.intercept:.2e}")
            ax4.set_title("Cathodic Fit")
            ax4.legend()
            st.pyplot(fig4)
            
            df_k_c = pd.DataFrame({"Scan_Rate": sr_arr, "sqrt_v": sqrt_v, "i_div_sqrt_v": y_c})
            st.download_button("⬇️ Export Cathodic k1/k2 Data", convert_df(df_k_c), "k1_k2_cathodic.csv", "text/csv")

        # Pass peak data silently to Tab 3 for the bar graphs
        st.session_state['sr_arr'] = sr_arr
        st.session_state['k_anodic'] = (fit_a_k.slope, fit_a_k.intercept)
        st.session_state['k_cathodic'] = (fit_c_k.slope, fit_c_k.intercept)

    # ==========================================
    # TAB 3: BAR GRAPHS & ORIGIN EXPORT
    # ==========================================
    with tab3:
        # --- Section 1: Contribution Bar Graph ---
        st.subheader("1. Peak Contribution Percentages (Bar Graph)")
        if 'sr_arr' in st.session_state:
            mode = st.radio("Select Peak for Bar Graph:", ["Anodic", "Cathodic"], horizontal=True)
            v = st.session_state['sr_arr']
            
            if mode == "Anodic":
                k1, k2 = st.session_state['k_anodic']
            else:
                k1, k2 = st.session_state['k_cathodic']
                
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
            
            df_bar = pd.DataFrame({"Scan Rate (mV/s)": v, "Capacitive (%)": cap_pct, "Diffusion (%)": diff_pct})
            st.download_button(f"⬇️ Export {mode} Bar Graph Data", convert_df(df_bar), f"bar_graph_{mode.lower()}.csv", "text/csv")
        
        # --- Section 2: Full Sweep Dunn's Data Export (NO PLOT) ---
        st.markdown("---")
        st.subheader("2. Full Sweep Processed Data (Origin Export)")
        st.write("Generates the complete interpolated CV datasets for manual shading in Origin.")
        
        if len(valid_data) > 1:
            v_min = max([np.min(item['voltage']) for item in valid_data])
            v_max = min([np.max(item['voltage']) for item in valid_data])
            
            v_grid_anodic = np.linspace(v_min, v_max, 500)
            v_grid_cathodic = np.linspace(v_max, v_min, 500)
            
            scan_rates_list, anodic_currents_interp, cathodic_currents_interp = [], [], []
            
            for item in valid_data:
                v_raw = item['voltage']
                i_raw = item['current_A'] * A_to_plot
                
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
                    i_an = i_an[unique_an]
                    
                    v_cat_sorted, unique_cat = np.unique(v_cat, return_index=True)
                    i_cat_sorted = i_cat[unique_cat]
                    
                    interp_an = interp1d(v_an, i_an, bounds_error=False, fill_value="extrapolate")
                    interp_cat = interp1d(v_cat_sorted, i_cat_sorted, bounds_error=False, fill_value="extrapolate")
                    
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
                # Reconstruct components for the chosen scan rate
                i_cap_anodic = k1_anodic * selected_sr
                i_diff_anodic = k2_anodic * np.sqrt(selected_sr)
                i_total_anodic = i_cap_anodic + i_diff_anodic
                
                i_cap_cathodic = k1_cathodic * selected_sr
                i_diff_cathodic = k2_cathodic * np.sqrt(selected_sr)
                i_total_cathodic = i_cap_cathodic + i_diff_cathodic
                
                df_anode = pd.DataFrame({
                    "Voltage (V)": v_grid_anodic,
                    "Total_Current": i_total_anodic,
                    "Capacitive_Current": i_cap_anodic,
                    "Diffusion_Current": i_diff_anodic
                })
                
                df_cathode = pd.DataFrame({
                    "Voltage (V)": v_grid_cathodic,
                    "Total_Current": i_total_cathodic,
                    "Capacitive_Current": i_cap_cathodic,
                    "Diffusion_Current": i_diff_cathodic
                })
                
                cx1, cx2 = st.columns(2)
                with cx1:
                    st.download_button(f"⬇️ Export Anode CV ({int(selected_sr)} mV/s)", convert_df(df_anode), f"anode_processed_CV_{int(selected_sr)}mVs.csv", "text/csv")
                with cx2:
                    st.download_button(f"⬇️ Export Cathode CV ({int(selected_sr)} mV/s)", convert_df(df_cathode), f"cathode_processed_CV_{int(selected_sr)}mVs.csv", "text/csv")
        else:
            st.warning("Upload a CSV with multiple scan rates to run Full Sweep interpolation.")
else:
    st.info("👈 Please upload a CSV file in the sidebar to begin analysis.")
