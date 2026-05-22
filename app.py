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

# --- Page Config ---
st.set_page_config(page_title="Electrochem Dashboard", layout="wide")
st.title("🔋 Advanced CV & Dunn Method Analyzer (Origin Export)")

# --- Global Sidebar Setup ---
st.sidebar.header("1. Upload & Global Settings")
uploaded_file = st.sidebar.file_uploader("Upload Raw CV (CSV)", type=["csv"])
mass = st.sidebar.number_input("Active Mass (grams)", min_value=0.0001, value=0.1000, format="%.4f")
data_unit = st.sidebar.selectbox("Current Unit in CSV", ["A", "mA", "uA"], index=1)
plot_unit = st.sidebar.selectbox("Desired Output Unit", ["A", "mA", "uA"], index=1)

# --- Interactive Tabs ---
tab1, tab2, tab3 = st.tabs(["1. CV & Capacitance", "2. Peak Extraction (b-value)", "3. Full Dunn's Method Data Export"])

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
        st.subheader("Cyclic Voltammetry Data & Capacitance")
        fig, ax = plt.subplots(figsize=(8, 6))
        results = []
        cv_export_dfs = [] # List to store data for Origin export
        
        for item in valid_data:
            v = item["voltage"]
            i_A = item["current_A"]
            sr = item["scan_rate"]
            i_plot = i_A * A_to_plot
            
            # Store data for CSV export
            temp_df = pd.DataFrame({
                "Scan_Rate_mVs": sr,
                "Voltage_V": v,
                f"Current_{plot_unit}_per_g": i_plot
            })
            cv_export_dfs.append(temp_df)
            
            # Use trapezoid for area calculation
            try:
                area = np.trapezoid(i_A, v)
            except AttributeError:
                area = np.trapz(i_A, v) # Fallback for older numpy
                
            V1, V2 = v.min(), v.max()
            spec_cap = np.abs(area) / (mass * sr * 1e-3 * (V2 - V1)) # sr in V/s for Farads
            
            results.append({"Scan Rate (mV/s)": sr, "Capacitance (F/g)": spec_cap})
            ax.plot(v, i_plot, label=f"{sr} mV/s")
            
        ax.set_xlabel("Potential (V)")
        ax.set_ylabel(f"Specific Current ({plot_unit}/g)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)
        
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        
        # Origin Export Button
        df_all_cvs = pd.concat(cv_export_dfs, ignore_index=True)
        st.download_button(
            label="⬇️ Download All Cleaned CV Plot Data (for Origin)",
            data=df_all_cvs.to_csv(index=False).encode('utf-8'),
            file_name="cleaned_cv_data_all_scans.csv",
            mime="text/csv"
        )

    # ==========================================
    # TAB 2: PEAK EXTRACTION
    # ==========================================
    with tab2:
        st.subheader("Peak Extraction & Reaction Kinetics ($i = a v^b$)")
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
        
        # Generate Export DataFrame for Origin
        df_peaks = pd.DataFrame({
            "Scan_Rate_mVs": sr_arr,
            "log_v": log_sr,
            "Anodic_Peak_Current": a_i_arr,
            "Cathodic_Peak_Current": c_i_arr,
            "log_Anodic_i": np.log(np.abs(a_i_arr)),
            "log_Cathodic_i": np.log(np.abs(c_i_arr))
        })
        
        st.dataframe(df_peaks, use_container_width=True)
        st.download_button(
            label="⬇️ Download Power Law Peak Data (for Origin)",
            data=df_peaks.to_csv(index=False).encode('utf-8'),
            file_name="power_law_peak_data.csv",
            mime="text/csv"
        )

    # ==========================================
    # TAB 3: FULL DUNN'S METHOD EXPORT
    # ==========================================
    with tab3:
        st.subheader("Dunn's Method Data Generator")
        st.write("This engine calculates k1 and k2 for the entire CV curve in the background. Select a scan rate below to generate the exact data files needed to plot the shaded capacitive contributions in Origin.")
        
        if len(valid_data) > 1:
            v_min = max([np.min(item['voltage']) for item in valid_data])
            v_max = min([np.max(item['voltage']) for item in valid_data])
            
            # Common voltage grid for interpolation
            v_grid_anodic = np.linspace(v_min, v_max, 500)
            v_grid_cathodic = np.linspace(v_max, v_min, 500)
            
            scan_rates_list = []
            anodic_currents_interp = []
            cathodic_currents_interp = []
            
            for item in valid_data:
                v = item['voltage']
                i = item['current_A'] * A_to_plot
                
                # Simple split based on voltage gradient
                dv = np.diff(v)
                split_idx = np.where(np.sign(dv[:-1]) != np.sign(dv[1:]))[0]
                
                if len(split_idx) >= 1:
                    idx_turn = split_idx[0] + 1
                    # Ensure correct sweep directions (anodic goes up, cathodic goes down)
                    if v[1] > v[0]: 
                        v_an, i_an = v[:idx_turn], i[:idx_turn]
                        v_cat, i_cat = v[idx_turn:], i[idx_turn:]
                    else:
                        v_cat, i_cat = v[:idx_turn], i[:idx_turn]
                        v_an, i_an = v[idx_turn:], i[idx_turn:]
                        
                    # Remove duplicates for interpolation
                    v_an, unique_an = np.unique(v_an, return_index=True)
                    i_an = i_an[unique_an]
                    
                    # Cathodic needs to be sorted for interpolation
                    v_cat_sorted, unique_cat = np.unique(v_cat, return_index=True)
                    i_cat_sorted = i_cat[unique_cat]
                    
                    interp_an = interp1d(v_an, i_an, bounds_error=False, fill_value="extrapolate")
                    interp_cat = interp1d(v_cat_sorted, i_cat_sorted, bounds_error=False, fill_value="extrapolate")
                    
                    anodic_currents_interp.append(interp_an(v_grid_anodic))
                    cathodic_currents_interp.append(interp_cat(v_grid_cathodic))
                    scan_rates_list.append(item['scan_rate'])
            
            sr_array = np.array(scan_rates_list)
            sqrt_sr = np.sqrt(sr_array)
            
            # 2. Calculate k1 and k2 for every voltage point
            k1_anodic, k2_anodic = np.zeros_like(v_grid_anodic), np.zeros_like(v_grid_anodic)
            k1_cathodic, k2_cathodic = np.zeros_like(v_grid_cathodic), np.zeros_like(v_grid_cathodic)
            
            for idx in range(len(v_grid_anodic)):
                i_an = np.array([curve[idx] for curve in anodic_currents_interp])
                res_an = linregress(sqrt_sr, i_an / sqrt_sr)
                k1_anodic[idx] = res_an.slope
                k2_anodic[idx] = res_an.intercept
                
                i_cat = np.array([curve[idx] for curve in cathodic_currents_interp])
                res_cat = linregress(sqrt_sr, i_cat / sqrt_sr)
                k1_cathodic[idx] = res_cat.slope
                k2_cathodic[idx] = res_cat.intercept

            # 3. User selects a scan rate to download
            st.markdown("---")
            selected_sr = st.selectbox("Select Scan Rate to Generate Processed CSVs:", sr_array)
            
            if selected_sr:
                # Reconstruct components for the chosen scan rate
                i_cap_anodic = k1_anodic * selected_sr
                i_diff_anodic = k2_anodic * np.sqrt(selected_sr)
                i_total_anodic = i_cap_anodic + i_diff_anodic
                
                i_cap_cathodic = k1_cathodic * selected_sr
                i_diff_cathodic = k2_cathodic * np.sqrt(selected_sr)
                i_total_cathodic = i_cap_cathodic + i_diff_cathodic
                
                # --- Export CSV Logic ---
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
                
                st.success(f"Data for {selected_sr} mV/s processed successfully. Ready for Origin import.")
                
                c3, c4 = st.columns(2)
                with c3:
                    st.download_button(
                        label="⬇️ Download Anode Processed CSV",
                        data=df_anode.to_csv(index=False).encode('utf-8'),
                        file_name=f"anode_processed_CV_{int(selected_sr)}mVs.csv",
                        mime="text/csv"
                    )
                with c4:
                    st.download_button(
                        label="⬇️ Download Cathode Processed CSV",
                        data=df_cathode.to_csv(index=False).encode('utf-8'),
                        file_name=f"cathode_processed_CV_{int(selected_sr)}mVs.csv",
                        mime="text/csv"
                    )
        else:
            st.warning("Upload a CSV with multiple scan rates to run Dunn's Method.")
else:
    st.info("👈 Please upload a CSV file in the sidebar to begin analysis.")
