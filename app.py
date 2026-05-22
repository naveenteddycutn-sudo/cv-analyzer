import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

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
st.title("🔋 Advanced CV & Dunn Method Analyzer")

# --- Global Sidebar Setup ---
st.sidebar.header("1. Upload & Global Settings")
uploaded_file = st.sidebar.file_uploader("Upload Raw CV (CSV)", type=["csv"])
mass = st.sidebar.number_input("Active Mass (grams)", min_value=0.0001, value=0.1000, format="%.4f")
data_unit = st.sidebar.selectbox("Current Unit in CSV", ["A", "mA", "uA"], index=1)
plot_unit = st.sidebar.selectbox("Plot Unit", ["A", "mA", "uA"], index=1)

# --- Interactive Tabs ---
tab1, tab2, tab3 = st.tabs(["1. CV & Capacitance", "2. Peak Extraction (b-value & k1/k2)", "3. Dunn's Method"])

if uploaded_file:
    # Read and clean the data globally so all tabs can use it
    uploaded_file.seek(0)
    data = pd.read_csv(uploaded_file).apply(pd.to_numeric, errors='coerce')
    to_A = factor_to_A(data_unit)
    A_to_plot = factor_from_A(plot_unit)
    
    potential_cols = data.columns[::2]
    current_cols = data.columns[1::2]
    
    # Store clean data arrays to avoid repeating loops
    valid_data = []
    for i, (pot_col, cur_col) in enumerate(zip(potential_cols, current_cols)):
        if "Unnamed" in str(cur_col): continue
        clean_data = pd.DataFrame({"V": data[pot_col], "I": data[cur_col]}).dropna()
        if clean_data.empty: continue
        
        sr = float((i + 1) * 10) # Assumes 10, 20, 30... mV/s
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
        
        for item in valid_data:
            v = item["voltage"]
            i_A = item["current_A"]
            sr = item["scan_rate"]
            
            area = np.trapezoid(i_A, v)
            V1, V2 = v.min(), v.max()
            
            spec_cap = area / (mass * sr * (V2 - V1))
            
            results.append({"Scan Rate (mV/s)": sr, "Area (A·V/g)": area, "Capacitance (F/g)": spec_cap})
            ax.plot(v, i_A * A_to_plot, label=f"{sr} mV/s")
            
        ax.set_xlabel("Potential (V)")
        ax.set_ylabel(f"Specific Current ({plot_unit}/g)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)
        
        st.dataframe(pd.DataFrame(results), use_container_width=True)

    # ==========================================
    # TAB 2: PEAK EXTRACTION
    # ==========================================
    with tab2:
        st.subheader("Peak Extraction & Reaction Kinetics")
        c1, c2 = st.columns(2)
        anodic_v = c1.number_input("Anodic Peak Potential (V)", value=0.50, step=0.05)
        cathodic_v = c2.number_input("Cathodic Peak Potential (V)", value=-0.50, step=0.05)
        
        scan_rates = []
        anodic_i = []
        cathodic_i = []
        
        for item in valid_data:
            v_arr = item["voltage"]
            i_arr_plot = item["current_A"] * A_to_plot 
            sr = item["scan_rate"]
            
            scan_rates.append(sr)
            
            # Find closest voltage to user input and extract that current
            idx_a = (np.abs(v_arr - anodic_v)).argmin()
            anodic_i.append(i_arr_plot[idx_a])
            
            idx_c = (np.abs(v_arr - cathodic_v)).argmin()
            cathodic_i.append(i_arr_plot[idx_c])
            
        sr_arr = np.array(scan_rates)
        a_i_arr = np.array(anodic_i)
        c_i_arr = np.array(cathodic_i)
        
        # --- 1. Power Law (b-value) ---
        st.markdown("---")
        st.markdown("### 1. Power Law ($i = a v^b$)")
        log_sr = np.log(sr_arr)
        log_a_i = np.log(np.abs(a_i_arr))
        log_c_i = np.log(np.abs(c_i_arr))
        
        fit_a_b = linregress(log_sr, log_a_i)
        fit_c_b = linregress(log_sr, log_c_i)
        
        cb1, cb2 = st.columns(2)
        with cb1:
            fig1, ax1 = plt.subplots(figsize=(5,4))
            ax1.scatter(log_sr, log_a_i, color='black')
            ax1.plot(log_sr, fit_a_b.slope * log_sr + fit_a_b.intercept, 'r-', label=f"b = {fit_a_b.slope:.3f}")
            ax1.set_xlabel("log(v)")
            ax1.set_ylabel("log(i)")
            ax1.set_title("Anodic Peak")
            ax1.legend()
            st.pyplot(fig1)
        with cb2:
            fig2, ax2 = plt.subplots(figsize=(5,4))
            ax2.scatter(log_sr, log_c_i, color='black')
            ax2.plot(log_sr, fit_c_b.slope * log_sr + fit_c_b.intercept, 'b-', label=f"b = {fit_c_b.slope:.3f}")
            ax2.set_xlabel("log(v)")
            ax2.set_ylabel("log(i)")
            ax2.set_title("Cathodic Peak")
            ax2.legend()
            st.pyplot(fig2)
            
        # --- 2. Dunn Method Parameters ---
        st.markdown("---")
        st.markdown("### 2. Dunn Method Parameters ($i/v^{1/2}$ vs $v^{1/2}$)")
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
            ax3.set_xlabel("$v^{1/2}$")
            ax3.set_ylabel("$i / v^{1/2}$")
            ax3.set_title("Anodic Fit")
            ax3.legend()
            st.pyplot(fig3)
        with ck2:
            fig4, ax4 = plt.subplots(figsize=(5,4))
            ax4.scatter(sqrt_v, y_c, color='black')
            ax4.plot(sqrt_v, fit_c_k.slope * sqrt_v + fit_c_k.intercept, 'b-', label=f"k1={fit_c_k.slope:.2e}\nk2={fit_c_k.intercept:.2e}")
            ax4.set_xlabel("$v^{1/2}$")
            ax4.set_ylabel("$i / v^{1/2}$")
            ax4.set_title("Cathodic Fit")
            ax4.legend()
            st.pyplot(fig4)
            
        # Silently pass data to Tab 3
        st.session_state['sr_arr'] = sr_arr
        st.session_state['k_anodic'] = (fit_a_k.slope, fit_a_k.intercept)
        st.session_state['k_cathodic'] = (fit_c_k.slope, fit_c_k.intercept)

    # ==========================================
    # TAB 3: DUNN's METHOD
    # ==========================================
    with tab3:
        st.subheader("Capacitive vs Diffusion Contributions")
        if 'sr_arr' in st.session_state:
            mode = st.radio("Select Peak to Visualize:", ["Anodic", "Cathodic"], horizontal=True)
            
            v = st.session_state['sr_arr']
            if mode == "Anodic":
                k1, k2 = st.session_state['k_anodic']
            else:
                k1, k2 = st.session_state['k_cathodic']
                
            sqrt_v = np.sqrt(v)
            i_cap = k1 * v
            i_diff = k2 * sqrt_v
            
            cap_mag = np.abs(i_cap)
            diff_mag = np.abs(i_diff)
            total_mag = cap_mag + diff_mag
            
            cap_pct = (cap_mag / total_mag) * 100.0
            diff_pct = (diff_mag / total_mag) * 100.0
            
            fig5, ax5 = plt.subplots(figsize=(8,6))
            width = 3.0
            
            bars1 = ax5.bar(v, cap_pct, width=width, color='black', label="Capacitive")
            bars2 = ax5.bar(v, diff_pct, width=width, bottom=cap_pct, color='red', label="Diffusion")
            
            # Add publication-standard data labels
            for bar, cap in zip(bars1, cap_pct):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, f"{cap:.1f}%", 
                         ha='center', va='center', rotation=90, color='white', fontsize=10, fontweight='bold')
            for bar, cap, diff in zip(bars2, cap_pct, diff_pct):
                ax5.text(bar.get_x() + bar.get_width()/2, cap + bar.get_height()/2, f"{diff:.1f}%", 
                         ha='center', va='center', rotation=90, color='white', fontsize=10, fontweight='bold')
                         
            ax5.set_ylim(0, 100)
            ax5.set_xlabel("Scan rate (mV/s)")
            ax5.set_ylabel("Contribution (%)")
            ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig5)
            
            df_cont = pd.DataFrame({"Scan Rate (mV/s)": v, "Capacitive (%)": cap_pct, "Diffusion (%)": diff_pct})
            st.dataframe(df_cont, use_container_width=True)
else:
    st.info("👈 Please upload a CSV file in the sidebar to begin analysis.")
