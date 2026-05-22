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

# --- Interactive Tabs ---
tab1, tab2, tab3 = st.tabs(["1. CV & Capacitance", "2. Peak Extraction (b-value)", "3. Dunn's Method"])

# ==========================================
# TAB 1: BASIC CV & CAPACITANCE
# ==========================================
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Experimental Parameters")
        uploaded_file = st.file_uploader("Upload Raw CV (CSV)", type=["csv"], key="file1")
        mass = st.number_input("Active Mass (grams)", min_value=0.0001, value=0.1000, format="%.4f")
        data_unit = st.selectbox("Current Unit in CSV", ["A", "mA", "uA"], index=1)
        plot_unit = st.selectbox("Plot Unit", ["A", "mA", "uA"], index=1)
        
    with col2:
        if uploaded_file:
            # Read the CSV
            data = pd.read_csv(uploaded_file)
            
            # Force everything to be a number. Strings like "V" or "mA" become NaN.
            data = data.apply(pd.to_numeric, errors='coerce') 
            
            to_A = factor_to_A(data_unit)
            A_to_plot = factor_from_A(plot_unit)
            
            potential_cols = data.columns[::2]
            current_cols = data.columns[1::2]
            
            results = []
            fig, ax = plt.subplots(figsize=(8, 6))
            
            for i, (pot_col, cur_col) in enumerate(zip(potential_cols, current_cols)):
                if "Unnamed" in str(cur_col): continue
                
                # Drop the NaNs (removes the unit row)
                clean_data = pd.DataFrame({"V": data[pot_col], "I": data[cur_col]}).dropna()
                
                if clean_data.empty: continue
                
                voltage = clean_data["V"].astype(float)
                current_A = clean_data["I"].astype(float) * to_A
                
                scan_rate = float((i + 1) * 10) # Assuming 10, 20, 30...
                
                # Calculate Area & Capacitance using updated numpy 2.0 syntax
                area = np.trapezoid(current_A, voltage)
                V1, V2 = voltage.min(), voltage.max()
                
                spec_cap = area / (mass * scan_rate * (V2 - V1))
                spec_capacity = area / (mass * scan_rate)
                
                results.append({"Scan Rate (mV/s)": scan_rate, "Capacitance (F/g)": spec_cap})
                
                # Plot
                ax.plot(voltage, current_A * A_to_plot, label=f"{scan_rate} mV/s")
            
            ax.set_xlabel("Potential (V)")
            ax.set_ylabel(f"Specific Current ({plot_unit}/g)")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)
            st.dataframe(pd.DataFrame(results), use_container_width=True)

# ==========================================
# TAB 2: PEAK EXTRACTION
# ==========================================
with tab2:
    st.subheader("Power Law Analysis ($i = a v^b$)")
    st.write("Extract anodic/cathodic currents at specific potentials.")
    if uploaded_file:
        c1, c2 = st.columns(2)
        anodic_v = c1.number_input("Anodic Potential (V)", value=0.5)
        cathodic_v = c2.number_input("Cathodic Potential (V)", value=-0.5)
        st.info("Upload processing logic for peak extraction goes here based on your 'start.py' logic.")

# ==========================================
# TAB 3: DUNN's METHOD
# ==========================================
with tab3:
    st.subheader("Dunn Method ($k_1, k_2$)")
    st.write("Upload the slope/intercept CSVs generated from Step 2 to calculate contributions.")
    dunn_file = st.file_uploader("Upload Anodic/Cathodic CSV", type=["csv"], key="file2")
    if dunn_file:
        df = pd.read_csv(dunn_file)
        st.write(df.head())
        st.success("Ready to calculate Capacitive vs Diffusion %!")
