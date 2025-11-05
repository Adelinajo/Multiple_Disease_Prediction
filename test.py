import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==============================================================
# 🧠 LOAD MODELS AND SCALERS 
# ==============================================================
@st.cache_resource(show_spinner=False)
def load_all_models():
    try:
        with open("kidney_model.pkl", "rb") as f:
            kidney_model = pickle.load(f)
        with open("kidney_scaler.pkl", "rb") as f:
            kidney_scaler = pickle.load(f)

        with open("parkinson_model.pkl", "rb") as f:
            parkinson_model = pickle.load(f)
        with open("parkinson_scaler.pkl", "rb") as f:
            parkinson_scaler = pickle.load(f)

        with open("liver2_model.pkl", "rb") as f:
            liver_model = pickle.load(f)
        with open("liver2_scaler.pkl", "rb") as f:
            liver_scaler = pickle.load(f)

        return kidney_model, kidney_scaler, parkinson_model, parkinson_scaler, liver_model, liver_scaler

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None


(kidney_model, kidney_scaler,
 parkinson_model, parkinson_scaler,
 liver_model, liver_scaler) = load_all_models()


# ==============================================================
# 🔠 MAPPING DICTIONARY (GLOBAL)
# ==============================================================
mapping_dict = {
    'yes': 1, 'no': 0,
    'abnormal': 1, 'normal': 0,
    'present': 1, 'notpresent': 0,
    'good': 1, 'poor': 0,
    'ckd': 1, 'notckd': 0,
    'male': 1, 'female': 0
}

# Expected columns for CKD model
expected_cols = ['al', 'bgr', 'bu', 'su', 'bp', 'sc', 'age', 'pot', 'sod']


# ==============================================================
# 🧹 PREPROCESSING FUNCTION FOR BATCH DATA
# ==============================================================
def preprocess_batch_data(df, mapping_dict, expected_cols):
    df = df.copy()

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # Map categorical values
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.lower().map(mapping_dict).fillna(df[col])

    # Keep only expected columns
    df = df[[c for c in expected_cols if c in df.columns]]

    # Convert numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Fill missing numeric values with median
    df = df.fillna(df.median(numeric_only=True))

    return df

# ===============================================================
# 🎨 PAGE STYLING
# ===============================================================
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #0da175;
            color: white;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
            color: white !important;
        }
        .main { background-color: #ECEFF1; }
        .header {
            text-align: center;
            background-color: #0da175;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .predict-box {
            background-color: #e0e0e0;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        }
        .stTabs [role="tablist"] { justify-content: center; }
    </style>
""", unsafe_allow_html=True)



# ==============================================================
# 🩺 Sidebar Selection
# ==============================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966481.png", width=80)
st.sidebar.title("🏥 Mutiple Disease Prediction Dashboard")

disease = st.sidebar.radio(
    "🧠 Select Disease to Predict",
    ["Chronic Kidney Disease", "Parkinson's Disease", "Indian Liver Disease"]
)

# ==============================================================
# 🧮 Chronic Kidney Disease Prediction
# ==============================================================
if disease == "Chronic Kidney Disease":
    st.markdown("<div class='header'><h1>🧫 Chronic Kidney Disease Prediction</h1></div>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🧍‍♂️ Single Prediction", "📂 Batch Prediction"])


# ==============================================================
# 🧍‍♂️ SINGLE PREDICTION
# ==============================================================
    with tab1:
       st.write("Enter the following patient details:")
       col1, col2 = st.columns(2)

       with col1:
           al = st.number_input("Albumin (al)", 0.000, 5.000, step=0.1)
           bgr = st.number_input("Blood Glucose Random (bgr)", 0.0, 500.0, step=1.0)
           bu = st.number_input("Blood Urea (bu)", 0.0, 300.0, step=1.0)
           su = st.number_input("Sugar (su)", 0.0, 5.0, step=0.1)
           bp = st.number_input("Blood Pressure (bp)", 0.0, 200.0, step=1.0)

       with col2:
           sc = st.number_input("Serum Creatinine (sc)", 0.00, 15.00, step=0.1)
           age = st.number_input("Age", 1, 120, step=1)
           pot = st.number_input("Potassium (pot)", 0.0, 10.0, step=0.1)
           sod = st.number_input("Sodium (sod)", 0.0, 200.0, step=1.0)

           st.markdown("<br>", unsafe_allow_html=True)

       if st.button("🔍 Predict Kidney Disease", use_container_width=True):
 
            if kidney_model and kidney_scaler:
                    input_data = np.array([[al, bgr, bu, su, bp, sc, age, pot, sod]])
                    scaled_input = kidney_scaler.transform(input_data)
                    pred = kidney_model.predict(scaled_input)[0]
                    prob = kidney_model.predict_proba(scaled_input)[0][1]

                    if pred == 1:
                      st.error(f"🚨 Likely to have **Chronic Kidney Disease** (Probability: {prob:.2f})")
                    else:
                      st.success(f"✅ Not likely to have Chronic Kidney Disease (Probability: {prob:.2f})")
            else:
                st.warning("⚠️ Model or scaler missing!")

    
# ==============================================================
# 📂 BATCH PREDICTION
# ==============================================================
    with tab2:
       st.subheader("📂 Batch Kidney Disease Prediction")

       uploaded_file = st.file_uploader("📁 Upload CSV file (with all 26 columns)", type=["csv"])

       if uploaded_file is not None:
            try:
                uploaded_file.seek(0)  # Reset pointer before reading
                df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")
                st.dataframe(df)

            # --- Batch Prediction Button ---
                if st.button("🔍 Run Batch Prediction"):
                
                    if kidney_model and kidney_scaler:
                        with st.spinner("🔄 Preprocessing and predicting..."):
                            expected_cols = ['al', 'bgr', 'bu', 'su', 'bp', 'sc', 'age', 'pot', 'sod']
                            df_clean = preprocess_batch_data(df, mapping_dict, expected_cols)

                            st.write("✅ After preprocessing:")
                            st.dataframe(df_clean.head())

                        # Scale & predict
                            scaled_df = kidney_scaler.transform(df_clean)
                            preds = kidney_model.predict(scaled_df)
                            probs = kidney_model.predict_proba(scaled_df)[:, 1]

                        # Add predictions
                            threshold = 0.34 
                            df["Probability_CKD"] = probs
                            df["Prediction"] = np.where(probs >= threshold, "ChronicKidneyDisease", "Healthy")

                            st.success("✅ Batch Prediction Complete!")
                            st.dataframe(df)

                        # Download button
                            csv = df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="⬇️ Download Predictions",
                                data=csv,
                                file_name="kidney_batch_predictions.csv",
                                mime="text/csv"
                                )
                    else:
                        st.warning("⚠️ Model or Scaler not found!")

            except Exception as e:
               st.error(f"❌ Error processing file: {e}")

# ==============================================================
# 🧠 Parkinson’s Disease Prediction
# ==============================================================
if disease == "Parkinson's Disease":
    st.markdown("<div class='header'><h1>🧠 Parkinson's Disease Prediction</h1></div>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🧍‍♂️ Single Prediction", "📂 Batch Prediction"])

    trained_features = [
        'spread1', 'PPE', 'spread2', 'MDVP:Shimmer', 'MDVP:APQ',
        'Shimmer:APQ5', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3',
        'Shimmer:DDA', 'D2'
    ]
    threshold = 0.6

     # ---------------- SINGLE PREDICTION ---------------- #

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            spread1 = st.number_input("Spread1", value=-6.0)
            PPE = st.number_input("PPE", value=0.15)
            spread2 = st.number_input("Spread2", value=0.18)
            MDVP_Shimmer = st.number_input("MDVP:Shimmer", value=0.016)
            MDVP_APQ = st.number_input("MDVP:APQ", value=0.013)
        with col2:
            Shimmer_APQ5 = st.number_input("Shimmer:APQ5", value=0.009)
            MDVP_Shimmer_dB = st.number_input("MDVP:Shimmer(dB)", value=0.14)
            Shimmer_APQ3 = st.number_input("Shimmer:APQ3", value=0.008)
            Shimmer_DDA = st.number_input("Shimmer:DDA", value=0.025)
            D2 = st.number_input("D2", value=2.2)

        if st.button("🔍 Predict Parkinson’s"):
            input_data = pd.DataFrame([[spread1, PPE, spread2, MDVP_Shimmer, MDVP_APQ,
                                        Shimmer_APQ5, MDVP_Shimmer_dB, Shimmer_APQ3,
                                        Shimmer_DDA, D2]], columns=trained_features)
            scaled = parkinson_scaler.transform(input_data)
            prob = parkinson_model.predict_proba(scaled)[:, 1][0]
            pred = 1 if prob >= threshold else 0
            if pred == 1:
                st.error(f"🧠 High chance of Parkinson’s Disease (Prob = {prob:.2f})")
            else:
                st.success(f"✅ Healthy (Prob = {prob:.2f})")

       # ---------------- BATCH PREDICTION ---------------- #            

    with tab2:
        st.subheader("Upload CSV File for Batch Prediction")
        uploaded_file = st.file_uploader("Upload Parkinson Dataset (CSV)", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
            run_batch = st.button("📊 Run Parkinson Batch Prediction", type="primary")
            if run_batch:
                df_batch = df[trained_features]
                scaled = parkinson_scaler.transform(df_batch)
                probs = parkinson_model.predict_proba(scaled)[:, 1]
                preds = (probs >= threshold).astype(int)
                df["Predicted Probability"] = probs
                df["Predicted Label"] = np.where(preds == 1, "Parkinson Disease", "Healthy")
                st.dataframe(df)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Predictions", csv, "Parkinson_Predictions.csv", "text/csv")


# ==============================================================
# 🩸 Indian Liver Disease Prediction
# ==============================================================
if disease == "Indian Liver Disease":
    st.markdown("<div class='header'><h1>🩸 Indian Liver Disease Prediction</h1></div>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🧍‍♂️ Single Prediction", "📂 Batch Prediction"])

    # ---------------- SINGLE PREDICTION ---------------- #
    with tab1:
        st.subheader("Enter Patient Data")
        col1, col2 = st.columns(2)
        with col1:
            Age = st.number_input("Age", value=45)
            Gender = st.selectbox("Gender", ["Male", "Female"])
            Total_Bilirubin = st.number_input("Total Bilirubin", value=1.2)
            Direct_Bilirubin = st.number_input("Direct Bilirubin", value=0.3)
            Alkaline_Phosphotase = st.number_input("Alkaline Phosphotase", value=200)
        with col2:
            Alamine_Aminotransferase = st.number_input("Alamine Aminotransferase", value=30)
            Aspartate_Aminotransferase = st.number_input("Aspartate Aminotransferase", value=35)
            Total_Protiens = st.number_input("Total Protiens", value=6.8)
            Albumin = st.number_input("Albumin", value=3.5)
            Albumin_and_Globulin_Ratio = st.number_input("Albumin and Globulin Ratio", value=1.0)

        if st.button("🔍 Predict Liver Disease"):
            input_data = pd.DataFrame([[Age, 1 if Gender == "Male" else 0, Total_Bilirubin,
                                        Direct_Bilirubin, Alkaline_Phosphotase,
                                        Alamine_Aminotransferase, Aspartate_Aminotransferase,
                                        Total_Protiens, Albumin, Albumin_and_Globulin_Ratio]],
                                      columns=['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
                                               'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                                               'Aspartate_Aminotransferase', 'Total_Protiens',
                                               'Albumin', 'Albumin_and_Globulin_Ratio'])

            scaled = liver_scaler.transform(input_data)
            prob = liver_model.predict_proba(scaled)[:, 1][0]
            pred = 1 if prob >= 0.10 else 0
            if pred == 1:
                st.error(f"⚠️ High chance of Liver Disease (Prob = {prob:.2f})")
            else:
                st.success(f"✅ Healthy (Prob = {prob:.2f})")

    # ---------------- BATCH PREDICTION ---------------- #
    with tab2:
        uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            if st.button("🚀 Run Batch Prediction"):
                df_original = df.copy()
                if 'Gender' in df.columns:
                    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0}).fillna(0)
                df = df.fillna(df.median(numeric_only=True))
                df = df.drop(columns=['Dataset'], errors='ignore')
                df_scaled = liver_scaler.transform(df)
                y_prob = liver_model.predict_proba(df_scaled)[:, 1]
                y_pred = (y_prob >= 0.10).astype(int)
                df_original['Prediction'] = np.where(y_pred == 1, 'Liver Disease', 'Healthy')
                df_original['Probability'] = y_prob.round(3)
                st.dataframe(df_original)
                csv = df_original.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Download Full Results as CSV", data=csv, file_name="Liver_Disease_Predictions.csv", mime="text/csv")

