import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
with open("best_kidney_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define input features (24 real + 1 dummy placeholder to match 25 input features)
feature_ranges = {
    'age': (0, 100),
    'bp': (50, 180),
    'sg': (1.000, 1.030),
    'al': (0, 5),
    'su': (0, 5),
    'bgr': (50, 500),
    'bu': (1, 300),
    'sc': (0.1, 15.0),
    'sod': (100, 160),
    'pot': (2.0, 8.0),
    'hemo': (3.0, 18.0),
    'pcv': (10, 60),
    'wc': (2000, 20000),
    'rc': (2.0, 7.0),
    'rbc': (0, 1),
    'pc': (0, 1),
    'pcc': (0, 1),
    'ba': (0, 1),
    'htn': (0, 1),
    'dm': (0, 1),
    'cad': (0, 1),
    'appet': (0, 1),
    'pe': (0, 1),
    'ane': (0, 1),
    'dummy_placeholder': (0, 1)  # TEMP fix to match model input shape
}

# Streamlit UI setup
st.set_page_config(page_title="Kidney Disease Predictor", layout="centered")
st.title("🧪 Kidney Disease Prediction App")
st.write("Enter patient data below to predict the likelihood of Chronic Kidney Disease (CKD).")

# User input form
with st.form("input_form"):
    user_input = []
    for feature, (min_val, max_val) in feature_ranges.items():
        if isinstance(min_val, float):
            val = st.number_input(f"{feature.upper()}", min_value=min_val, max_value=max_val, step=0.1)
        elif max_val == 1:
            val = st.selectbox(f"{feature.upper()} (Yes=1, No=0)", options=[0, 1])
        else:
            val = st.number_input(f"{feature.upper()}", min_value=min_val, max_value=max_val)
        user_input.append(val)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([user_input], columns=list(feature_ranges.keys()))

    try:
        # Check model input shape
        model_input_shape = model.n_features_in_ if hasattr(model, 'n_features_in_') else len(input_df.columns)
        if input_df.shape[1] != model_input_shape:
            raise ValueError(f"Feature mismatch: Model expects {model_input_shape} features, but input has {input_df.shape[1]} features.")
    except AttributeError:
        pass

    # Predict and show results
    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
    except ValueError as e:
        st.error(f"Error in prediction: {e}")
        st.stop()

    if prediction == 1:
        st.error("🚨 Warning: Patient is likely to have Chronic Kidney Disease.")
    else:
        st.success("🎉 Good news: Patient is unlikely to have CKD.")

    if proba is not None:
        st.metric(label="Prediction Confidence", value=f"{proba * 100:.2f}%")

    # Visual summary
    st.subheader("📊 Input Feature Summary")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=input_df.columns, y=input_df.values[0], ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    # Mock correlation heatmap
    st.subheader("📈 Feature Correlation (Synthetic Sample)")
    sample_data = pd.DataFrame(np.random.normal(loc=input_df.values, scale=0.1, size=(100, len(input_df.columns))), columns=input_df.columns)
    corr = sample_data.corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax_corr)
    st.pyplot(fig_corr)

    # Pairplot example
    st.subheader("📉 Feature Distribution (Sample Data)")
    sample_data['CKD'] = np.random.choice([0, 1], size=len(sample_data))
    selected_features = ['age', 'bp', 'sc', 'hemo', 'bgr', 'CKD']
    fig_pair = sns.pairplot(sample_data[selected_features], hue="CKD")
    st.pyplot(fig_pair)

st.caption("Developed using Streamlit and scikit-learn.")
