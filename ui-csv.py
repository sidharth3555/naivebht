import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Load the trained model
with open("best_kidney_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="CKD CSV Predictor", layout="wide")
st.title("🧪 Chronic Kidney Disease model tester")
st.write("Upload a CSV file with  to predict CKD and view visual summaries.")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload CSV file ", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("🔍 Uploaded Data Preview")
        st.dataframe(df.head())

        expected_cols = 25
        if df.shape[1] != expected_cols:
            st.error(f"❌ This file has {df.shape[1]} columns, but 25 are required.")
            st.stop()
        else:
            st.success("✅ CSV column count is correct.")

        # Make predictions
        preds = model.predict(df)
        probas = model.predict_proba(df)[:, 1] if hasattr(model, "predict_proba") else None

        result_df = df.copy()
        result_df["Prediction"] = ["CKD" if p == 1 else "No CKD" for p in preds]
        if probas is not None:
            result_df["Confidence"] = (probas * 100).round(2).astype(str) + "%"

        st.subheader("✅ Prediction Results")
        st.dataframe(result_df)

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results as CSV", csv, "ckd_predictions.csv", "text/csv")

        # Binary feature plot
        st.subheader("📊 Binary Feature Distribution")
        binary_cols = [col for col in df.columns if set(df[col].dropna().unique()).issubset({0, 1})]
        if binary_cols:
            bin_selected = st.selectbox("Select a binary feature", binary_cols)
            st.bar_chart(df[bin_selected].value_counts())
        else:
            st.warning("No binary features found.")

        # Correlation heatmap
        st.subheader("📈 Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] >= 2:
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
            st.pyplot(fig)

        # Pairplot with fake labels
        st.subheader("🔬 Pairplot of Key Features")
        sel = st.multiselect("Select 2-5 features", numeric_df.columns.tolist(), default=numeric_df.columns[:5].tolist())
        if 2 <= len(sel) <= 5:
            plot_df = df.copy()
            plot_df["Label"] = result_df["Prediction"]
            fig_pair = sns.pairplot(plot_df[sel + ["Label"]], hue="Label")
            st.pyplot(fig_pair)

    except Exception as e:
        st.error(f"⚠️ Error reading or processing file: {e}")
else:
    st.info("Awaiting CSV file upload...")
