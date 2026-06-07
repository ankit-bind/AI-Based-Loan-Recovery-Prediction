import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from utils.styles import inject_css, set_page_config
from utils.components import render_section_label, render_section_title, render_health_banner
from utils.insights import get_probability_interpretation, log_prediction

set_page_config("Portfolio Analysis", icon="", layout="wide")
inject_css()

render_section_label("Batch Processing")
render_section_title("Portfolio Analysis")

st.markdown("""
<p style="font-size:14px;         color:#F0F2F6; line-height:1.65; margin-bottom:1rem;">
Upload a CSV of borrower accounts to process predictions in bulk. Results are risk-sorted with critical accounts surfaced first.
</p>
""", unsafe_allow_html=True)

# Check model readiness
model_ready = (
    Path("models/best_model.pkl").exists() and
    Path("models/threshold.pkl").exists() and
    Path("models/preprocessor.pkl").exists()
)

if not model_ready:
    st.warning("Model artifacts not found. Please run the training pipeline first:")
    st.code("python main.py", language="bash")
    st.stop()

# ── Upload Zone ──
st.markdown("### Upload Portfolio CSV")
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="File must contain the same raw columns as the Single Borrower page. "
         "Up to 10,000 rows supported per batch."
)

if uploaded_file is not None:
    try:
        from src.loanrecovery.pipeline.prediction_pipeline import PredictPipeline

        df = pd.read_csv(uploaded_file)
        st.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

        if df.empty:
            st.error("Uploaded file is empty.")
            st.stop()

        # Show sample
        with st.expander("Preview uploaded data", expanded=False):
            st.dataframe(df.head(20))

        if st.button("Run Batch Prediction", type="primary", use_container_width=True):
            with st.spinner(f"Processing {len(df):,} borrowers..."):
                # Save uploaded file to temp path for predict_batch
                import tempfile, os
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as tmp:
                    tmp_path = tmp.name
                    df.to_csv(tmp_path, index=False)
                try:
                    pipeline = PredictPipeline()
                    results = pipeline.predict_batch(tmp_path)
                finally:
                    os.remove(tmp_path)

            # Log each row to history
            for _, row in results.iterrows():
                log_prediction(
                    borrower_id=str(row.get("SK_ID_CURR", f"BATCH-{_.index}")),
                    probability=float(row["PROBABILITY"]),
                    prediction=int(row["PREDICTION"]),
                    threshold=float(row["THRESHOLD_USED"]),
                    risk_label=row["RISK_LABEL"],
                    input_features=len(df.columns)
                )

            st.success(f"Batch complete! {len(results):,} borrowers processed.")

            # ── Summary Metrics ──
            total = len(results)
            recovered = (results["PREDICTION"] == 1).sum()
            not_recovered = total - recovered
            avg_prob = results["PROBABILITY"].mean()

            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Total Accounts", f"{total:,}")
            mcol2.metric("Recovered", f"{recovered:,}", f"{recovered/total:.1%}")
            mcol3.metric("Not Recovered", f"{not_recovered:,}", f"{not_recovered/total:.1%}")
            mcol4.metric("Avg Probability", f"{avg_prob:.2%}")

            # ── Risk Distribution ──
            st.markdown("### Risk Distribution")
            risk_colors = {"Recovered": "#6BBF2A", "Not Recovered": "#E24B4A"}
            chart_data = results["RISK_LABEL"].value_counts().reset_index()
            chart_data.columns = ["Risk Label", "Count"]
            st.bar_chart(chart_data.set_index("Risk Label"))

            # ── Top Critical Accounts ──
            st.markdown("### Top Critical Accounts (Lowest Probability)")
            critical = results.nsmallest(20, "PROBABILITY")[
                ["SK_ID_CURR" if "SK_ID_CURR" in results.columns else results.columns[0],
                 "PROBABILITY", "RISK_LABEL"]
            ]
            st.dataframe(critical, use_container_width=True)

            # ── Download Results ──
            st.markdown("### Download Results")
            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Full Results CSV",
                data=csv,
                file_name=f"portfolio_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

            # ── Detailed Results Table ──
            with st.expander("View Full Results Table", expanded=False):
                st.dataframe(results, use_container_width=True)

    except Exception as e:
        st.error(f"Error during batch prediction: {e}")
        st.info("Please ensure the CSV contains valid raw borrower columns matching the training data.")

else:
    st.info("Upload a CSV file to begin batch prediction.")

st.markdown("---")
st.caption("**Tip:** For best results, ensure your CSV contains the same column names as the Single Borrower input form.")
