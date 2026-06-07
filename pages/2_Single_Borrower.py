import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
from pathlib import Path

from utils.styles import inject_css, set_page_config
from utils.components import (
    render_section_label, render_section_title, render_borrower_card,
    render_verdict_card, render_shap_factors, risk_badge
)
from utils.charts import gauge_chart, plotly_config
from utils.insights import (
    generate_borrower_verdict, get_probability_interpretation,
    get_risk_color, get_risk_gradient_css, load_model_info,
    log_prediction
)

set_page_config("Single Borrower", icon="")
inject_css()

render_section_label("Analysis")
render_section_title("Single Borrower Intelligence")

st.markdown("""
<p style="font-size:14px; color:#F0F2F6; line-height:1.65; margin-bottom:1rem;">
Enter borrower details to generate a recovery probability assessment with inline explainability and recommended action.
</p>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  Model readiness check
# ═══════════════════════════════════════════════════════════════════════════════
model_ready = (
    Path("models/best_model.pkl").exists() and
    Path("models/threshold.pkl").exists() and
    Path("models/preprocessor.pkl").exists()
)

if not model_ready:
    st.warning("Model artifacts not found. Please run the training pipeline first:")
    st.code("python main.py", language="bash")
    st.info("The prediction feature will be available after training completes.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL INFO PANEL
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("Model Info", expanded=False):
    info = load_model_info()
    cols = st.columns(4)
    cols[0].metric("Model", info["model_name"])
    cols[1].metric("Recall", f"{info['recall']:.2f}")
    cols[2].metric("Threshold", f"{info['threshold']:.3f}")
    cols[3].metric("Features", info["features"])
    st.caption(f"Last trained: {info['last_trained']}  |  SMOTE sampling_strategy: {info['smote_strategy']}")

# ═══════════════════════════════════════════════════════════════════════════════
#  UI Input Collection
# ═══════════════════════════════════════════════════════════════════════════════

with st.expander("Borrower Profile", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["M", "F"], help="Select borrower's gender")
        age_years = st.slider("Age (Years)", 18, 80, 35, help="Borrower age in years")
        days_birth = int(-age_years * 365.25)
        children = st.number_input("Children Count", 0, 14, 0, help="Number of dependent children")
    with col2:
        family_members = st.number_input("Family Members", 1, 15, 2, help="Total family size")
        own_car = st.checkbox("Owns Car", help="Does borrower own a vehicle?")
        own_house = st.checkbox("Owns House", help="Does borrower own real estate?")
        occupation = st.selectbox("Occupation Type",
            ['Laborers', 'Sales staff', 'Core staff', 'Managers', 'Drivers',
             'High skill tech staff', 'Accountants', 'Medicine staff', 'Security staff',
             'Cooking staff', 'Cleaning staff', 'Private service staff', 'Low-skill Laborers',
             'Waiters/barmen staff', 'Secretaries', 'Realty agents', 'HR staff', 'IT staff'],
            help="Primary occupation category")
    with col3:
        employment_years = st.slider("Employment Years", 0, 50, 5, help="Years in current employment")
        days_employed = int(-employment_years * 365.25)
        family_status = st.selectbox("Family Status",
            ["Married", "Single / not married", "Separated", "Widow", "Unknown"],
            help="Current marital status")
        housing_type = st.selectbox("Housing Type",
            ["House / apartment", "With parents", "Rented apartment",
             "Municipal apartment", "Office apartment"],
            help="Current housing situation")

with st.expander("Employment & Income"):
    col4, col5, col6 = st.columns(3)
    with col4:
        income_type = st.selectbox("Income Type",
            ["Working", "Commercial associate", "State servant",
             "Student", "Unemployed", "Maternity leave"],
            help="Primary source of income")
        education_type = st.selectbox("Education",
            ["Secondary / secondary special", "Higher education",
             "Incomplete higher", "Lower secondary"],
            help="Highest education level")
        organization_type = st.selectbox("Organization Type",
            ['Business Entity Type 3', 'Business Entity Type 2', 'Self-employed',
             'Government', 'Industry: type 1', 'Industry: type 4', 'Industry: type 7',
             'Trade: type 3', 'Trade: type 2', 'Construction', 'Transport: type 2',
             'Bank', 'Services', 'School', 'Police', 'Medicine', 'Military',
             'Security', 'Realtor', 'Hotel', 'Restaurant', 'Other'],
            help="Type of employer organization")
    with col5:
        income = st.number_input("Annual Income (INR)", min_value=0.0, value=150000.0, step=5000.0,
                                 help="Total annual income")
    with col6:
        region_population = st.slider("Region Population (relative)", 0.0, 0.1, 0.02, step=0.001,
                                      help="Normalized population of the region where borrower lives")

with st.expander("Loan Details"):
    col7, col8, col9 = st.columns(3)
    with col7:
        contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"],
                                     help="Type of loan contract")
        credit = st.number_input("Credit Amount (INR)", min_value=0.0, value=500000.0, step=10000.0,
                                 help="Total loan amount requested")
    with col8:
        annuity = st.number_input("Loan Annuity (INR)", min_value=0.0, value=25000.0, step=1000.0,
                                help="Monthly loan payment obligation")
        goods_price = st.number_input("Goods Price (INR)", min_value=0.0, value=450000.0, step=10000.0,
                                      help="Price of goods being financed")
    with col9:
        credit_income_ratio = credit / (income + 1)
        annuity_income_ratio = annuity / (income + 1)
        st.metric("Credit / Income Ratio", f"{credit_income_ratio:.2f}")
        st.metric("Annuity / Income Ratio", f"{annuity_income_ratio:.2f}")

with st.expander("Credit History"):
    col10, col11, col12 = st.columns(3)
    with col10:
        existing_debt = st.number_input("Existing Debt (INR)", min_value=0.0, value=100000.0, step=5000.0,
                                        help="Total existing debt obligations")
        total_loans = st.number_input("Total Bureau Loans", min_value=0, value=3,
                                       help="Number of active bureau loans")
        bureau_credit_sum = st.number_input("Bureau Credit Sum (INR)", min_value=0.0, value=200000.0, step=10000.0,
                                            help="Total credit amount in bureau records")
    with col11:
        bureau_debt_sum = st.number_input("Bureau Debt Sum (INR)", min_value=0.0, value=80000.0, step=5000.0,
                                          help="Total debt in bureau records")
        bureau_inquiries = st.number_input("Credit Bureau Inquiries (Yearly)", min_value=0, value=2,
                                           help="Number of credit bureau inquiries in last year")
        overdue_days = st.slider("Max Overdue Days", 0, 365, 0,
                                 help="Maximum days any payment is overdue")
    with col12:
        ext_source_1 = st.slider("External Credit Score 1", 0.0, 1.0, 0.5,
                                 help="External credit bureau score 1 (normalized)")
        ext_source_2 = st.slider("External Credit Score 2", 0.0, 1.0, 0.5,
                                 help="External credit bureau score 2 (normalized)")
        ext_source_3 = st.slider("External Credit Score 3", 0.0, 1.0, 0.5,
                                 help="External credit bureau score 3 (normalized)")

with st.expander("Optional Flags (defaults applied if skipped)"):
    col13, col14, col15 = st.columns(3)
    with col13:
        flag_work_phone = st.checkbox("Work Phone", value=False)
        flag_cont_mobile = st.checkbox("Contact Mobile", value=True)
        flag_phone = st.checkbox("Home Phone", value=False)
        flag_email = st.checkbox("Email", value=False)
    with col14:
        region_rating = st.selectbox("Region Rating", [1, 2, 3], index=1,
                                     help="Rating of the region (1=best, 3=worst)")
        region_rating_city = st.selectbox("Region Rating with City", [1, 2, 3], index=1,
                                          help="Rating of region + city")
        reg_region_not_live = st.checkbox("Reg Region != Live Region", value=False)
        reg_region_not_work = st.checkbox("Reg Region != Work Region", value=False)
    with col15:
        live_region_not_work = st.checkbox("Live Region != Work Region", value=False)
        reg_city_not_live = st.checkbox("Reg City != Live City", value=False)
        reg_city_not_work = st.checkbox("Reg City != Work City", value=False)
        live_city_not_work = st.checkbox("Live City != Work City", value=False)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

if st.button("Predict Recovery Risk", type="primary", use_container_width=True):
    try:
        from src.loanrecovery.pipeline.prediction_pipeline import PredictPipeline

        input_data = pd.DataFrame({
            'CODE_GENDER': [gender],
            'DAYS_BIRTH': [days_birth],
            'CNT_CHILDREN': [children],
            'CNT_FAM_MEMBERS': [family_members],
            'FLAG_OWN_CAR': ['Y' if own_car else 'N'],
            'FLAG_OWN_REALTY': ['Y' if own_house else 'N'],
            'OCCUPATION_TYPE': [occupation],
            'DAYS_EMPLOYED': [days_employed],
            'NAME_FAMILY_STATUS': [family_status],
            'NAME_HOUSING_TYPE': [housing_type],
            'NAME_INCOME_TYPE': [income_type],
            'NAME_EDUCATION_TYPE': [education_type],
            'ORGANIZATION_TYPE': [organization_type],
            'AMT_INCOME_TOTAL': [income],
            'REGION_POPULATION_RELATIVE': [region_population],
            'NAME_CONTRACT_TYPE': [contract_type],
            'AMT_CREDIT': [credit],
            'AMT_ANNUITY': [annuity],
            'AMT_GOODS_PRICE': [goods_price],
            'AMT_DEBT_TOTAL': [existing_debt],
            'CNT_TOTAL_LOANS': [total_loans],
            'AMT_CREDIT_SUM_sum': [bureau_credit_sum],
            'AMT_CREDIT_SUM_DEBT_sum': [bureau_debt_sum],
            'AMT_REQ_CREDIT_BUREAU_YEAR': [bureau_inquiries],
            'CREDIT_DAY_OVERDUE_max': [overdue_days],
            'EXT_SOURCE_1': [ext_source_1],
            'EXT_SOURCE_2': [ext_source_2],
            'EXT_SOURCE_3': [ext_source_3],
            'FLAG_WORK_PHONE': [1 if flag_work_phone else 0],
            'FLAG_CONT_MOBILE': [1 if flag_cont_mobile else 0],
            'FLAG_PHONE': [1 if flag_phone else 0],
            'FLAG_EMAIL': [1 if flag_email else 0],
            'REGION_RATING_CLIENT': [region_rating],
            'REGION_RATING_CLIENT_W_CITY': [region_rating_city],
            'REG_REGION_NOT_LIVE_REGION': [1 if reg_region_not_live else 0],
            'REG_REGION_NOT_WORK_REGION': [1 if reg_region_not_work else 0],
            'LIVE_REGION_NOT_WORK_REGION': [1 if live_region_not_work else 0],
            'REG_CITY_NOT_LIVE_CITY': [1 if reg_city_not_live else 0],
            'REG_CITY_NOT_WORK_CITY': [1 if reg_city_not_work else 0],
            'LIVE_CITY_NOT_WORK_CITY': [1 if live_city_not_work else 0],
        })

        pipeline = PredictPipeline()
        prediction, probability, risk_label, aligned_features = pipeline.predict(input_data)
        threshold = pipeline.threshold

        # ── Confidence calculation ──
        dist = abs(probability - threshold)
        confidence = "High" if dist > 0.20 else "Medium" if dist > 0.10 else "Low"

        # ── Probability interpretation ──
        interp = get_probability_interpretation(probability)

        # ── Build SHAP-like risk factors for UI display ──
        shap_factors = []
        if credit_income_ratio > 3:
            shap_factors.append({"name": "Debt-to-income ratio", "value": f"{credit_income_ratio:.2f}", "direction": "up", "pct": 85, "impact": 0.34})
        if annuity_income_ratio > 0.3:
            shap_factors.append({"name": "Annuity burden", "value": f"{annuity_income_ratio:.2f}", "direction": "up", "pct": 65, "impact": 0.21})
        if ext_source_2 < 0.3:
            shap_factors.append({"name": "External credit score", "value": f"{ext_source_2:.2f}", "direction": "up", "pct": 55, "impact": 0.18})
        if overdue_days > 30:
            shap_factors.append({"name": "Days overdue", "value": f"{overdue_days}", "direction": "up", "pct": 70, "impact": 0.25})
        if own_house or own_car:
            shap_factors.append({"name": "Collateral present", "value": "Yes", "direction": "down", "pct": 40, "impact": -0.12})
        if employment_years < 2:
            shap_factors.append({"name": "Employment history", "value": f"{employment_years} yrs", "direction": "up", "pct": 30, "impact": 0.10})
        if not shap_factors:
            shap_factors.append({"name": "Stable profile", "value": "No dominant risks", "direction": "down", "pct": 10, "impact": -0.05})

        # ── Log prediction to history ──
        log_prediction(
            borrower_id="LN-2847",
            probability=probability,
            prediction=int(prediction),
            threshold=threshold,
            risk_label=risk_label,
            input_features=input_data.shape[1]
        )

        # ── Render Results ──
        render_section_label("Result")

        # Risk color banner
        risk_color = get_risk_color(probability)
        st.markdown(f"""
        <div style="background: {risk_color}15; border: 1px solid {risk_color}40; border-radius: 10px; padding: 12px 16px; margin: 10px 0;">
            <div style="font-size: 13px; font-weight: 500; color: {risk_color}; margin-bottom: 4px;">
                {interp['label']} — {probability*100:.1f}%
            </div>
            <div style="font-size: 12px;         color: #F0F2F6;">
                {interp['meaning']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        render_borrower_card(
            loan_id="LN-2847",
            loan_type=contract_type,
            outstanding=f"₹{credit:,.0f}",
            probability=probability,
            shap_factors=shap_factors,
            confidence=confidence
        )

        verdict = generate_borrower_verdict(probability, shap_factors)
        render_verdict_card(verdict["verdict"], verdict["explanation"], verdict["severity"])

        if confidence == "Low":
            st.warning("Low model confidence — prediction is near threshold. Manual review recommended.")

        # Gauge
        st.markdown("---")
        render_section_label("Visualization")
        render_section_title("Recovery Probability Gauge")
        fig_gauge = gauge_chart(probability, threshold=threshold)
        st.plotly_chart(fig_gauge, use_container_width=True, config=plotly_config)

        # SHAP waterfall (real SHAP)
        st.markdown("---")
        render_section_label("Explainability")
        render_section_title("SHAP Waterfall Explanation")
        st.markdown(f"""
        <p style="font-size:12px;         color:#B8C4D4; margin-bottom:8px;">
        This waterfall shows how each feature pushed the prediction away from the base value toward the final probability of <strong>{probability:.1%}</strong>.
        </p>
        """, unsafe_allow_html=True)

        try:
            # Generate real SHAP waterfall for this single prediction
            model = joblib.load("models/best_model.pkl")
            preprocessor = pipeline.preprocessor
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(aligned_features.values)

            # For LightGBM binary classifier, shap_values might be a list
            if isinstance(shap_values, list):
                shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_vals = shap_values

            # Plot waterfall for the first (and only) instance
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                data=aligned_features.iloc[0].values,
                feature_names=aligned_features.columns.tolist()
            ), show=False)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()
        except Exception as e:
            st.info(f"SHAP waterfall not available: {e}")
            # Fallback to rule-based factors
            render_shap_factors(shap_factors)

        # Raw prediction details
        with st.expander("Raw Prediction Details", expanded=False):
            st.write(f"**Probability:** {probability:.4f}")
            st.write(f"**Threshold:** {threshold:.4f}")
            st.write(f"**Prediction:** {risk_label}")
            st.write(f"**Confidence:** {confidence}")
            st.write(f"**Aligned Features:** {aligned_features.shape[1]} columns")
            st.write(f"**Interpretation:** {interp['label']}")

        st.success("Prediction complete. Review the recommended action above.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("Please ensure the model has been trained and all artifacts exist in the models/ folder.")

# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE DEBUG EXPANDER
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("Pipeline Debug 🔧", expanded=False):
    st.markdown("""
    <p style="font-size:12px;         color:#B8C4D4;">
    This panel shows internal pipeline state for debugging purposes.
    </p>
    """, unsafe_allow_html=True)

    try:
        from src.loanrecovery.pipeline.prediction_pipeline import PredictPipeline
        pipeline = PredictPipeline()
        st.write(f"**Model features expected:** {len(pipeline.feature_names)}")
        st.write(f"**Threshold:** {pipeline.threshold}")
        st.write(f"**Scaler loaded:** {pipeline.scaler is not None}")

        if st.checkbox("Show feature list"):
            st.write(pipeline.feature_names)
    except Exception as e:
        st.error(f"Could not load pipeline for debug: {e}")

st.markdown("---")
st.caption("**Disclaimer:** This prediction is for decision support only. Final loan decisions should include business review, financial analysis, and human verification.")
