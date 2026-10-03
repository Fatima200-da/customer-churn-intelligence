import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
        letter-spacing: -1px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-top: 5px;
        margin-bottom: 20px;
    }

    .section-description {
        color: #6b7280;
        margin-top: -8px;
        margin-bottom: 20px;
        font-size: 15px;
    }

    div[data-testid="stMetric"] {
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(128, 128, 128, 0.18);
        background-color: rgba(128, 128, 128, 0.05);
    }

    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 750;
    }

    .risk-card {
        padding: 24px;
        border-radius: 18px;
        border: 1px solid rgba(128, 128, 128, 0.18);
        background: rgba(128, 128, 128, 0.05);
        min-height: 145px;
        text-align: center;
    }

    .risk-card-title {
        font-size: 14px;
        font-weight: 600;
        color: #6b7280;
        margin-bottom: 10px;
    }

    .risk-card-value {
        font-size: 32px;
        font-weight: 800;
    }

    .risk-card-small {
        font-size: 14px;
        color: #6b7280;
        margin-top: 4px;
    }

    .insight-box {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(128, 128, 128, 0.18);
        background-color: rgba(128, 128, 128, 0.05);
        margin-bottom: 12px;
    }

    .sidebar-title {
        font-size: 21px;
        font-weight: 750;
    }

    .small-label {
        color: #6b7280;
        font-size: 13px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 46px;
    }

    div[data-testid="stDownloadButton"] > button {
        border-radius: 10px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model_path = BASE_DIR / "churn_model.pkl"

    return joblib.load(model_path)


model = load_model()

preprocessor = model.named_steps["preprocessor"]
gb_model = model.named_steps["model"]

explainer = shap.TreeExplainer(gb_model)


# =========================================================
# LOAD SUPPORTING FILES
# =========================================================

@st.cache_data
def load_supporting_files():

    with open(BASE_DIR / "model_metrics.json", "r") as f:
        loaded_metrics = json.load(f)

    loaded_risk_data = pd.read_csv(
        BASE_DIR / "risk_distribution.csv"
    )

    loaded_high_risk_data = pd.read_csv(
        BASE_DIR / "high_risk_customers.csv"
    )

    loaded_business_data = pd.read_csv(
        BASE_DIR / "business_insights.csv"
    )

    loaded_roc_data = pd.read_csv(
        BASE_DIR / "roc_curve.csv"
    )

    return (
        loaded_metrics,
        loaded_risk_data,
        loaded_high_risk_data,
        loaded_business_data,
        loaded_roc_data
    )


(
    metrics,
    risk_data,
    high_risk_data,
    business_data,
    roc_data
) = load_supporting_files()


# =========================================================
# CALCULATE HIGH-RISK SUMMARY
# =========================================================

high_risk_rows = risk_data[
    risk_data["RiskLevel"] == "High Risk"
]

if not high_risk_rows.empty:

    high_risk_count = int(
        high_risk_rows["CustomerCount"].iloc[0]
    )

    high_risk_percentage = float(
        high_risk_rows["Percentage"].iloc[0]
    )

else:

    high_risk_count = 0
    high_risk_percentage = 0.0


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">📊 Churn Intelligence</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "AI-powered customer retention analytics"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Customer Prediction",
            "Model Performance",
            "Business Insights",
            "High-Risk Customers"
        ],
        key="navigation"
    )

    st.divider()

    st.markdown(
        "**Technology Stack**"
    )

    st.caption(
        "Python • Scikit-learn • SHAP • Plotly • Streamlit"
    )

    st.caption(
        "Gradient Boosting • Explainable AI"
    )


# =========================================================
# COMMON HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📊 Customer Churn Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered customer retention and churn risk analysis'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    st.caption(
        "Predict customer churn, understand risk drivers, "
        "and identify retention opportunities."
    )

    st.divider()

    # =====================================================
    # KPI CARDS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Model Accuracy",
            f"{metrics['accuracy'] * 100:.2f}%"
        )

    with col2:

        st.metric(
            "ROC-AUC",
            f"{metrics['roc_auc'] * 100:.2f}%"
        )

    with col3:

        st.metric(
            "High-Risk Customers",
            high_risk_count
        )

        st.caption("Held-out test set")

    with col4:

        st.metric(
            "Overall Churn Rate",
            "26.54%"
        )

        st.caption("Full dataset")

    st.divider()

    # =====================================================
    # PROJECT OVERVIEW
    # =====================================================

    st.header("🎯 Project Overview")

    st.markdown(
        """
        This project uses machine learning to predict customer
        churn and identify customers who may require retention
        attention.

        The solution combines:

        - **Machine Learning** for churn prediction
        - **SHAP Explainability** for understanding model decisions
        - **Risk Segmentation** for customer prioritization
        - **Business Analytics** for identifying churn patterns
        - **Retention Recommendations** for customer-level actions
        """
    )

    st.divider()

    # =====================================================
    # RISK DISTRIBUTION
    # =====================================================

    st.header("📊 Customer Risk Distribution")

    st.markdown(
        '<div class="section-description">'
        'Predicted churn-risk categories across the held-out test set.'
        '</div>',
        unsafe_allow_html=True
    )

    risk_chart = px.bar(
        risk_data,
        x="RiskLevel",
        y="CustomerCount",
        text="CustomerCount",
        title="Predicted Customer Risk Distribution"
    )

    risk_chart.update_traces(
        textposition="outside"
    )

    risk_chart.update_layout(
        xaxis_title="Risk Level",
        yaxis_title="Customer Count",
        height=430,
        showlegend=False
    )

    st.plotly_chart(
        risk_chart,
        use_container_width=True
    )

    st.caption(
        "Risk categories use 40% and 70% predicted churn probability "
        "thresholds as illustrative business segmentation rules."
    )

    st.divider()

    # =====================================================
    # EXECUTIVE SUMMARY
    # =====================================================

    st.header("🎯 Executive Summary")

    st.subheader("Key Business Findings")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            "**Contract**\n\n"
            "Month-to-month customers show an observed churn "
            "rate of 42.71%, compared with 11.27% for one-year "
            "contracts and 2.83% for two-year contracts."
        )

    with col2:

        st.warning(
            "**Internet Service**\n\n"
            "Fiber optic customers show an observed churn rate "
            "of 41.89%, compared with 18.96% for DSL customers "
            "and 7.40% for customers without internet service."
        )

    with col3:

        st.warning(
            "**Payment Method**\n\n"
            "Electronic check customers show an observed churn "
            "rate of 45.29%, compared with 16.71% for automatic "
            "bank transfer and 15.24% for automatic credit card payments."
        )

    st.subheader("Retention Focus Areas")

    st.markdown(
        """
        - **Contract:** Monitor month-to-month customers for elevated churn risk.
        - **Customer lifecycle:** Prioritize early-stage customers during onboarding.
        - **Payment:** Consider encouraging alternative automatic payment methods.
        - **Internet service:** Review the customer experience of fiber optic users.
        - **Support services:** Consider targeted support and security offers for customers without these services.
        """
    )

    st.caption(
        "These findings describe observed patterns in the dataset "
        "and should not be interpreted as causal relationships."
    )


# =========================================================
# CUSTOMER PREDICTION
# =========================================================

elif page == "Customer Prediction":

    st.caption(
        "Enter customer characteristics to estimate churn risk "
        "and understand the main model drivers."
    )

    st.divider()

    # =====================================================
    # CUSTOMER INFORMATION
    # =====================================================

    st.header("👤 Customer Information")

    st.markdown(
        '<div class="section-description">'
        'Enter customer characteristics to estimate their churn risk.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    # =====================================================
    # BASIC INFORMATION
    # =====================================================

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            key="prediction_gender"
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            key="prediction_senior_citizen"
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"],
            key="prediction_partner"
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"],
            key="prediction_dependents"
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=1,
            key="prediction_tenure"
        )

    # =====================================================
    # PHONE & INTERNET
    # =====================================================

    with col2:

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"],
            key="prediction_phone_service"
        )

        if phone_service == "No":

            multiple_lines = "No phone service"

            st.selectbox(
                "Multiple Lines",
                ["No phone service"],
                disabled=True,
                key="prediction_multiple_lines_disabled"
            )

        else:

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["Yes", "No"],
                key="prediction_multiple_lines"
            )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
            key="prediction_internet_service"
        )

    # =====================================================
    # INTERNET SERVICES
    # =====================================================

    with col3:

        if internet_service == "No":

            online_security = "No internet service"
            online_backup = "No internet service"

            st.selectbox(
                "Online Security",
                ["No internet service"],
                disabled=True,
                key="prediction_online_security_disabled"
            )

            st.selectbox(
                "Online Backup",
                ["No internet service"],
                disabled=True,
                key="prediction_online_backup_disabled"
            )

        else:

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No"],
                key="prediction_online_security"
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No"],
                key="prediction_online_backup"
            )

    st.divider()

    # =====================================================
    # ADDITIONAL SERVICES
    # =====================================================

    st.subheader("🛠️ Additional Services")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        if internet_service == "No":

            device_protection = "No internet service"

            st.selectbox(
                "Device Protection",
                ["No internet service"],
                disabled=True,
                key="prediction_device_protection_disabled"
            )

        else:

            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No"],
                key="prediction_device_protection"
            )

    with col2:

        if internet_service == "No":

            tech_support = "No internet service"

            st.selectbox(
                "Tech Support",
                ["No internet service"],
                disabled=True,
                key="prediction_tech_support_disabled"
            )

        else:

            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No"],
                key="prediction_tech_support"
            )

    with col3:

        if internet_service == "No":

            streaming_tv = "No internet service"

            st.selectbox(
                "Streaming TV",
                ["No internet service"],
                disabled=True,
                key="prediction_streaming_tv_disabled"
            )

        else:

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No"],
                key="prediction_streaming_tv"
            )

    with col4:

        if internet_service == "No":

            streaming_movies = "No internet service"

            st.selectbox(
                "Streaming Movies",
                ["No internet service"],
                disabled=True,
                key="prediction_streaming_movies_disabled"
            )

        else:

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No"],
                key="prediction_streaming_movies"
            )

    st.divider()

    # =====================================================
    # CONTRACT & BILLING
    # =====================================================

    st.header("💳 Contract & Billing")

    col1, col2, col3 = st.columns(3)

    with col1:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ],
            key="prediction_contract"
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
            key="prediction_paperless_billing"
        )

    with col2:

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            key="prediction_payment_method"
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=0.01,
            key="prediction_monthly_charges"
        )

    with col3:

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=70.0,
            step=0.01,
            key="prediction_total_charges"
        )

    st.divider()

    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    predict_button = st.button(
        "🔮 Predict Churn Risk",
        type="primary",
        use_container_width=True,
        key="prediction_button"
    )

    # =====================================================
    # PREDICTION
    # =====================================================

    if predict_button:

        customer_data = pd.DataFrame(
            [{
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }]
        )

        # =================================================
        # PREDICTION
        # =================================================

        probability = model.predict_proba(
            customer_data
        )[0][1]

        risk_percent = probability * 100

        if probability >= 0.70:

            risk_level = "High Risk"

        elif probability >= 0.40:

            risk_level = "Medium Risk"

        else:

            risk_level = "Low Risk"

        # =================================================
        # RESULT HEADER
        # =================================================

        st.divider()

        st.header("🎯 Churn Prediction")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(
                f"""
                <div class="risk-card">
                    <div class="risk-card-title">
                        Churn Probability
                    </div>
                    <div class="risk-card-value">
                        {risk_percent:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                f"""
                <div class="risk-card">
                    <div class="risk-card-title">
                        Risk Level
                    </div>
                    <div class="risk-card-value">
                        {risk_level}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:

            st.markdown(
                f"""
                <div class="risk-card">
                    <div class="risk-card-title">
                        Customer Tenure
                    </div>
                    <div class="risk-card-value">
                        {tenure}
                    </div>
                    <div class="risk-card-small">
                        months
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # =================================================
        # RISK GAUGE
        # =================================================

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_percent,
                number={
                    "suffix": "%",
                    "font": {"size": 38}
                },
                title={
                    "text": "Predicted Churn Risk"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "bar": {
                        "thickness": 0.35
                    },
                    "steps": [
                        {
                            "range": [0, 40],
                            "name": "Low Risk"
                        },
                        {
                            "range": [40, 70],
                            "name": "Medium Risk"
                        },
                        {
                            "range": [70, 100],
                            "name": "High Risk"
                        }
                    ],
                    "threshold": {
                        "line": {
                            "width": 4
                        },
                        "thickness": 0.8,
                        "value": risk_percent
                    }
                }
            )
        )

        gauge.update_layout(
            height=320,
            margin=dict(
                l=30,
                r=30,
                t=70,
                b=20
            )
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

        st.progress(
            probability,
            text=f"Churn Risk: {risk_percent:.2f}%"
        )

        if risk_level == "High Risk":

            st.error(
                "🚨 This customer has a high predicted churn risk. "
                "Immediate retention attention may be appropriate."
            )

        elif risk_level == "Medium Risk":

            st.warning(
                "⚠️ This customer shows a moderate predicted churn risk. "
                "Consider proactive retention actions."
            )

        else:

            st.success(
                "✅ This customer currently shows a relatively low "
                "predicted churn risk."
            )

        # =================================================
        # SHAP EXPLANATION
        # =================================================

        transformed_customer = preprocessor.transform(
            customer_data
        )

        if hasattr(transformed_customer, "toarray"):

            transformed_customer = transformed_customer.toarray()

        shap_values = explainer.shap_values(
            transformed_customer
        )

        shap_array = np.asarray(
            shap_values
        )

        if shap_array.ndim == 3:

            shap_row = shap_array[0, :, 1]

        elif shap_array.ndim == 2:

            shap_row = shap_array[0]

        else:

            shap_row = shap_array

        feature_names = (
            preprocessor.get_feature_names_out()
        )

        shap_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "SHAP": shap_row
            }
        )

        # =================================================
        # FEATURE LABELS
        # =================================================

        feature_labels = {

            "Contract_Month-to-month":
                "Month-to-month contract",

            "Contract_One year":
                "One-year contract",

            "Contract_Two year":
                "Two-year contract",

            "PaymentMethod_Electronic check":
                "Electronic check payment",

            "PaymentMethod_Mailed check":
                "Mailed check payment",

            "PaymentMethod_Bank transfer (automatic)":
                "Automatic bank transfer",

            "PaymentMethod_Credit card (automatic)":
                "Automatic credit card payment",

            "InternetService_Fiber optic":
                "Fiber optic internet",

            "InternetService_DSL":
                "DSL internet",

            "InternetService_No":
                "No internet service",

            "OnlineSecurity_No":
                "No online security",

            "OnlineSecurity_Yes":
                "Online security",

            "TechSupport_No":
                "No technical support",

            "TechSupport_Yes":
                "Technical support",

            "PaperlessBilling_Yes":
                "Paperless billing",

            "PaperlessBilling_No":
                "No paperless billing",

            "tenure":
                "Customer tenure",

            "MonthlyCharges":
                "Monthly charges",

            "TotalCharges":
                "Total charges",

            "SeniorCitizen":
                "Senior citizen status",

            "MultipleLines_Yes":
                "Multiple phone lines",

            "MultipleLines_No":
                "No multiple lines"
        }

        # =================================================
        # TOP POSITIVE FACTORS
        # =================================================

        categorical_columns = [
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod"
        ]

        candidate_features = []

        for _, row in shap_df.sort_values(
            "SHAP",
            ascending=False
        ).iterrows():

            feature = row["Feature"]
            shap_value = float(row["SHAP"])

            clean_feature = (
                feature
                .replace("cat__", "")
                .replace("num__", "")
            )

            matched = False

            for column in categorical_columns:

                prefix = column + "_"

                if clean_feature.startswith(prefix):

                    selected_value = (
                        clean_feature[len(prefix):]
                    )

                    actual_value = str(
                        customer_data.iloc[0][column]
                    )

                    if (
                        selected_value == actual_value
                        and shap_value > 0
                    ):

                        candidate_features.append(
                            row
                        )

                    matched = True

                    break

            if not matched and shap_value > 0:

                candidate_features.append(
                    row
                )

            if len(candidate_features) >= 5:

                break

        top_positive = pd.DataFrame(
            candidate_features
        )

        # =================================================
        # SHAP VISUALIZATION
        # =================================================

        st.divider()

        st.subheader(
            "🔍 Key Risk Factors"
        )

        if not top_positive.empty:

            plot_df = top_positive.copy()

            plot_df["CleanFeature"] = (
                plot_df["Feature"]
                .str.replace("cat__", "", regex=False)
                .str.replace("num__", "", regex=False)
            )

            plot_df["ReadableFeature"] = (
                plot_df["CleanFeature"]
                .map(feature_labels)
                .fillna(plot_df["CleanFeature"])
            )

            plot_df = plot_df.sort_values(
                "SHAP",
                ascending=True
            )

            shap_chart = px.bar(
                plot_df,
                x="SHAP",
                y="ReadableFeature",
                orientation="h",
                text="SHAP",
                title="Features Increasing Predicted Churn Risk"
            )

            shap_chart.update_traces(
                texttemplate="+%{text:.3f}",
                textposition="outside"
            )

            shap_chart.update_layout(
                xaxis_title="SHAP Contribution",
                yaxis_title="",
                height=360,
                showlegend=False
            )

            st.plotly_chart(
                shap_chart,
                use_container_width=True
            )

            for _, row in top_positive.iterrows():

                feature = (
                    row["Feature"]
                    .replace("cat__", "")
                    .replace("num__", "")
                )

                readable_feature = feature_labels.get(
                    feature,
                    feature
                )

                st.write(
                    f"**{readable_feature}** — "
                    f"model contribution: +{float(row['SHAP']):.3f}"
                )

        else:

            st.write(
                "No strong positive risk factors were identified."
            )

        st.caption(
            "SHAP values describe how model features contributed "
            "to this individual prediction. They do not establish causality."
        )

        # =================================================
        # RETENTION RECOMMENDATIONS
        # =================================================

        st.divider()

        st.subheader(
            "💡 Retention Recommendations"
        )

        recommendations = []

        if contract == "Month-to-month":

            recommendations.append(
                "Consider offering a longer-term contract "
                "with a retention incentive."
            )

        if tech_support == "No":

            recommendations.append(
                "Offer technical support assistance "
                "or a support trial."
            )

        if online_security == "No":

            recommendations.append(
                "Promote an online security add-on or bundle."
            )

        if payment_method == "Electronic check":

            recommendations.append(
                "Consider promoting automatic payment options."
            )

        if internet_service == "Fiber optic":

            recommendations.append(
                "Review the customer's fiber service "
                "experience and pricing."
            )

        if tenure <= 3:

            recommendations.append(
                "Prioritize early-stage customer onboarding "
                "and retention outreach."
            )

        if recommendations:

            for recommendation in recommendations:

                st.write(
                    f"• {recommendation}"
                )

        else:

            st.write(
                "No specific retention actions were "
                "triggered for this customer."
            )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.header("📈 Model Performance")

    st.markdown(
        '<div class="section-description">'
        'Evaluation metrics calculated on the held-out test set.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # =====================================================
    # METRICS
    # =====================================================

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Accuracy",
            f"{metrics['accuracy'] * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Precision",
            f"{metrics['precision'] * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Recall",
            f"{metrics['recall'] * 100:.2f}%"
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{metrics['f1'] * 100:.2f}%"
        )

    with col5:

        st.metric(
            "ROC-AUC",
            f"{metrics['roc_auc'] * 100:.2f}%"
        )

    st.divider()

    # =====================================================
    # METRIC EXPLANATIONS
    # =====================================================

    st.subheader("📌 Metric Interpretation")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:

        st.info(
            "**Precision**\n\n"
            "Among customers predicted as likely to churn, "
            "the proportion that actually churned."
        )

    with metric_col2:

        st.info(
            "**Recall**\n\n"
            "Among customers who actually churned, "
            "the proportion identified by the model."
        )

    with metric_col3:

        st.info(
            "**ROC-AUC**\n\n"
            "Measures how well the model separates "
            "customers who churn from those who do not."
        )

    st.divider()

    # =====================================================
    # ROC CURVE
    # =====================================================

    st.subheader("📈 ROC Curve")

    st.markdown(
        "The ROC curve illustrates the model's ability to distinguish "
        "between customers who churn and customers who do not churn."
    )

    roc_fig = go.Figure()

    roc_fig.add_trace(
        go.Scatter(
            x=roc_data["fpr"],
            y=roc_data["tpr"],
            mode="lines",
            name=f"Gradient Boosting (AUC = {metrics['roc_auc']:.3f})"
        )
    )

    roc_fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline",
            line=dict(
                dash="dash"
            )
        )
    )

    roc_fig.update_layout(
        title="Receiver Operating Characteristic",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
        xaxis=dict(
            range=[0, 1]
        ),
        yaxis=dict(
            range=[0, 1]
        )
    )

    st.plotly_chart(
        roc_fig,
        use_container_width=True
    )

    st.caption(
        "ROC-AUC summarizes ranking performance across classification thresholds."
    )

    st.divider()

    # =====================================================
    # RISK DISTRIBUTION
    # =====================================================

    st.subheader("📊 Customer Risk Distribution")

    risk_fig = px.bar(
        risk_data,
        x="RiskLevel",
        y="CustomerCount",
        text="CustomerCount",
        title="Risk Segmentation"
    )

    risk_fig.update_traces(
        textposition="outside"
    )

    risk_fig.update_layout(
        xaxis_title="Risk Level",
        yaxis_title="Customers",
        height=430,
        showlegend=False
    )

    st.plotly_chart(
        risk_fig,
        use_container_width=True
    )


# =========================================================
# BUSINESS INSIGHTS
# =========================================================

elif page == "Business Insights":

    st.header("💼 Business Insights")

    st.markdown(
        '<div class="section-description">'
        'Observed churn patterns across important customer segments.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # =====================================================
    # CONTRACT
    # =====================================================

    st.subheader(
        "📄 Churn Rate by Contract"
    )

    contract_data = business_data[
        business_data["Category"] == "Contract"
    ]

    contract_fig = px.bar(
        contract_data,
        x="Segment",
        y="ChurnRate",
        text="ChurnRate",
        title="Observed Churn Rate by Contract"
    )

    contract_fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    contract_fig.update_layout(
        xaxis_title="Contract Type",
        yaxis_title="Churn Rate (%)",
        height=430,
        showlegend=False
    )

    st.plotly_chart(
        contract_fig,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # INTERNET
    # =====================================================

    st.subheader(
        "🌐 Churn Rate by Internet Service"
    )

    internet_data = business_data[
        business_data["Category"] == "Internet Service"
    ]

    internet_fig = px.bar(
        internet_data,
        x="Segment",
        y="ChurnRate",
        text="ChurnRate",
        title="Observed Churn Rate by Internet Service"
    )

    internet_fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    internet_fig.update_layout(
        xaxis_title="Internet Service",
        yaxis_title="Churn Rate (%)",
        height=430,
        showlegend=False
    )

    st.plotly_chart(
        internet_fig,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # PAYMENT
    # =====================================================

    st.subheader(
        "💳 Churn Rate by Payment Method"
    )

    payment_data = business_data[
        business_data["Category"] == "Payment Method"
    ]

    payment_fig = px.bar(
        payment_data,
        x="Segment",
        y="ChurnRate",
        text="ChurnRate",
        title="Observed Churn Rate by Payment Method"
    )

    payment_fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    payment_fig.update_layout(
        xaxis_title="Payment Method",
        yaxis_title="Churn Rate (%)",
        height=480,
        showlegend=False
    )

    st.plotly_chart(
        payment_fig,
        use_container_width=True
    )

    st.caption(
        "Churn rates are calculated from the full dataset and "
        "represent observed associations between customer "
        "segments and churn."
    )

    st.divider()

    # =====================================================
    # BUSINESS SUMMARY
    # =====================================================

    st.subheader(
        "🎯 Business Summary"
    )

    st.markdown(
        """
        **Contract:** Month-to-month customers show a higher
        observed churn rate than customers on longer-term contracts.

        **Customer lifecycle:** Customers with shorter tenure
        represent an important segment for onboarding and retention analysis.

        **Payment:** Electronic check users show a higher observed
        churn rate than customers using automatic payment methods.

        **Internet service:** Fiber optic customers show a higher
        observed churn rate than DSL customers in this dataset.
        """
    )

    st.caption(
        "These observations describe associations in the dataset "
        "and do not establish causal relationships."
    )


# =========================================================
# HIGH-RISK CUSTOMERS
# =========================================================

elif page == "High-Risk Customers":

    st.header("🚨 High-Risk Customers")

    st.markdown(
        '<div class="section-description">'
        'Customers with the highest predicted churn probability '
        'on the held-out test set.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # =====================================================
    # SUMMARY
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "High-Risk Customers",
            high_risk_count
        )

    with col2:

        st.metric(
            "Share of Test Set",
            f"{high_risk_percentage:.2f}%"
        )

    st.divider()

    # =====================================================
    # TABLE
    # =====================================================

    st.subheader(
        "📋 Highest-Risk Customers"
    )

    display_columns = [
        "tenure",
        "Contract",
        "InternetService",
        "MonthlyCharges",
        "RiskPercent"
    ]

    display_data = high_risk_data[
        display_columns
    ].copy()

    display_data["RiskPercent"] = (
        display_data["RiskPercent"].astype(float)
    )

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "tenure": st.column_config.NumberColumn(
                "Tenure",
                format="%d months"
            ),
            "Contract": st.column_config.TextColumn(
                "Contract"
            ),
            "InternetService": st.column_config.TextColumn(
                "Internet Service"
            ),
            "MonthlyCharges": st.column_config.NumberColumn(
                "Monthly Charges",
                format="$%.2f"
            ),
            "RiskPercent": st.column_config.ProgressColumn(
                "Churn Risk",
                min_value=0,
                max_value=100,
                format="%.2f%%"
            )
        }
    )

    # =====================================================
    # DOWNLOAD
    # =====================================================

    csv_data = display_data.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download High-Risk Customers",
        data=csv_data,
        file_name="high_risk_customers.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_high_risk"
    )

    st.caption(
        "The table contains the top 20 customers with the highest "
        "predicted churn probability in the held-out test set."
    )

    st.divider()

    # =====================================================
    # RISK DISTRIBUTION
    # =====================================================

    st.subheader(
        "📊 Risk Segment Overview"
    )

    risk_fig = px.pie(
        risk_data,
        names="RiskLevel",
        values="CustomerCount",
        hole=0.45,
        title="Customer Risk Segmentation"
    )

    risk_fig.update_layout(
        height=450
    )

    st.plotly_chart(
        risk_fig,
        use_container_width=True
    )