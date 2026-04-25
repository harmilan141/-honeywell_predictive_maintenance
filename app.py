import streamlit as st
import pandas as pd
import joblib
import html
import plotly.express as px
from datetime import datetime
from pathlib import Path
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="SmartPredict Maintenance",
    page_icon="⚙️",
    layout="wide"
)

MODEL_PATH = "models/predictive_maintenance_model.h5"
SCALER_PATH = "models/scaler.pkl"
DATA_PATH = "data/processed/train_features.csv"
METRICS_PATH = "models/metrics.pkl"
BOOKINGS_PATH = Path("data/maintenance_bookings.csv")
DROP_COLS = ["unit", "cycle", "RUL", "failure_risk"]

# Strict statistical thresholds catch unusual readings without adding new dependencies.
ANOMALY_LOW_QUANTILE = 0.01
ANOMALY_HIGH_QUANTILE = 0.99
ANOMALY_SENSOR_COUNT_THRESHOLD = 2

# A high-risk alert is treated as real only when it persists across recent readings.
HIGH_RISK_THRESHOLD = 0.7
FALSE_ALARM_WINDOW = 3
MIN_HIGH_RISK_READINGS = 2

BOOKING_COLUMNS = [
    "created_at",
    "machine_id",
    "issue_type",
    "preferred_date",
    "technician_name_status",
    "booking_status",
    "failure_probability",
    "failure_risk",
    "alert_check",
]
ISSUE_TYPES = [
    "Sensor anomaly",
    "High failure risk",
    "Temperature issue",
    "Pressure issue",
    "Vibration/instability",
    "Routine inspection",
]
BOOKING_STATUSES = ["Pending", "Scheduled", "Completed"]
MAINTENANCE_ACTIVE_STATUSES = ["Pending", "Scheduled"]
MACHINE_ID_COLUMNS = [
    "machine_id",
    "Machine ID",
    "machine",
    "Machine",
    "unit",
    "engine_id",
    "asset_id",
]
RISK_COLORS = {
    "Low": "#1f9d55",
    "Medium": "#c97706",
    "High": "#d64545",
}

SENSOR_LABELS = {
    "sensor_2": "Temperature",
    "sensor_3": "Temperature",
    "sensor_4": "Temperature",
    "sensor_6": "Pressure",
    "sensor_7": "Pressure",
    "sensor_8": "Rotational speed",
    "sensor_9": "Rotational speed",
    "sensor_11": "Pressure",
    "sensor_12": "Airflow",
    "sensor_13": "Rotational speed",
    "sensor_14": "Rotational speed",
    "sensor_20": "Airflow",
    "sensor_21": "Airflow",
}


def apply_dashboard_style():
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 2rem;
                max-width: 1280px;
            }
            .dashboard-hero {
                border: 1px solid #d9e2ec;
                border-radius: 8px;
                padding: 22px 24px;
                background: #f8fafc;
                margin-bottom: 18px;
            }
            .hero-kicker {
                color: #486581;
                font-size: 0.82rem;
                font-weight: 700;
                letter-spacing: 0;
                text-transform: uppercase;
                margin-bottom: 6px;
            }
            .hero-title {
                color: #102a43;
                font-size: 2rem;
                font-weight: 800;
                line-height: 1.15;
                margin-bottom: 8px;
            }
            .hero-subtitle {
                color: #486581;
                font-size: 0.98rem;
                margin-bottom: 16px;
            }
            .hero-meta {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                align-items: center;
            }
            .status-pill {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                padding: 7px 12px;
                font-size: 0.82rem;
                font-weight: 700;
            }
            .risk-low {
                color: #0b6b3a;
                background: #e3f9e5;
                border: 1px solid #a8e6b1;
            }
            .risk-medium {
                color: #8a4b00;
                background: #fff7cc;
                border: 1px solid #f7d070;
            }
            .risk-high {
                color: #9b1c1c;
                background: #ffe3e3;
                border: 1px solid #ffb3b3;
            }
            .section-title {
                color: #102a43;
                font-size: 1.18rem;
                font-weight: 800;
                margin: 8px 0 12px;
            }
            .metric-card {
                border: 1px solid #d9e2ec;
                border-radius: 8px;
                background: #ffffff;
                padding: 16px;
                min-height: 108px;
            }
            .metric-label {
                color: #627d98;
                font-size: 0.78rem;
                font-weight: 700;
                text-transform: uppercase;
                margin-bottom: 8px;
            }
            .metric-value {
                color: #102a43;
                font-size: 1.55rem;
                font-weight: 800;
                line-height: 1.1;
            }
            .metric-caption {
                color: #627d98;
                font-size: 0.82rem;
                margin-top: 8px;
            }
            .action-panel {
                border-left: 5px solid #1f9d55;
                border-radius: 8px;
                background: #ffffff;
                border-top: 1px solid #d9e2ec;
                border-right: 1px solid #d9e2ec;
                border-bottom: 1px solid #d9e2ec;
                padding: 16px;
                margin: 14px 0 4px;
            }
            .action-panel.risk-medium {
                border-left-color: #c97706;
                background: #fffdf2;
            }
            .action-panel.risk-high {
                border-left-color: #d64545;
                background: #fffafa;
            }
            .chart-shell {
                border: 1px solid #d9e2ec;
                border-radius: 8px;
                background: #ffffff;
                padding: 12px;
                margin-top: 10px;
            }
            div[data-testid="stTabs"] button {
                font-weight: 700;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


def get_risk_class(risk):
    return {
        "Low": "risk-low",
        "Medium": "risk-medium",
        "High": "risk-high",
    }.get(risk, "risk-low")


def render_header(risk, prediction_prob, machine_id):
    risk_class = get_risk_class(risk)
    st.markdown(
        f"""
        <div class="dashboard-hero">
            <div class="hero-kicker">Predictive Maintenance Command Center</div>
            <div class="hero-title">SmartPredict Maintenance Dashboard</div>
            <div class="hero-subtitle">
                Live model scoring, anomaly checks, maintenance planning, and fleet status in one view.
            </div>
            <div class="hero-meta">
                <span class="status-pill {risk_class}">Current Risk: {html.escape(risk)}</span>
                <span class="status-pill {risk_class}">Failure Probability: {prediction_prob:.2%}</span>
                <span class="status-pill">Machine: {html.escape(str(machine_id))}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_metric_card(label, value, caption=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(str(label))}</div>
            <div class="metric-value">{html.escape(str(value))}</div>
            <div class="metric-caption">{html.escape(str(caption))}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_model_performance(metrics):
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    metric_values = [
        ("Accuracy", f"{metrics['accuracy']:.2%}", "Overall correct predictions"),
        ("Precision", f"{metrics['precision']:.2%}", "How reliable high-risk alerts are"),
        ("Recall", f"{metrics['recall']:.2%}", "How many failures are caught"),
        ("F1 Score", f"{metrics['f1']:.2%}", "Balanced model score"),
    ]

    for col, (label, value, caption) in zip(cols, metric_values):
        with col:
            render_metric_card(label, value, caption)


def render_prediction_panel(
    current_machine_id,
    prediction_prob,
    risk,
    priority,
    suggested_action,
    alert_result,
    prediction_probabilities,
    machine_ids,
    sample,
):
    st.markdown('<div class="section-title">Prediction Panel</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    with cols[0]:
        render_metric_card("Machine ID", current_machine_id, "Latest selected machine")
    with cols[1]:
        render_metric_card("Failure Probability", f"{prediction_prob:.2%}", "Current model output")
    with cols[2]:
        render_metric_card("Risk Level", risk, "Green safe, yellow warning, red high")
    with cols[3]:
        render_metric_card("Priority", priority, alert_result["status"])

    risk_class = get_risk_class(risk)
    st.markdown(
        f"""
        <div class="action-panel {risk_class}">
            <strong>Suggested action:</strong> {html.escape(suggested_action)}
        </div>
        """,
        unsafe_allow_html=True
    )

    chart_cols = st.columns(2)
    with chart_cols[0]:
        render_probability_chart(prediction_probabilities, machine_ids)
    with chart_cols[1]:
        render_sensor_trend_chart(sample)


def render_probability_chart(probabilities, machine_ids):
    st.markdown('<div class="section-title">Failure Probability Trend</div>', unsafe_allow_html=True)

    chart_df = pd.DataFrame({
        "Reading": list(range(1, len(probabilities) + 1)),
        "Failure Probability": [float(value) * 100 for value in probabilities],
        "Machine ID": machine_ids[:len(probabilities)],
    })

    fig = px.line(
        chart_df,
        x="Reading",
        y="Failure Probability",
        hover_data=["Machine ID"],
        markers=True,
        color_discrete_sequence=["#1f9d55"],
    )
    fig.add_hrect(y0=0, y1=40, fillcolor="#e3f9e5", opacity=0.35, line_width=0)
    fig.add_hrect(y0=40, y1=70, fillcolor="#fff7cc", opacity=0.35, line_width=0)
    fig.add_hrect(y0=70, y1=100, fillcolor="#ffe3e3", opacity=0.35, line_width=0)
    fig.update_layout(
        height=330,
        margin=dict(l=8, r=8, t=16, b=8),
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        xaxis_title="Reading",
        yaxis_title="Failure probability",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, width="stretch")


def render_sensor_trend_chart(sample):
    st.markdown('<div class="section-title">Sensor Trends</div>', unsafe_allow_html=True)

    sensor_cols = get_raw_sensor_columns(sample.columns)[:6]

    if not sensor_cols:
        st.info("No raw sensor columns are available for trend charts.")
        return

    if len(sample) > 1:
        trend_df = sample[sensor_cols].reset_index(drop=True).copy()
        trend_df["Reading"] = range(1, len(trend_df) + 1)
        melted_df = trend_df.melt(
            id_vars="Reading",
            var_name="Sensor",
            value_name="Value"
        )
        fig = px.line(
            melted_df,
            x="Reading",
            y="Value",
            color="Sensor",
            markers=True,
        )
        fig.update_layout(
            height=330,
            margin=dict(l=8, r=8, t=16, b=8),
            xaxis_title="Reading",
            yaxis_title="Sensor value",
            paper_bgcolor="white",
            plot_bgcolor="white",
            legend_title_text="Sensor",
        )
    else:
        current_values = sample[sensor_cols].iloc[0].reset_index()
        current_values.columns = ["Sensor", "Value"]
        fig = px.bar(
            current_values,
            x="Sensor",
            y="Value",
            color_discrete_sequence=["#3a7ca5"],
        )
        fig.update_layout(
            height=330,
            margin=dict(l=8, r=8, t=16, b=8),
            xaxis_title="Sensor",
            yaxis_title="Current value",
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

    st.plotly_chart(fig, width="stretch")


@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    metrics = joblib.load(METRICS_PATH)
    return model, scaler, metrics


@st.cache_data
def load_feature_data():
    df = pd.read_csv(DATA_PATH)
    feature_df = df.drop(columns=DROP_COLS)
    return feature_df.select_dtypes(include=["int64", "float64"])


def align_features(input_df, feature_columns):
    sample = input_df.select_dtypes(include=["int64", "float64"])
    missing_cols = [col for col in feature_columns if col not in sample.columns]

    if missing_cols:
        return None, missing_cols

    return sample[feature_columns], []


def get_risk_details(probability):
    if probability < 0.4:
        return "Low", "Low", "Machine is operating normally. No immediate maintenance required."
    if probability < 0.7:
        return "Medium", "Medium", "Early signs of degradation detected. Schedule inspection soon."
    return "High", "High", "High failure risk detected. Immediate maintenance recommended."


def get_risk_level(probability):
    return get_risk_details(probability)[0]


def predict_failure_probability(sample, model, scaler):
    sample_scaled = scaler.transform(sample)
    return float(model.predict(sample_scaled)[-1][0])


def predict_failure_probabilities(sample, model, scaler):
    sample_scaled = scaler.transform(sample)
    return model.predict(sample_scaled).ravel()


def get_model_feature_importance(model, feature_columns):
    importance = pd.Series(1.0, index=feature_columns)

    try:
        weights = model.layers[0].get_weights()[0]
        if weights.shape[0] != len(feature_columns):
            return importance

        importance = pd.Series(abs(weights).mean(axis=1), index=feature_columns)
        max_importance = importance.max()

        if max_importance > 0:
            importance = importance / max_importance
    except (AttributeError, IndexError, ValueError):
        return pd.Series(1.0, index=feature_columns)

    return importance


def get_feature_stats(feature_df):
    stats = pd.DataFrame({
        "median": feature_df.median(numeric_only=True),
        "std": feature_df.std(numeric_only=True).replace(0, 1),
        "lower": feature_df.quantile(0.05, numeric_only=True),
        "upper": feature_df.quantile(0.95, numeric_only=True),
    })
    return stats.fillna(0)


def get_anomaly_thresholds(feature_df):
    sensor_cols = get_raw_sensor_columns(feature_df.columns)
    sensor_df = feature_df[sensor_cols]

    thresholds = pd.DataFrame({
        "median": sensor_df.median(numeric_only=True),
        "std": sensor_df.std(numeric_only=True).replace(0, 1),
        "lower": sensor_df.quantile(ANOMALY_LOW_QUANTILE, numeric_only=True),
        "upper": sensor_df.quantile(ANOMALY_HIGH_QUANTILE, numeric_only=True),
    })
    return thresholds.fillna(0)


def get_raw_sensor_columns(feature_columns):
    return [
        col for col in feature_columns
        if col.startswith("sensor_") and "_rolling_" not in col
    ]


def get_base_sensor_name(feature):
    if "_rolling_" in feature:
        return feature.split("_rolling_")[0]
    return feature


def get_friendly_feature_name(feature):
    base_feature = get_base_sensor_name(feature)

    if base_feature in SENSOR_LABELS:
        return SENSOR_LABELS[base_feature]

    if base_feature.startswith("sensor_"):
        return base_feature.replace("_", " ").title()

    if base_feature.startswith("op_setting_"):
        return base_feature.replace("_", " ").title()

    return base_feature.replace("_", " ").title()


def build_explanation_message(feature, value, median, lower, upper):
    friendly_name = get_friendly_feature_name(feature)
    direction = "above" if value >= median else "below"

    if feature.endswith("_rolling_std"):
        return (
            f"{friendly_name} has been changing more than normal. "
            "Unstable readings can increase failure risk."
        )

    if feature.endswith("_rolling_mean"):
        return (
            f"{friendly_name} has stayed {direction} normal recently. "
            "This pattern can increase failure risk."
        )

    if value > upper:
        return f"{friendly_name} is above normal range, which can increase failure risk."

    if value < lower:
        return f"{friendly_name} is below normal range, which can increase failure risk."

    return (
        f"{friendly_name} is one of the most important readings for this prediction. "
        "The model uses it together with other sensor values."
    )


def explain_prediction(sample, feature_df, model, top_n=4):
    stats = get_feature_stats(feature_df)
    importance = get_model_feature_importance(model, feature_df.columns)
    row = sample.iloc[0]
    explanations = []

    for feature in feature_df.columns:
        if not feature.startswith("sensor_"):
            continue

        value = row.get(feature)
        if pd.isna(value):
            continue

        median = stats.at[feature, "median"]
        std = stats.at[feature, "std"] or 1
        lower = stats.at[feature, "lower"]
        upper = stats.at[feature, "upper"]
        deviation_score = abs(value - median) / std
        importance_score = importance.get(feature, 1.0)
        score = deviation_score * (0.5 + importance_score)

        explanations.append({
            "feature": feature,
            "title": get_friendly_feature_name(feature),
            "message": build_explanation_message(feature, value, median, lower, upper),
            "value": value,
            "lower": lower,
            "upper": upper,
            "score": score,
        })

    explanations = sorted(explanations, key=lambda item: item["score"], reverse=True)
    return explanations[:top_n]


def detect_anomalies(sample, feature_df):
    thresholds = get_anomaly_thresholds(feature_df)
    row_anomaly_counts = []
    current_reading = sample.tail(1).iloc[0]
    current_anomalies = []

    # Compare each raw sensor against the normal range learned from training data.
    for _, row in sample.iterrows():
        anomaly_count = 0

        for feature in thresholds.index:
            value = row.get(feature)
            if pd.isna(value):
                continue

            lower = thresholds.at[feature, "lower"]
            upper = thresholds.at[feature, "upper"]

            if value < lower or value > upper:
                anomaly_count += 1

        row_anomaly_counts.append(anomaly_count)

    for feature in thresholds.index:
        value = current_reading.get(feature)
        if pd.isna(value):
            continue

        median = thresholds.at[feature, "median"]
        std = thresholds.at[feature, "std"] or 1
        lower = thresholds.at[feature, "lower"]
        upper = thresholds.at[feature, "upper"]

        if value < lower or value > upper:
            direction = "above" if value > upper else "below"
            severity = abs(value - median) / std
            current_anomalies.append({
                "feature": feature,
                "title": get_friendly_feature_name(feature),
                "message": (
                    f"{get_friendly_feature_name(feature)} is {direction} normal range. "
                    "This abnormal reading may increase failure risk."
                ),
                "value": value,
                "lower": lower,
                "upper": upper,
                "severity": severity,
            })

    current_anomalies = sorted(
        current_anomalies,
        key=lambda item: item["severity"],
        reverse=True
    )
    current_count = row_anomaly_counts[-1] if row_anomaly_counts else 0
    status = "Anomaly" if current_count >= ANOMALY_SENSOR_COUNT_THRESHOLD else "Normal"

    return {
        "status": status,
        "current_count": current_count,
        "row_counts": row_anomaly_counts,
        "top_anomalies": current_anomalies[:4],
    }


def get_alert_confidence(current_probability, anomaly_result, recent_high_count, recent_window_size):
    anomaly_signal = min(
        anomaly_result["current_count"] / ANOMALY_SENSOR_COUNT_THRESHOLD,
        1.0
    )
    persistence_signal = (
        recent_high_count / recent_window_size
        if recent_window_size else 0
    )

    if current_probability < 0.4:
        confidence = (1 - current_probability) * 0.7 + (1 - anomaly_signal) * 0.3
    else:
        confidence = (
            current_probability * 0.5
            + anomaly_signal * 0.25
            + persistence_signal * 0.25
        )

    return min(max(confidence, 0.0), 0.99)


def reduce_false_alarm(probabilities, anomaly_result):
    current_probability = float(probabilities[-1])
    recent_probabilities = probabilities[-FALSE_ALARM_WINDOW:]
    recent_high_count = int((recent_probabilities >= HIGH_RISK_THRESHOLD).sum())
    recent_window_size = len(recent_probabilities)
    persistent_high_risk = recent_high_count >= MIN_HIGH_RISK_READINGS
    has_anomaly = anomaly_result["status"] == "Anomaly"

    confidence = get_alert_confidence(
        current_probability,
        anomaly_result,
        recent_high_count,
        recent_window_size
    )

    # False alarm reduction: high model risk alone is not enough for a real alert.
    if current_probability >= HIGH_RISK_THRESHOLD and persistent_high_risk and has_anomaly:
        alert_status = "Real alert"
        message = (
            "Failure risk stayed high across recent readings and abnormal sensors were found. "
            "Treat this as a real maintenance alert."
        )
    elif current_probability >= HIGH_RISK_THRESHOLD:
        alert_status = "Possible false alarm"
        message = (
            "The model shows high risk, but the signal did not stay high enough "
            "or abnormal sensors were not strong enough. Recheck with more readings."
        )
    elif has_anomaly:
        alert_status = "Monitor closely"
        message = (
            "Some sensors look abnormal, but failure risk is not high yet. "
            "Keep monitoring before raising a maintenance alert."
        )
    else:
        alert_status = "No alert needed"
        message = "Current readings look stable and do not need an urgent alert."

    return {
        "status": alert_status,
        "message": message,
        "confidence": confidence,
        "recent_high_count": recent_high_count,
        "recent_window_size": recent_window_size,
        "persistent_high_risk": persistent_high_risk,
    }


def render_anomaly_alert_cards(anomaly_result, alert_result):
    st.subheader("Anomaly and False Alarm Check")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Anomaly Status", anomaly_result["status"])
    col2.metric("Confidence Level", f"{alert_result['confidence']:.0%}")
    col3.metric("Alert Check", alert_result["status"])
    col4.metric(
        "High-Risk Readings",
        f"{alert_result['recent_high_count']}/{alert_result['recent_window_size']}"
    )

    if alert_result["status"] == "Real alert":
        st.error(alert_result["message"])
    elif alert_result["status"] in ["Possible false alarm", "Monitor closely"]:
        st.warning(alert_result["message"])
    else:
        st.success(alert_result["message"])

    if not anomaly_result["top_anomalies"]:
        st.info("No major abnormal sensor values were found in the latest reading.")
        return

    st.markdown(
        """
        <style>
            .anomaly-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 12px;
                margin-bottom: 18px;
            }
            .anomaly-card {
                border: 1px solid #e0b4b4;
                border-radius: 8px;
                padding: 14px;
                background: #fffafa;
            }
            .anomaly-title {
                font-size: 0.98rem;
                font-weight: 700;
                margin-bottom: 6px;
                color: #7f1d1d;
            }
            .anomaly-body {
                font-size: 0.9rem;
                color: #334e68;
                line-height: 1.4;
                margin-bottom: 10px;
            }
            .anomaly-meta {
                font-size: 0.78rem;
                color: #627d98;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    cards = []
    for item in anomaly_result["top_anomalies"]:
        title = html.escape(item["title"])
        message = html.escape(item["message"])

        cards.append(
            f"""
            <div class="anomaly-card">
                <div class="anomaly-title">{title}</div>
                <div class="anomaly-body">{message}</div>
                <div class="anomaly-meta">
                    Current: {item["value"]:.2f} | Normal range: {item["lower"]:.2f} to {item["upper"]:.2f}
                </div>
            </div>
            """
        )

    st.markdown(
        f"<div class=\"anomaly-grid\">{''.join(cards)}</div>",
        unsafe_allow_html=True
    )


def load_maintenance_history():
    if not BOOKINGS_PATH.exists():
        return pd.DataFrame(columns=BOOKING_COLUMNS)

    history = pd.read_csv(BOOKINGS_PATH)

    for column in BOOKING_COLUMNS:
        if column not in history.columns:
            history[column] = ""

    return history[BOOKING_COLUMNS]


def save_maintenance_booking(booking):
    BOOKINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    history = load_maintenance_history()
    updated_history = pd.concat([history, pd.DataFrame([booking])], ignore_index=True)
    updated_history.to_csv(BOOKINGS_PATH, index=False)


def get_machine_ids(input_df, row_count, fallback_prefix="Machine"):
    for column in MACHINE_ID_COLUMNS:
        if column in input_df.columns:
            machine_ids = input_df[column].fillna("").astype(str).tolist()
            return [
                machine_id.strip() or f"{fallback_prefix}-{index + 1:03d}"
                for index, machine_id in enumerate(machine_ids[:row_count])
            ]

    return [f"{fallback_prefix}-{index + 1:03d}" for index in range(row_count)]


def get_maintenance_status_map():
    history = load_maintenance_history()

    if history.empty:
        return {}

    history = history.sort_values("created_at", ascending=False)
    latest_bookings = history.drop_duplicates(subset=["machine_id"], keep="first")
    return dict(zip(latest_bookings["machine_id"], latest_bookings["booking_status"]))


def build_fleet_dashboard(machine_ids, probabilities, anomaly_result):
    maintenance_status_map = get_maintenance_status_map()
    row_counts = anomaly_result["row_counts"]
    records = []

    for index, probability in enumerate(probabilities):
        machine_id = machine_ids[index] if index < len(machine_ids) else f"Machine-{index + 1:03d}"
        anomaly_count = row_counts[index] if index < len(row_counts) else 0
        anomaly_status = (
            "Anomaly"
            if anomaly_count >= ANOMALY_SENSOR_COUNT_THRESHOLD
            else "Normal"
        )

        records.append({
            "Machine ID": machine_id,
            "Failure Probability": float(probability),
            "Probability": f"{float(probability):.2%}",
            "Risk Level": get_risk_level(float(probability)),
            "Anomaly Status": anomaly_status,
            "Maintenance Status": maintenance_status_map.get(machine_id, "Not booked"),
        })

    fleet_df = pd.DataFrame(records)

    if fleet_df.empty:
        return fleet_df

    # For repeated machine readings, show the latest reading per machine in the fleet table.
    return fleet_df.drop_duplicates(subset=["Machine ID"], keep="last").reset_index(drop=True)


def render_fleet_dashboard(fleet_df):
    st.subheader("Fleet Monitoring Dashboard")

    if fleet_df.empty:
        st.info("No machine readings are available for fleet monitoring.")
        return

    total_machines = len(fleet_df)
    high_risk_machines = int((fleet_df["Risk Level"] == "High").sum())
    under_maintenance = int(
        fleet_df["Maintenance Status"].isin(MAINTENANCE_ACTIVE_STATUSES).sum()
    )
    normal_machines = int(
        (
            (fleet_df["Risk Level"] == "Low")
            & (fleet_df["Anomaly Status"] == "Normal")
        ).sum()
    )

    summary_cols = st.columns(4)
    summary_cols[0].metric("Total Machines", total_machines)
    summary_cols[1].metric("High-Risk Machines", high_risk_machines)
    summary_cols[2].metric("Under Maintenance", under_maintenance)
    summary_cols[3].metric("Normal Machines", normal_machines)

    selected_risks = st.multiselect(
        "Filter by risk",
        ["Low", "Medium", "High"],
        default=["Low", "Medium", "High"]
    )

    filtered_fleet = fleet_df[fleet_df["Risk Level"].isin(selected_risks)].copy()
    display_columns = [
        "Machine ID",
        "Probability",
        "Risk Level",
        "Anomaly Status",
        "Maintenance Status",
    ]

    st.dataframe(
        filtered_fleet[display_columns],
        width="stretch",
        hide_index=True
    )

    render_fleet_cards(filtered_fleet)


def render_fleet_cards(fleet_df):
    if fleet_df.empty:
        st.info("No machines match the selected risk filter.")
        return

    st.markdown(
        """
        <style>
            .fleet-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
                gap: 12px;
                margin: 14px 0 20px;
            }
            .fleet-card {
                border: 1px solid #d9e2ec;
                border-radius: 8px;
                padding: 14px;
                background: #ffffff;
            }
            .fleet-machine {
                font-size: 1rem;
                font-weight: 700;
                color: #102a43;
                margin-bottom: 8px;
            }
            .fleet-row {
                display: flex;
                justify-content: space-between;
                gap: 12px;
                font-size: 0.86rem;
                color: #334e68;
                padding: 3px 0;
            }
            .fleet-label {
                color: #627d98;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    cards = []

    for _, row in fleet_df.iterrows():
        cards.append(
            f"""
            <div class="fleet-card">
                <div class="fleet-machine">{html.escape(str(row["Machine ID"]))}</div>
                <div class="fleet-row">
                    <span class="fleet-label">Failure probability</span>
                    <strong>{html.escape(str(row["Probability"]))}</strong>
                </div>
                <div class="fleet-row">
                    <span class="fleet-label">Risk level</span>
                    <strong>{html.escape(str(row["Risk Level"]))}</strong>
                </div>
                <div class="fleet-row">
                    <span class="fleet-label">Anomaly status</span>
                    <strong>{html.escape(str(row["Anomaly Status"]))}</strong>
                </div>
                <div class="fleet-row">
                    <span class="fleet-label">Maintenance</span>
                    <strong>{html.escape(str(row["Maintenance Status"]))}</strong>
                </div>
            </div>
            """
        )

    st.markdown(
        f"<div class=\"fleet-grid\">{''.join(cards)}</div>",
        unsafe_allow_html=True
    )


def render_maintenance_booking(risk, prediction_prob, alert_result, default_machine_id=""):
    st.subheader("Maintenance Booking")

    if risk == "High":
        with st.form("maintenance_booking_form", clear_on_submit=True):
            machine_id = st.text_input("Machine ID", value=default_machine_id)
            issue_type = st.selectbox("Issue type", ISSUE_TYPES)
            preferred_date = st.date_input("Preferred date")
            technician_name_status = st.text_input(
                "Technician name/status",
                placeholder="Example: Priya - available"
            )
            booking_status = st.selectbox("Booking status", BOOKING_STATUSES)
            submitted = st.form_submit_button("Book Maintenance")

        if submitted:
            if not machine_id.strip():
                st.error("Please enter a Machine ID before booking maintenance.")
            else:
                # Save a lightweight booking record with the current prediction snapshot.
                save_maintenance_booking({
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "machine_id": machine_id.strip(),
                    "issue_type": issue_type,
                    "preferred_date": preferred_date.isoformat(),
                    "technician_name_status": technician_name_status.strip() or "Unassigned",
                    "booking_status": booking_status,
                    "failure_probability": f"{prediction_prob:.2%}",
                    "failure_risk": risk,
                    "alert_check": alert_result["status"],
                })
                st.success("Maintenance booking saved.")
    else:
        st.info("Book Maintenance becomes available when failure risk is High.")

    history = load_maintenance_history()
    st.subheader("Maintenance History")

    if history.empty:
        st.info("No maintenance bookings yet.")
    else:
        st.dataframe(history.sort_values("created_at", ascending=False), width="stretch")


def render_explanation_cards(explanations):
    st.subheader("Why This Prediction?")
    st.caption(
        "These are the sensor readings that looked most unusual compared with training data. "
        "They are a simple feature-importance explanation, not a guarantee that one sensor alone caused the result."
    )

    if not explanations:
        st.info("No unusual sensor pattern was found for this input.")
        return

    st.markdown(
        """
        <style>
            .explanation-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 12px;
                margin-bottom: 18px;
            }
            .explanation-card {
                border: 1px solid #d9e2ec;
                border-radius: 8px;
                padding: 14px;
                background: #ffffff;
            }
            .explanation-title {
                font-size: 0.98rem;
                font-weight: 700;
                margin-bottom: 6px;
                color: #102a43;
            }
            .explanation-body {
                font-size: 0.9rem;
                color: #334e68;
                line-height: 1.4;
                margin-bottom: 10px;
            }
            .explanation-meta {
                font-size: 0.78rem;
                color: #627d98;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    cards = []
    for item in explanations:
        title = html.escape(item["title"])
        message = html.escape(item["message"])
        current_value = item["value"]
        lower_value = item["lower"]
        upper_value = item["upper"]

        cards.append(
            f"""
            <div class="explanation-card">
                <div class="explanation-title">{title}</div>
                <div class="explanation-body">{message}</div>
                <div class="explanation-meta">
                    Current: {current_value:.2f} | Normal range: {lower_value:.2f} to {upper_value:.2f}
                </div>
            </div>
            """
        )

    st.markdown(
        f"<div class=\"explanation-grid\">{''.join(cards)}</div>",
        unsafe_allow_html=True
    )


model, scaler, metrics = load_artifacts()
feature_df = load_feature_data()

apply_dashboard_style()

st.sidebar.header("Input Options")

mode = st.sidebar.radio(
    "Choose prediction mode",
    ["Random equipment sample", "Manual sensor input", "Upload CSV"]
)

if mode == "Random equipment sample":
    random_machine_count = st.sidebar.slider(
        "Machines to sample",
        min_value=1,
        max_value=min(25, len(feature_df)),
        value=5
    )
    sample = feature_df.sample(random_machine_count, random_state=None)
    machine_ids = [f"Machine-{int(index) + 1:04d}" for index in sample.index]
    st.subheader("Selected Sensor Data")
    st.dataframe(sample, width="stretch")

elif mode == "Manual sensor input":
    st.subheader("Manual Sensor Input")
    manual_machine_id = st.text_input("Machine ID", value="Manual-001")
    default_sample = feature_df.median(numeric_only=True).to_frame().T
    default_sample = default_sample[feature_df.columns]

    sample = st.data_editor(
        default_sample,
        hide_index=True,
        num_rows="fixed",
        width="stretch",
        key="manual_sensor_input"
    )

    sample = sample[feature_df.columns]
    machine_ids = [manual_machine_id.strip() or "Manual-001"]

else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        machine_ids = get_machine_ids(uploaded_df, len(uploaded_df))

        for col in DROP_COLS:
            if col in uploaded_df.columns:
                uploaded_df = uploaded_df.drop(columns=[col])

        sample, missing_cols = align_features(uploaded_df, feature_df.columns)

        if missing_cols:
            st.error("Uploaded CSV is missing required feature-engineered columns.")
            st.write("Missing columns:", missing_cols[:20])
            st.stop()

        machine_ids = machine_ids[:len(sample)]
        st.subheader("Uploaded Sensor Data")
        st.dataframe(sample.head(10), width="stretch")
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

current_sample = sample.tail(1)
prediction_probabilities = predict_failure_probabilities(sample, model, scaler)
prediction_prob = float(prediction_probabilities[-1])
risk, priority, suggested_action = get_risk_details(prediction_prob)
anomaly_result = detect_anomalies(sample, feature_df)
alert_result = reduce_false_alarm(prediction_probabilities, anomaly_result)
fleet_df = build_fleet_dashboard(machine_ids, prediction_probabilities, anomaly_result)
current_machine_id = machine_ids[-1] if machine_ids else "Machine-001"

explanations = explain_prediction(current_sample, feature_df, model)

render_header(risk, prediction_prob, current_machine_id)
render_model_performance(metrics)

prediction_tab, explanation_tab, anomaly_tab, booking_tab, fleet_tab = st.tabs([
    "Prediction Panel",
    "Failure Explanation",
    "Anomaly Detection",
    "Maintenance Booking",
    "Fleet Monitoring",
])

with prediction_tab:
    render_prediction_panel(
        current_machine_id,
        prediction_prob,
        risk,
        priority,
        suggested_action,
        alert_result,
        prediction_probabilities,
        machine_ids,
        sample,
    )

    with st.expander("Current input data", expanded=False):
        st.dataframe(sample.head(25), width="stretch")

with explanation_tab:
    render_explanation_cards(explanations)

with anomaly_tab:
    render_anomaly_alert_cards(anomaly_result, alert_result)

with booking_tab:
    render_maintenance_booking(risk, prediction_prob, alert_result, current_machine_id)

with fleet_tab:
    render_fleet_dashboard(fleet_df)
