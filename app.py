import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forgetfulness Predictor",
    page_icon="🧠",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f5f7fb; }
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .suggestion-box {
        border-radius: 12px;
        padding: 1.4rem;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
        margin-top: 1rem;
    }
    .red-box   { background:#ffe5e5; color:#c0392b; border:2px solid #e74c3c; }
    .amber-box { background:#fff3cd; color:#856404; border:2px solid #ffc107; }
    .green-box { background:#d4edda; color:#155724; border:2px solid #28a745; }
    h1 { color: #2c3e50; }
    h2, h3 { color: #34495e; }
</style>
""", unsafe_allow_html=True)

# ── Load & preprocess data ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    le = LabelEncoder()
    df["Difficulty_enc"] = le.fit_transform(df["Difficulty"])   # Easy=0, Hard=1, Medium=2
    return df, le

@st.cache_resource
def train_models(df):
    features = ["Time_since_study", "Revisions", "Study_duration", "Difficulty_enc", "CGPA"]
    X = df[features]
    y = df["Retention"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_rmse = mean_squared_error(y_test, lr.predict(X_test)) ** 0.5

    # Random Forest
    rf = RandomForestRegressor(n_estimators=150, random_state=42)
    rf.fit(X_train, y_train)
    rf_rmse = mean_squared_error(y_test, rf.predict(X_test)) ** 0.5

    return lr, rf, lr_rmse, rf_rmse, X_test, y_test

df, le = load_data()
lr_model, rf_model, lr_rmse, rf_rmse, X_test, y_test = train_models(df)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 🧠 Forgetfulness Predictor")
st.markdown("*Predict how much you remember — inspired by the **Ebbinghaus Forgetting Curve***")
st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Performance", "📈 Forgetting Curve"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Your Study Details")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        days = st.slider("📅 Days since you studied", 0, 30, 7,
                         help="How many days ago did you study this topic?")
        revisions = st.slider("🔁 Number of revisions done", 0, 10, 2)
        duration = st.slider("⏱️ Study duration (hours)", 0.5, 8.0, 2.0, step=0.5)

    with col2:
        difficulty = st.selectbox("🎯 Topic difficulty", ["Easy", "Medium", "Hard"])
        cgpa = st.slider("📚 Your CGPA", 2.0, 4.0, 3.0, step=0.1,
                         help="Overall academic performance")
        model_choice = st.radio("🤖 Model to use",
                                ["Random Forest (recommended)", "Linear Regression"],
                                horizontal=True)

    st.markdown("")
    predict_btn = st.button("🔮 Predict Retention", use_container_width=True, type="primary")

    if predict_btn:
        diff_enc = le.transform([difficulty])[0]
        input_data = pd.DataFrame([[days, revisions, duration, diff_enc, cgpa]],
                                  columns=["Time_since_study", "Revisions",
                                           "Study_duration", "Difficulty_enc", "CGPA"])

        model = rf_model if "Random Forest" in model_choice else lr_model
        retention = float(model.predict(input_data)[0])
        retention = max(0, min(100, retention))

        st.divider()
        st.subheader("📋 Prediction Result")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2.5rem;">💡</div>
                <div style="font-size:2rem; font-weight:700; color:#2980b9;">{retention:.1f}%</div>
                <div style="color:#7f8c8d;">Estimated Retention</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2.5rem;">📅</div>
                <div style="font-size:2rem; font-weight:700; color:#8e44ad;">{days} days</div>
                <div style="color:#7f8c8d;">Since Last Study</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2.5rem;">🔁</div>
                <div style="font-size:2rem; font-weight:700; color:#27ae60;">{revisions}</div>
                <div style="color:#7f8c8d;">Revisions Done</div>
            </div>""", unsafe_allow_html=True)

        # Suggestion
        st.markdown("")
        if retention < 40:
            msg = "⚠️ Retention is LOW — Revise Immediately!"
            css = "red-box"
            tip = "Your memory of this topic is fading fast. Schedule a revision session today."
        elif retention < 70:
            msg = "🕐 Retention is MODERATE — Revise Soon"
            css = "amber-box"
            tip = "You still remember a fair amount, but don't wait too long. Revise within 1–2 days."
        else:
            msg = "✅ Retention is GOOD — Keep it up!"
            css = "green-box"
            tip = "Great job! Consider a quick review in a few days to lock in long-term memory."

        st.markdown(f'<div class="suggestion-box {css}">{msg}</div>', unsafe_allow_html=True)
        st.info(f"💬 {tip}")

        # Retention gauge bar
        st.markdown("")
        st.markdown(f"**Retention Level:** {retention:.1f}%")
        st.progress(int(retention))

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Comparison")

    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:2rem;">📐</div>
            <div style="font-size:1.8rem; font-weight:700; color:#e74c3c;">{lr_rmse:.2f}</div>
            <div style="color:#7f8c8d;">Linear Regression RMSE</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:2rem;">🌲</div>
            <div style="font-size:1.8rem; font-weight:700; color:#27ae60;">{rf_rmse:.2f}</div>
            <div style="color:#7f8c8d;">Random Forest RMSE</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    winner = "Random Forest" if rf_rmse < lr_rmse else "Linear Regression"
    improvement = abs(lr_rmse - rf_rmse)
    st.success(f"✅ **{winner}** performs better with {improvement:.2f} lower RMSE.")

    st.divider()

    # Actual vs Predicted scatter plot
    st.subheader("Actual vs Predicted — Random Forest")
    rf_preds = rf_model.predict(X_test)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_test, rf_preds, alpha=0.4, color="#3498db", s=20, label="Predictions")
    ax.plot([0, 100], [0, 100], "r--", lw=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Retention (%)", fontsize=11)
    ax.set_ylabel("Predicted Retention (%)", fontsize=11)
    ax.set_title("Actual vs Predicted Retention", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

    # Feature importance
    st.subheader("Feature Importance (Random Forest)")
    features = ["Time_since_study", "Revisions", "Study_duration", "Difficulty", "CGPA"]
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)

    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    colors = ["#3498db" if i != sorted_idx[-1] else "#e74c3c" for i in sorted_idx]
    ax2.barh([features[i] for i in sorted_idx], importances[sorted_idx], color=colors)
    ax2.set_xlabel("Importance", fontsize=11)
    ax2.set_title("Which factors affect retention the most?", fontsize=13)
    ax2.grid(True, axis="x", alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — FORGETTING CURVE
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📈 Ebbinghaus Forgetting Curve Simulation")
    st.markdown("See how retention changes over time based on your study habits.")

    col_a, col_b = st.columns(2)
    with col_a:
        sim_revisions = st.slider("Revisions", 0, 10, 3, key="sim_rev")
        sim_duration  = st.slider("Study duration (hrs)", 0.5, 8.0, 3.0, step=0.5, key="sim_dur")
    with col_b:
        sim_difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], key="sim_diff")
        sim_cgpa       = st.slider("CGPA", 2.0, 4.0, 3.0, step=0.1, key="sim_cgpa")

    days_range = np.arange(0, 31)
    diff_enc_sim = le.transform([sim_difficulty])[0]

    sim_input = pd.DataFrame({
        "Time_since_study": days_range,
        "Revisions":        [sim_revisions] * 31,
        "Study_duration":   [sim_duration]  * 31,
        "Difficulty_enc":   [diff_enc_sim]  * 31,
        "CGPA":             [sim_cgpa]      * 31,
    })
    sim_preds = rf_model.predict(sim_input)
    sim_preds = np.clip(sim_preds, 0, 100)

    # Also show 0-revision curve for comparison
    zero_rev_input = sim_input.copy()
    zero_rev_input["Revisions"] = 0
    zero_preds = np.clip(rf_model.predict(zero_rev_input), 0, 100)

    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    ax3.plot(days_range, sim_preds, color="#3498db", lw=2.5,
             label=f"With {sim_revisions} revision(s)")
    ax3.plot(days_range, zero_preds, color="#e74c3c", lw=2, linestyle="--",
             label="No revisions (baseline)")
    ax3.fill_between(days_range, zero_preds, sim_preds, alpha=0.15, color="#3498db")
    ax3.axhline(70, color="#27ae60", linestyle=":", lw=1.5, label="Good retention threshold (70%)")
    ax3.axhline(40, color="#f39c12", linestyle=":", lw=1.5, label="Low retention threshold (40%)")
    ax3.set_xlabel("Days Since Study", fontsize=11)
    ax3.set_ylabel("Estimated Retention (%)", fontsize=11)
    ax3.set_title("Retention Over Time — Ebbinghaus Forgetting Curve", fontsize=13)
    ax3.set_ylim(0, 105)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    st.pyplot(fig3)

    st.caption("📌 The shaded area shows the memory improvement gained by revising. "
               "More revisions = slower forgetting.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#95a5a6; font-size:0.85rem;'>"
    "🧠 Forgetfulness Predictor · Inspired by the Ebbinghaus Forgetting Curve · "
    "Powered by Random Forest ML"
    "</div>",
    unsafe_allow_html=True,
)
