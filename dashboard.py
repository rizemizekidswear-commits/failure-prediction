
import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# ─── تنظیمات صفحه ───────────────────────────────
st.set_page_config(
    page_title="سیستم پیش‌بینی خرابی تجهیزات",
    page_icon="⚙️",
    layout="wide"
)

# ─── لود مدل ────────────────────────────────────
model = joblib.load("failure_model.pkl")
scaler = joblib.load("scaler.pkl")

# ─── هدر ────────────────────────────────────────
st.title(" سیستم پیش‌بینی خرابی تجهیزات صنعتی")
st.markdown("---")

# ─── ورودی‌ها ────────────────────────────────────
st.subheader(" مقادیر سنسورها را وارد کنید")

col1, col2 = st.columns(2)

with col1:
    air_temp = st.slider(" دمای هوا (K)", 295.0, 310.0, 298.0, 0.1)
    process_temp = st.slider(" دمای پروسه (K)", 305.0, 320.0, 308.0, 0.1)
    rot_speed = st.slider(" سرعت چرخش (RPM)", 1000, 2000, 1500, 10)

with col2:
    torque = st.slider(" گشتاور (Nm)", 10.0, 80.0, 40.0, 0.1)
    tool_wear = st.slider("⏱ فرسودگی ابزار (min)", 0, 250, 100, 1)

st.markdown("---")

# ─── پیش‌بینی ────────────────────────────────────
if st.button("🔍 پیش‌بینی کن", use_container_width=True):
    
    features = np.array([[air_temp, process_temp, rot_speed, torque, tool_wear]])
    features_scaled = scaler.transform(features)
    prob = model.predict_proba(features_scaled)[0][1]
    threshold = 0.3
    status = "خراب" if prob >= threshold else "سالم"

    col1, col2 = st.columns(2)

    # ─── گیج ────────────────────────────────────
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            title={"text": "احتمال خرابی (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "red" if prob >= threshold else "green"},
                "steps": [
                    {"range": [0, 30], "color": "#d4edda"},
                    {"range": [30, 60], "color": "#fff3cd"},
                    {"range": [60, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": 30
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # ─── وضعیت ──────────────────────────────────
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if status == "خراب":
            st.error(f"🔴 وضعیت: {status}")
            st.warning("⚠️ نیاز به بررسی فوری دارد!")
        else:
            st.success(f"🟢 وضعیت: {status}")
            st.info("✅ تجهیز در وضعیت مناسب است")
        
        st.metric("احتمال خرابی", f"{prob*100:.1f}%")
        st.metric("آستانه تصمیم", "30%")
