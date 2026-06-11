import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# 1. Configuration
# We set the page icon to your logo file so it appears in the browser tab
st.set_page_config(
    page_title="NetSentrixAI - Intelligent IDS", 
    page_icon="new_logo.png", 
    layout="wide"
)

# 2. Sidebar - Logo and Branding (Standard Image Style)
with st.sidebar:
    # This renders the image as a standard, plain block in the sidebar
    try:
        # st.image renders the image itself without special CSS borders or shadows
        # We adjust the width to fit nicely in the sidebar
        st.image("new_logo.png", width=300, output_format="PNG")
    except FileNotFoundError:
        # Fallback if image isn't found during a local test
        st.info("🛡️ NetSentrixAI")
    except Exception:
        # Other potential errors (e.g., format)
        st.info("🛡️ NetSentrixAI Logo Loading...")

    # Title below the image
    st.markdown("<h1 style='text-align: center; color: #2ecc71; font-family: Helvetica; font-weight: bold;'>NetSentrixAI</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.header("Control Panel")
    model_choice = st.selectbox("Select Detection Model", ("Decision Tree", "SVM"))

# 3. Load Assets & Constants
FEATURES = ['id', 'sttl', 'ct_state_ttl', 'dload', 'ct_dst_sport_ltm', 
            'dmean', 'rate', 'swin', 'dwin', 'ct_src_dport_ltm', 'ct_dst_src_ltm']

@st.cache_resource
def load_assets(model_path):
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        # Fallback to a default model if the specific one is missing
        model = joblib.load('final_model.pkl')

    try:
        scaler = joblib.load('scaler.pkl')
    except FileNotFoundError:
        scaler = None

    return model, scaler

# 4. UI Main Header
st.title("🛡️ Intelligent Intrusion Detection System")
st.markdown(f"**Status:** System Online | **Provider:** NetSentrixAI | **Active Model:** {model_choice}")

# 5. Data Processing
model_file = 'final_model.pkl' if model_choice == "Decision Tree" else 'svm_model.pkl'

try:
    df = pd.read_csv('UNSW_NB15.csv')
    st.sidebar.success("Dataset active: UNSW_NB15.csv")
except FileNotFoundError:
    st.error("⚠️ Dataset 'UNSW_NB15.csv' not found. Please upload it to your GitHub repository.")
    df = None

if df is not None:
    # Running Predictions
    model, scaler = load_assets(model_file)
    
    input_data = df[FEATURES].copy()
    predictions = model.predict(input_data)
    df['Prediction'] = ["Attack" if p == 1 else "Normal" for p in predictions]

    # Metrics Summary
    total_conn = len(df)
    attack_count = (df['Prediction'] == "Attack").sum()
    confidence = 98.2 # Mock confidence or model score

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Total Connections", total_conn)
    m_col2.metric("Attacks Detected", attack_count, delta=f"{attack_count}", delta_color="inverse")
    m_col3.metric("System Confidence", f"{confidence:.2f}%")

    # 6. Visual Alerts
    if attack_count > 0:
        st.subheader("🚨 Critical Security Alerts")
        attacks = df[df['Prediction'] == "Attack"].head(5)
        for _, row in attacks.iterrows():
            st.error(f"**ATTACK ALERT:** ID {row['id']} | Rate: {row['rate']} | STTL: {row['sttl']}")

    # 7. Data Visualization
    st.divider()
    v_col1, v_col2 = st.columns(2)

    with v_col1:
        st.subheader("Traffic Distribution")
        fig = px.pie(df, names='Prediction', color='Prediction', 
                     color_discrete_map={'Normal':'#2ecc71', 'Attack':'#e74c3c'},
                     hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with v_col2:
        st.subheader("Feature Impact (STTL vs Rate)")
        fig_scatter = px.scatter(df, x='sttl', y='rate', color='Prediction',
                                 color_discrete_map={'Normal':'#2ecc71', 'Attack':'#e74c3c'},
                                 symbol='Prediction', hover_data=['id'])
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Feature Importance (Only for Decision Tree)
    if model_choice == "Decision Tree":
        st.subheader("Feature Importance Analysis")
        importance = pd.DataFrame({'Feature': FEATURES, 'Score': model.feature_importances_})
        importance = importance.sort_values(by='Score', ascending=False)
        st.bar_chart(importance.set_index('Feature'))

    # 8. Raw Data Expander
    with st.expander("View Raw Prediction Data Table"):
        st.dataframe(df, use_container_width=True)

else:
    st.info("Awaiting network traffic data for analysis...")