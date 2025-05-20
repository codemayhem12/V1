import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# ─── 1) Page config & global CSS ─────────────────────────────────────────────────
st.set_page_config(
    page_title="🔧 Predictive Maintenance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
      /* Base background and sidebar */
      .stApp { background-color: #e8f0fe; }
      .sidebar .sidebar-content { background-color: #003d99; color: white; }
      /* Buttons and sliders */
      .stButton>button { background-color: #1a75ff; color: white; border-radius: 8px; }
      .stSlider>div input { accent-color: #1a75ff; }
      /* Info cards */
      .card { background-color: white; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
      .card-title { color: #003d99; font-size: 18px; font-weight: 600; }
      .card-value { color: #1a75ff; font-size: 32px; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

# ─── 2) Load & prepare data ───────────────────────────────────────────────────────
data = pd.read_csv('machinery_data.csv').fillna(method='ffill')
features = ['sensor_1', 'sensor_2', 'sensor_3', 'operational_hours']
target_rul = 'RUL'
target_maint = 'maintenance'

scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Train/test split for models
Xr, Xr_test, yr, yr_test = train_test_split(
    data[features], data[target_rul], test_size=0.2, random_state=42)
Xc, Xc_test, yc, yc_test = train_test_split(
    data[features], data[target_maint], test_size=0.2, random_state=42)

# Fit models
reg = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xr, yr)
clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xc, yc)
km = KMeans(n_clusters=2, random_state=42).fit(data[features])
data['cluster'] = km.labels_

def predict(vals):
    rul_pred = reg.predict([vals])[0]
    maint_pred = clf.predict([vals])[0]
    anom_pred = km.predict([vals])[0]
    return rul_pred, ('Needs Maintenance' if maint_pred else 'OK'), ('Anomaly' if anom_pred else 'Normal')

# ─── 3) Sidebar menu ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Menu")
    selected = option_menu(
        menu_title=None,
        options=["Home", "Historical", "Input", "Results", "Visuals"],
        icons=["house", "table", "gear", "check-circle", "bar-chart"],
        default_index=0,
        menu_icon="cast",
        orientation="vertical"
    )

# ─── 4) Home ─────────────────────────────────────────────────────────────────────
if selected == "Home":
    st.markdown("## 🔍 Dashboard Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("""
        <div class="card">
          <div class="card-title">Avg RUL</div>
          <div class="card-value">{:.1f} hrs</div>
        </div>
    """.format(data['RUL'].mean()), unsafe_allow_html=True)
    col2.markdown("""
        <div class="card">
          <div class="card-title">Maintenance Rate</div>
          <div class="card-value">{:.1f}%</div>
        </div>
    """.format(data['maintenance'].mean()*100), unsafe_allow_html=True)
    col3.markdown("""
        <div class="card">
          <div class="card-title">Anomaly Rate</div>
          <div class="card-value">{:.1f}%</div>
        </div>
    """.format((data['cluster']==1).mean()*100), unsafe_allow_html=True)
    col4.markdown("""
        <div class="card">
          <div class="card-title">Data Points</div>
          <div class="card-value">{}</div>
        </div>
    """.format(len(data)), unsafe_allow_html=True)

# ─── 5) Historical Data ─────────────────────────────────────────────────────────
elif selected == "Historical":
    st.markdown("## 📂 Historical Data")
    st.dataframe(data.head(15), use_container_width=True)

# ─── 6) Input Data ───────────────────────────────────────────────────────────────
elif selected == "Input":
    st.markdown("## 📝 Input Sensor Readings")
    cols = st.columns(4)
    vals = []
    for i, feat in enumerate(features):
        if feat == 'operational_hours':
            v = cols[i].number_input(
                'Operational Hours',
                float(data[feat].min()), float(data[feat].max()),
                float(data[feat].mean())
            )
        else:
            v = cols[i].slider(
                feat.replace("_", " ").title(),
                float(data[feat].min()), float(data[feat].max()),
                float(data[feat].mean())
            )
        vals.append(v)
    if st.button("Submit Readings"):
        st.session_state['input'] = vals
        st.success("Values saved! Check Results.")

# ─── 7) Results ─────────────────────────────────────────────────────────────────
elif selected == "Results":
    st.markdown("## 📊 Prediction Results")
    if 'input' not in st.session_state:
        st.warning("Please enter sensor readings first.")
    else:
        rul_val, maint_val, anom_val = predict(st.session_state['input'])
        st.metric("Remaining Useful Life", f"{rul_val:.1f} hrs")
        st.metric("Maintenance Status", maint_val)
        st.metric("Anomaly Detection", anom_val)

# ─── 8) Visualizations ──────────────────────────────────────────────────────────
elif selected == "Visuals":
    st.markdown("## 📈 Data Visualizations")

    # Row 1: RUL trend and correlation
    r1c1, r1c2 = st.columns((2, 3))
    with r1c1:
        st.subheader("RUL vs Operational Hours")
        fig_rul = px.line(
            data, x='operational_hours', y='RUL',
            title="Remaining Useful Life over Time",
            template="plotly_white",
            color_discrete_sequence=['#1a75ff']
        )
        fig_rul.update_layout(xaxis_title="Operational Hours", yaxis_title="RUL (hrs)",
                              plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_rul, use_container_width=True)

    with r1c2:
        st.subheader("Feature Correlations")
        corr = data[features].corr()
        fig_corr = px.imshow(
            corr, text_auto=True, color_continuous_scale="Blues",
            title="Sensor Correlation Matrix"
        )
        fig_corr.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_corr, use_container_width=True)

    # Row 2: Distributions and gauge
    r2c1, r2c2 = st.columns((3, 2))
    with r2c1:
        st.subheader("Sensor Distributions")
        df_long = data.melt(
            id_vars=['operational_hours'], value_vars=features[:-1],
            var_name="Sensor", value_name="Reading"
        )
        fig_dist = px.histogram(
            df_long, x="Reading", facet_col="Sensor", facet_col_wrap=2,
            nbins=30, color_discrete_sequence=['#1a75ff'],
            title="Histogram of Sensor Readings", template="plotly_white"
        )
        fig_dist.update_layout(showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

    with r2c2:
        st.subheader("Live Fault Probability")
        prob = 0.0
        if 'input' in st.session_state:
            rul_val, _, _ = predict(st.session_state['input'])
            prob = np.clip((100 - rul_val) / 100, 0, 1)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob*100,
            title={'text': "Fault Risk (%)", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0,100], 'tickcolor': '#1a75ff'},
                'bar': {'color': '#1a75ff'},
                'bgcolor': 'white',
                'borderwidth': 2, 'bordercolor': '#003d99'
            },
            number={'suffix': '%', 'font': {'size': 36, 'color': '#003d99'}}
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Row 3: Detailed scatter in expander
    with st.expander("🔍 Detailed Sensor vs RUL Scatter"):
        fig_scatter = px.scatter(
            data, x='sensor_1', y='RUL', color='cluster',
            title="Sensor 1 vs RUL by Cluster",
            color_discrete_map={0: '#003d99', 1: '#1a75ff'},
            template="plotly_white"
        )
        fig_scatter.update_traces(marker=dict(size=6, opacity=0.7))
        st.plotly_chart(fig_scatter, use_container_width=True)
