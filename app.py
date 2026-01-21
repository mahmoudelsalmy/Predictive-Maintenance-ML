"""
Prediction App - Users use this to make predictions
The model is already trained and loaded
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from scipy.stats import skew, kurtosis
import os


# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #06b6d4;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
    }
    h1, h2, h3 {
        color: #f1f5f9;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

@st.cache_resource
def load_model_components():
    try:
        required_files = [
            'trained_model.pkl',
            'scaler.pkl',
            'pca.pkl',
            'model_metadata.json'
        ]

        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            return None, None, None, None, None, f"Missing files: {', '.join(missing_files)}"

        model = joblib.load('trained_model.pkl')
        scaler = joblib.load('scaler.pkl')
        pca = joblib.load('pca.pkl')

        
        label_encoder = None

        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)

        return model, scaler, pca, label_encoder, metadata, None

    except Exception as e:
        return None, None, None, None, None, str(e)

def create_gauge_chart(value, title, max_value=100):
    """Create a gauge chart for confidence"""
    if value > 80:
        color = "#22c55e"  # Green
    elif value > 60:
        color = "#f59e0b"  # Orange
    else:
        color = "#ef4444"  # Red
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20, 'color': '#f1f5f9'}},
        number = {'suffix': "%", 'font': {'size': 40, 'color': '#f1f5f9'}},
        gauge = {
            'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
            'bar': {'color': color},
            'bgcolor': "rgba(30, 41, 59, 0.5)",
            'borderwidth': 2,
            'bordercolor': "#475569",
            'steps': [
                {'range': [0, 60], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [60, 80], 'color': 'rgba(251, 146, 60, 0.2)'},
                {'range': [80, 100], 'color': 'rgba(34, 197, 94, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#f1f5f9"},
        height=300
    )
    
    return fig

def single_value_to_window(value, window_size=256):
    noise = np.random.normal(
        loc=0,
        scale=0.05 * abs(value + 1e-6),
        size=window_size
    )
    return value + noise

def create_windows(signal, window_size=256, step=128):
    """
    Split 1D signal into overlapping windows
    """
    return [
        signal[i:i + window_size]
        for i in range(0, len(signal) - window_size + 1, step)
    ]

def extract_features(window):
    feats = [
        np.mean(window),
        np.std(window),
        np.sqrt(np.mean(window**2)),
        np.min(window),
        np.max(window),
        skew(window),
        kurtosis(window)
    ]
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    # Replace NaN or inf with 0
   

def build_feature_vector(a1, a2, a3, a4):
    features = []
    for val in [a1, a2, a3, a4]:
        fake_window = np.ones(256) * val
        features.extend(extract_features(fake_window))
    return np.array(features).reshape(1, -1)

def main():
    # Header
    st.markdown("<h1 style='text-align: center; color: #06b6d4;'>⚙️ Predictive Maintenance System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 24px;'>AI-Powered Equipment Health Monitoring</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model

    model, scaler, pca, label_encoder, metadata, error = load_model_components()

    
    if error:
        st.error(f"❌ Error loading model: {error}")
        st.info("💡 Please run `python train_model.py` first to train the model!")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=100)
        st.title("🎛️ Model Information")
        # st.info("🔬 Model Pipeline: Scaling → PCA → SVM (RBF)")

        # Model info
        st.success("✅ Model Loaded Successfully!")
        
        with st.expander("📊 Model Details", expanded=True):
            st.metric("Model Type", metadata['model_type'])
            # st.metric("Accuracy", f"{metadata['accuracy']:.2%}")
            # st.metric("Features",
            # metadata.get("features_per_sensor", 0) * len(metadata.get("sensors", [])))
            st.metric("Classes", 2)
            # st.write("**Trained:** ", metadata['trained_date'][:19])
        
        with st.expander("🎯 Target Classes"):
            target_classes = metadata.get(
                "target_classes",
                ["Healthy", "Faulty"]
            )
            for i, cls in enumerate(target_classes, 1):
                st.write(f"{i}. {cls}")
        
        with st.expander("📋 Feature List"):
            FEATURES = ["Motor vibration - X direction", "Motor vibration - Y direction", "Gearbox - vibration", "Bearing vibration"]
            for i, feat in enumerate(FEATURES, 1):
                st.write(f"{i}. {feat}")
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "📍 Select Mode",
            ["🔍 Single Prediction",
            "📁 Batch Prediction",
            "📊 History",
            "📈 PCA Visualization"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.metric("Total Predictions", len(st.session_state.predictions_history))
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.predictions_history = []
            st.rerun()
    
    # Main content
    if page == "🔍 Single Prediction":
        show_single_prediction(model, scaler, pca, label_encoder, metadata)

    elif page == "📁 Batch Prediction":
        show_batch_prediction(model, scaler, pca, label_encoder, metadata)

    elif page == "📈 PCA Visualization":
        show_pca_visualization(scaler, pca, label_encoder)

    else:
        show_history()

def show_single_prediction(model, scaler, pca, label_encoder, metadata):
    
    st.header("🔍 Single Equipment Prediction")
    st.info("💡 Enter sensor readings to predict equipment condition")
    
    SENSOR_LABELS = {
        "a1": "Motor Vibration X (mm/s)",
        "a2": "Motor Vibration Y (mm/s)",
        "a3": "Gearbox Vibration (mm/s)",
        "a4": "Bearing Vibration (mm/s)"
    }
    # Create dynamic input fields
    feature_names = metadata.get(
        "feature_names",
        ["a1", "a2", "a3", "a4"]
    )
    n_features = len(feature_names)
    
    # Calculate number of columns (3 inputs per row)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    st.subheader("📊 Enter Sensor Readings")
    
    input_values = []
    idx = 0
    
    for row in range(n_rows):
        cols = st.columns(n_cols)
        for col_idx in range(n_cols):
            if idx < n_features:
                with cols[col_idx]:
                    label = SENSOR_LABELS.get(feature_names[idx], feature_names[idx])

                    value = st.number_input(
                        label,
                        value=0.0,
                        step=0.01,
                        format="%.4f",
                        key=f"input_{feature_names[idx]}",
                        help=f"Enter {label} sensor reading"
                    )
                    input_values.append(value)
                idx += 1
    
    st.markdown("---")
    
    # Predict button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🎯 Predict Condition", type="primary", use_container_width=True)
    
    if predict_button:
        # Validate inputs
        if all(v == 0.0 for v in input_values):
            st.warning("⚠️ Please enter sensor readings (all values are zero)")
            return
        
        # Prepare input
        a1, a2, a3, a4 = input_values

        # Convert single values to pseudo windows (DEMO MODE)
        windows = [
            single_value_to_window(a1),
            single_value_to_window(a2),
            single_value_to_window(a3),
            single_value_to_window(a4)
        ]

        # Feature extraction
        features = []
        for w in windows:
            features.extend(extract_features(w))

        X_input = np.array(features).reshape(1, -1)

        # Scaling + PCA
        X_scaled = scaler.transform(X_input)
        X_pca = pca.transform(X_scaled)
                
        # Predict
        prediction = int(model.predict(X_pca)[0])
        probabilities = model.predict_proba(X_pca)[0]

        # Decode prediction
        class_map = {0: "Healthy", 1: "Faulty"}
        predicted_class = class_map[prediction]

        # Confidence
        confidence = probabilities[prediction] * 100

        # Save to history
        st.session_state.predictions_history.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'predicted_class': predicted_class,
            'confidence': confidence,
            'mode': 'Single'
        })
        
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        
        # Result metrics
        col1, col2 = st.columns(2)
        with col1:
            if predicted_class.lower() == "healthy":
                st.metric(
                "Predicted Condition",
                predicted_class,
                delta="Safe",
                delta_color="normal"
            )
            else:
                st.metric(
                    "Predicted Condition",
                    predicted_class,
                    delta="High Risk",
                    delta_color="inverse"
                )

        with col2:
            st.metric("Confidence Level", f"{confidence:.1f}%")
        
        # Gauge chart
        col1, col2 = st.columns([1, 1])
        with col1:
            fig = create_gauge_chart(confidence, "Confidence Score")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 All Class Probabilities")

            class_names = ["Healthy", "Faulty"]

            prob_df = pd.DataFrame({
                'Condition': class_names,
                'Probability (%)': probabilities * 100
            })
            
            fig = px.bar(prob_df, x='Condition', y='Probability (%)',
                        color='Probability (%)',
                        color_continuous_scale='RdYlGn',
                        text='Probability (%)')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(
                paper_bgcolor='rgba(30, 41, 59, 0.5)',
                plot_bgcolor='rgba(30, 41, 59, 0.5)',
                font={'color': "#f1f5f9"},
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("---")
        st.subheader("📋 Recommendations")

        if predicted_class.lower() == "healthy":
            st.success("✅ Equipment is operating normally")
            st.write("**Recommended Actions:**")
            st.write("• Continue regular monitoring")
            st.write("• Follow standard maintenance schedule")
            st.write("• No immediate action required")

        else:
            if confidence > 80:
                st.error(f"🚨 High confidence fault detected: **{predicted_class}**")
                st.write("**Immediate Actions Required:**")
                st.write("• Schedule immediate inspection")
                st.write("• Prepare for equipment shutdown if necessary")
                st.write("• Alert maintenance team")
            elif confidence > 60:
                st.warning(f"⚠️ Moderate confidence fault detected: **{predicted_class}**")
                st.write("**Recommended Actions:**")
                st.write("• Schedule inspection within 48 hours")
                st.write("• Increase monitoring frequency")
            else:
                st.info(f"ℹ️ Low confidence fault indication: **{predicted_class}**")
                st.write("**Recommended Actions:**")
                st.write("• Verify sensor readings")
                st.write("• Collect additional data")

def show_batch_prediction(model, scaler, pca, label_encoder, metadata):
    
    st.header("📁 Batch Equipment Prediction")
    st.info("📤 Upload a CSV file with sensor readings for multiple equipment units")
    
    # Show expected format
    with st.expander("📋 Expected CSV Format"):
        st.write("**Required columns (Raw Signal):**")
        st.write("• Motor vibration - X direction")
        st.write("• Motor vibration - Y direction")
        st.write("• Gearbox - vibration")
        st.write("• Bearing vibration")

        st.info(
            "📌 CSV must contain RAW time-series data.\n"
            "Each row = one time step (not averaged values)."
        )

        sample_df = pd.DataFrame({
            "Motor vibration - X direction": np.random.randn(300),
            "Motor vibration - Y direction": np.random.randn(300),
            "Gearbox - vibration": np.random.randn(300),
            "Bearing vibration": np.random.randn(300),
        })

        st.dataframe(sample_df.head(10))

        csv = sample_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Sample CSV Template",
            csv,
            "sample_template.csv",
            "text/csv",
        )
        
    
    # File uploader
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Found {len(df)} records")
            
            # Validate columns
            missing_cols = [col for col in metadata['feature_names'] if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
                return
            
            with st.expander("👁️ Preview Uploaded Data"):
                st.dataframe(df.head(10))
            
            if st.button("🚀 Predict All    Equipment", type="primary"):
                with st.spinner("Making predictions..."):

                    SENSORS = ["a1", "a2", "a3", "a4"]

                    # =============================
                    # CREATE WINDOWS (RAW SIGNAL)
                    # =============================
                    sensor_windows = {}
                    for s in SENSORS:
                        sensor_windows[s] = create_windows(df[s].values)

                    num_windows = min(len(sensor_windows[s]) for s in SENSORS)

                    if num_windows == 0:
                        st.error("❌ Not enough data (need at least 256 samples)")
                        return

                    # =============================
                    # FEATURE EXTRACTION
                    # =============================
                    X_features = []
                    for i in range(num_windows):
                        feats = []
                        for s in SENSORS:
                            feats.extend(extract_features(sensor_windows[s][i]))
                        X_features.append(feats)

                    X_features = np.array(X_features)

                    # =============================
                    # PREDICTION PIPELINE
                    # =============================
                    X_scaled = scaler.transform(X_features)
                    X_pca = pca.transform(X_scaled)

                    predictions = model.predict(X_pca)
                    probabilities = model.predict_proba(X_pca)

                    class_map = {0: "Healthy", 1: "Faulty"}
                    predicted_classes = np.array([class_map[int(p)] for p in predictions])
                    confidences = probabilities.max(axis=1) * 100

                    # =============================
                    # RESULTS
                    # =============================
                    results_df = pd.DataFrame({
                        "Predicted_Condition": predicted_classes,
                        "Confidence (%)": confidences
                    })

                    st.success("✅ Prediction completed!")
                    st.dataframe(results_df.head(20))

                    # Summary
                    class_counts = results_df["Predicted_Condition"].value_counts()

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Windows", len(results_df))
                    with col2:
                        st.metric("Faulty Windows", class_counts.get("Faulty", 0))

                    # Pie chart
                    fig = px.pie(
                        values=class_counts.values,
                        names=class_counts.index,
                        title="Condition Distribution"
                    )
                    st.plotly_chart(fig)

        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            
def show_pca_visualization(scaler, pca, label_encoder):
    st.header("📈 PCA Visualization")
    st.write(
        "This visualization shows the dataset projected onto two principal components "
        "(PC1 and PC2) to illustrate class separability."
    )

    # Load original data
    df_healthy = pd.read_csv("Data/fullHealthy.csv")
    df_broken = pd.read_csv("Data/fullBroken.csv")

    df_healthy["label"] = 0
    df_broken["label"] = 1

    df = pd.concat([df_healthy, df_broken], ignore_index=True)

    FEATURES = ["a1", "a2", "a3", "a4"]
    df = df[FEATURES + ["label"]]
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    df.dropna(inplace=True)

    # Sample for performance
    df = df.sample(n=5000, random_state=42)

    # -------- Helper functions --------
    def extract_features(window):
        feats = [
            np.mean(window),
            np.std(window),
            np.sqrt(np.mean(window**2)),
            np.min(window),
            np.max(window),
            skew(window),
            kurtosis(window)
        ]
        return np.nan_to_num(feats)

    def create_windows(signal, window_size=256, step=256):
        return [
            signal[i:i+window_size]
            for i in range(0, len(signal) - window_size, step)
        ]

    # -------- Feature extraction --------
    X_features = []
    y_labels = []

    for label, group in df.groupby("label"):
        sensor_windows = []
        for col in FEATURES:
            sensor_windows.append(create_windows(group[col].values))

        num_windows = min(len(w) for w in sensor_windows)

        for i in range(num_windows):
            feats = []
            for sw in sensor_windows:
                feats.extend(extract_features(sw[i]))
            X_features.append(feats)
            y_labels.append(label)

    X_features = np.array(X_features)
    y_labels = np.array(y_labels)

    # Apply trained pipeline
    X_scaled = scaler.transform(X_features)
    X_pca = pca.transform(X_scaled)

    # Prepare DataFrame for plotting
    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Condition": y_labels
    })

    pca_df["Condition"] = pca_df["Condition"].map({
        0: "Healthy",
        1: "Faulty"
    })

    # Plot
    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Condition",
        title="PCA Projection of Gearbox Data",
        opacity=0.7
    )

    fig.update_layout(
        paper_bgcolor='rgba(30, 41, 59, 0.5)',
        plot_bgcolor='rgba(30, 41, 59, 0.5)',
        font={'color': "#f1f5f9"}
    )

    st.plotly_chart(fig, use_container_width=True)
def show_history():
    st.header("📊 Prediction History")
    
    if len(st.session_state.predictions_history) == 0:
        st.info("📭 No predictions yet. Make some predictions to see history!")
        return
    
    # Convert to dataframe
    df_history = pd.DataFrame(st.session_state.predictions_history)
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predictions", len(df_history))
    with col2:
        st.metric("Avg Confidence", f"{df_history['confidence'].mean():.1f}%")
    with col3:
        unique_classes = df_history['predicted_class'].nunique()
        st.metric("Unique Conditions", unique_classes)
    with col4:
        latest = df_history.iloc[-1]['predicted_class']
        st.metric("Latest Prediction", latest)
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Condition distribution
        class_counts = df_history['predicted_class'].value_counts()
        fig = px.bar(x=class_counts.index, y=class_counts.values,
                    title='Prediction Distribution',
                    labels={'x': 'Condition', 'y': 'Count'},
                    color=class_counts.values,
                    color_continuous_scale='Viridis')
        fig.update_layout(
            paper_bgcolor='rgba(30, 41, 59, 0.5)',
            plot_bgcolor='rgba(30, 41, 59, 0.5)',
            font={'color': "#f1f5f9"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confidence over time
        fig = px.scatter(df_history, x=df_history.index, y='confidence',
                        color='predicted_class',
                        title='Confidence Levels Over Time',
                        labels={'index': 'Prediction #', 'confidence': 'Confidence (%)'})
        fig.update_layout(
            paper_bgcolor='rgba(30, 41, 59, 0.5)',
            plot_bgcolor='rgba(30, 41, 59, 0.5)',
            font={'color': "#f1f5f9"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # History table
    st.markdown("---")
    st.subheader("📋 All Predictions")
    
    # Sort by timestamp (newest first)
    df_display = df_history.sort_values('timestamp', ascending=False).reset_index(drop=True)
    st.dataframe(df_display, use_container_width=True)
    
    # Download history
    csv = df_display.to_csv(index=False)
    st.download_button(
        "⬇️ Download History",
        csv,
        f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv",
        use_container_width=True
    )

if __name__ == "__main__":

    main()

