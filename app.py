import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import joblib
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineer import AutoFeatureEngineer
from auto_ml_engine import detect_problem_type, train_all_models, select_best_model, get_feature_importance
from predictor import predict_single, predict_batch, get_prediction_confidence

# Page Config
st.set_page_config(page_title="Auto-ML Predictor", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown("""
<style>
    :root {
        --primary-color: #00f2fe;
        --secondary-color: #4facfe;
        --bg-color: #0e1117;
        --card-bg: rgba(255, 255, 255, 0.05);
    }
    
    .stApp {
        background-color: var(--bg-color);
        color: white;
    }
    
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .gradient-text {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(0,242,254,0.1) 0%, rgba(79,172,254,0.1) 100%);
        border-left: 4px solid var(--primary-color);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #00f2fe, #4facfe);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 242, 254, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'df' not in st.session_state:
    st.session_state.df = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None

# Header
col1, col2 = st.columns([1, 5])
with col1:
    logo_path = os.path.join(os.path.dirname(__file__), '..', 'Logo.webp')
    if os.path.exists(logo_path):
        image = Image.open(logo_path)
        st.image(image, width=100)
with col2:
    st.markdown('<h1 class="gradient-text">🤖 Auto-ML Predictor Platform</h1>', unsafe_allow_html=True)
    st.markdown('Build, train, and deploy machine learning models instantly.')

st.markdown("---")

# Sidebar navigation
st.sidebar.title("Navigation")
step = st.sidebar.radio("Go to:", 
    ["1. Data Upload", "2. Data Preview", "3. Target Selection", 
     "4. Feature Selection", "5. Train Models", "6. Results & Analysis", "7. Predict"]
)

sample_data_path = os.path.join(os.path.dirname(__file__), '..', 'sample_data', 'Almunajem_Stock_Full_Features.xlsx')

def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file)
        elif file.name.endswith('.json'):
            return pd.read_json(file)
        elif file.name.endswith('.parquet'):
            return pd.read_parquet(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

if step == "1. Data Upload":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, JSON, Parquet)", type=['csv', 'xlsx', 'json', 'parquet'])
    
    if st.button("Load Sample Dataset"):
        if os.path.exists(sample_data_path):
            st.session_state.df = pd.read_excel(sample_data_path)
            st.success("Sample dataset loaded successfully!")
        else:
            st.error("Sample dataset not found.")
            
    if uploaded_file is not None:
        st.session_state.df = load_data(uploaded_file)
        st.success(f"Successfully loaded {uploaded_file.name}")
        
    st.markdown('</div>', unsafe_allow_html=True)

elif step == "2. Data Preview":
    if st.session_state.df is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.df.head())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><h3>Rows</h3><h2>{st.session_state.df.shape[0]}</h2></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>Columns</h3><h2>{st.session_state.df.shape[1]}</h2></div>', unsafe_allow_html=True)
        with col3:
            missing = st.session_state.df.isnull().sum().sum()
            st.markdown(f'<div class="metric-card"><h3>Missing Values</h3><h2>{missing}</h2></div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Please upload data in Step 1.")

elif step == "3. Target Selection":
    if st.session_state.df is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Select Target Variable")
        columns = st.session_state.df.columns.tolist()
        target_col = st.selectbox("Select the column you want to predict:", columns)
        
        st.session_state.target_col = target_col
        auto_ptype = detect_problem_type(st.session_state.df[target_col])
        
        ptype = st.selectbox("Problem Type:", ['regression', 'classification'], index=0 if auto_ptype == 'regression' else 1)
        st.session_state.problem_type = ptype
        
        st.success(f"Target selected: {target_col}. Problem type: {ptype}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Please upload data in Step 1.")

elif step == "4. Feature Selection":
    if st.session_state.df is not None and hasattr(st.session_state, 'target_col'):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Select Features")
        
        all_cols = st.session_state.df.columns.tolist()
        all_cols.remove(st.session_state.target_col)
        
        # Simple auto-exclude datetimes
        default_features = [c for c in all_cols if not pd.api.types.is_datetime64_any_dtype(st.session_state.df[c])]
        
        selected_features = st.multiselect("Select features to use:", all_cols, default=default_features)
        st.session_state.selected_features = selected_features
        
        time_series = st.checkbox("Is this a time-series dataset? (Enable advanced TS features)")
        st.session_state.time_series = time_series
        if time_series:
            date_col = st.selectbox("Select Date/Time column (optional):", ['None'] + all_cols)
            st.session_state.date_col = date_col if date_col != 'None' else None
        else:
            st.session_state.date_col = None
            
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Please complete previous steps.")

elif step == "5. Train Models":
    if st.session_state.df is not None and hasattr(st.session_state, 'selected_features'):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Train Models")
        
        if st.button("🚀 Train Models"):
            with st.spinner("Engineering features and training models... This might take a while."):
                df = st.session_state.df.copy()
                X = df[st.session_state.selected_features]
                y = df[st.session_state.target_col]
                
                # Drop rows where target is missing
                valid_idx = y.dropna().index
                X = X.loc[valid_idx]
                y = y.loc[valid_idx]
                
                # Split data
                from sklearn.model_selection import train_test_split
                if st.session_state.time_series and st.session_state.date_col:
                    X = X.sort_values(by=st.session_state.date_col)
                    y = y.loc[X.index]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                elif st.session_state.problem_type == 'classification':
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                st.session_state.y_test = y_test
                    
                # Feature Engineering
                fe = AutoFeatureEngineer(
                    time_series=st.session_state.time_series, 
                    target_col=st.session_state.target_col if st.session_state.time_series else None,
                    date_col=st.session_state.date_col
                )
                
                X_train_processed = fe.fit_transform(X_train)
                X_test_processed = fe.transform(X_test)
                
                st.session_state.pipeline = fe
                st.session_state.X_test_processed = X_test_processed
                
                # Train Models
                results = train_all_models(X_train_processed, y_train, X_test_processed, y_test, st.session_state.problem_type)
                st.session_state.results = results
                
                best_name, best_model = select_best_model(results, st.session_state.problem_type)
                st.session_state.best_model_name = best_name
                st.session_state.best_model = best_model
                
                st.success(f"Training complete! Best model: {best_name}")
                
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Please complete previous steps.")

elif step == "6. Results & Analysis":
    if st.session_state.results is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Model Leaderboard")
        
        results_data = []
        for name, data in st.session_state.results.items():
            row = {'Model': name}
            row.update(data['metrics'])
            results_data.append(row)
            
        results_df = pd.DataFrame(results_data)
        if st.session_state.problem_type == 'regression':
            results_df = results_df.sort_values(by='R2', ascending=False)
            metric_col = 'R2'
        else:
            results_df = results_df.sort_values(by='Accuracy', ascending=False)
            metric_col = 'Accuracy'
            
        st.dataframe(results_df.style.highlight_max(axis=0, subset=[metric_col]))
        
        st.markdown(f"### Best Model: **{st.session_state.best_model_name}** 🏆")
        
        # Charts
        fig = px.bar(results_df, x='Model', y=metric_col, title=f"Model Comparison ({metric_col})", 
                     color=metric_col, color_continuous_scale="Viridis")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance
        st.subheader("Feature Importance")
        fi_df = get_feature_importance(st.session_state.best_model, st.session_state.pipeline.get_feature_names_out())
        if fi_df is not None:
            fig2 = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title="Top 15 Features")
            fig2.update_layout(template="plotly_dark", yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")
            
        # Actual vs Predicted
        st.subheader("Actual vs Predicted")
        best_preds = st.session_state.results[st.session_state.best_model_name]['y_pred']
        
        if st.session_state.problem_type == 'regression':
            fig3 = px.scatter(x=st.session_state.y_test, y=best_preds, labels={'x': 'Actual', 'y': 'Predicted'}, 
                             title="Actual vs Predicted Values")
            fig3.add_shape(type="line", x0=min(st.session_state.y_test), y0=min(st.session_state.y_test), 
                           x1=max(st.session_state.y_test), y1=max(st.session_state.y_test), line=dict(color="red", dash="dash"))
            fig3.update_layout(template="plotly_dark")
            st.plotly_chart(fig3, use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Please train models first.")

elif step == "7. Predict":
    if st.session_state.best_model is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Make Predictions")
        
        tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
        
        with tab1:
            st.write("Enter values for features:")
            input_dict = {}
            cols = st.columns(3)
            for i, feature in enumerate(st.session_state.selected_features):
                # get some representative value
                example_val = st.session_state.df[feature].iloc[0]
                with cols[i % 3]:
                    if pd.api.types.is_numeric_dtype(st.session_state.df[feature]):
                        input_dict[feature] = st.number_input(f"{feature}", value=float(example_val))
                    else:
                        unique_vals = st.session_state.df[feature].dropna().unique().tolist()
                        input_dict[feature] = st.selectbox(f"{feature}", options=unique_vals)
                        
            if st.button("Predict"):
                pred = predict_single(st.session_state.best_model, st.session_state.pipeline, input_dict)
                st.markdown(f'<div class="metric-card"><h3>Prediction</h3><h2>{pred}</h2></div>', unsafe_allow_html=True)
                
                conf = get_prediction_confidence(st.session_state.best_model, st.session_state.pipeline.transform(pd.DataFrame([input_dict])))
                if conf is not None:
                    st.info(f"Confidence: {conf[0]:.2%}")
                    
        with tab2:
            st.write("Upload a CSV file for batch predictions.")
            batch_file = st.file_uploader("Upload Data", type=['csv', 'xlsx'])
            if batch_file:
                batch_df = load_data(batch_file)
                if st.button("Run Batch Prediction"):
                    preds = predict_batch(st.session_state.best_model, st.session_state.pipeline, batch_df)
                    batch_df['Prediction'] = preds
                    st.dataframe(batch_df)
                    
                    csv = batch_df.to_csv(index=False)
                    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
                    
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Please train models first to make predictions.")
