import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pickle
from io import BytesIO
import shap

# Enhanced Data Loading
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("SCM_Dataset_Updated_with_Green_Logistics.csv")
        
        # Critical cleaning
        df = df.drop_duplicates()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # If date exists
        
        # Convert percentages (handle '95%' → 0.95)
        pct_cols = [c for c in df.columns if '(%)' in c]
        for col in pct_cols:
            df[col] = df[col].astype(str).str.replace('%','').astype(float) / 100
            
        # Calculate derived metrics
        df['Efficiency_Ratio'] = df['Order_Fulfillment_Rate_(%)'] / df['Lead_Time_(days)']
        df['Sustainability_Score'] = (
            df['Recycling_Rate_(%)'] * 0.4 + 
            df['Use_of_Renewable_Energy_(%)'] * 0.6
        )
        
        # Normalize key metrics for comparison
        scaler = MinMaxScaler()
        score_cols = ['Operational_Efficiency_Score', 'Sustainability_Score']
        df[score_cols] = scaler.fit_transform(df[score_cols])
        
        return df
    
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()
st.markdown("""
<style>
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.2rem;
        font-weight: bold;
    }
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)
# New Overview Page
if page == "Overview":
    st.title("Supply Chain Performance Dashboard")
    
    df = load_data()
    if df.empty:
        st.stop()
    
    # Key Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Efficiency Score", f"{df['Operational_Efficiency_Score'].mean():.2f}")
    with col2:
        st.metric("Avg Lead Time", f"{df['Lead_Time_(days)'].mean():.1f} days")
    with col3:
        st.metric("Top Sustainability", df.loc[df['Sustainability_Score'].idxmax()]['Company_Name'])
    
    # Performance Matrix
    st.subheader("Efficiency vs Sustainability")
    fig = px.scatter(df, 
                    x='Operational_Efficiency_Score',
                    y='Sustainability_Score',
                    color='SCM_Type',
                    hover_name='Company_Name',
                    size='Revenue_Growth_Rate_out_of_(15)',
                    labels={
                        'Operational_Efficiency_Score': 'Efficiency (0-1)',
                        'Sustainability_Score': 'Sustainability (0-1)'
                    })
    st.plotly_chart(fig, use_container_width=True)
    
    # Gap Analysis
    st.subheader("Performance Gaps")
    benchmark = df.quantile(0.75)  # Top 25% as benchmark
    gap_df = pd.DataFrame({
        'Metric': ['Lead Time', 'Fulfillment Rate', 'Carbon Emissions'],
        'Your Avg': [
            df['Lead_Time_(days)'].mean(),
            df['Order_Fulfillment_Rate_(%)'].mean(),
            df['Carbon_Emissions_(kg_CO2e)'].mean()
        ],
        'Top 25% Benchmark': [
            benchmark['Lead_Time_(days)'],
            benchmark['Order_Fulfillment_Rate_(%)'],
            benchmark['Carbon_Emissions_(kg_CO2e)']
        ]
    })
    st.dataframe(gap_df, hide_index=True)
elif page == "Predictive Modeling":
    st.title("Smart Supply Chain Predictions")
    
    # Load preprocessed data
    df = load_data() 
    if df.empty:
        st.stop()
    
    # Model Configuration
    st.sidebar.header("Model Settings")
    model_choice = st.sidebar.selectbox(
        "Prediction Task",
        options=[
            "Operational Efficiency",
            "Carbon Emissions",
            "Lead Time Reduction"
        ]
    )
    
    # Dynamic target selection
    target_map = {
        "Operational Efficiency": "Operational_Efficiency_Score",
        "Carbon Emissions": "Carbon_Emissions_(kg_CO2e)",
        "Lead Time Reduction": "Lead_Time_(days)"
    }
    target = target_map[model_choice]
    
    # Smart feature selection
    feature_presets = {
        "Operational Efficiency": [
            'Order_Fulfillment_Rate_(%)',
            'Inventory_Accuracy_(%)',
            'Supplier_Lead_Time_Variability_(days)',
            'Transportation_Cost_Efficiency_(%)'
        ],
        "Carbon Emissions": [
            'Energy_Consumption_(MWh)',
            'Use_of_Renewable_Energy_(%)',
            'Green_Packaging_Usage_(%)',
            'Total_Implementation_Cost'
        ],
        "Lead Time Reduction": [
            'Supplier_Count',
            'Inventory_Turnover_Ratio',
            'Supply_Chain_Complexity_Index',
            'SCM_Type_Encoded'  # Will create dynamically
        ]
    }
    
    # Prepare data
    model_df = df.copy()
    
    # Encode categorical SCM types if needed
    if 'SCM_Type_Encoded' not in model_df.columns:
        model_df['SCM_Type_Encoded'] = pd.factorize(model_df['SCM_Type'])[0]
    
    # Feature selection UI
    with st.expander("Advanced Feature Selection"):
        default_features = [f for f in feature_presets[model_choice] if f in model_df.columns]
        features = st.multiselect(
            "Select predictors",
            options=[f for f in model_df.columns 
                    if f not in [target, 'Company_Name', 'Date']],
            default=default_features
        )
    
    if not features:
        st.warning("Please select at least one feature")
        st.stop()
    
    # Train-test split
    X = model_df[features]
    y = model_df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Model training - simplified interface
    if st.sidebar.checkbox("Show Model Options"):
        n_estimators = st.sidebar.slider("Number of trees", 50, 200, 100)
        max_depth = st.sidebar.slider("Max depth", 2, 10, 4)
    else:
        n_estimators = 100
        max_depth = 4
    
    # Model dictionary with tuned defaults
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        ),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=42
        )
    }
    
    # Model training and evaluation
    results = []
    feature_importances = {}
    
    for name, model in models.items():
        with st.spinner(f"Training {name}..."):
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results.append({
                    "Model": name,
                    "MAE": mae,
                    "R²": r2
                })
                
                # Store feature importances
                if hasattr(model, 'feature_importances_'):
                    feature_importances[name] = {
                        "features": features,
                        "importance": model.feature_importances_
                    }
                    
            except Exception as e:
                st.error(f"{name} failed: {str(e)}")
    
    # Results display
    st.subheader("Model Performance")
    results_df = pd.DataFrame(results).sort_values("R²", ascending=False)
    st.dataframe(
        results_df.style.format({
            "MAE": "{:.2f}",
            "R²": "{:.2f}"
        }),
        hide_index=True
    )
    
    # Feature importance visualization
    if feature_importances:
        st.subheader("Key Predictive Factors")
        selected_model = st.selectbox(
            "View feature importance for:",
            list(feature_importances.keys())
        )
        
        fi_data = feature_importances[selected_model]
        fi_df = pd.DataFrame({
            "Feature": fi_data["features"],
            "Importance": fi_data["importance"]
        }).sort_values("Importance", ascending=False)
        
        fig = px.bar(
            fi_df.head(10),
            x="Importance",
            y="Feature",
            orientation='h',
            title=f"{selected_model} - Top Features"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show top feature impact
        top_feature = fi_df.iloc[0]
        st.info(
            f"**{top_feature['Feature']}** has the highest impact "
            f"({top_feature['Importance']*100:.1f}% of model decisions)"
        )
    
    # Best model insights
    if not results_df.empty:
        best_model = results_df.iloc[0]
        st.success(
            f"**Recommended Model:** {best_model['Model']}\n\n"
            f"- **Accuracy:** R² = {best_model['R²']:.2f}\n"
            f"- **Avg Error:** ±{best_model['MAE']:.2f} units\n"
            f"- **Best for:** {model_choice} prediction"
        )
# Add to imports at the top
import shap
from io import BytesIO
import pickle

# Add this right after the model training in Part 2 (before st.success())

# -------------------------------------
# Recommendation Engine
# -------------------------------------
st.markdown("---")
st.subheader("📈 Actionable Recommendations")

if not results_df.empty and feature_importances:
    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]
    fi_data = feature_importances[best_model_name]
    
    # Generate recommendations based on model type
    if model_choice == "Operational Efficiency":
        rec_feature = fi_data["features"][np.argmax(fi_data["importance"])]
        rec_impact = fi_data["importance"].max() * 100
        
        st.markdown(f"""
        ### 🎯 Focus Area: **{rec_feature.replace('_', ' ').replace('(%)','')}**
        - Contributes **{rec_impact:.1f}%** to efficiency scores
        - **10% improvement** could boost efficiency by **{(rec_impact/10):.1f} points**
        """)
        
        # What-if simulator
        with st.expander("🚀 Improvement Simulator"):
            current_val = st.number_input(
                f"Current {rec_feature}",
                value=float(X[rec_feature].mean()),
                step=0.1
            )
            target_val = st.number_input(
                "Target improvement (%)",
                min_value=1.0,
                max_value=100.0,
                value=10.0,
                step=1.0
            )
            
            # Create sample input for prediction
            sample = X_test.mean().to_dict()
            sample[rec_feature] = current_val * (1 + target_val/100)
            
            # Predict new efficiency
            current_eff = best_model.predict([list(sample.values())])[0]
            new_eff = best_model.predict([list(sample.values())])[0]
            
            st.metric(
                "Predicted Efficiency Change",
                f"{new_eff - current_eff:.2f} points",
                delta=f"{target_val}% {rec_feature} improvement"
            )

    elif model_choice == "Carbon Emissions":
        top_3 = np.argsort(fi_data["importance"])[-3:][::-1]
        st.markdown("""
        ### 🌱 Sustainability Levers
        Focus on these for maximum emissions reduction:
        """)
        
        for i, idx in enumerate(top_3):
            feature = fi_data["features"][idx]
            impact = fi_data["importance"][idx] * 100
            
            with st.container(border=True):
                st.markdown(f"""
                **{i+1}. {feature.replace('_', ' ')}**  
                ▸ Impact: {impact:.1f}% of emissions  
                ▸ Typical range: {X[feature].min():.1f} to {X[feature].max():.1f}  
                ▸ **Action:** {'Increase' if 'Renewable' in feature else 'Reduce'} this metric
                """)

    # Model explanation with SHAP
    st.markdown("---")
    st.subheader("🔍 Why These Recommendations?")
    with st.spinner("Generating explanations..."):
        try:
            explainer = shap.Explainer(best_model, X_train)
            shap_values = explainer(X_test.iloc[:50])  # Sample for speed
            
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_test.iloc[:50], plot_type="bar")
            st.pyplot(fig)
            
            st.caption("""
            How each feature contributes to predictions. Longer bars = stronger impact.
            Red/blue shows high/low values of each feature.
            """)
        except Exception as e:
            st.warning(f"Couldn't generate explanations: {str(e)}")

    # Model deployment
    st.markdown("---")
    st.subheader("📤 Export Model")
    if st.button("Save Best Model"):
        with BytesIO() as buffer:
            pickle.dump(best_model, buffer)
            st.download_button(
                label="Download Model",
                data=buffer.getvalue(),
                file_name=f"{model_choice.replace(' ', '_')}_model.pkl",
                mime="application/octet-stream"
            )
        st.toast("Model saved successfully!", icon="✅")
