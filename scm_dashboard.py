# Add these imports at the top
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Enhanced data loading with feature engineering
@st.cache_data
def load_data():
    df = pd.read_csv("SCM_Dataset_Updated_with_Green_Logistics.csv")
    
    # Clean customer satisfaction (handle % and NA)
    if 'Customer_Satisfaction_(%)' in df.columns:
        df['Customer_Satisfaction'] = (
            df['Customer_Satisfaction_(%)']
            .astype(str).str.replace('%','')
            .astype(float) / 100
        ).fillna(df['Customer_Satisfaction_(%)'].median())
    
    # Create enhanced features
    df['Lead_Time_Squared'] = df['Lead_Time_(days)'] ** 2
    df['Inventory_Ratio'] = (
        df['Inventory_Turnover_Ratio'] / 
        (df['Supplier_Count'] + 1)  # Avoid division by zero
    )
    df['SCM_Type_Encoded'] = pd.factorize(df['SCM_Type'])[0]
    
    return df

# Diagnostic visualization
def show_diagnostics(X, y):
    st.subheader("🧪 Data Diagnostics")
    
    tab1, tab2, tab3 = st.tabs(["Relationships", "Multicollinearity", "Residuals"])
    
    with tab1:
        fig = px.scatter_matrix(
            pd.concat([X.iloc[:,:3], y], axis=1),
            dimensions=X.columns[:3].tolist() + [y.name],
            title="Pairwise Relationships"
        )
        st.plotly_chart(fig)
    
    with tab2:
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(len(X.columns))]
        st.dataframe(vif_data.style.highlight_between(
            subset="VIF", left=5, right=10, color="orange"
        ).highlight_between(
            subset="VIF", left=10, color="red"
        ))
    
    with tab3:
        # Dummy model for initial residual check
        pipe = make_pipeline(StandardScaler(), LinearRegression())
        pipe.fit(X, y)
        y_pred = pipe.predict(X)
        residuals = y - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.scatter(y_pred, residuals)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_title("Residual Plot")
        sns.histplot(residuals, kde=True, ax=ax2)
        ax2.set_title("Residual Distribution")
        st.pyplot(fig)

# Main prediction function
def predict_customer_satisfaction():
    st.title("Customer Satisfaction Prediction")
    df = load_data()
    
    if 'Customer_Satisfaction' not in df.columns:
        st.error("Customer Satisfaction data not found")
        return
    
    # Feature selection
    st.sidebar.header("Feature Selection")
    base_features = [
        'Lead_Time_(days)', 'Lead_Time_Squared',
        'Inventory_Turnover_Ratio', 'Inventory_Ratio',
        'Order_Fulfillment_Rate_(%)', 'SCM_Type_Encoded'
    ]
    available_features = [f for f in base_features if f in df.columns]
    selected_features = st.sidebar.multiselect(
        "Select features",
        options=[f for f in df.columns 
                if f not in ['Customer_Satisfaction', 'Company_Name']],
        default=available_features
    )
    
    if not selected_features:
        st.warning("Please select at least one feature")
        return
    
    # Prepare data
    X = df[selected_features]
    y = df['Customer_Satisfaction']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Show diagnostics
    if st.sidebar.checkbox("Show Diagnostics"):
        show_diagnostics(X_train, y_train)
    
    # Model selection
    st.sidebar.header("Model Selection")
    models = {
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Polynomial Regression": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2),
            LinearRegression()
        ),
        "Ridge Regression": make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=[0.1, 1.0, 10.0])
        ),
        "Support Vector Regression": make_pipeline(
            StandardScaler(),
            SVR(kernel='rbf', C=1.0)
        )
    }
    
    # Model training
    results = []
    for name, model in models.items():
        with st.spinner(f"Training {name}..."):
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                results.append({
                    "Model": name,
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "R²": r2_score(y_test, y_pred),
                    "Features": ", ".join(selected_features[:3]) + ("..." if len(selected_features)>3 else "")
                })
            except Exception as e:
                st.error(f"{name} failed: {str(e)}")
    
    # Results display
    st.subheader("Model Performance")
    results_df = pd.DataFrame(results).sort_values("R²", ascending=False)
    st.dataframe(
        results_df.style.format({
            "RMSE": "{:.3f}",
            "R²": "{:.3f}"
        }).background_gradient(cmap='Blues', subset=["R²"]),
        hide_index=True,
        use_container_width=True
    )
    
    # Best model explanation
    if not results_df.empty:
        best_model_name = results_df.iloc[0]["Model"]
        st.success(f"**Best Model:** {best_model_name} (R² = {results_df.iloc[0]['R²']:.3f})")
        
        # Feature importance
        if hasattr(models[best_model_name].steps[-1][1], 'feature_importances_') if isinstance(models[best_model_name], Pipeline) else hasattr(models[best_model_name], 'feature_importances_'):
            st.subheader("Feature Importance")
            
            if isinstance(models[best_model_name], Pipeline):
                model = models[best_model_name].steps[-1][1]
                features = selected_features
            else:
                model = models[best_model_name]
                features = selected_features
            
            try:
                importances = model.feature_importances_
                fi_df = pd.DataFrame({
                    "Feature": features,
                    "Importance": importances
                }).sort_values("Importance", ascending=False)
                
                fig = px.bar(
                    fi_df,
                    x="Importance",
                    y="Feature",
                    orientation='h',
                    title=f"{best_model_name} Feature Importance"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendation based on top feature
                top_feature = fi_df.iloc[0]
                st.info(
                    f"**Key Driver:** {top_feature['Feature']} accounts for "
                    f"{top_feature['Importance']*100:.1f}% of satisfaction predictions"
                )
            except Exception as e:
                st.warning(f"Couldn't get feature importance: {str(e)}")

# Run the app
if __name__ == "__main__":
    predict_customer_satisfaction()
    
