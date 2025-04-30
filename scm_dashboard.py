import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, VotingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from math import sqrt
from wordcloud import WordCloud
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Page configuration
st.set_page_config(
    page_title="Supply Chain Management Analytics",
    page_icon="📊",
    layout="wide"
)

# Add CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0D47A1;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 0.15rem 0.3rem rgba(0,0,0,0.1);
    }
    .insight-text {
        font-style: italic;
        color: #555;
        padding: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown("<h1 class='main-header'>Supply Chain Management Analytics Dashboard</h1>", unsafe_allow_html=True)
st.markdown("""
This dashboard provides comprehensive insights into supply chain management practices,
sustainability metrics, and operational efficiency across various companies.
""")

# Load data
@st.cache_data
def load_data():
    file_paths = ["paste.txt", "SCM_Dataset_Updated_with_Green_Logistics.csv"]
    
    for path in file_paths:
        try:
            df = pd.read_csv(path, sep='\t' if path == "paste.txt" else ',', encoding='utf-8')
            st.sidebar.success(f"Successfully loaded: {path}")
            break
        except Exception as e:
            st.sidebar.warning(f"Failed to load {path}: {e}")
            df = None

    # Last resort: let user manually provide path
    if df is None:
        user_path = st.sidebar.text_input("Provide path to your CSV file:")
        if user_path:
            try:
                df = pd.read_csv(user_path, encoding='utf-8')
                st.sidebar.success(f"Successfully loaded: {user_path}")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")
                return pd.DataFrame()
        else:
            return pd.DataFrame()

    # Handle missing values
    df.fillna('Not Specified', inplace=True)

    # Map SCM Practices to Type
    type_mapping = {
        'Agile SCM': 'Strategic',
        'Lean Manufacturing': 'Operational',
        'Cross-Docking': 'Logistics',
        'Sustainable SCM': 'Green',
        'Demand-Driven SCM': 'Customer-centric',
        'Vendor Managed Inventory': 'Tactical',
        'Efficient Consumer Response': 'Customer-centric',
        'Just-In-Time': 'Operational',
        'Six Sigma': 'Operational',
        'CPFR': 'Strategic',
        'Green Logistics': 'Green'
    }
    if 'SCM Practices' in df.columns:
        df['SCM_Type'] = df['SCM Practices'].map(type_mapping)

    # Clean column names
    df.columns = df.columns.str.strip().str.replace(' ', '_')

    # Convert numeric columns
    numeric_cols = [
        'Supplier_Count', 'Inventory_Turnover_Ratio', 'Lead_Time_(days)',
        'Order_Fulfillment_Rate_(%)', 'Customer_Satisfaction_(%)',
        'Environmental_Impact_Score', 'Supplier_Lead_Time_Variability_(days)',
        'Inventory_Accuracy_(%)', 'Transportation_Cost_Efficiency_(%)',
        'Supply_Chain_Complexity_Index', 'Operational_Efficiency_Score',
        'Revenue_Growth_Rate_out_of_(15)', 'Supply_Chain_Risk_(%)',
        'Supply_Chain_Resilience_Score', 'Supplier_Relationship_Score',
        'Total_Implementation_Cost', 'Carbon_Emissions_(kg_CO2e)',
        'Recycling_Rate_(%)', 'Energy_Consumption_(MWh)', 
        'Use_of_Renewable_Energy_(%)', 'Green_Packaging_Usage_(%)'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.replace('%', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Clean and convert COGS
    if 'Cost_of_Goods_Sold_(COGS)' in df.columns:
        df['COGS_Value'] = (
          df['Cost_of_Goods_Sold_(COGS)']
          .astype(str)
          .str.replace('$', '', regex=False)
          .str.replace(',', '', regex=False)
          .str.replace('B', '', regex=False)
    )
    # Convert to numeric AFTER all string cleaning
        df['COGS_Value'] = pd.to_numeric(df['COGS_Value'], errors='coerce') * 1e9

    return df

# Load and display the data
df = load_data()
if df.empty:
    st.error("Failed to load dataset. Please check the file path and format.")
    st.stop()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", 
    ["Overview", "SCM Practices Analysis", "Sustainability Metrics", "Operational Efficiency", "Predictive Modeling", "About"])

# Overview page
if page == "Overview":
    st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)
    
    # Display dataset info
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"*Total Companies:* {df['Company_Name'].nunique()}")
        st.write(f"*Total SCM Practices:* {df['SCM_Practices'].nunique()}")
        st.write(f"*Average Lead Time:* {df['Lead_Time_(days)'].mean():.2f} days")
    
    with col2:
        st.write(f"*Average Customer Satisfaction:* {df['Customer_Satisfaction_(%)'].mean():.2f}%")
        st.write(f"*Average Inventory Accuracy:* {df['Inventory_Accuracy_(%)'].mean():.2f}%")
        st.write(f"*Average Order Fulfillment Rate:* {df['Order_Fulfillment_Rate_(%)'].mean():.2f}%")
    
    # Show dataset
    with st.expander("View Dataset", expanded=False):
        st.dataframe(df)
    
    # Key metrics visualization
    st.markdown("<h2 class='sub-header'>Key Performance Indicators</h2>", unsafe_allow_html=True)
    
    # KPI metrics in cards
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Avg. Operational Efficiency", f"{df['Operational_Efficiency_Score'].mean():.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with metric_cols[1]:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Avg. Revenue Growth Rate", f"{df['Revenue_Growth_Rate_out_of_(15)'].mean():.2f}/15")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with metric_cols[2]:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Avg. Supply Chain Risk", f"{df['Supply_Chain_Risk_(%)'].mean():.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with metric_cols[3]:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Avg. Environmental Impact", f"{df['Environmental_Impact_Score'].mean():.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Top performing companies
    st.markdown("<h2 class='sub-header'>Top Performing Companies</h2>", unsafe_allow_html=True)
    
    metric_option = st.selectbox(
        "Select performance metric:",
        ["Operational_Efficiency_Score", "Revenue_Growth_Rate_out_of_(15)", 
         "Customer_Satisfaction_(%)", "Environmental_Impact_Score"]
    )
    
    top_companies = df.sort_values(by=metric_option, ascending=False).head(10)
    
    fig = px.bar(
        top_companies, 
        x='Company_Name', 
        y=metric_option,
        color='SCM_Type',
        title=f'Top 10 Companies by {metric_option}',
        labels={'Company_Name': 'Company', metric_option: metric_option.replace('_', ' ')},
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution of key metrics
    st.markdown("<h2 class='sub-header'>Distribution of Key Metrics</h2>", unsafe_allow_html=True)
    
    dist_col1, dist_col2 = st.columns(2)
    
    with dist_col1:
        fig = px.histogram(
            df, 
            x="Operational_Efficiency_Score",
            nbins=20,
            color_discrete_sequence=['skyblue'],
            labels={"Operational_Efficiency_Score": "Operational Efficiency Score"},
            title="Distribution of Operational Efficiency Scores"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with dist_col2:
        fig = px.histogram(
            df, 
            x="Revenue_Growth_Rate_out_of_(15)",
            nbins=20,
            color_discrete_sequence=['lightgreen'],
            labels={"Revenue_Growth_Rate_out_of_(15)": "Revenue Growth Rate (out of 15)"},
            title="Distribution of Revenue Growth Rates"
        )
        st.plotly_chart(fig, use_container_width=True)

# SCM Practices Analysis page
elif page == "SCM Practices Analysis":
    st.markdown("<h2 class='sub-header'>SCM Practices Analysis</h2>", unsafe_allow_html=True)
    
    # SCM Types Distribution
    st.markdown("<h3>Distribution of SCM Types</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        scm_counts = df['SCM_Type'].value_counts().reset_index()
        scm_counts.columns = ['SCM Type', 'Count']
        
        fig = px.pie(
            scm_counts, 
            names='SCM Type', 
            values='Count',
            title='Distribution of SCM Types',
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<div class='insight-text'>", unsafe_allow_html=True)
        st.write("""
        The pie chart shows the distribution of various Supply Chain Management types across companies. 
        Strategic approaches like Agile SCM are prevalent, highlighting the industry's focus on adaptability. 
        Operational approaches like Lean Manufacturing follow closely, emphasizing efficiency and waste reduction.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # SCM performance comparison
    st.markdown("<h3>SCM Type Performance Comparison</h3>", unsafe_allow_html=True)
    
    # Performance metrics by SCM Type
    performance_metric = st.selectbox(
        "Select performance metric to compare across SCM Types:",
        ["Revenue_Growth_Rate_out_of_(15)", "Operational_Efficiency_Score", 
         "Customer_Satisfaction_(%)", "Lead_Time_(days)", "Supply_Chain_Risk_(%)"]
    )
    
    avg_by_scm = df.groupby('SCM_Type')[performance_metric].mean().reset_index()
    
    fig = px.bar(
        avg_by_scm,
        x='SCM_Type',
        y=performance_metric,
        color='SCM_Type',
        title=f'Average {performance_metric.replace("_", " ")} by SCM Type',
        labels={'SCM_Type': 'SCM Type', performance_metric: performance_metric.replace('_', ' ')},
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Best performing SCM practice for each metric
    st.markdown("<h3>Best Performing SCM Practices</h3>", unsafe_allow_html=True)
    
    metrics = ["Revenue_Growth_Rate_out_of_(15)", "Operational_Efficiency_Score", 
               "Customer_Satisfaction_(%)", "Lead_Time_(days)", "Supply_Chain_Risk_(%)"]
    
    best_practices = {}
    for metric in metrics:
        if metric in ['Lead_Time_(days)', 'Supply_Chain_Risk_(%)']:
            # Lower is better
            best = df.groupby('SCM_Practices')[metric].mean().sort_values().reset_index().iloc[0]
        else:
            # Higher is better
            best = df.groupby('SCM_Practices')[metric].mean().sort_values(ascending=False).reset_index().iloc[0]
        
        best_practices[metric] = {
            'practice': best['SCM_Practices'],
            'value': best[metric]
        }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("Best Practices by Metric")
        for metric, data in best_practices.items():
            st.write(f"{metric.replace('_', ' ')}:** {data['practice']} ({data['value']:.2f})")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Word cloud of SCM Terms
        if 'SCM_Practices' in df.columns:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.subheader("Word Cloud of SCM Practices")
            
            scm_text = ' '.join(term for term in df['SCM_Practices'].dropna().astype(str))
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white', 
                colormap='viridis'
            ).generate(scm_text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

# Sustainability Metrics page
elif page == "Sustainability Metrics":
    st.markdown("<h2 class='sub-header'>Sustainability Analysis</h2>", unsafe_allow_html=True)
    
    st.write("""
    This section analyzes the sustainability metrics across companies,
    including carbon emissions, recycling rates, energy consumption,
    and use of renewable energy.
    """)
    
    # Correlation heatmap of sustainability metrics
    st.markdown("<h3>Correlation Between Sustainability Metrics</h3>", unsafe_allow_html=True)
    
    sustainability_cols = [
        'Carbon_Emissions_(kg_CO2e)', 'Recycling_Rate_(%)', 
        'Energy_Consumption_(MWh)', 'Use_of_Renewable_Energy_(%)', 
        'Green_Packaging_Usage_(%)', 'Environmental_Impact_Score'
    ]
    
    # Check if all columns exist in the dataset
    existing_cols = [col for col in sustainability_cols if col in df.columns]
    
    if len(existing_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[existing_cols].corr(), annot=True, cmap='YlGnBu', ax=ax)
        plt.title('Correlation Between Sustainability Metrics')
        st.pyplot(fig)
        
        st.markdown("<div class='insight-text'>", unsafe_allow_html=True)
        st.write("""
        The heatmap shows correlations between sustainability metrics. 
        Positive correlations suggest metrics that tend to improve together, 
        while negative correlations may indicate trade-offs or different sustainability approaches.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # SCM Type vs Sustainability Metrics
    st.markdown("<h3>Sustainability Metrics by SCM Type</h3>", unsafe_allow_html=True)
    
    sustainability_metric = st.selectbox(
        "Select sustainability metric:",
        existing_cols
    )
    
    fig = px.box(
        df, 
        x='SCM_Type', 
        y=sustainability_metric,
        color='SCM_Type',
        title=f'{sustainability_metric.replace("_", " ")} by SCM Type',
        labels={'SCM_Type': 'SCM Type', sustainability_metric: sustainability_metric.replace('_', ' ')},
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Renewable Energy vs Carbon Emissions
    st.markdown("<h3>Renewable Energy Usage vs Carbon Emissions</h3>", unsafe_allow_html=True)
    
    if (
    'Use_of_Renewable_Energy_(%)' in df.columns and
    'Carbon_Emissions_(kg_CO2e)' in df.columns and
    'Environmental_Impact_Score' in df.columns and
    'SCM_Type' in df.columns and
    'Company_Name' in df.columns
):
    # Drop rows with missing values in the required columns
      plot_df = df[['Use_of_Renewable_Energy_(%)', 'Carbon_Emissions_(kg_CO2e)', 'Environmental_Impact_Score', 'SCM_Type', 'Company_Name']].dropna()

      fig = px.scatter(
        plot_df,
        x='Use_of_Renewable_Energy_(%)',
        y='Carbon_Emissions_(kg_CO2e)',
        color='SCM_Type',
        size='Environmental_Impact_Score',
        hover_name='Company_Name',
        title='Renewable Energy Usage vs Carbon Emissions',
        labels={
            'Use_of_Renewable_Energy_(%)': 'Use of Renewable Energy (%)',
            'Carbon_Emissions_(kg_CO2e)': 'Carbon Emissions (kg CO2e)',
            'SCM_Type': 'SCM Type'
        },
        template='plotly_white',
        size_max=60
      )
      st.plotly_chart(fig, use_container_width=True)

      st.markdown("<div class='insight-text'>", unsafe_allow_html=True)
      st.write("""
      This scatter plot visualizes the relationship between renewable energy usage and carbon emissions.
      Companies with higher renewable energy usage tend to have different carbon emission profiles.
      The size of each point represents the Environmental Impact Score of the company.
      """)
      st.markdown("</div>", unsafe_allow_html=True)

# Top Sustainable Companies
    st.markdown("<h3>Top Sustainable Companies</h3>", unsafe_allow_html=True)

    if 'Environmental_Impact_Score' in df.columns:
        top_sustainable = df.sort_values(by='Environmental_Impact_Score', ascending=False).head(10)
        
        fig = px.bar(
            top_sustainable,
            x='Company_Name',
            y='Environmental_Impact_Score',
            color='SCM_Type',
            title='Top 10 Companies by Environmental Impact Score',
            labels={
                'Company_Name': 'Company',
                'Environmental_Impact_Score': 'Environmental Impact Score',
                'SCM_Type': 'SCM Type'
            },
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

# Operational Efficiency page
elif page == "Operational Efficiency":
    st.markdown("<h2 class='sub-header'>Operational Efficiency Analysis</h2>", unsafe_allow_html=True)
    
    st.write("""
    This section analyzes operational efficiency metrics across companies,
    including lead time, inventory turnover, order fulfillment rate,
    and customer satisfaction.
    """)
    
    # Operational metrics comparison
    st.markdown("<h3>Operational Metrics Comparison</h3>", unsafe_allow_html=True)
    
    operational_cols = [
        'Lead_Time_(days)', 'Inventory_Turnover_Ratio', 
        'Order_Fulfillment_Rate_(%)', 'Customer_Satisfaction_(%)',
        'Inventory_Accuracy_(%)', 'Transportation_Cost_Efficiency_(%)'
    ]
    
    # Check for existing columns
    existing_op_cols = [col for col in operational_cols if col in df.columns]
    
    if existing_op_cols:
        selected_companies = st.multiselect(
            "Select companies to compare:",
            options=df['Company_Name'].unique(),
            default=df['Company_Name'].unique()[:5]
        )
        
        if selected_companies:
            filtered_df = df[df['Company_Name'].isin(selected_companies)]
            
            for col in existing_op_cols:
                fig = px.bar(
                    filtered_df,
                    x='Company_Name',
                    y=col,
                    color='SCM_Type',
                    title=f'{col.replace("_", " ")} by Company',
                    labels={'Company_Name': 'Company', col: col.replace('_', ' ')},
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Relationship between metrics
    st.markdown("<h3>Relationships Between Operational Metrics</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_metric = st.selectbox(
            "Select X-axis metric:",
            existing_op_cols,
            index=0
        )
    
    with col2:
        y_metric = st.selectbox(
            "Select Y-axis metric:",
            existing_op_cols,
            index=1
        )
    
    if x_metric != y_metric:
        fig = px.scatter(
            df,
            x=x_metric,
            y=y_metric,
            color='SCM_Type',
            hover_name='Company_Name',
            title=f'Relationship between {x_metric.replace("", " ")} and {y_metric.replace("", " ")}',
            labels={x_metric: x_metric.replace('', ' '), y_metric: y_metric.replace('', ' ')},
            trendline="ols",
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<div class='insight-text'>", unsafe_allow_html=True)
        st.write(f"""
        This scatter plot shows the relationship between {x_metric.replace('', ' ')} and {y_metric.replace('', ' ')} across companies.
        The trendline indicates the general relationship between these metrics.
        Colors represent different SCM types to identify any patterns by strategy.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Operational Efficiency Score breakdown
    st.markdown("<h3>Operational Efficiency Score Breakdown</h3>", unsafe_allow_html=True)
    
    if 'Operational_Efficiency_Score' in df.columns:
        # Create efficiency categories
        df['Efficiency_Category'] = pd.cut(
            df['Operational_Efficiency_Score'],
            bins=[0, 70, 80, 90, 100],
            labels=['Low', 'Medium', 'High', 'Excellent']
        )
        
        eff_counts = df['Efficiency_Category'].value_counts().reset_index()
        eff_counts.columns = ['Efficiency Category', 'Count']
        
        fig = px.pie(
            eff_counts, 
            names='Efficiency Category', 
            values='Count',
            title='Distribution of Operational Efficiency Categories',
            color_discrete_sequence=px.colors.sequential.RdBu,
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Efficiency score vs Revenue Growth Rate
        if 'Revenue_Growth_Rate_out_of_(15)' in df.columns:
            fig = px.scatter(
                df,
                x='Operational_Efficiency_Score',
                y='Revenue_Growth_Rate_out_of_(15)',
                color='SCM_Type',
                size='Supplier_Count',
                hover_name='Company_Name',
                title='Operational Efficiency vs Revenue Growth',
                labels={
                    'Operational_Efficiency_Score': 'Operational Efficiency Score',
                    'Revenue_Growth_Rate_out_of_(15)': 'Revenue Growth Rate (out of 15)',
                    'SCM_Type': 'SCM Type'
                },
                trendline="ols",
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)

# Predictive Modeling page
# Predictive Modeling page
# Predictive Modeling page# Predictive Modeling page
# In your imports section, add these if not already present:


# Then update your Predictive Modeling section:

elif page == "Predictive Modeling":
    st.markdown("<h2 class='sub-header'>Advanced Predictive Modeling</h2>", unsafe_allow_html=True)
    
    st.write("""
    This section demonstrates robust machine learning approaches for supply chain analytics,
    with improved data preprocessing and model evaluation.
    """)
    
    # Model type selection
    model_type = st.selectbox(
        "Select modeling approach:",
        [
            "Revenue Growth Prediction (XGBoost)",
            "Supply Chain Risk Classification (Random Forest)",
            "Carbon Emissions Prediction (Gradient Boosting)",
            "SCM Type Classification (Ensemble)",
            "Operational Clustering (K-Means)"
        ]
    )
    
    # Common preprocessing function
    def preprocess_data(df, target_col, feature_cols):
        """Handle missing values, scaling, and encoding"""
        df = df.dropna(subset=[target_col])
        
        # Only use features that exist in dataframe
        existing_features = [f for f in feature_cols if f in df.columns]
        X = df[existing_features]
        y = df[target_col]
        
        # Handle missing values in features
        numeric_cols = X.select_dtypes(include=np.number).columns
        cat_cols = X.select_dtypes(exclude=np.number).columns
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, cat_cols)
            ])
        
        return X, y, preprocessor, existing_features
    
    if model_type == "Revenue Growth Prediction (XGBoost)":
        st.markdown("<h3>Revenue Growth Rate Prediction</h3>", unsafe_allow_html=True)
        st.write("Predicts revenue growth using XGBoost with feature importance analysis")
        
        # Prepare data
        feature_cols = [
            'Order_Fulfillment_Rate_(%)', 'Operational_Efficiency_Score',
            'Customer_Satisfaction_(%)', 'Supply_Chain_Resilience_Score',
            'Supplier_Relationship_Score', 'Inventory_Turnover_Ratio',
            'Lead_Time_(days)', 'SCM_Type'
        ]
        
        X, y, preprocessor, existing_features = preprocess_data(
            df, 'Revenue_Growth_Rate_out_of_(15)', feature_cols)
        
        if len(existing_features) > 0:
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            
            # Create pipeline with XGBoost
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', XGBRegressor(random_state=42))
            ])
            
            # Train model
            with st.spinner('Training model...'):
                model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{r2:.3f}", 
                         help="Proportion of variance explained by model (1 is perfect)")
            with col2:
                st.metric("RMSE", f"{rmse:.3f}", 
                         help="Root Mean Squared Error (lower is better)")
            with col3:
                st.metric("MAE", f"{mae:.3f}", 
                         help="Mean Absolute Error (lower is better)")
            
            # Feature importance
            try:
                # For XGBoost, we need to get feature names after preprocessing
                feature_names = []
                for name, trans, cols in preprocessor.transformers_:
                    if name == 'num':
                        feature_names.extend(cols)
                    elif name == 'cat':
                        # Get one-hot encoded feature names
                        ohe = trans.named_steps['encoder']
                        feature_names.extend(ohe.get_feature_names_out(cols))
                
                importances = model.named_steps['regressor'].feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(10)
                
                st.write("Top 10 Feature Importances:")
                fig = px.bar(
                    importance_df,
                    x='Feature',
                    y='Importance',
                    title='Feature Importance for Revenue Growth Prediction',
                    labels={'Feature': 'Feature', 'Importance': 'Importance'},
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display feature importance: {str(e)}")
            
            # Actual vs Predicted plot with trendline
            fig = px.scatter(
                x=y_test, y=y_pred,
                labels={'x': 'Actual', 'y': 'Predicted'},
                title='Actual vs Predicted Revenue Growth',
                trendline="ols",
                trendline_color_override="red"
            )
            fig.update_layout(
                xaxis_title="Actual Revenue Growth Rate",
                yaxis_title="Predicted Revenue Growth Rate"
            )
            fig.add_shape(
                type="line", line=dict(dash='dash'),
                x0=y_test.min(), y0=y_test.min(),
                x1=y_test.max(), y1=y_test.max()
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Residual plot
            residuals = y_test - y_pred
            fig = px.scatter(
                x=y_pred, y=residuals,
                labels={'x': 'Predicted', 'y': 'Residuals'},
                title='Residual Plot',
                trendline="ols",
                trendline_color_override="red"
            )
            fig.add_hline(y=0, line_dash="dash")
            fig.update_layout(
                xaxis_title="Predicted Values",
                yaxis_title="Residuals (Actual - Predicted)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Required features not found in dataset")
    
    elif model_type == "Supply Chain Risk Classification (Random Forest)":
        st.markdown("<h3>Supply Chain Risk Classification</h3>", unsafe_allow_html=True)
        st.write("Classifies companies into risk categories using Random Forest")
        
        # Create risk categories if not exists
        if 'Supply_Chain_Risk_(%)' in df.columns:
            df['Risk_Category'] = pd.cut(
                df['Supply_Chain_Risk_(%)'],
                bins=[0, 25, 50, 75, 100],
                labels=['Low', 'Medium', 'High', 'Critical']
            )
            
            # Prepare data
            feature_cols = [
                'Lead_Time_(days)', 'Supplier_Count', 
                'Inventory_Turnover_Ratio', 'Order_Fulfillment_Rate_(%)',
                'Customer_Satisfaction_(%)', 'SCM_Type'
            ]
            
            X, y, preprocessor, existing_features = preprocess_data(
                df, 'Risk_Category', feature_cols)
            
            if len(existing_features) > 0:
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y)
                
                # Create pipeline with Random Forest
                model = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', RandomForestClassifier(random_state=42))
                ])
                
                # Train model
                with st.spinner('Training model...'):
                    model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                # Display metrics
                st.metric("Accuracy", f"{accuracy:.3f}")
                
                # Confusion matrix
                st.write("Confusion Matrix:")
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=model.classes_,
                    y=model.classes_,
                    text_auto=True,
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Classification report
                st.write("Classification Report:")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.3f}"))
                
                # Feature importance
                try:
                    feature_names = []
                    for name, trans, cols in preprocessor.transformers_:
                        if name == 'num':
                            feature_names.extend(cols)
                        elif name == 'cat':
                            ohe = trans.named_steps['encoder']
                            feature_names.extend(ohe.get_feature_names_out(cols))
                    
                    importances = model.named_steps['classifier'].feature_importances_
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    st.write("Top 10 Feature Importances:")
                    fig = px.bar(
                        importance_df,
                        x='Feature',
                        y='Importance',
                        title='Feature Importance for Risk Classification',
                        labels={'Feature': 'Feature', 'Importance': 'Importance'},
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display feature importance: {str(e)}")
            else:
                st.warning("Required features not found in dataset")
        else:
            st.warning("Supply_Chain_Risk_(%) column not found in dataset")
    
    elif model_type == "Carbon Emissions Prediction (Gradient Boosting)":
        st.markdown("<h3>Carbon Emissions Prediction</h3>", unsafe_allow_html=True)
        st.write("Predicts carbon emissions using Gradient Boosting with early stopping")
        
        if 'Carbon_Emissions_(kg_CO2e)' in df.columns:
            # Prepare data
            feature_cols = [
                'Energy_Consumption_(MWh)', 'Use_of_Renewable_Energy_(%)',
                'Recycling_Rate_(%)', 'Green_Packaging_Usage_(%)',
                'Total_Implementation_Cost', 'SCM_Type'
            ]
            
            X, y, preprocessor, existing_features = preprocess_data(
                df, 'Carbon_Emissions_(kg_CO2e)', feature_cols)
            
            if len(existing_features) > 0:
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42)
                
                # Create pipeline with Gradient Boosting
                model = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', GradientBoostingRegressor(
                        n_estimators=500,
                        learning_rate=0.01,
                        random_state=42,
                        validation_fraction=0.2,
                        n_iter_no_change=10
                    ))
                ])
                
                # Train model with progress bar
                with st.spinner('Training model (this may take a few minutes)...'):
                    model.fit(X_train, y_train)
                    n_estimators = len(model.named_steps['regressor'].estimators_)
                    st.success(f"Training completed with {n_estimators} estimators")
                
                # Evaluate
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{r2:.3f}")
                with col2:
                    st.metric("RMSE", f"{rmse:.3f}")
                with col3:
                    st.metric("MAE", f"{mae:.3f}")
                
                # Learning curve
                st.write("Model Learning Curve:")
                regressor = model.named_steps['regressor']
                train_score = regressor.train_score_
                val_score = regressor.validation_score_
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=np.arange(len(train_score)),
                    y=train_score,
                    name='Training Score'
                ))
                fig.add_trace(go.Scatter(
                    x=np.arange(len(val_score)),
                    y=val_score,
                    name='Validation Score'
                ))
                fig.update_layout(
                    title='Gradient Boosting Learning Curve',
                    xaxis_title='Number of Boosting Iterations',
                    yaxis_title='Score (R²)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance
                try:
                    feature_names = []
                    for name, trans, cols in preprocessor.transformers_:
                        if name == 'num':
                            feature_names.extend(cols)
                        elif name == 'cat':
                            ohe = trans.named_steps['encoder']
                            feature_names.extend(ohe.get_feature_names_out(cols))
                    
                    importances = model.named_steps['regressor'].feature_importances_
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    st.write("Top 10 Feature Importances:")
                    fig = px.bar(
                        importance_df,
                        x='Feature',
                        y='Importance',
                        title='Feature Importance for Carbon Emissions Prediction',
                        labels={'Feature': 'Feature', 'Importance': 'Importance'},
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display feature importance: {str(e)}")
            else:
                st.warning("Required features not found in dataset")
        else:
            st.warning("Carbon_Emissions_(kg_CO2e) column not found in dataset")
    
    elif model_type == "SCM Type Classification (Ensemble)":
        st.markdown("<h3>SCM Type Classification</h3>", unsafe_allow_html=True)
        st.write("Classifies SCM types using Voting Classifier with multiple algorithms")
        
        if 'SCM_Type' in df.columns:
            # Prepare data
            feature_cols = [
                'Supplier_Count', 'Inventory_Turnover_Ratio', 
                'Lead_Time_(days)', 'Order_Fulfillment_Rate_(%)',
                'Customer_Satisfaction_(%)', 'Environmental_Impact_Score',
                'Inventory_Accuracy_(%)', 'Transportation_Cost_Efficiency_(%)'
            ]
            
            X, y, preprocessor, existing_features = preprocess_data(
                df, 'SCM_Type', feature_cols)
            
            if len(existing_features) > 0:
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y)
                
                # Create ensemble model
                estimators = [
                    ('rf', RandomForestClassifier(random_state=42)),
                    ('gb', GradientBoostingClassifier(random_state=42)),
                    ('xgb', XGBClassifier(random_state=42, eval_metric='mlogloss'))
                ]
                
                model = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', VotingClassifier(estimators=estimators, voting='soft'))
                ])
                
                # Train model
                with st.spinner('Training ensemble model (this may take a few minutes)...'):
                    model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                # Display metrics
                st.metric("Accuracy", f"{accuracy:.3f}")
                
                # Confusion matrix
                st.write("Confusion Matrix:")
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=model.classes_,
                    y=model.classes_,
                    text_auto=True,
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Classification report
                st.write("Classification Report:")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.3f}"))
                
                # Feature importance (average from all models)
                try:
                    feature_names = []
                    for name, trans, cols in preprocessor.transformers_:
                        if name == 'num':
                            feature_names.extend(cols)
                        elif name == 'cat':
                            ohe = trans.named_steps['encoder']
                            feature_names.extend(ohe.get_feature_names_out(cols))
                    
                    # Get feature importances from each model
                    importances = []
                    for name, clf in model.named_steps['classifier'].named_estimators_.items():
                        if hasattr(clf, 'feature_importances_'):
                            importances.append(clf.feature_importances_)
                    
                    if importances:
                        avg_importances = np.mean(importances, axis=0)
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': avg_importances
                        }).sort_values('Importance', ascending=False).head(10)
                        
                        st.write("Top 10 Feature Importances (Average):")
                        fig = px.bar(
                            importance_df,
                            x='Feature',
                            y='Importance',
                            title='Feature Importance for SCM Type Classification',
                            labels={'Feature': 'Feature', 'Importance': 'Importance'},
                            template='plotly_white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display feature importance: {str(e)}")
            else:
                st.warning("Required features not found in dataset")
        else:
            st.warning("SCM_Type column not found in dataset")
    
    elif model_type == "Operational Clustering (K-Means)":
        st.markdown("<h3>Operational Performance Clustering</h3>", unsafe_allow_html=True)
        st.write("Groups companies into clusters based on operational metrics using K-Means with silhouette analysis")
        
        # Prepare data
        feature_cols = [
            'Inventory_Accuracy_(%)', 'Transportation_Cost_Efficiency_(%)',
            'Operational_Efficiency_Score', 'Supply_Chain_Risk_(%)',
            'Supplier_Relationship_Score'
        ]
        
        existing_features = [f for f in feature_cols if f in df.columns]
        
        if len(existing_features) > 0:
            # Select only numeric features and drop missing values
            df_cluster = df[existing_features].select_dtypes(include=np.number).dropna()
            
            # Scale data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_cluster)
            
            # Silhouette analysis to determine optimal clusters
            st.write("Silhouette Analysis for Optimal Cluster Count:")
            
            range_n_clusters = range(2, 8)
            silhouette_scores = []
            
            for n_clusters in range_n_clusters:
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = clusterer.fit_predict(X_scaled)
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            
            # Plot silhouette scores
            fig = px.line(
                x=list(range_n_clusters), 
                y=silhouette_scores,
                labels={'x': 'Number of Clusters', 'y': 'Average Silhouette Score'},
                title='Silhouette Analysis for Optimal K'
            )
            fig.update_layout(
                xaxis=dict(tickmode='linear', tick0=2, dtick=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Let user select number of clusters
            optimal_k = np.argmax(silhouette_scores) + 2  # +2 because range starts at 2
            n_clusters = st.slider(
                "Select number of clusters", 
                min_value=2, 
                max_value=7, 
                value=optimal_k
            )
            
            # Fit K-Means with selected clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to dataframe
            df_cluster['Cluster'] = cluster_labels
            
            # Show cluster distribution
            st.write("Cluster Distribution:")
            st.bar_chart(df_cluster['Cluster'].value_counts())
            
            # Show cluster characteristics
            cluster_means = df_cluster.groupby('Cluster').mean()
            st.write("Cluster Characteristics (mean values):")
            st.dataframe(cluster_means.style.background_gradient(cmap='Blues'))
            
            # PCA for visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            # Create plot
            plot_df = pd.DataFrame({
                'PCA1': X_pca[:, 0],
                'PCA2': X_pca[:, 1],
                'Cluster': cluster_labels
            })
            
            fig = px.scatter(
                plot_df,
                x='PCA1',
                y='PCA2',
                color='Cluster',
                title='Cluster Visualization (PCA Reduced)',
                labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'},
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Parallel coordinates plot
            st.write("Parallel Coordinates Plot:")
            parallel_df = df_cluster.copy()
            parallel_df['Cluster'] = parallel_df['Cluster'].astype(str)
            
            fig = px.parallel_coordinates(
                parallel_df,
                color='Cluster',
                labels={col: col.replace('_', ' ') for col in parallel_df.columns},
                title='Parallel Coordinates Plot of Clusters'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Required features not found in dataset")
# About page
elif page == "About":
    st.markdown("<h2 class='sub-header'>About This Dashboard</h2>", unsafe_allow_html=True)
    
    st.write("""
    This Supply Chain Management Analytics Dashboard provides a comprehensive analysis of SCM practices,
    sustainability metrics, and operational efficiency across various companies.
    
    The dashboard is built using Streamlit, a Python library for creating web applications for data science and machine learning.
    
    ### Data Sources
    
    The dataset used in this dashboard contains information about various companies and their supply chain management practices, including:
    
    - SCM Practices (Agile SCM, Lean Manufacturing, etc.)
    - Operational metrics (Lead Time, Inventory Turnover Ratio, etc.)
    - Sustainability metrics (Carbon Emissions, Recycling Rate, etc.)
    - Technology adoption (ERP, AI, Blockchain, etc.)
    
    ### Key Features
    
    - *Overview*: High-level view of key metrics and top-performing companies
    - *SCM Practices Analysis*: In-depth analysis of different SCM practices and their performance
    - *Sustainability Metrics*: Analysis of environmental impact and sustainability practices
    - *Operational Efficiency*: Analysis of operational metrics and their relationships
    - *Predictive Modeling*: Machine learning models to predict operational efficiency and SCM type
    
    ### Technologies Used
    
    - *Streamlit*: For application framework and UI
    - *Pandas*: For data manipulation and analysis
    - *Plotly & Matplotlib*: For data visualization
    - *Scikit-learn*: For predictive modeling
    
    ### Contact
    
    For more information about this dashboard, please contact the development team.
    """)
    
    # Add some statistics about the dataset
    st.markdown("<h3>Dataset Statistics</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Number of Companies", df['Company_Name'].nunique())
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Number of SCM Practices", df['SCM_Practices'].nunique())
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Average Supplier Count", f"{df['Supplier_Count'].mean():.1f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Show sustainability stats
    if all(col in df.columns for col in ['Carbon_Emissions_(kg_CO2e)', 'Recycling_Rate_(%)', 'Use_of_Renewable_Energy_(%)']):
        st.markdown("<h3>Sustainability Statistics</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Avg. Carbon Emissions (kg CO2e)", f"{df['Carbon_Emissions_(kg_CO2e)'].mean():.1f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Avg. Recycling Rate", f"{df['Recycling_Rate_(%)'].mean():.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Avg. Renewable Energy Use", f"{df['Use_of_Renewable_Energy_(%)'].mean():.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("---")
    st.markdown("Note: This dashboard is for demonstration purposes only.")

# Run the app
if __name__ == "__main__":
    # This part is automatically handled by Streamlit when the script is run
    pass
