import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
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
from sklearn.ensemble import GradientBoostingRegressor
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
    st.markdown("<h2 class='sub-header'>Predictive Model Comparison</h2>", unsafe_allow_html=True)
    
    st.write("""
    Compare how different models perform at predicting key supply chain metrics.
    """)
    
    try:
        # Target selection
        available_targets = [
            'Revenue_Growth_Rate_out_of_(15)',
            'Operational_Efficiency_Score', 
            'Customer_Satisfaction_(%)',
            'Lead_Time_(days)'
        ]
        # Only show targets that exist in the dataframe
        valid_targets = [t for t in available_targets if t in df.columns]
        
        if not valid_targets:
            st.warning("No valid target variables found in the dataset")
            st.stop()
            
        target_var = st.selectbox("Select target variable to predict:", valid_targets)
        
        # Feature selection - auto-select relevant numeric features
        numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        feature_options = [f for f in numeric_features if f != target_var and f in df.columns]
        
        if not feature_options:
            st.warning("No valid features found for prediction")
            st.stop()
            
        selected_features = st.multiselect(
            "Select features to use:", 
            feature_options,
            default=feature_options[:min(5, len(feature_options))]
        )
        
        if not selected_features:
            st.warning("Please select at least one feature")
            st.stop()

        # Prepare data
        model_df = df[[target_var] + selected_features].dropna()
        
        if len(model_df) < 20:
            st.warning(f"Only {len(model_df)} samples available after cleaning - too few for reliable modeling")
            st.stop()
            
        X = model_df[selected_features]
        y = model_df[target_var]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Model selection - simplified with constrained parameters
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=3, random_state=42),
            "Random Forest": RandomForestRegressor(
                n_estimators=50, 
                max_depth=3, 
                random_state=42
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=50, 
                max_depth=3, 
                random_state=42
            )
        }
        
        # Evaluate models
        results = []
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                results.append({
                    "Model": name,
                    "RMSE": f"{rmse:.2f}",
                    "R²": f"{r2:.2f}",
                    "Features": ", ".join(selected_features[:3]) + ("..." if len(selected_features)>3 else "")
                })
            except Exception as e:
                st.error(f"Error with {name}: {str(e)}")
                continue
        
        if not results:
            st.error("No models could be successfully trained")
            st.stop()
            
        # Display results
        st.markdown("### Model Performance")
        results_df = pd.DataFrame(results)
        st.dataframe(
            results_df.sort_values("R²", ascending=False),
            hide_index=True
        )
        
        # Best model insights
        best_row = results_df.iloc[0]
        st.markdown(f"""
        **Best Model:** `{best_row['Model']}`  
        - **R²:** {best_row['R²']} (1.0 is perfect)  
        - **RMSE:** {best_row['RMSE']} (lower is better)
        """)
        
        # Feature importance for tree models
        st.markdown("### Feature Importance")
        tree_models = {
            "Decision Tree": models["Decision Tree"],
            "Random Forest": models["Random Forest"],
            "Gradient Boosting": models["Gradient Boosting"]
        }
        
        tabs = st.tabs(list(tree_models.keys()))
        for tab, (name, model) in zip(tabs, tree_models.items()):
            with tab:
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        importance_df,
                        x='Feature',
                        y='Importance',
                        title=f'{name} Feature Importance'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"{name} doesn't provide feature importances")
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.stop()
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
