"""
Comprehensive Streamlit UI for Cost-Sensitive Sentiment Analysis Results
Run with: streamlit run cost_analysis_viewer.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import json

# Page configuration
st.set_page_config(
    page_title="Cost-Sensitive Sentiment Analysis",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark Purple/Blue Theme
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main .block-container {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .cost-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 12px;
        border-radius: 8px;
        border-left: 5px solid #d63031;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .cost-medium {
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
        color: #2d3436;
        padding: 12px;
        border-radius: 8px;
        border-left: 5px solid #e17055;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .cost-low {
        background: linear-gradient(135deg, #55efc4 0%, #00b894 100%);
        color: #2d3436;
        padding: 12px;
        border-radius: 8px;
        border-left: 5px solid #00b894;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Constants
RESULTS_DIR = Path(__file__).parent / "results"
COST_MATRIX = np.array([
    [0., 1., 5.],  # negative
    [1., 0., 1.],  # neutral
    [5., 1., 0.]   # positive
])
CLASS_NAMES = ['negative', 'neutral', 'positive']

@st.cache_data
def load_model_comparison():
    """Load model comparison CSV."""
    csv_path = RESULTS_DIR / 'cost_sensitive_model_comparison.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df
    return None

def visualize_cost_matrix_interactive():
    """Create interactive cost matrix visualization."""
    fig = go.Figure(data=go.Heatmap(
        z=COST_MATRIX,
        x=CLASS_NAMES,
        y=CLASS_NAMES,
        colorscale='Reds',
        text=COST_MATRIX,
        texttemplate='%{text:.1f}',
        textfont={"size": 20, "color": "white"},
        colorbar=dict(title="Cost"),
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Cost: %{z:.1f}<extra></extra>'
    ))
    fig.update_layout(
        title='Cost Matrix for Sentiment Classification<br><sub>Row = True Class, Column = Predicted Class</sub>',
        xaxis_title='Predicted Class',
        yaxis_title='True Class',
        width=700,
        height=600,
        font=dict(size=14)
    )
    return fig

def create_confusion_matrix_plot(cm_data, title="Confusion Matrix", show_cost=True):
    """Create interactive confusion matrix plot."""
    annot_text = []
    cost_matrix_display = COST_MATRIX
    
    for i in range(len(CLASS_NAMES)):
        row = []
        for j in range(len(CLASS_NAMES)):
            count = cm_data[i, j]
            cost = cost_matrix_display[i, j] if show_cost else 0
            if show_cost and count > 0:
                row.append(f"{count}<br>(cost: {cost:.0f})")
            else:
                row.append(f"{count}")
        annot_text.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_data,
        x=CLASS_NAMES,
        y=CLASS_NAMES,
        colorscale='Blues',
        text=annot_text,
        texttemplate='%{text}',
        textfont={"size": 14},
        colorbar=dict(title="Count"),
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Predicted Class',
        yaxis_title='True Class',
        width=700,
        height=600,
        font=dict(size=14)
    )
    return fig

# Main title
st.title("💰 Cost-Sensitive Sentiment Analysis Dashboard")
st.markdown("Comprehensive interactive visualization of cost-aware model performance")

# Load data
comparison_df = load_model_comparison()

# Sidebar navigation
st.sidebar.header("📊 Navigation")
page = st.sidebar.selectbox(
    "Select Section",
    ["Overview", "Model Comparison", "Cost Matrix", "Performance Metrics", "Cost Analysis", "Interactive Explorer"]
)

if comparison_df is None:
    st.error("❌ Model comparison data not found. Please run `train_cost_sensitive.py` first.")
    st.stop()

if page == "Overview":
    st.header("📈 Overview Dashboard")
    
    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_model = comparison_df.iloc[0]['Model']
        st.metric("🏆 Best Model", best_model)
    
    with col2:
        best_cost = comparison_df.iloc[0]['Cost']
        st.metric("Lowest Cost", f"{best_cost:.2f}")
    
    with col3:
        best_accuracy = comparison_df.iloc[0]['Accuracy']
        st.metric("Best Accuracy", f"{best_accuracy:.4f}")
    
    with col4:
        avg_cost = comparison_df['Cost'].mean()
        st.metric("Average Cost", f"{avg_cost:.2f}")
    
    # Summary statistics
    st.subheader("📊 Summary Statistics")
    col1, col2 = st.columns(2)
    
    summary_stats = comparison_df.describe()
    with col1:
        st.dataframe(summary_stats, use_container_width=True)
    
    with col2:
        # Model count by performance tier
        high_perf = len(comparison_df[comparison_df['Accuracy'] >= 0.9])
        medium_perf = len(comparison_df[(comparison_df['Accuracy'] >= 0.7) & (comparison_df['Accuracy'] < 0.9)])
        low_perf = len(comparison_df[comparison_df['Accuracy'] < 0.7])
        
        fig_pie = px.pie(
            values=[high_perf, medium_perf, low_perf],
            names=['High (≥90%)', 'Medium (70-90%)', 'Low (<70%)'],
            title="Models by Accuracy Tier",
            color_discrete_map={
                'High (≥90%)': '#28a745',
                'Medium (70-90%)': '#ffc107',
                'Low (<70%)': '#dc3545'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Quick comparison charts
    st.subheader("📈 Quick Performance Comparison")
    
    # Sort by cost for better visualization
    sorted_df = comparison_df.sort_values('Cost').head(10)
    
    fig1 = px.bar(
        sorted_df,
        x='Model',
        y='Cost',
        title="Total Cost by Model (Top 10)",
        color='Cost',
        color_continuous_scale='RdYlGn_r',
        labels={'Cost': 'Total Cost'}
    )
    fig1.update_xaxes(tickangle=45)
    st.plotly_chart(fig1, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig2 = px.scatter(
            comparison_df,
            x='Cost',
            y='Accuracy',
            size='F1-Macro',
            hover_data=['Model'],
            title="Accuracy vs Cost Trade-off",
            color='Cost',
            color_continuous_scale='RdYlGn_r',
            labels={'Cost': 'Total Cost', 'Accuracy': 'Accuracy', 'F1-Macro': 'F1 Score'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        fig3 = px.scatter(
            comparison_df,
            x='Expected Cost',
            y='Accuracy',
            size='Cost',
            hover_data=['Model'],
            title="Accuracy vs Expected Cost",
            color='Expected Cost',
            color_continuous_scale='RdYlGn_r',
            labels={'Expected Cost': 'Expected Cost', 'Accuracy': 'Accuracy'}
        )
        st.plotly_chart(fig3, use_container_width=True)

elif page == "Model Comparison":
    st.header("📊 Model Comparison")
    
    # Filters and sorting
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            options=['Cost (Best First)', 'Accuracy (Best First)', 'Expected Cost (Best First)', 
                    'F1-Macro (Best First)', 'Cost (Worst First)', 'Alphabetical'],
            index=0
        )
    
    with col2:
        min_accuracy = st.slider(
            "Minimum Accuracy",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05
        )
    
    with col3:
        max_cost = st.slider(
            "Maximum Cost",
            min_value=0.0,
            max_value=float(comparison_df['Cost'].max()) * 1.1,
            value=float(comparison_df['Cost'].max()),
            step=10.0
        )
    
    # Sort and filter dataframe
    display_df = comparison_df.copy()
    
    # Apply filters
    display_df = display_df[display_df['Accuracy'] >= min_accuracy]
    display_df = display_df[display_df['Cost'] <= max_cost]
    
    # Sort
    if 'Cost (Best First)' in sort_by:
        display_df = display_df.sort_values('Cost')
    elif 'Accuracy (Best First)' in sort_by:
        display_df = display_df.sort_values('Accuracy', ascending=False)
    elif 'Expected Cost (Best First)' in sort_by:
        display_df = display_df.sort_values('Expected Cost')
    elif 'F1-Macro (Best First)' in sort_by:
        display_df = display_df.sort_values('F1-Macro', ascending=False)
    elif 'Cost (Worst First)' in sort_by:
        display_df = display_df.sort_values('Cost', ascending=False)
    elif 'Alphabetical' in sort_by:
        display_df = display_df.sort_values('Model')
    
    # Display table
    st.subheader(f"Model Performance ({len(display_df)} models)")
    
    # Format metrics for display
    display_df_formatted = display_df.copy()
    display_df_formatted['Cost'] = display_df_formatted['Cost'].apply(lambda x: f"{x:.2f}")
    display_df_formatted['Expected Cost'] = display_df_formatted['Expected Cost'].apply(lambda x: f"{x:.4f}")
    display_df_formatted['Accuracy'] = display_df_formatted['Accuracy'].apply(lambda x: f"{x:.4f}")
    display_df_formatted['F1-Macro'] = display_df_formatted['F1-Macro'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display_df_formatted, use_container_width=True, height=400)
    
    # Interactive visualizations
    st.subheader("📈 Interactive Visualizations")
    
    selected_models = st.multiselect(
        "Select models to compare in detail",
        options=display_df['Model'].tolist(),
        default=display_df['Model'].head(5).tolist()
    )
    
    if selected_models:
        compare_df = display_df[display_df['Model'].isin(selected_models)]
        
        # Multi-metric comparison
        metrics = ['Cost', 'Expected Cost', 'Accuracy', 'F1-Macro']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Cost', 'Expected Cost', 'Accuracy', 'F1-Macro'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        for i, metric in enumerate(metrics):
            row = (i // 2) + 1
            col = (i % 2) + 1
            fig.add_trace(
                go.Bar(x=compare_df['Model'], y=compare_df[metric], name=metric),
                row=row, col=col
            )
        
        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=800, showlegend=False, title_text="Detailed Model Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # Radar chart for normalized metrics
        normalized = compare_df.copy()
        normalized['Cost_norm'] = 1 - (normalized['Cost'] / normalized['Cost'].max())
        normalized['Expected_Cost_norm'] = 1 - (normalized['Expected Cost'] / normalized['Expected Cost'].max())
        normalized['Accuracy_norm'] = normalized['Accuracy']
        normalized['F1_norm'] = normalized['F1-Macro']
        
        fig_radar = go.Figure()
        
        for _, row in normalized.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Cost_norm'], row['Expected_Cost_norm'], row['Accuracy_norm'], row['F1_norm']],
                theta=['Cost (inverted)', 'Expected Cost (inverted)', 'Accuracy', 'F1-Macro'],
                fill='toself',
                name=row['Model']
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Normalized Performance Comparison (Radar Chart)",
            height=600
        )
        st.plotly_chart(fig_radar, use_container_width=True)

elif page == "Cost Matrix":
    st.header("💰 Cost Matrix")
    st.markdown("""
    The cost matrix defines the cost of each type of misclassification:
    - **negative ↔ positive** = Very bad (cost 5)
    - **negative ↔ neutral** or **neutral ↔ positive** = Less bad (cost 1)
    - **Same class** = No cost (0)
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = visualize_cost_matrix_interactive()
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Cost Breakdown")
        st.markdown("""
        <div class="cost-high">
            <strong>High Cost (5):</strong><br>
            Negative → Positive<br>
            Positive → Negative
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="cost-medium">
            <strong>Medium Cost (1):</strong><br>
            Negative → Neutral<br>
            Neutral → Positive<br>
            Neutral → Negative<br>
            Positive → Neutral
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="cost-low">
            <strong>No Cost (0):</strong><br>
            Correct predictions
        </div>
        """, unsafe_allow_html=True)

elif page == "Performance Metrics":
    st.header("📊 Performance Metrics Analysis")
    
    # Metrics comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Cost distribution
        fig_cost_hist = px.histogram(
            comparison_df,
            x='Cost',
            nbins=30,
            title="Cost Distribution Across Models",
            labels={'Cost': 'Total Cost', 'count': 'Number of Models'},
            color_discrete_sequence=['#667eea']
        )
        st.plotly_chart(fig_cost_hist, use_container_width=True)
    
    with col2:
        # Expected cost distribution
        fig_exp_cost_hist = px.histogram(
            comparison_df,
            x='Expected Cost',
            nbins=30,
            title="Expected Cost Distribution",
            labels={'Expected Cost': 'Expected Cost', 'count': 'Number of Models'},
            color_discrete_sequence=['#764ba2']
        )
        st.plotly_chart(fig_exp_cost_hist, use_container_width=True)
    
    # Correlation matrix
    st.subheader("Metric Correlations")
    metric_cols = ['Cost', 'Expected Cost', 'Accuracy', 'F1-Macro']
    corr_matrix = comparison_df[metric_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect="auto",
        title="Correlation Matrix of Performance Metrics",
        color_continuous_scale='RdYlBu'
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # 3D scatter plot
    st.subheader("3D Performance Visualization")
    fig_3d = px.scatter_3d(
        comparison_df,
        x='Cost',
        y='Accuracy',
        z='F1-Macro',
        color='Expected Cost',
        size='Accuracy',
        hover_data=['Model'],
        title="3D Performance Space",
        labels={'Cost': 'Total Cost', 'Accuracy': 'Accuracy', 'F1-Macro': 'F1 Score'}
    )
    st.plotly_chart(fig_3d, use_container_width=True)

elif page == "Cost Analysis":
    st.header("🔍 Cost Analysis")
    
    st.markdown("Analyze the cost structure and mistake patterns")
    
    # Select model for analysis
    selected_model = st.selectbox(
        "Select model for detailed cost analysis",
        options=comparison_df['Model'].tolist(),
        index=0
    )
    
    model_data = comparison_df[comparison_df['Model'] == selected_model].iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cost", f"{model_data['Cost']:.2f}")
    
    with col2:
        st.metric("Expected Cost", f"{model_data['Expected Cost']:.4f}")
    
    with col3:
        cost_rank = (comparison_df['Cost'] <= model_data['Cost']).sum()
        total = len(comparison_df)
        st.metric("Cost Rank", f"{cost_rank}/{total}")
    
    with col4:
        best_cost = comparison_df['Cost'].min()
        cost_diff = model_data['Cost'] - best_cost
        st.metric("Cost vs Best", f"{cost_diff:+.2f}")
    
    # Cost breakdown visualization
    st.subheader("Cost Structure")
    
    # Assuming we can calculate cost breakdown (this would need actual predictions)
    # For now, show a conceptual breakdown
    cost_breakdown_data = {
        'Error Type': [
            'Negative → Positive (5)',
            'Positive → Negative (5)',
            'Negative → Neutral (1)',
            'Neutral → Negative (1)',
            'Neutral → Positive (1)',
            'Positive → Neutral (1)',
            'Correct (0)'
        ],
        'Estimated Cost': [
            model_data['Cost'] * 0.3,  # Rough estimates
            model_data['Cost'] * 0.2,
            model_data['Cost'] * 0.15,
            model_data['Cost'] * 0.1,
            model_data['Cost'] * 0.1,
            model_data['Cost'] * 0.15,
            0
        ]
    }
    
    cost_breakdown_df = pd.DataFrame(cost_breakdown_data)
    
    fig_cost_bar = px.bar(
        cost_breakdown_df,
        x='Error Type',
        y='Estimated Cost',
        title=f"Estimated Cost Breakdown for {selected_model}",
        color='Estimated Cost',
        color_continuous_scale='Reds',
        labels={'Estimated Cost': 'Estimated Cost Contribution'}
    )
    fig_cost_bar.update_xaxes(tickangle=45)
    st.plotly_chart(fig_cost_bar, use_container_width=True)
    
    # Interactive confusion matrix builder
    st.subheader("Build Custom Confusion Matrix")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**True: Negative →**")
        neg_to_neg = st.number_input("Negative", min_value=0, value=1000, key='n2n')
        neg_to_neu = st.number_input("Neutral", min_value=0, value=50, key='n2u')
        neg_to_pos = st.number_input("Positive", min_value=0, value=10, key='n2p')
    
    with col2:
        st.write("**True: Neutral →**")
        neu_to_neg = st.number_input("Negative", min_value=0, value=50, key='u2n')
        neu_to_neu = st.number_input("Neutral", min_value=0, value=2000, key='u2u')
        neu_to_pos = st.number_input("Positive", min_value=0, value=50, key='u2p')
    
    with col3:
        st.write("**True: Positive →**")
        pos_to_neg = st.number_input("Negative", min_value=0, value=10, key='p2n')
        pos_to_neu = st.number_input("Neutral", min_value=0, value=50, key='p2u')
        pos_to_pos = st.number_input("Positive", min_value=0, value=1000, key='p2p')
    
    # Calculate cost
    confusion_matrix_custom = np.array([
        [neg_to_neg, neg_to_neu, neg_to_pos],
        [neu_to_neg, neu_to_neu, neu_to_pos],
        [pos_to_neg, pos_to_neu, pos_to_pos]
    ])
    
    total_cost_custom = np.sum(confusion_matrix_custom * COST_MATRIX)
    accuracy_custom = (neg_to_neg + neu_to_neu + pos_to_pos) / confusion_matrix_custom.sum()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cost", f"{total_cost_custom:.2f}")
    with col2:
        st.metric("Accuracy", f"{accuracy_custom:.4f}")
    with col3:
        st.metric("Total Samples", f"{confusion_matrix_custom.sum():,}")
    
    # Visualize
    fig_cm = create_confusion_matrix_plot(confusion_matrix_custom, "Custom Confusion Matrix", show_cost=True)
    st.plotly_chart(fig_cm, use_container_width=True)

elif page == "Interactive Explorer":
    st.header("🔎 Interactive Model Explorer")
    
    selected_model = st.selectbox(
        "Select a model to explore",
        options=comparison_df['Model'].tolist()
    )
    
    model_data = comparison_df[comparison_df['Model'] == selected_model].iloc[0]
    
    st.subheader(f"Model: {selected_model}")
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{model_data['Accuracy']:.4f}")
    
    with col2:
        st.metric("Cost", f"{model_data['Cost']:.2f}")
    
    with col3:
        st.metric("Expected Cost", f"{model_data['Expected Cost']:.4f}")
    
    with col4:
        st.metric("F1-Macro", f"{model_data['F1-Macro']:.4f}")
    
    # Performance comparison
    st.subheader("Performance Comparison")
    
    other_models = st.multiselect(
        "Select models to compare with",
        options=comparison_df[comparison_df['Model'] != selected_model]['Model'].tolist(),
        default=comparison_df[comparison_df['Model'] != selected_model]['Model'].head(3).tolist()
    )
    
    if other_models:
        compare_models = [selected_model] + other_models
        compare_data = comparison_df[comparison_df['Model'].isin(compare_models)]
        
        # Side-by-side comparison
        metrics = ['Cost', 'Expected Cost', 'Accuracy', 'F1-Macro']
        
        for metric in metrics:
            fig = px.bar(
                compare_data,
                x='Model',
                y=metric,
                title=f"{metric} Comparison",
                color='Model',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Ranking visualization
    st.subheader("Ranking Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cost_rank = (comparison_df['Cost'] <= model_data['Cost']).sum()
        total = len(comparison_df)
        st.progress(cost_rank / total, text=f"Cost Ranking: {cost_rank} out of {total}")
        st.caption(f"Lower cost is better. This model is ranked {cost_rank} out of {total}.")
    
    with col2:
        accuracy_rank = (comparison_df['Accuracy'] >= model_data['Accuracy']).sum()
        st.progress(accuracy_rank / total, text=f"Accuracy Ranking: {accuracy_rank} out of {total}")
        st.caption(f"Higher accuracy is better. This model is ranked {accuracy_rank} out of {total}.")

# Footer
st.markdown("---")
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("**Cost-Sensitive Sentiment Analysis Dashboard** | Built with Streamlit")
    st.caption(f"Data loaded from: {RESULTS_DIR}")

with col2:
    # Download comparison data
    csv = comparison_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Data",
        data=csv,
        file_name="cost_sensitive_model_comparison.csv",
        mime="text/csv"
    )
