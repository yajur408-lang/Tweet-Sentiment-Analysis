"""
Streamlit UI for viewing tweets with sentiment labels
Run with: streamlit run tweet_viewer.py
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Tweet Sentiment Viewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    .sentiment-positive {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
    .sentiment-neutral {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    .tweet-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load tweets with sentiment data."""
    csv_path = Path(__file__).parent / "results" / "tweets_with_sentiment.csv"
    if not csv_path.exists():
        st.error(f"❌ File not found: {csv_path}")
        st.stop()
    
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

# Main title
st.title("📊 Tweet Sentiment Viewer")
st.markdown("Browse and analyze tweets with sentiment labels")

# Load data
with st.spinner("Loading tweets..."):
    df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Stock filter
stocks = sorted(df['stock name'].unique().tolist())
selected_stocks = st.sidebar.multiselect(
    "Select Stocks",
    options=stocks,
    default=stocks[:5] if len(stocks) > 5 else stocks
)

# Sentiment filter
sentiment_options = ['All', 'Positive', 'Neutral', 'Negative']
textblob_sentiment = st.sidebar.selectbox(
    "TextBlob Sentiment",
    options=sentiment_options,
    index=0
)

vader_sentiment = st.sidebar.selectbox(
    "VADER Sentiment",
    options=sentiment_options,
    index=0
)

# Target filter (if available)
target_filter = None
if 'target' in df.columns:
    target_options = ['All', 'Up (1)', 'Down (0)']
    target_filter = st.sidebar.selectbox(
        "Target (Price Direction)",
        options=target_options,
        index=0
    )

# Date range filter
min_date = df['date'].min().date()
max_date = df['date'].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Search filter
search_query = st.sidebar.text_input("🔎 Search in tweets", "")

# Number of tweets to display
num_tweets = st.sidebar.slider(
    "Number of tweets to display",
    min_value=10,
    max_value=500,
    value=50,
    step=10
)

# Apply filters
filtered_df = df.copy()

if selected_stocks:
    filtered_df = filtered_df[filtered_df['stock name'].isin(selected_stocks)]

if textblob_sentiment != 'All':
    filtered_df = filtered_df[filtered_df['textblob_sentiment'].str.lower() == textblob_sentiment.lower()]

if vader_sentiment != 'All':
    filtered_df = filtered_df[filtered_df['vader_sentiment'].str.lower() == vader_sentiment.lower()]

if target_filter and target_filter != 'All':
    if target_filter == 'Up (1)':
        filtered_df = filtered_df[filtered_df['target'] == 1]
    elif target_filter == 'Down (0)':
        filtered_df = filtered_df[filtered_df['target'] == 0]

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df['date'].dt.date >= start_date) & 
        (filtered_df['date'].dt.date <= end_date)
    ]

if search_query:
    filtered_df = filtered_df[
        filtered_df['tweet'].str.contains(search_query, case=False, na=False)
    ]

# Main content area
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Tweets", f"{len(filtered_df):,}")

with col2:
    positive_count = len(filtered_df[filtered_df['textblob_sentiment'].str.lower() == 'positive'])
    st.metric("Positive", f"{positive_count:,}", f"{(positive_count/len(filtered_df)*100):.1f}%")

with col3:
    neutral_count = len(filtered_df[filtered_df['textblob_sentiment'].str.lower() == 'neutral'])
    st.metric("Neutral", f"{neutral_count:,}", f"{(neutral_count/len(filtered_df)*100):.1f}%")

with col4:
    negative_count = len(filtered_df[filtered_df['textblob_sentiment'].str.lower() == 'negative'])
    st.metric("Negative", f"{negative_count:,}", f"{(negative_count/len(filtered_df)*100):.1f}%")

# Visualizations
tab1, tab2, tab3 = st.tabs(["📈 Charts", "📝 Tweets", "📊 Statistics"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution pie chart
        sentiment_counts = filtered_df['textblob_sentiment'].value_counts()
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="TextBlob Sentiment Distribution",
            color_discrete_map={
                'positive': '#28a745',
                'neutral': '#ffc107',
                'negative': '#dc3545'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # VADER sentiment distribution
        vader_counts = filtered_df['vader_sentiment'].value_counts()
        fig_pie_vader = px.pie(
            values=vader_counts.values,
            names=vader_counts.index,
            title="VADER Sentiment Distribution",
            color_discrete_map={
                'positive': '#28a745',
                'neutral': '#ffc107',
                'negative': '#dc3545'
            }
        )
        st.plotly_chart(fig_pie_vader, use_container_width=True)
    
    # Sentiment over time
    daily_sentiment = filtered_df.groupby([filtered_df['date'].dt.date, 'textblob_sentiment']).size().reset_index(name='count')
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    fig_line = px.line(
        daily_sentiment,
        x='date',
        y='count',
        color='textblob_sentiment',
        title="Sentiment Trends Over Time",
        labels={'count': 'Number of Tweets', 'date': 'Date'},
        color_discrete_map={
            'positive': '#28a745',
            'neutral': '#ffc107',
            'negative': '#dc3545'
        }
    )
    st.plotly_chart(fig_line, use_container_width=True)
    
    # Sentiment by stock
    stock_sentiment = filtered_df.groupby(['stock name', 'textblob_sentiment']).size().reset_index(name='count')
    fig_bar = px.bar(
        stock_sentiment,
        x='stock name',
        y='count',
        color='textblob_sentiment',
        title="Sentiment Distribution by Stock",
        labels={'count': 'Number of Tweets'},
        color_discrete_map={
            'positive': '#28a745',
            'neutral': '#ffc107',
            'negative': '#dc3545'
        }
    )
    fig_bar.update_xaxes(tickangle=45)
    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    # Display tweets
    st.subheader(f"📝 Tweets ({len(filtered_df):,} total)")
    
    # View mode toggle
    view_mode = st.radio("View Mode", ["Card View", "Table View"], horizontal=True)
    
    # Sort options
    sort_by = st.selectbox(
        "Sort by",
        options=['Date (Newest)', 'Date (Oldest)', 'TextBlob Score (High)', 'TextBlob Score (Low)', 
                 'VADER Score (High)', 'VADER Score (Low)'],
        index=0
    )
    
    # Sort dataframe
    display_df = filtered_df.copy()
    if sort_by == 'Date (Newest)':
        display_df = display_df.sort_values('date', ascending=False)
    elif sort_by == 'Date (Oldest)':
        display_df = display_df.sort_values('date', ascending=True)
    elif sort_by == 'TextBlob Score (High)':
        display_df = display_df.sort_values('textblob_score', ascending=False)
    elif sort_by == 'TextBlob Score (Low)':
        display_df = display_df.sort_values('textblob_score', ascending=True)
    elif sort_by == 'VADER Score (High)':
        display_df = display_df.sort_values('vader_score', ascending=False)
    elif sort_by == 'VADER Score (Low)':
        display_df = display_df.sort_values('vader_score', ascending=True)
    
    if view_mode == "Table View":
        # Table view with color coding
        display_cols = ['date', 'stock name', 'tweet', 'textblob_sentiment', 'vader_sentiment']
        if 'target' in display_df.columns:
            display_cols.append('target')
        
        table_df = display_df[display_cols].head(num_tweets).copy()
        
        # Color code sentiment
        def color_sentiment(val):
            if pd.isna(val):
                return ''
            val_str = str(val).lower()
            if val_str == 'positive':
                return 'background-color: #d4edda; color: #000'
            elif val_str == 'negative':
                return 'background-color: #f8d7da; color: #000'
            elif val_str == 'neutral':
                return 'background-color: #fff3cd; color: #000'
            return ''
        
        # Color code target
        def color_target(val):
            if pd.isna(val):
                return ''
            if val == 1:
                return 'background-color: #d4edda; color: #000'
            elif val == 0:
                return 'background-color: #f8d7da; color: #000'
            return ''
        
        styled_df = table_df.style.applymap(
            color_sentiment, 
            subset=['textblob_sentiment', 'vader_sentiment']
        )
        
        if 'target' in table_df.columns:
            styled_df = styled_df.applymap(
                color_target,
                subset=['target']
            )
        
        st.dataframe(styled_df, use_container_width=True, height=400)
    else:
        # Card view
        for idx, row in display_df.head(num_tweets).iterrows():
            # Determine sentiment color
            sentiment = row['textblob_sentiment'].lower()
            if sentiment == 'positive':
                sentiment_class = 'sentiment-positive'
                sentiment_emoji = '✅'
            elif sentiment == 'negative':
                sentiment_class = 'sentiment-negative'
                sentiment_emoji = '❌'
            else:
                sentiment_class = 'sentiment-neutral'
                sentiment_emoji = '➖'
            
            # Create tweet card
            st.markdown(f"""
            <div class="tweet-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div>
                        <strong>📅 {row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'N/A'}</strong> | 
                        <strong>📈 {row['stock name']}</strong>
                    </div>
                    <div>
                        {sentiment_emoji} <strong>{row['textblob_sentiment'].upper()}</strong>
                    </div>
                </div>
                <p style="font-size: 14px; line-height: 1.6; margin: 10px 0;">{row['tweet']}</p>
                <div style="display: flex; gap: 20px; margin-top: 10px; font-size: 12px; color: #666;">
                    <span>📊 TextBlob: {row['textblob_sentiment']} ({row['textblob_score']:.3f})</span>
                    <span>📊 VADER: {row['vader_sentiment']} ({row['vader_score']:.3f})</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    if len(display_df) > num_tweets:
        st.info(f"Showing {num_tweets} of {len(display_df):,} tweets. Use the slider in the sidebar to see more.")

with tab3:
    st.subheader("📊 Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**TextBlob Sentiment Statistics**")
        tb_stats = filtered_df.groupby('textblob_sentiment').agg({
            'textblob_score': ['count', 'mean', 'std', 'min', 'max']
        }).round(3)
        tb_stats.columns = ['Count', 'Mean Score', 'Std Dev', 'Min Score', 'Max Score']
        st.dataframe(tb_stats, use_container_width=True)
    
    with col2:
        st.write("**VADER Sentiment Statistics**")
        vader_stats = filtered_df.groupby('vader_sentiment').agg({
            'vader_score': ['count', 'mean', 'std', 'min', 'max']
        }).round(3)
        vader_stats.columns = ['Count', 'Mean Score', 'Std Dev', 'Min Score', 'Max Score']
        st.dataframe(vader_stats, use_container_width=True)
    
    # Agreement between TextBlob and VADER
    st.write("**Sentiment Agreement Analysis**")
    filtered_df['agreement'] = filtered_df['textblob_sentiment'] == filtered_df['vader_sentiment']
    agreement_rate = filtered_df['agreement'].mean() * 100
    st.metric("TextBlob-VADER Agreement Rate", f"{agreement_rate:.2f}%")
    
    # Agreement matrix
    agreement_matrix = pd.crosstab(
        filtered_df['textblob_sentiment'],
        filtered_df['vader_sentiment'],
        normalize='index'
    ) * 100
    st.dataframe(agreement_matrix.round(2), use_container_width=True)
    
    # Download button
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=filtered_df.to_csv(index=False),
        file_name=f"filtered_tweets_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("**Tweet Sentiment Viewer** | Built with Streamlit")

