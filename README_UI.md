# Tweet Sentiment Viewer UI

A simple web interface to browse and analyze tweets with sentiment labels.

## Installation

Make sure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

## Running the UI

To start the web interface, run:

```bash
streamlit run tweet_viewer.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

## Features

### 🔍 Filtering
- **Stock Filter**: Select one or more stocks to view
- **Sentiment Filter**: Filter by TextBlob or VADER sentiment (Positive, Neutral, Negative)
- **Date Range**: Filter tweets by date range
- **Search**: Search for specific keywords in tweets
- **Display Count**: Choose how many tweets to display (10-500)

### 📈 Visualizations
- **Sentiment Distribution**: Pie charts showing sentiment breakdown
- **Sentiment Trends**: Line chart showing sentiment over time
- **Stock Analysis**: Bar chart showing sentiment by stock

### 📝 Tweet Display
- **Card View**: Each tweet displayed in a clean card format
- **Color Coding**: 
  - 🟢 Green for positive sentiment
  - 🟡 Yellow for neutral sentiment
  - 🔴 Red for negative sentiment
- **Sorting Options**: Sort by date, sentiment scores, etc.

### 📊 Statistics
- Detailed statistics for each sentiment category
- Agreement analysis between TextBlob and VADER
- Download filtered data as CSV

## Usage Tips

1. **Start with filters**: Use the sidebar to narrow down the tweets you want to see
2. **Explore trends**: Check the "Charts" tab to see sentiment patterns over time
3. **Compare stocks**: Filter by multiple stocks to compare sentiment
4. **Search**: Use the search box to find tweets about specific topics
5. **Download**: Export your filtered results as CSV for further analysis

## File Location

The app reads from: `results/tweets_with_sentiment.csv`

Make sure this file exists before running the app!

