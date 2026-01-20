# Simple Tweet Sentiment Viewer (HTML)

A simple, no-installation-required web interface to view tweets with sentiment labels.

## How to Use

1. **Open the HTML file**: Double-click `simple_tweet_viewer.html` in your file explorer
   - OR right-click and select "Open with" → your web browser (Chrome, Firefox, Edge, etc.)

2. **Load your CSV file**: 
   - Click the file input button
   - Navigate to `results/tweets_with_sentiment.csv`
   - Select and open the file

3. **Start browsing**:
   - Use the filters in the top section to narrow down tweets
   - View statistics and charts
   - Scroll through tweets below

## Features

- ✅ **No installation required** - Just open the HTML file in any browser
- 🔍 **Filter by stock, sentiment, or search keywords**
- 📊 **Interactive charts** showing sentiment distribution
- 📈 **Real-time statistics** updated as you filter
- 🎨 **Color-coded sentiment badges** (green=positive, yellow=neutral, red=negative)
- 📱 **Responsive design** - works on desktop and mobile

## Advantages over Streamlit

- No Python dependencies needed
- Works offline (after loading the CSV)
- Faster startup - just open the file
- Works on any device with a web browser

## File Location

Make sure `results/tweets_with_sentiment.csv` exists before loading it in the UI.

