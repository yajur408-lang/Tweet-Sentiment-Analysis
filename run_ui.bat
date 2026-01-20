
@echo off
echo Checking if streamlit is installed...
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Streamlit not found. Installing...
    pip install streamlit plotly
    echo.
)

echo Starting Tweet Sentiment Viewer UI...
python -m streamlit run tweet_viewer.py
pause

