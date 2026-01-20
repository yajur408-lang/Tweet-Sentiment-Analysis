@echo off
echo Checking if streamlit is installed...
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Streamlit not found. Installing...
    python -m pip install streamlit plotly
    echo.
)

echo Starting Cost-Sensitive Sentiment Analysis Viewer...
cd /d "%~dp0"
python -m streamlit run cost_analysis_viewer.py
pause
