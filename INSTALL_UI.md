# Installing the Tweet Sentiment Viewer UI

## Quick Install

Run this command in your terminal:

```bash
pip install streamlit plotly
```

Or use the provided script:
- **Windows**: Double-click `install_ui_dependencies.bat`
- **Linux/Mac**: `pip install streamlit plotly`

## Verify Installation

Check if streamlit is installed:

```bash
python -c "import streamlit; print('Streamlit installed successfully!')"
```

## Running the UI

After installation, run:

```bash
streamlit run tweet_viewer.py
```

Or use the run script:
- **Windows**: Double-click `run_ui.bat`
- **Linux/Mac**: `bash run_ui.sh`

## Troubleshooting

### If you get "streamlit not found" error:

1. Make sure you're using the correct Python environment
2. Try: `python -m streamlit run tweet_viewer.py` instead
3. Reinstall: `pip install --upgrade streamlit plotly`

### If you're using a virtual environment:

Make sure your virtual environment is activated before installing:

```bash
# Activate your venv first
# Then install
pip install streamlit plotly
```

### Alternative: Install all requirements

If you want to install everything at once:

```bash
pip install -r requirements.txt
```

This will install streamlit, plotly, and all other project dependencies.

