# Setup Instructions

## Virtual Environment

A virtual environment has been created for this project. **Always activate it before running scripts.**

### Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### Verify Installation

After activating the virtual environment, verify that packages are installed:
```bash
python -c "import pandas; print('pandas version:', pandas.__version__)"
```

### Running Scripts

**IMPORTANT: Always activate the virtual environment first!**

```bash
# Activate virtual environment first (REQUIRED)
source venv/bin/activate

# You should see (venv) in your terminal prompt

# Load data from Kaggle
python scripts/load_data_from_kaggle.py

# Run integration script
python scripts/integrate_datasets.py

# Run analysis script
python scripts/analyze_datasets.py

# Run interactive plot script
python scripts/interactive_plot.py
```

**Note**: If you get `ModuleNotFoundError`, make sure you've activated the virtual environment!

### Deactivate Virtual Environment

When you're done, deactivate the virtual environment:
```bash
deactivate
```

## Dependencies

All required packages are listed in `requirements.txt` and have been installed in the virtual environment.

## Troubleshooting

If you get `ModuleNotFoundError`, make sure:
1. The virtual environment is activated (you should see `(venv)` in your terminal prompt)
2. You're using the correct Python interpreter: `which python` should point to `venv/bin/python`

