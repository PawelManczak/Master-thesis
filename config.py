"""
Configuration settings for the Graph Neural Network Emotion Analysis project.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Data files
RDF_FILE = RAW_DATA_DIR / "GraphNeuralNetwork.rdf"
HTML_INDEX_FILE = RAW_DATA_DIR / "Index of _datasets_GraphNeuralNetwork.html"
CSV_DATA_DIR = PROCESSED_DATA_DIR / "csv_data"

# Analysis settings
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

# Visualization settings
PLOT_DPI = 300
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_SIZE = (15, 10)

# Ensure directories exist
def ensure_directories():
    """Create all necessary directories if they don't exist."""
    for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CSV_DATA_DIR,
                      PLOTS_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

