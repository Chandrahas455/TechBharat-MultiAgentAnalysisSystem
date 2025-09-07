# TechBharat-MultiAgentAnalysisSystem


A comprehensive data processing pipeline built with CrewAI that automatically processes CSV files through multiple specialized agents.

## Apology/Disclaimer
I could not keep up with the checkpoint commits as all the work I did was in the last few hours of the second day, my laptop had some issues
so I had to wait till I got it back from the repair store.

## Dataset
The entire system is built to be data agnostic but I have tested the workflow on slices of the following few datasets:
https://data.telangana.gov.in/dataset/haritha-haram
https://data.telangana.gov.in/dataset/tgspdcl-agriculture-consumption-data
https://data.telangana.gov.in/dataset/telangana-ground-water-department-water-level-data
## Overview

This crew consists of 5 specialized agents that work together to process and analyze CSV data end-to-end:

1. **Ingestion Agent** - Ingests data from user-uploaded CSV files
2. **Standardization Agent** - Fixes column names, formats, and units
3. **Cleaning Agent** - Removes duplicates, handles missing data, resolves errors
4. **Transformation Agent** - Creates derived fields, aggregations, ratios, and reformatting
5. **Analysis Agent** - Runs final analysis and produces a concise Markdown report

## Demo Video
  - [▶️ Watch the walkthrough](media/SubmissionVideo.mp4)

## Features

- **Automated Data Pipeline**: Complete end-to-end data processing
- **Data Quality Improvement**: Automatic standardization and cleaning
- **Feature Engineering**: Derived variables including ratios (e.g., A/B)
- **Statistical Analysis**: Trends, correlations, patterns, gaps, imbalances
- **Single Output Report**: A Markdown report `report.md` summarizing findings
- **Minimal Plots (optional)**: Only when relationships are strong, with clear titles

## Installation

1. Install dependencies:
```bash
uv init
```
```bash
uv venv
```
```bash
uv sync
```
```bash
source .venv/Scripts/activate
```

2. Set up environment variables (optional):
```bash
# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Usage

### Usage

From the project virtual environment, run:
```bash
cd cleanup_and_analysis
```
Upload the csv you want to analyze in the "inputs" folder

```bash
python src/cleanup_and_analysis/main.py --file "YOUR_FILENAME" --context "A SHORT DESCRIPTION ABOUT THE DATASET"
```


## Output Files

The crew generates the following outputs:

- `input/*_transformed.csv` - Final transformed data with engineered features/ratios
- `report.md` - Final Markdown analysis report
- `charts/*.png` - Only if the analysis finds strong relationships (optional)
- `processing_log.json` - Processing step log


## Agent Details

### Ingestion Agent
- Loads CSV files with various encodings
- Validates data structure and format
- Provides comprehensive data overview
- Identifies potential issues early

### Standardization Agent
- Standardizes column names (lowercase, underscores)
- Converts data types appropriately
- Handles date/time formatting
- Ensures consistent formatting

### Cleaning Agent
- Removes duplicate rows
- Handles missing values intelligently
- Detects and documents outliers
- Improves data quality metrics

### Transformation Agent
- Creates derived features automatically
- Performs aggregations and groupings
- Engineers new variables for analysis
- Optimizes data structure

### Analysis Agent
- Performs statistical analysis (trends, correlations, patterns)
- Detects data gaps and imbalances
- Selects at most two strong relationships (|r| ≥ 0.6) to plot (optional)
- Generates a concise Markdown report (`report.md`)

## Configuration

The crew behavior can be customized by modifying:
- `config/agents.yaml` - Agent roles and behaviors
- `config/tasks.yaml` - Task descriptions and outputs

