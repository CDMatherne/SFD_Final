# AIS Shipping Fraud Detection (SFD) System

**Version:** 2.1 Beta  
**Team:** Dreadnaught (Alex Giacomello, Christopher Matherne, Rupert Rigg, Zachary Zhao)

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Project Features](#project-features)
4. [Usage Guide](#usage-guide)
   - [SFD_GUI.py - Main Interface](#sfd_guipy---main-interface)
   - [advanced_analysis.py - Advanced Analysis Tools](#advanced_analysispy---advanced-analysis-tools)
5. [Usage Flow](#usage-flow)
6. [Machine Learning Integration](#machine-learning-integration)
7. [Model Training Details](#model-training-details)
8. [Data Flow Diagrams](#data-flow-diagrams)
9. [Output Descriptions](#output-descriptions)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The AIS Shipping Fraud Detection (SFD) System is a comprehensive maritime analysis platform designed to detect anomalous vessel behavior patterns from Automatic Identification System (AIS) data. The system combines traditional data analysis techniques with advanced machine learning to predict vessel courses up to 48 hours in advance, enabling proactive law enforcement interdiction of suspicious vessels.

### Key Capabilities

- **Data Processing**: Download, process, and analyze AIS data from multiple sources (NOAA, local files, AWS S3)
- **Anomaly Detection**: Identify various types of suspicious vessel behaviors
- **Geographic Analysis**: Zone violation detection and geographic filtering
- **Advanced Analytics**: Statistical analysis, clustering, temporal pattern analysis
- **Machine Learning Predictions**: 48-hour vessel course prediction with uncertainty quantification
- **Interactive Visualizations**: Maps, charts, and reports for data exploration

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Windows, Linux, or macOS
- Minimum 8GB RAM (16GB+ recommended for large datasets)
- Optional: NVIDIA GPU with CUDA support or AMD GPU with ROCm support for ML acceleration

### Step 1: Clone or Download the Repository

```bash
# If using git
git clone <repository-url>
cd SFD_Beta_release

# Or download and extract the ZIP file
```

### Step 2: Install Core Dependencies

Install the main project requirements:

```bash
pip install -r requirements.txt
```

**Core Dependencies Include:**
- `pandas>=1.3.0` - Data processing
- `numpy>=1.20.0` - Numerical computations
- `dask>=2022.1.0` - Parallel processing
- `geopy>=2.0.0` - Geospatial calculations
- `folium>=0.12.0` - Interactive maps
- `matplotlib>=3.4.0` - Plotting
- `plotly>=5.0.0` - Interactive visualizations
- `openpyxl>=3.0.0` - Excel file support
- `tkcalendar>=1.6.0` - Date selection widgets
- And more (see `requirements.txt`)

### Step 3: Install ML Course Prediction Dependencies

For machine learning features, install additional dependencies:

```bash
pip install -r ml_course_prediction/requirements.txt
```

**ML Dependencies Include:**
- `torch>=2.0.0` - PyTorch deep learning framework
- `torchvision>=0.15.0` - Computer vision utilities
- `geopandas>=0.13.0` - Geospatial data processing
- `scikit-learn>=1.3.0` - Machine learning utilities
- `plotly>=5.14.0` - Advanced visualizations
- And more (see `ml_course_prediction/requirements.txt`)

### Step 4: Optional GPU Support

**For NVIDIA GPUs:**
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For AMD GPUs (ROCm):**
```bash
# Install PyTorch with ROCm support (Linux recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
# Also install ROCm drivers from: https://rocm.docs.amd.com/
```

### Step 5: Verify Installation

Run the dependency checker:

```bash
python -c "from utils import check_dependencies; check_dependencies()"
```

### Step 6: Configure the System

1. Copy `config.ini` template if needed (or create from scratch)
2. Configure data directories, output paths, and analysis parameters
3. Set up AWS credentials if using S3 data sources

---

## Project Features

### Core Analysis Features

1. **Multi-Source Data Integration**
   - NOAA AIS data downloads
   - Local file processing (CSV, Parquet)
   - AWS S3 data access
   - Automatic data caching and conversion

2. **Anomaly Detection**
   - Speed anomalies (excessive speed, sudden changes)
   - Course anomalies (unusual turns, erratic behavior)
   - Position anomalies (AIS spoofing indicators)
   - Zone violations (entering restricted areas)
   - Time-based anomalies (missing reports, gaps)

3. **Vessel Type Filtering**
   - Cargo vessels (types 70-79)
   - Tanker vessels (types 80-89)
   - Other vessel types (types 90-94)
   - Hazardous material categorization

4. **Geographic Analysis**
   - Custom zone definition and violation detection
   - Geographic bounding box filtering
   - Interactive map-based zone drawing
   - Coordinate-based zone entry

5. **Statistical Analysis**
   - Vessel statistics and summaries
   - Anomaly frequency analysis
   - Temporal pattern detection
   - Correlation analysis between vessel types and anomalies

6. **Visualization Tools**
   - Interactive Folium maps with vessel paths
   - Heat maps for anomaly density
   - Vessel-specific trajectory maps
   - Statistical charts and graphs

### Machine Learning Features

1. **48-Hour Course Prediction**
   - Predicts vessel positions up to 48 hours ahead
   - 8 prediction points at 6-hour intervals
   - Uncertainty quantification (1σ confidence intervals)
   - Autoregressive prediction for multi-step forecasting

2. **Hybrid LSTM-Transformer Architecture**
   - LSTM encoder for short-term temporal patterns
   - Transformer encoder for long-term dependencies
   - Feature fusion layer combining both architectures
   - Uncertainty head for confidence estimation

3. **Physics-Constrained Predictions**
   - Maximum speed constraints (50 knots)
   - Turn rate limitations
   - Maritime physics validation

---

## Usage Guide

### SFD_GUI.py - Main Interface

The main GUI provides a tabbed interface for configuring and running AIS data analysis.

#### Starting the Application

```bash
python SFD_GUI.py
```

#### Tab Overview

**1. Startup Tab**
- **Purpose**: Initial configuration and analysis execution
- **Key Controls**:
  - **Run Analysis**: Starts the main analysis pipeline with current settings
  - **Save Configuration**: Saves all settings to `config.ini`
  - **Check for GPU Acceleration**: Verifies GPU availability for ML features
  - **Exit**: Closes the application

**2. Date Selection Tab**
- **Purpose**: Define the analysis time range
- **Controls**:
  - Start Date: Beginning of analysis period
  - End Date: End of analysis period
  - Date picker widgets for easy selection
- **Output**: Date range stored in configuration

**3. Ship Types Tab**
- **Purpose**: Select vessel types to include in analysis
- **Available Ship Types by Category**:

  **Wing in Ground (WIG) - Types 20-24:**
  - 20: Wing in ground (WIG), all ships of this type
  - 21: Wing in ground (WIG), Hazardous category A
  - 22: Wing in ground (WIG), Hazardous category B
  - 23: Wing in ground (WIG), Hazardous category C
  - 24: Wing in ground (WIG), Hazardous category D

  **Special Craft - Types 30-39:**
  - 30: Fishing
  - 31: Towing
  - 32: Towing: length exceeds 200m or breadth exceeds 25m
  - 33: Dredging or underwater ops
  - 34: Diving ops
  - 35: Military ops
  - 36: Sailing
  - 37: Pleasure Craft
  - 38: Reserved
  - 39: Reserved

  **High Speed Craft (HSC) - Types 40-49:**
  - 40: High speed craft (HSC), all ships of this type
  - 41: High speed craft (HSC), Hazardous category A
  - 42: High speed craft (HSC), Hazardous category B
  - 43: High speed craft (HSC), Hazardous category C
  - 44: High speed craft (HSC), Hazardous category D
  - 45: High speed craft (HSC), Reserved
  - 46: High speed craft (HSC), Reserved
  - 47: High speed craft (HSC), Reserved
  - 48: High speed craft (HSC), Reserved
  - 49: High speed craft (HSC), No additional information

  **Special Purpose - Types 50-59:**
  - 50: Pilot Vessel
  - 51: Search and Rescue vessel
  - 52: Tug
  - 53: Port Tender
  - 54: Anti-pollution equipment
  - 55: Law Enforcement
  - 56: Spare - Local Vessel
  - 57: Spare - Local Vessel
  - 58: Medical Transport
  - 59: Noncombatant ship (RR Resolution No. 18)

  **Passenger - Types 60-69:**
  - 60: Passenger, all ships of this type
  - 61: Passenger, Hazardous category A
  - 62: Passenger, Hazardous category B
  - 63: Passenger, Hazardous category C
  - 64: Passenger, Hazardous category D
  - 69: Passenger, No additional information

  **Cargo - Types 70-79:**
  - 70: Cargo, all ships of this type
  - 71: Cargo, Hazardous category A
  - 72: Cargo, Hazardous category B
  - 73: Cargo, Hazardous category C
  - 74: Cargo, Hazardous category D
  - 79: Cargo, No additional information

  **Tanker - Types 80-89:**
  - 80: Tanker, all ships of this type
  - 81: Tanker, Hazardous category A
  - 82: Tanker, Hazardous category B
  - 83: Tanker, Hazardous category C
  - 84: Tanker, Hazardous category D
  - 89: Tanker, No additional information

  **Other Type - Types 90-94:**
  - 90: Other Type, all ships of this type
  - 91: Other Type, Hazardous category A
  - 92: Other Type, Hazardous category B
  - 93: Other Type, Hazardous category C
  - 94: Other Type, Hazardous category D

- **Controls**:
  - Individual checkboxes for each vessel type
  - Category select/deselect buttons (Select All Cargo, Deselect All Cargo, Select All Tanker, Deselect All Tanker, etc.)
  - Select All Ships / Deselect All Ships buttons
- **Output**: Selected vessel types filter the analysis dataset

**4. Anomaly Types Tab**
- **Purpose**: Select which anomaly types to detect
- **Anomaly Types**:
  - Speed Anomalies
  - Course Anomalies
  - Position Anomalies
  - Zone Violations
  - Time-based Anomalies
- **Controls**:
  - Checkboxes for each anomaly type
  - Select All / Deselect All buttons
- **Output**: Only selected anomaly types are detected and reported

**5. Analysis Filters Tab**
- **Purpose**: Configure analysis parameters and thresholds
- **Key Filters**:
  - Speed thresholds (maximum, minimum)
  - Course change thresholds
  - Time gap thresholds
  - Geographic bounds
  - Data quality filters
- **Controls**:
  - Numeric input fields for thresholds
  - Reset to Defaults button
- **Output**: Filter parameters applied during analysis

**6. Data Tab**
- **Purpose**: Configure data sources and storage
- **Data Sources**:
  - **Local Directory**: Path to local AIS data files
  - **AWS S3**: S3 bucket URI for cloud data
  - **NOAA Download**: Automatic download from NOAA servers
- **Controls**:
  - Directory browser buttons
  - S3 URI input field
  - Data source selection
- **Output**: Data source configuration saved

**7. Zone Violations Tab**
- **Purpose**: Define geographic zones for violation detection
- **Features**:
  - Add/Edit/Delete zones
  - Interactive map-based zone drawing
  - Coordinate-based zone entry
  - Zone selection for analysis
- **Controls**:
  - Zone list with selection checkboxes
  - Add Zone button
  - Edit/Delete buttons for each zone
  - Draw Zone button (opens map interface)
  - Select All / Deselect All zones
- **Output**: Zone definitions saved to configuration

**8. Output Controls Tab**
- **Purpose**: Configure output generation options
- **Output Types**:
  - Summary reports
  - Statistics CSV files
  - Anomaly reports
  - Map visualizations
  - Excel workbooks
- **Controls**:
  - Checkboxes for each output type
  - Output directory selection
  - Select All / Deselect All buttons
- **Output**: Generated files saved to specified directory

**9. Instructions Tab**
- **Purpose**: User guide and help information
- **Content**:
  - Step-by-step usage instructions
  - Feature descriptions
  - Troubleshooting tips
  - Contact information

#### Running an Analysis

1. **Configure Settings**:
   - Set date range (Startup or Date Selection tab)
   - Select vessel types (Ship Types tab)
   - Select anomaly types (Anomaly Types tab)
   - Configure filters (Analysis Filters tab)
   - Set data source (Data tab)
   - Define zones if needed (Zone Violations tab)
   - Configure outputs (Output Controls tab)

2. **Execute Analysis**:
   - Click "Run Analysis" button on Startup tab
   - Progress window shows download/processing status
   - Analysis runs in background thread

3. **View Results**:
   - Progress window shows completion status
   - "Open Results" button opens output directory
   - "Conduct Additional Analysis" launches advanced tools

#### Outputs from SFD_GUI.py

- **Cached Data**: Processed AIS data stored in `.ais.cache_data` directories
- **Anomaly Reports**: CSV files with detected anomalies
- **Summary Statistics**: Statistical summaries of vessels and anomalies
- **Configuration File**: Updated `config.ini` with run parameters
- **Log Files**: Processing logs for debugging

---

### advanced_analysis.py - Advanced Analysis Tools

The Advanced Analysis GUI provides post-processing tools for previously analyzed datasets.

#### Launching Advanced Analysis

**From SFD_GUI.py:**
- Click "Conduct Additional Analysis" after main analysis completes

**Standalone:**
```bash
python run_advanced_analysis.py
```

#### Tab Overview

**1. Additional Outputs Tab**
- **Purpose**: Generate additional reports and exports
- **Features**:
  - **Export Full Dataset**: Export complete processed dataset to CSV/Parquet
  - **Generate Summary Report**: Create comprehensive HTML summary report
  - **Export Vessel Statistics**: Export per-vessel statistics to CSV
  - **Generate Anomaly Timeline**: Create timeline visualization of anomalies
- **Outputs**: Files saved to output directory

**2. Further Analysis Tab**
- **Purpose**: Advanced statistical and pattern analysis
- **Features**:
  - **Correlation Analysis**: Analyze relationships between vessel types and anomalies
  - **Temporal Pattern Analysis**: Identify time-based patterns in anomalies
  - **Vessel Behavior Clustering**: Group vessels by behavior patterns (K-means)
  - **Anomaly Frequency Analysis**: Analyze frequency distributions of anomalies
- **Outputs**: Analysis reports, charts, and statistical summaries

**3. Mapping Tools Tab**
- **Purpose**: Create interactive maps and visualizations
- **Features**:
  - **Full Spectrum Anomaly Map**: Map showing all anomalies with heat map overlay
  - **Vessel-Specific Maps**: Individual vessel trajectory maps
  - **Filtered Maps**: Maps filtered by vessel type, anomaly type, or specific MMSI
- **Outputs**: Interactive HTML maps (Folium) saved to output directory

**4. Vessel Analysis Tab**
- **Purpose**: Deep-dive analysis for specific vessels
- **Features**:
  - **Extended Time Range Analysis**: Analyze vessel behavior over extended periods
  - **ML Course Prediction**: Generate 48-hour course predictions (if ML available)
  - **Vessel MMSI Input**: Enter specific vessel MMSI for analysis
- **Outputs**: Vessel-specific reports, predictions, and visualizations

**5. Anomaly Types Tab**
- **Purpose**: Filter analysis by anomaly types
- **Features**:
  - Select/deselect specific anomaly types
  - Apply filters to all analysis operations
- **Usage**: Check desired anomaly types before running analyses

**6. Analysis Filters Tab**
- **Purpose**: Apply additional filters to cached data
- **Features**:
  - Vessel type filters
  - Date range filters
  - Geographic bounds
  - Speed/course thresholds
- **Usage**: Refine analysis scope without re-running main analysis

**7. Zone Violations Tab**
- **Purpose**: Manage and analyze zone violations
- **Features**:
  - View/edit zone definitions
  - Filter by selected zones
  - Generate zone-specific maps
- **Usage**: Focus analysis on specific geographic areas

**8. Vessel Selection Tab**
- **Purpose**: Select specific vessels for analysis
- **Features**:
  - MMSI input field
  - Vessel type selection
  - Category-based selection (Cargo/Tanker/Other)
- **Usage**: Narrow analysis to specific vessels or categories

#### ML Course Prediction Feature

**Access**: Vessel Analysis Tab → ML Course Prediction section

**Requirements**:
- Trained model (`best_model.pt`) in `ml_course_prediction/models/trained/`
- At least 2 data points for the vessel (minimum 1 hour span)
- Valid trajectory data (no gaps > 6 hours)

**Usage**:
1. Enter vessel MMSI in Vessel Analysis tab
2. Click "Generate ML Prediction"
3. View prediction results:
   - Predicted positions (8 points, 6-hour intervals)
   - Uncertainty bounds (1σ confidence intervals)
   - Prediction map with trajectory overlay
   - Speed and course predictions (if available)

**Outputs**:
- Interactive map showing:
  - Historical trajectory (blue line)
  - Predicted path (green line)
  - Uncertainty ellipses (red shaded areas)
  - Prediction points (green markers)
- Prediction report with coordinates and uncertainties

---

## Usage Flow

### Complete Workflow: From Initialization to Outputs

```
┌─────────────────────────────────────────────────────────────────┐
│                   1. INITIALIZATION                             │
│                   (SFD_GUI.py Startup)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   2. CONFIGURATION                              │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Date Range   │  │ Vessel Types │  │ Anomaly Types│           │
│  │ Selection    │  │ Selection    │  │ Selection    │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Analysis     │  │ Data Source  │  │ Zone         │           │
│  │ Filters      │  │ Config       │  │ Definitions  │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                 │
│  ┌──────────────┐                                               │
│  │ Output       │                                               │
│  │ Controls     │                                               │
│   └──────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   3. DATA ACQUISITION                           │
│                                                                 │
│  Input Sources:                                                 │
│  • NOAA AIS Data (download)                                     │
│  • Local Files (CSV/Parquet)                                    │
│  • AWS S3 Buckets                                               │
│                                                                 │
│  Data Elements Required:                                        │
│  • MMSI (Maritime Mobile Service Identity)                      │
│  • BaseDateTime (timestamp)                                     │
│  • LAT (latitude)                                               │
│  • LON (longitude)                                              │
│  • SOG (speed over ground)                                      │
│  • COG (course over ground)                                     │
│  • Heading                                                      │
│  • VesselType                                                   │
│                                                                 │
│  Processing:                                                    │
│  • Download/load data files                                     │
│  • Convert CSV to Parquet (if needed)                           │
│  • Cache processed data                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   4. DATA PROCESSING                            │
│                                                                 │
│  Steps:                                                         │
│  1. Data validation and cleaning                                │
│  2. Filter by date range                                        │
│  3. Filter by vessel types                                      │
│  4. Apply geographic filters                                    │
│  5. Calculate derived features                                  │
│                                                                 │
│  Outputs:                                                       │
│  • Processed DataFrame (pandas)                                 │
│  • Cached data files (.parquet)                                 │
│  • Data quality metrics                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   5. ANOMALY DETECTION                          │
│                                                                 │
│  Detection Algorithms:                                          │
│  • Speed Anomaly Detection                                      │
│    - Excessive speed (> threshold)                              │
│    - Sudden speed changes                                       │
│  • Course Anomaly Detection                                     │
│    - Unusual turns                                              │
│    - Erratic course changes                                     │
│  • Position Anomaly Detection                                   │
│    - AIS spoofing indicators                                    │
│    - Impossible movements                                       │
│  • Zone Violation Detection                                     │
│    - Entry into restricted zones                                │
│  • Time-based Anomaly Detection                                 │
│    - Missing AIS reports                                        │
│    - Unusual time gaps                                          │
│                                                                 │
│  Outputs:                                                       │
│  • Anomaly DataFrame with detected anomalies                    │
│  • Anomaly summary statistics                                   │
│  • Per-vessel anomaly counts                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   6. OUTPUT GENERATION                          │
│                                                                 │
│  Generated Files:                                               │
│  • anomaly_summary.csv - Summary of all anomalies               │
│  • vessel_statistics.csv - Per-vessel statistics                │
│  • summary_report.html - Comprehensive HTML report              │
│  • Configuration files (config.ini updated)                     │
│  • Cached data (for advanced analysis)                          │
│                                                                 │
│  Storage:                                                       │
│  • Output directory (user-specified)                            │
│  • Cache directory (~/.ais_data_cache)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   7. ADVANCED ANALYSIS                          │
│                   (advanced_analysis.py)                        │
│                                                                 │
│  Available Operations:                                          │
│  • Export full dataset                                          │
│  • Generate detailed reports                                    │
│  • Statistical analysis                                         │
│  • Clustering analysis                                          │
│  • Temporal pattern analysis                                    │
│  • Interactive map generation                                   │
│  • ML course predictions                                        │
│                                                                 │
│  Inputs:                                                        │
│  • Cached data from main analysis                               │
│  • Anomaly data                                                 │
│  • Configuration from previous run                              │
│                                                                 │
│  Outputs:                                                       │
│  • Additional CSV exports                                       │
│  • HTML reports and visualizations                              │
│  • Interactive maps (Folium)                                    │
│  • Statistical charts                                           │
│  • ML prediction results                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Data Elements at Each Stage

**Input Data (Stage 3):**
- Raw AIS records with required fields
- File format: CSV or Parquet
- Date range: User-specified

**Processed Data (Stage 4):**
- Cleaned and validated records
- Filtered by vessel type and date
- Derived features calculated
- Format: Pandas DataFrame, cached Parquet

**Anomaly Data (Stage 5):**
- Anomaly records with:
  - MMSI
  - Timestamp
  - Anomaly type
  - Anomaly details (speed, course, position, etc.)
  - Severity scores
- Format: Pandas DataFrame, CSV export

**Output Data (Stage 6):**
- Summary statistics
- Anomaly reports
- Vessel statistics
- HTML reports
- Configuration files

**Advanced Analysis Outputs (Stage 7):**
- Extended datasets
- Statistical analyses
- Clustering results
- Interactive visualizations
- ML predictions

---

## Machine Learning Integration

### Overview

The ML Course Prediction system uses a hybrid LSTM-Transformer architecture to predict vessel positions up to 48 hours in advance. The system is integrated into the Advanced Analysis GUI and can be accessed through the Vessel Analysis tab.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML PREDICTION PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

Input: Vessel AIS Data (Historical Trajectory)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. DATA PREPARATION                                             │
│    • Filter vessel data by MMSI                                 │
│    • Extract last N hours (default: 24 hours)                   │
│    • Sort by timestamp                                          │
│    • Validate minimum data requirements (2+ points, 1+ hour)    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. TRAJECTORY PROCESSING                                        │
│    • Segment trajectories (handle gaps > 6 hours)               │
│    • Select most recent trajectory segment                      │
│    • Validate trajectory quality                                │
│    • Handle missing data                                        │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. FEATURE EXTRACTION                                           │
│    • Position features (LAT, LON)                               │
│    • Speed features (SOG)                                       │
│    • Course features (COG, Heading)                             │
│    • Temporal features (hour, day of week, etc.)                │
│    • Vessel type encoding                                       │
│    • Create sequence features (24-hour history)                 │
│    • Normalize features                                         │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. MODEL INFERENCE                                              │
│                                                                 │
│    Input Sequence (batch_size, seq_len, features)               │
│         │                                                       │
│         ├──────────────────┬──────────────────┐                 │
│         ▼                  ▼                  ▼                 │
│    ┌──────────┐      ┌─────────────┐    ┌──────────┐            │
│    │  LSTM    │      │Transformer  │    │  Feature │            │
│    │ Encoder  │      │ Encoder     │    │  Fusion  │            │
│    │          │      │             │    │          │            │
│    │ Short-   │      │ Long-term   │    │ Combines │            │
│    │ term     │      │ Dependencies│    │ LSTM +   │            │
│    │ Patterns │      │             │    │ Trans.   │            │
│    └──────────┘      └─────────────┘    └──────────┘            │
│         │                  │                  │                 │
│         └──────────────────┴──────────────────┘                 │
│                            │                                    │
│                            ▼                                    │
│                   ┌─────────────────┐                           │
│                   │  Output Heads   │                           │
│                   │                 │                           │
│                   │ • Position Head │                           │
│                   │   (with         │                           │
│                   │    uncertainty) │                           │
│                   │ • Speed Head    │                           │
│                   │ • Course Head   │                           │
│                   └─────────────────┘                           │
│                            │                                    │
│                            ▼                                    │
│              Autoregressive Prediction (8 steps)                │
│              • Step 1: Predict t+6h                             │
│              • Step 2: Use prediction, predict t+12h            │
│              • ...                                              │
│              • Step 8: Predict t+48h                            │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. POST-PROCESSING                                              │
│    • Apply physics constraints                                  │
│    • Calculate uncertainty bounds (1σ)                          │
│    • Format predictions (8 points, 6-hour intervals)            │
│    • Generate confidence intervals                              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. VISUALIZATION                                                │
│    • Create interactive map (Folium)                            │
│    • Plot historical trajectory                                 │
│    • Plot predicted path                                        │
│    • Display uncertainty ellipses                               │
│    • Generate prediction report                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Model Architecture Details

**Hybrid LSTM-Transformer Model:**

1. **LSTM Encoder**
   - Layers: 2
   - Hidden size: 128
   - Dropout: 0.2
   - Bidirectional: False
   - Purpose: Captures short-term temporal patterns (recent movements)

2. **Transformer Encoder**
   - Layers: 4
   - Hidden size: 256
   - Attention heads: 8
   - Feedforward size: 1024
   - Dropout: 0.1
   - Purpose: Captures long-term dependencies via self-attention

3. **Feature Fusion**
   - Combines LSTM and Transformer outputs
   - Hidden size: 256
   - Activation: ReLU
   - Dropout: 0.1

4. **Output Heads**
   - **Position Head**: Predicts LAT/LON with uncertainty (variational method)
   - **Speed Head**: Predicts SOG (Speed Over Ground)
   - **Course Head**: Predicts COG (Course Over Ground)

### Prediction Process

1. **Input Requirements**:
   - Minimum 2 data points
   - Minimum 1 hour time span
   - Maximum 6-hour gap between points
   - Valid position data (LAT, LON)

2. **Autoregressive Prediction**:
   - Model predicts one time step (6 hours) at a time
   - Each prediction uses previous predictions as input
   - Total: 8 predictions covering 48 hours
   - Each prediction includes uncertainty estimates

3. **Uncertainty Quantification**:
   - Method: Variational inference
   - Output: Mean and standard deviation for each position
   - Confidence intervals: 1σ (68% confidence)
   - Uncertainty grows with prediction horizon

4. **Physics Constraints**:
   - Maximum speed: 50 knots
   - Maximum turn rate: 180°/hour
   - Predictions validated against physical constraints

### Integration Points

**In Advanced Analysis GUI:**
- Vessel Analysis Tab → ML Course Prediction section
- Requires: Vessel MMSI input
- Output: Interactive map and prediction report

**In Code:**
- `ml_prediction_integration.py`: Integration module
- `ml_course_prediction/`: ML module directory
- `best_model.pt`: Trained model file

---

## Model Training Details

### Training Process Overview

The model was trained on AIS data from January 2024, using a hybrid LSTM-Transformer architecture. Training was performed using PyTorch on CPU (GPU support available but not used in final training).

### Training Data

**Dataset:**
- **Date Range**: January 1-31, 2024 (31 days)
- **Total Records**: 221,495,593 AIS records
- **Unique Vessels**: 33,578 vessels
- **Trajectory Segments**: 111,144 segments
- **Training Sequences**: Generated from trajectory segments

**Data Processing:**
- Day-by-day loading with caching
- Preprocessing: Data cleaning, validation, feature engineering
- Sequence creation: 24-hour history windows
- Trajectory segmentation: Handles gaps > 6 hours

### Training Configuration

**Model Architecture:**
- **LSTM**: 2 layers, 128 hidden size, 0.2 dropout
- **Transformer**: 4 layers, 256 hidden size, 8 attention heads, 1024 feedforward
- **Fusion**: 256 hidden size
- **Input Size**: Variable (based on features)
- **Prediction Horizon**: 48 hours (8 time steps)

**Training Parameters:**
- **Batch Size**: 32
- **Initial Learning Rate**: 0.001
- **Optimizer**: Adam
- **Learning Rate Schedule**: ReduceLROnPlateau (reduces on validation plateau)
- **Early Stopping**: Patience of 10 epochs
- **Gradient Clipping**: 1.0
- **Number of Epochs**: 100 (stopped early at epoch 81)

**Loss Function:**
- **Position Loss**: Weight 1.0 (primary)
- **Speed Loss**: Weight 0.5
- **Course Loss**: Weight 0.5
- **Uncertainty Loss**: Weight 0.3
- **Physics Loss**: Weight 0.2

### Training Progress

**Key Training Iterations:**

- **Epoch 1**: Train Loss: 302.61, Val Loss: 71.17 → **Best Model Saved**
- **Epoch 2**: Train Loss: 102.96, Val Loss: 40.92 → **Best Model Saved**
- **Epoch 4**: Train Loss: 81.58, Val Loss: 39.31 → **Best Model Saved**
- **Epoch 5**: Train Loss: 71.11, Val Loss: 28.91 → **Best Model Saved**
- **Epoch 7**: Train Loss: 65.14, Val Loss: 26.03 → **Best Model Saved**
- **Epoch 8**: Train Loss: 57.37, Val Loss: 25.11 → **Best Model Saved**
- **Epoch 10**: Train Loss: 50.xx, Val Loss: 24.xx → **Best Model Saved**
- **Epoch 13**: Train Loss: 45.xx, Val Loss: 23.xx → **Best Model Saved**

**Final Training Results (Epoch 81):**
- **Best Validation Loss**: 15.3570 (achieved at epoch 71)
- **Final Train Loss**: 34.24
- **Final Val Loss**: 15.80
- **Final Learning Rate**: 0.000016 (reduced from initial 0.001)
- **Training Stopped**: Early stopping triggered (no improvement for 10 epochs)

**Validation Metrics (Final Epoch):**
- **MAE (Mean Absolute Error)**: 71.09 nautical miles at 48 hours
- **RMSE (Root Mean Square Error)**: 98.56 nautical miles at 48 hours
- **Position Loss Component**: 1.53
- **Speed Loss Component**: 22.19
- **Course Loss Component**: 0.85
- **Uncertainty Loss Component**: 7.90

### Best Model Characteristics

**File**: `ml_course_prediction/models/trained/best_model.pt`

**Achieved at**: Epoch 71

**Performance:**
- **Validation Loss**: 15.3570 (lowest achieved)
- **Validation MAE**: ~70-73 nautical miles at 48 hours
- **Validation RMSE**: ~98-100 nautical miles at 48 hours

**Model State:**
- All model parameters (LSTM, Transformer, Fusion, Output Heads)
- Optimizer state
- Training metadata
- Best epoch information

**Learning Rate Evolution:**
- **Initial**: 0.001
- **Epoch 76**: 0.000031 (reduced)
- **Epoch 77-81**: 0.000016 (final, reduced further)
- **Final**: 0.000016

### Training Challenges and Solutions

1. **Memory Management**:
   - Challenge: Large dataset (221M records) caused memory issues
   - Solution: Day-by-day processing with caching, sequence batching

2. **Variable Sequence Lengths**:
   - Challenge: AIS reports every 3 hours, sequences vary (2-8 points)
   - Solution: Length-grouped batching, dynamic padding

3. **Data Quality**:
   - Challenge: Missing data, invalid positions, gaps
   - Solution: Preprocessing pipeline with validation and filtering

4. **Training Stability**:
   - Challenge: Loss fluctuations, gradient issues
   - Solution: Gradient clipping, learning rate scheduling, early stopping

### Model Evaluation

**Target Metrics (from config):**
- **MAE 48h**: Target < 50 nm (Achieved: ~70-73 nm)
- **RMSE 48h**: Target < 50 nm (Achieved: ~98-100 nm)
- **Uncertainty Coverage**: Target 68% within 1σ

**Performance Notes:**
- Model achieves reasonable accuracy for 48-hour predictions
- Uncertainty quantification provides confidence bounds
- Performance degrades with prediction horizon (expected)
- Model handles various vessel types and behaviors

---

## Data Flow Diagrams

### Complete System Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA FLOW DIAGRAM                            │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐
│  Raw AIS     │  (CSV/Parquet files, NOAA, S3)
│  Data        │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  SFD_GUI.py - Data Manager                                      │
│  • Download from NOAA                                           │
│  • Load from local files                                        │
│  • Access S3 buckets                                            │
│  • Convert CSV to Parquet                                       │
│  • Cache processed data                                         │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  SFD.py - Main Analysis Engine                                  │
│  • Data validation and cleaning                                 │
│  • Filtering (date, vessel type, geography)                     │
│  • Feature engineering                                          │
│  • Anomaly detection                                            │
│  • Zone violation detection                                     │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ├──────────────────┬──────────────────┬──────────────────┐
       ▼                  ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Anomaly     │  │  Vessel      │  │  Cached      │  │  Output      │
│  Data        │  │  Statistics  │  │  Data        │  │  Reports     │
└──────────────┘  └──────────────┘  └──────┬───────┘  └──────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  advanced_analysis.py - Advanced Analysis                       │
│  • Load cached data                                             │
│  • Statistical analysis                                         │
│  • Clustering                                                   │
│  • Temporal patterns                                            │
│  • Map generation                                               │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ├──────────────────┬──────────────────┐
       ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  ML          │  │  Advanced    │  │  Interactive │
│  Predictions │  │  Reports     │  │  Maps        │
└──────────────┘  └──────────────┘  └──────────────┘
```

### ML Prediction Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│              ML PREDICTION DATA FLOW                            │
└─────────────────────────────────────────────────────────────────┘

Cached AIS Data (from main analysis)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  ml_prediction_integration.py                                   │
│  • MLPredictionIntegrator.prepare_vessel_data()                 │
│    - Filter by MMSI                                             │
│    - Extract last N hours                                       │
│    - Validate data requirements                                 │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Trajectory Processing                                          │
│  • TrajectoryProcessor.segment_trajectories()                   │
│    - Handle gaps > 6 hours                                      │
│    - Select most recent segment                                 │
│    - Validate trajectory quality                                │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Feature Extraction                                             │
│  • FeatureEngineer.create_sequence_features()                   │
│    - Position features (LAT, LON)                               │
│    - Speed features (SOG)                                       │
│    - Course features (COG, Heading)                             │
│    - Temporal features                                          │
│    - Vessel type encoding                                       │
│    - Normalization                                              │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Model Inference                                                │
│  • Load best_model.pt                                           │
│  • Forward pass through hybrid model                            │
│  • Autoregressive prediction (8 steps)                          │
│  • Uncertainty calculation                                      │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Post-Processing                                                │
│  • Format predictions (8 points, 6-hour intervals)              │
│  • Calculate confidence intervals                               │
│  • Apply physics constraints                                    │
└──────┬──────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Visualization                                                  │
│  • Create Folium map                                            │
│  • Plot historical trajectory                                   │
│  • Plot predicted path with uncertainty                         │
│  • Generate prediction report                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Output Descriptions

### Main Analysis Outputs (SFD_GUI.py)

1. **anomaly_summary.csv**
   - **Content**: All detected anomalies with details
   - **Columns**: MMSI, Timestamp, AnomalyType, Details, Severity, etc.
   - **Location**: Output directory

2. **vessel_statistics.csv**
   - **Content**: Per-vessel statistics
   - **Columns**: MMSI, VesselType, TotalRecords, AnomalyCount, AvgSpeed, etc.
   - **Location**: Output directory

3. **summary_report.html**
   - **Content**: Comprehensive HTML report
   - **Sections**: Overview, Statistics, Anomaly Summary, Vessel Summary
   - **Location**: Output directory

4. **Cached Data Files**
   - **Format**: Parquet files
   - **Location**: `.ais.cache_data` directories
   - **Content**: Processed AIS data for advanced analysis

5. **config.ini**
   - **Content**: Updated configuration with run parameters
   - **Location**: Project root

### Advanced Analysis Outputs

1. **Full Dataset Export**
   - **Format**: CSV or Parquet
   - **Content**: Complete processed dataset
   - **Location**: Output directory

2. **Correlation Analysis Report**
   - **Format**: HTML with charts
   - **Content**: Relationships between vessel types and anomalies
   - **Location**: Output directory

3. **Temporal Pattern Analysis**
   - **Format**: HTML with time-series charts
   - **Content**: Time-based patterns in anomalies
   - **Location**: Output directory

4. **Clustering Results**
   - **Format**: CSV and HTML report
   - **Content**: Vessel behavior clusters (K-means)
   - **Location**: Output directory

5. **Interactive Maps**
   - **Format**: HTML (Folium)
   - **Content**: 
     - Full spectrum anomaly map
     - Vessel-specific maps
     - Filtered maps
     - ML prediction maps
   - **Location**: Output directory

6. **ML Prediction Reports**
   - **Format**: HTML map and text report
   - **Content**: 
     - Predicted positions (8 points)
     - Uncertainty bounds
     - Historical trajectory
     - Prediction visualization
   - **Location**: Output directory

---

## Troubleshooting

### Common Issues

1. **"Advanced analysis module not available"**
   - **Solution**: Ensure `advanced_analysis.py` is in the same directory as `SFD_GUI.py`
   - Check that all dependencies are installed

2. **"ML Course Prediction not available"**
   - **Solution**: Install ML dependencies: `pip install -r ml_course_prediction/requirements.txt`
   - Ensure `best_model.pt` exists in `ml_course_prediction/models/trained/`

3. **Memory Errors**
   - **Solution**: Reduce date range, process smaller datasets
   - Close other applications to free memory
   - Use data caching to avoid reloading

4. **GPU Not Detected**
   - **Solution**: Install appropriate PyTorch version (CUDA or ROCm)
   - Verify GPU drivers are installed
   - System will fall back to CPU automatically

5. **Data Loading Failures**
   - **Solution**: Check data file paths in configuration
   - Verify file formats (CSV or Parquet)
   - Check network connection for NOAA/S3 downloads

6. **ML Prediction Errors**
   - **Solution**: Ensure vessel has sufficient data (2+ points, 1+ hour span)
   - Check for data gaps (max 6 hours)
   - Verify trajectory quality

### Getting Help

- Check log files for detailed error messages
- Review configuration file settings
- Verify all dependencies are installed
- Check data file formats and paths

---

## License and Credits

**Version**: 2.1 Beta  
**Team**: Dreadnaught  
**Developers**: Alex Giacomello, Christopher Matherne, Rupert Rigg, Zachary Zhao

---

## Additional Resources

- Configuration file: `config.ini`
- Training logs: `ml_course_prediction/training/training.log`
- Model configuration: `ml_course_prediction/models/configs/default_config.yaml`
- ML module README: `ml_course_prediction/README.md`

---

*Last Updated: Based on training log analysis from 26 November 2025*

