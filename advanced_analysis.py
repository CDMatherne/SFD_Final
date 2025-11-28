#!/usr/bin/env python3
"""
Advanced Analysis Module for SFD Project

[VERSION}
Team = Dreadnaught
Alex Giacomello, Christopher Matherne, Rupert Rigg, Zachary Zhao
version = 2.1 Beta

This module provides advanced analysis capabilities on cached AIS data
from previous analysis runs.
"""

import os
import sys
import glob
import logging
import configparser
import pandas as pd
import numpy as np
import traceback
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
import re
import time
import math
from utils import validate_config
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import traceback
import platform
import subprocess

# Suppress common deprecation warnings from pandas and openpyxl
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='openpyxl')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='openpyxl')

# Import local utility modules
from utils import get_cache_dir, check_dependencies, format_file_size, log_memory_usage

# Set up logging
logger = logging.getLogger("Advanced_Analysis")

# Check for tkcalendar
try:
    from tkcalendar import DateEntry
    TKCALENDAR_AVAILABLE = True
except ImportError:
    TKCALENDAR_AVAILABLE = False


# Try to import optional dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from openpyxl import Workbook
    from openpyxl.chart import BarChart, LineChart, Reference
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

try:
    import folium
    from folium.plugins import MarkerCluster, HeatMap
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    from sklearn.cluster import KMeans  # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try to import ML Course Prediction integration
try:
    from ml_prediction_integration import (
        MLPredictionIntegrator, 
        MLPredictionError, 
        is_available as ml_prediction_available
    )
    ML_PREDICTION_AVAILABLE = ml_prediction_available()
except ImportError as e:
    ML_PREDICTION_AVAILABLE = False
    # Define a fallback exception class if import fails
    class MLPredictionError(Exception):
        """Fallback exception if ML prediction module is not available"""
        pass
    logger.warning(f"ML Course Prediction integration not available: {e}")




# ============================================================================
# HELPER FUNCTIONS AND VALIDATION
# ============================================================================

def open_file(file_path):
    """Open a file with the default application based on the operating system."""
    if not file_path or not os.path.exists(file_path):
        logger.error(f"Cannot open file: {file_path} - File not found")
        return False
        
    try:
        logger.info(f"Opening file: {file_path}")
        if platform.system() == 'Windows':
            os.startfile(file_path)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.call(['open', file_path])
        else:  # Linux
            subprocess.call(['xdg-open', file_path])
        return True
    except Exception as e:
        logger.error(f"Error opening file {file_path}: {e}")
        return False

def get_config_path(config_path='config.ini'):
    """Get the absolute path to config.ini, resolving relative to script directory."""
    if os.path.isabs(config_path):
        return config_path
    
    # Get the script directory
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        script_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try script directory first
    abs_config_path = os.path.join(script_dir, config_path)
    if os.path.exists(abs_config_path):
        return abs_config_path
    
    # Try current working directory as fallback
    if os.path.exists(config_path):
        return os.path.abspath(config_path)
    
    # Return the script directory path even if it doesn't exist (for error messages)
    return abs_config_path




def validate_data_availability(run_info):
    """Validate that required data is available for analysis."""
    warnings = []
    
    output_dir = run_info.get('output_directory')
    if not output_dir:
        return False, "Output directory not specified", warnings
    
    if not os.path.exists(output_dir):
        warnings.append(f"Output directory does not exist: {output_dir}")
        try:
            os.makedirs(output_dir, exist_ok=True)
            warnings.append(f"Created output directory: {output_dir}")
        except Exception as e:
            return False, f"Cannot create output directory: {e}", warnings
    
    cache_dir = get_cache_dir()
    if not os.path.exists(cache_dir):
        warnings.append(f"Cache directory does not exist: {cache_dir}")
    
    try:
        # Extract date range and ship types with safe defaults
        start_date = run_info.get('start_date', '2024-10-15')
        end_date = run_info.get('end_date', '2024-10-17')
        ship_types = run_info.get('ship_types', [])
        
        logger.info(f"Searching for cache files: dates={start_date} to {end_date}, ship types={ship_types}")
        cache_files = find_cache_files_for_date_range(
            start_date,
            end_date,
            ship_types,
            cache_dir
        )
        
        if not cache_files:
            warnings.append("No cached data files found for the specified date range")
            logger.warning("No cache files found for the specified parameters")
        else:
            logger.info(f"Found {len(cache_files)} matching cache files")
    except Exception as e:
        logger.error(f"Error searching for cache files: {e}")
        warnings.append(f"Error locating cache files: {str(e)}")
        cache_files = []
    
    anomaly_summary_path = os.path.join(output_dir, "AIS_Anomalies_Summary.csv")
    if not os.path.exists(anomaly_summary_path):
        warnings.append("Anomaly summary file not found. Some features may be limited.")
    
    return True, None, warnings




def cleanup_large_dataframe(df, keep_columns=None):
    """Clean up a DataFrame to reduce memory usage."""
    if df is None or df.empty:
        return df
    
    original_size = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:
                try:
                    df[col] = df[col].astype('category')
                except:
                    pass
    
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    if keep_columns:
        df = df[keep_columns]
    
    new_size = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info(f"DataFrame memory reduced from {original_size:.2f} MB to {new_size:.2f} MB")
    
    return df




def get_all_vessel_types():
    """Get list of all possible vessel types (20-99) as defined in GUI."""
    return list(range(20, 100))


def get_vessel_type_name(vessel_type):
    """Get the human-readable name for a vessel type number."""
    vessel_type_names = {
        20: 'Wing in ground (WIG), all ships of this type',
        21: 'Wing in ground (WIG), Hazardous category A',
        22: 'Wing in ground (WIG), Hazardous category B',
        23: 'Wing in ground (WIG), Hazardous category C',
        24: 'Wing in ground (WIG), Hazardous category D',
        30: 'Fishing',
        31: 'Towing',
        32: 'Towing: length exceeds 200m or breadth exceeds 25m',
        33: 'Dredging or underwater ops',
        34: 'Diving ops',
        35: 'Military ops',
        36: 'Sailing',
        37: 'Pleasure Craft',
        38: 'Reserved',
        39: 'Reserved',
        40: 'High speed craft (HSC), all ships of this type',
        41: 'High speed craft (HSC), Hazardous category A',
        42: 'High speed craft (HSC), Hazardous category B',
        43: 'High speed craft (HSC), Hazardous category C',
        44: 'High speed craft (HSC), Hazardous category D',
        45: 'High speed craft (HSC), Reserved',
        46: 'High speed craft (HSC), Reserved',
        47: 'High speed craft (HSC), Reserved',
        48: 'High speed craft (HSC), Reserved',
        49: 'High speed craft (HSC), No additional information',
        50: 'Pilot Vessel',
        51: 'Search and Rescue vessel',
        52: 'Tug',
        53: 'Port Tender',
        54: 'Anti-pollution equipment',
        55: 'Law Enforcement',
        56: 'Spare - Local Vessel',
        57: 'Spare - Local Vessel',
        58: 'Medical Transport',
        59: 'Noncombatant ship (RR Resolution No. 18)',
        60: 'Passenger, all ships of this type',
        61: 'Passenger, Hazardous category A',
        62: 'Passenger, Hazardous category B',
        63: 'Passenger, Hazardous category C',
        64: 'Passenger, Hazardous category D',
        69: 'Passenger, No additional information',
        70: 'Cargo, all ships of this type',
        71: 'Cargo, Hazardous category A',
        72: 'Cargo, Hazardous category B',
        73: 'Cargo, Hazardous category C',
        74: 'Cargo, Hazardous category D',
        79: 'Cargo, No additional information',
        80: 'Tanker, all ships of this type',
        81: 'Tanker, Hazardous category A',
        82: 'Tanker, Hazardous category B',
        83: 'Tanker, Hazardous category C',
        84: 'Tanker, Hazardous category D',
        89: 'Tanker, No additional information',
        90: 'Other, all ships of this type',
        91: 'Other, Hazardous category A',
        92: 'Other, Hazardous category B',
        93: 'Other, Hazardous category C',
        94: 'Other, Hazardous category D',
        99: 'Other, No additional information'
    }
    return vessel_type_names.get(int(vessel_type), f'Vessel Type {int(vessel_type)}')


def get_all_anomaly_types():
    """Get list of all possible anomaly types as defined in GUI."""
    return [
        "AIS Beacon Off",
        "AIS Beacon On",
        "Excessive Travel Distance (Fast)",
        "Excessive Travel Distance (Slow)",
        "Course over Ground-Heading Inconsistency",
        "Loitering",
        "Rendezvous",
        "Identity Spoofing",
        "Zone Violations"
    ]


def map_anomaly_type_gui_to_data(gui_name):
    """Map GUI anomaly type name to data anomaly type name."""
    mapping = {
        "AIS Beacon Off": "AIS_Beacon_Off",
        "AIS Beacon On": "AIS_Beacon_On",
        "Excessive Travel Distance (Fast)": "Speed",
        "Excessive Travel Distance (Slow)": "Speed",
        "Course over Ground-Heading Inconsistency": "Course",
        "Loitering": "Loitering",
        "Rendezvous": "Rendezvous",
        "Identity Spoofing": "Identity_Spoofing",
        "Zone Violations": "Zone_Violation"
    }
    return mapping.get(gui_name, gui_name)


def map_anomaly_type_data_to_gui(data_name):
    """Map data anomaly type name to GUI anomaly type name."""
    mapping = {
        "AIS_Beacon_Off": "AIS Beacon Off",
        "AIS_Beacon_On": "AIS Beacon On",
        "Speed": "Excessive Travel Distance (Fast)",
        "Course": "Course over Ground-Heading Inconsistency",
        "Loitering": "Loitering",
        "Rendezvous": "Rendezvous",
        "Identity_Spoofing": "Identity Spoofing",
        "Zone_Violation": "Zone Violations"
    }
    return mapping.get(data_name, data_name)


def get_available_vessel_types(df):
    """Get list of available vessel types from data."""
    if df.empty or 'VesselType' not in df.columns:
        return []
    return sorted(df['VesselType'].dropna().unique().tolist())


def get_available_anomaly_types(anomaly_df):
    """Get list of available anomaly types from data."""
    if anomaly_df.empty or 'AnomalyType' not in anomaly_df.columns:
        return []
    return sorted(anomaly_df['AnomalyType'].dropna().unique().tolist())


def validate_mmsi(mmsi_value):
    """Validate MMSI value."""
    try:
        mmsi = int(mmsi_value)
        if mmsi < 100000000 or mmsi > 999999999:
            return False, None, "MMSI must be a 9-digit number"
        return True, mmsi, None
    except (ValueError, TypeError):
        return False, None, "MMSI must be a valid number"


def validate_date_range(start_date, end_date):
    """Validate date range."""
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start > end:
            return False, "Start date must be before end date"
        
        if (end - start).days > 365:
            return False, "Date range cannot exceed 365 days"
        
        return True, None
    except ValueError as e:
        return False, f"Invalid date format: {e}"


def get_dependency_warnings():
    """Get warnings about missing optional dependencies."""
    warnings = []
    
    # Initialize our own dependency check since utils.check_dependencies() returns a boolean
    deps = {}
    optional_deps = {
        'plotly': 'Interactive visualizations will use basic HTML tables',
        'folium': 'Map generation will not be available',
        'sklearn': 'Vessel clustering will not be available',
        'openpyxl': 'Excel export will use CSV format instead',
    }
    
    # Check each dependency
    for dep in optional_deps.keys():
        try:
            __import__(dep)
            deps[dep] = True
        except ImportError:
            deps[dep] = False
            warnings.append(f"{dep}: {optional_deps[dep]}")
    
    return warnings


# ============================================================================
# DATA ACCESS LAYER
# ============================================================================

def get_last_run_info(config_path='config.ini'):
    """Extract information about the last analysis run from config.ini."""
    # Resolve config path relative to script directory
    config_path = get_config_path(config_path)
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return None
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    def get_config_value(section, key, fallback=None, value_type='str'):
        try:
            if value_type == 'float':
                return config.getfloat(section, key, fallback=fallback)
            elif value_type == 'int':
                return config.getint(section, key, fallback=fallback)
            elif value_type == 'boolean':
                return config.getboolean(section, key, fallback=fallback)
            else:
                return config.get(section, key, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    
    # First try to get from config, then use a fallback that's more likely to be found by run_advanced_analysis.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.join(script_dir, 'output')
    
    output_dir = get_config_value('DEFAULT', 'output_directory', 
                                fallback=get_config_value('Paths', 'output_directory', 
                                                        fallback=default_output))
    
    # Try START_DATE/END_DATE first (as saved by SFD_GUI.py), then start_date/end_date, then fallback
    start_date = get_config_value('DEFAULT', 'START_DATE', 
                                 fallback=get_config_value('DEFAULT', 'start_date', fallback='2024-10-01'))
    end_date = get_config_value('DEFAULT', 'END_DATE',
                               fallback=get_config_value('DEFAULT', 'end_date', fallback='2024-10-03'))
    
    data_dir = get_config_value('DEFAULT', 'data_directory',
                               fallback=get_config_value('Paths', 'data_directory', fallback=''))
    
    ship_types_str = get_config_value('SHIP_FILTERS', 'selected_ship_types', fallback='')
    if ship_types_str:
        try:
            ship_types = [int(t.strip()) for t in ship_types_str.split(',') if t.strip()]
        except ValueError:
            ship_types = []
    else:
        ship_types = []
    
    anomaly_types = {
        'ais_beacon_off': get_config_value('ANOMALY_TYPES', 'ais_beacon_off', fallback=False, value_type='boolean'),
        'ais_beacon_on': get_config_value('ANOMALY_TYPES', 'ais_beacon_on', fallback=False, value_type='boolean'),
        'excessive_travel_distance_fast': get_config_value('ANOMALY_TYPES', 'excessive_travel_distance_fast', fallback=False, value_type='boolean'),
        'excessive_travel_distance_slow': get_config_value('ANOMALY_TYPES', 'excessive_travel_distance_slow', fallback=False, value_type='boolean'),
        'cog-heading_inconsistency': get_config_value('ANOMALY_TYPES', 'cog-heading_inconsistency', fallback=False, value_type='boolean'),
        'loitering': get_config_value('ANOMALY_TYPES', 'loitering', fallback=False, value_type='boolean'),
        'rendezvous': get_config_value('ANOMALY_TYPES', 'rendezvous', fallback=False, value_type='boolean'),
        'identity_spoofing': get_config_value('ANOMALY_TYPES', 'identity_spoofing', fallback=False, value_type='boolean'),
        'zone_violations': get_config_value('ANOMALY_TYPES', 'zone_violations', fallback=False, value_type='boolean'),
    }
    
    return {
        'output_directory': output_dir,
        'start_date': start_date,
        'end_date': end_date,
        'ship_types': ship_types,
        'anomaly_types': anomaly_types,
        'data_directory': data_dir
    }


def find_cache_files_for_date_range(start_date, end_date, ship_types, cache_dir=None):
    """Find all cache files that match the date range and ship types."""
    if cache_dir is None:
        cache_dir = get_cache_dir()
    
    if not os.path.exists(cache_dir):
        logger.warning(f"Cache directory does not exist: {cache_dir}")
        return []
    
    # Format dates for subfolder name (YYYYMMDD-YYYYMMDD)
    try:
        # Ensure we have valid date strings
        if not start_date or not isinstance(start_date, str):
            logger.error(f"Invalid start_date: {start_date}")
            start_date = '2024-10-15'  # Use a default date
            logger.info(f"Using default start_date: {start_date}")
            
        if not end_date or not isinstance(end_date, str):
            logger.error(f"Invalid end_date: {end_date}")
            end_date = '2024-10-17'  # Use a default date
            logger.info(f"Using default end_date: {end_date}")
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_fmt = start_dt.strftime('%Y%m%d')
        end_fmt = end_dt.strftime('%Y%m%d')
        date_subfolder = f"{start_fmt}-{end_fmt}"
        
        # Check if date-specific subfolder exists
        date_cache_dir = os.path.join(cache_dir, date_subfolder)
        if os.path.exists(date_cache_dir):
            logger.info(f"Found date-specific cache subfolder: {date_cache_dir}")
            cache_files = glob.glob(os.path.join(date_cache_dir, "*.parquet"))
        else:
            # If subfolder doesn't exist, fall back to the main directory
            logger.info(f"Date-specific subfolder not found, searching in main cache directory")
            cache_files = glob.glob(os.path.join(cache_dir, "*.parquet"))
    except Exception as e:
        logger.warning(f"Error finding date subfolder: {e}")
        cache_files = glob.glob(os.path.join(cache_dir, "*.parquet"))
    
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        return []
    
    matching_files = []
    
    for cache_file in cache_files:
        try:
            # Using head() instead of nrows parameter which isn't supported by pd.read_parquet
            df_sample = pd.read_parquet(cache_file).head(1)
            
            if 'MMSI' not in df_sample.columns or 'BaseDateTime' not in df_sample.columns:
                continue
            
            # Read the full file only if the sample check passes
            df_check = pd.read_parquet(cache_file)
            if 'BaseDateTime' in df_check.columns:
                # Ensure datetime format for comparison
                df_check['BaseDateTime'] = pd.to_datetime(df_check['BaseDateTime'])
                file_dates = df_check['BaseDateTime'].dt.date.unique()
                
                # Check if file contains data within the date range
                date_match = any(start_dt <= d <= end_dt for d in file_dates)
                
                # Check for vessel type match if ship_types are specified
                vessel_type_match = True  # Default to True if no ship_types specified
                if ship_types and len(ship_types) > 0:
                    # Check both VesselType and MainVesselType columns
                    if 'MainVesselType' in df_check.columns:
                        # If MainVesselType exists, it's likely a post-analysis file
                        vessel_types = df_check['MainVesselType'].unique()
                        vessel_type_match = any(int(vt) in [int(st) for st in ship_types] for vt in vessel_types if not pd.isna(vt))
                        logger.debug(f"File {os.path.basename(cache_file)} has MainVesselTypes: {vessel_types}, match: {vessel_type_match}")
                    elif 'VesselType' in df_check.columns:
                        # Check regular VesselType column
                        vessel_types = df_check['VesselType'].unique()
                        vessel_type_match = any(int(vt) in [int(st) for st in ship_types] for vt in vessel_types if not pd.isna(vt))
                        logger.debug(f"File {os.path.basename(cache_file)} has VesselTypes: {vessel_types}, match: {vessel_type_match}")
                    else:
                        # No vessel type column found
                        vessel_type_match = False
                        logger.debug(f"File {os.path.basename(cache_file)} has no vessel type columns")
                
                if date_match and vessel_type_match:
                    matching_files.append(cache_file)
                    logger.info(f"File matches criteria: {os.path.basename(cache_file)}")
                else:
                    logger.debug(f"File does not match criteria: {os.path.basename(cache_file)}, date_match: {date_match}, vessel_type_match: {vessel_type_match}")
                    
            
        except Exception as e:
            logger.debug(f"Error checking cache file {cache_file}: {e}")
            continue
    
    logger.info(f"Found {len(matching_files)} matching cache files for date range {start_date} to {end_date}")
    
    if matching_files:
        logger.info(f"Matching cache files: {[os.path.basename(f) for f in matching_files]}")
    else:
        logger.warning(f"No cache files found matching the date range {start_date} to {end_date} for selected vessel types")
        if 'date_cache_dir' in locals() and os.path.exists(date_cache_dir):
            logger.info(f"Date subfolder exists at: {date_cache_dir}")
            logger.info(f"Files in date subfolder: {os.listdir(date_cache_dir) if os.path.exists(date_cache_dir) else 'none'}")
    
    return matching_files


def load_cached_data_for_date_range(start_date, end_date, ship_types, cache_dir=None):
    """Load all cached parquet files matching the date range and ship types."""
    cache_files = find_cache_files_for_date_range(start_date, end_date, ship_types, cache_dir)
    
    if not cache_files:
        logger.warning("No matching cache files found")
        return pd.DataFrame()
    
    dataframes = []
    for cache_file in cache_files:
        try:
            df = pd.read_parquet(cache_file)
            
            if ship_types and 'VesselType' in df.columns:
                df = df[df['VesselType'].isin(ship_types)]
            
            if 'BaseDateTime' in df.columns:
                df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
                df = df[(df['BaseDateTime'] >= start_dt) & (df['BaseDateTime'] < end_dt)]
            
            if not df.empty:
                dataframes.append(df)
                logger.info(f"Loaded {len(df)} records from {os.path.basename(cache_file)}")
        except Exception as e:
            logger.error(f"Error loading cache file {cache_file}: {e}")
            continue
    
    if not dataframes:
        logger.warning("No data loaded from cache files")
        return pd.DataFrame()
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Total records loaded: {len(combined_df)}")
    return combined_df


def load_anomaly_summary(output_dir):
    """Load the AIS_Anomalies_Summary.csv file if it exists."""
    # First, try the given output directory
    summary_path = os.path.join(output_dir, "AIS_Anomalies_Summary.csv")
    
    # If not found, try alternative locations
    if not os.path.exists(summary_path):
        logger.warning(f"Anomaly summary file not found: {summary_path}")
        
        # Try looking in C:/AIS_Data/Output directory
        alt_path = "C:/AIS_Data/Output/AIS_Anomalies_Summary.csv"
        if os.path.exists(alt_path):
            logger.info(f"Found anomaly summary file in alternate location: {alt_path}")
            summary_path = alt_path
        else:
            # Try looking in the script directory's output folder
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.join(script_dir, "output", "AIS_Anomalies_Summary.csv")
            if os.path.exists(alt_path):
                logger.info(f"Found anomaly summary file in script output directory: {alt_path}")
                summary_path = alt_path
    
    if not os.path.exists(summary_path):
        logger.warning(f"Anomaly summary file not found in any location")
        # Create an empty dataframe with the expected columns
        columns = ['MMSI', 'VesselName', 'BaseDateTime', 'AnomalyType', 'Confidence', 
                  'LAT', 'LON', 'Description', 'Severity']
        return pd.DataFrame(columns=columns)
    
    try:
        df = pd.read_csv(summary_path)
        logger.info(f"Loaded {len(df)} anomalies from summary file")
        return df
    except Exception as e:
        logger.error(f"Error loading anomaly summary: {e}")
        # Create an empty dataframe with the expected columns on error
        columns = ['MMSI', 'VesselName', 'BaseDateTime', 'AnomalyType', 'Confidence', 
                   'LAT', 'LON', 'Description', 'Severity']
        return pd.DataFrame(columns=columns)


# ============================================================================
# ADVANCED ANALYSIS CLASS
# ============================================================================

class AdvancedAnalysis:
    """Main class for Advanced Analysis functionality."""
    
    def __init__(self, parent_window, output_directory=None, config_path='config.ini'):
        """Initialize Advanced Analysis with enhanced validation."""
        self.parent_window = parent_window
        # Resolve config path relative to script directory
        self.config_path = get_config_path(config_path)
        
        is_valid, error_msg = validate_config(self.config_path)
        if not is_valid:
            raise ValueError(f"Config validation failed: {error_msg}")
        
        self.run_info = get_last_run_info(self.config_path)
        
        if self.run_info is None:
            raise ValueError("Could not load last run information from config.ini")
        
        self.output_directory = output_directory or self.run_info['output_directory']
        os.makedirs(self.output_directory, exist_ok=True)
        
        # Validate data availability with enhanced error handling
        is_valid, error_msg, warnings = validate_data_availability(self.run_info)
        if not is_valid:
            logger.warning(f"Data validation warning: {error_msg}")
            # Don't raise an exception - continue with limited functionality
        
        for warning in warnings:
            logger.warning(warning)
        
        # Create cache directory if it doesn't exist
        cache_dir = get_cache_dir()
        if not os.path.exists(cache_dir):
            try:
                os.makedirs(cache_dir, exist_ok=True)
                logger.info(f"Created missing cache directory: {cache_dir}")
            except Exception as e:
                logger.error(f"Failed to create cache directory: {e}")
        
        dep_warnings = get_dependency_warnings()
        for warning in dep_warnings:
            logger.warning(f"Dependency: {warning}")
        
        self._cached_data = None
        self._anomaly_data = None
        
        logger.info(f"Advanced Analysis initialized with output directory: {self.output_directory}")
        log_memory_usage("after initialization")
    
    def get_map_output_directory(self):
        """
        Get the directory for saving maps. Maps should go to Path_Maps subdirectory.
        Uses either hardcoded path "C:\\AIS_Data\\Output\\Path_Maps" or 
        output_directory from config + "Path_Maps".
        
        Returns:
            Path object for the map output directory
        """
        from pathlib import Path
        
        # Try hardcoded path first
        hardcoded_path = Path("C:\\AIS_Data\\Output\\Path_Maps")
        hardcoded_parent = Path("C:\\AIS_Data\\Output")
        
        # Use hardcoded path if parent directory exists (we can create Path_Maps)
        if hardcoded_parent.exists():
            map_dir = hardcoded_path
        else:
            # Use output_directory from config + Path_Maps subdirectory
            map_dir = Path(self.output_directory) / "Path_Maps"
        
        # Create directory if it doesn't exist
        map_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Map output directory: {map_dir}")
        return map_dir
    
    def load_cached_data(self, force_reload=False):
        """Load cached data from the last run with enhanced error handling."""
        if self._cached_data is not None and not force_reload:
            return self._cached_data
        
        logger.info("Loading cached data...")
        log_memory_usage("before loading cached data")
    
        try:
            # Get information about what we're looking for with robust error handling
            cache_dir = get_cache_dir()
            if not cache_dir:
                logger.warning("Failed to get cache directory, using fallback")
                cache_dir = os.path.expanduser("~/.ais_data_cache")
                os.makedirs(cache_dir, exist_ok=True)
                
            # Safely extract run info with defaults
            start_date = self.run_info.get('start_date', '2024-10-15')
            end_date = self.run_info.get('end_date', '2024-10-17')
            ship_types = self.run_info.get('ship_types', [])
            
            logger.info(f"Searching for cached data with: date range={start_date} to {end_date}, vessel types={ship_types}")
        
            # First try to find the consolidated dataframe
            consolidated_found = False
            
            # Check if date-specific subfolder might exist
            try:
                start_fmt = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')
                end_fmt = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d')
                date_subfolder = f"{start_fmt}-{end_fmt}"
                date_cache_dir = os.path.join(cache_dir, date_subfolder)
                consolidated_path = os.path.join(date_cache_dir, "consolidated_data.parquet")
                
                # Try to load from the date subfolder first
                if os.path.exists(consolidated_path):
                    logger.info(f"Found consolidated dataframe at: {consolidated_path}")
                    self._cached_data = pd.read_parquet(consolidated_path)
                    
                    # If ship_types are specified, filter the dataframe
                    if ship_types and len(ship_types) > 0:
                        # Check if any selected type is in the 30-39 or 50-59 ranges (requires exact matching)
                        has_types_30_39 = any(30 <= int(st) <= 39 for st in ship_types)
                        has_types_50_59 = any(50 <= int(st) <= 59 for st in ship_types)
                        requires_exact_matching = has_types_30_39 or has_types_50_59
                        
                        if requires_exact_matching and 'VesselType' in self._cached_data.columns:
                            # For types 30-39 and 50-59, always use exact VesselType matching
                            range_desc = []
                            if has_types_30_39:
                                range_desc.append("30-39")
                            if has_types_50_59:
                                range_desc.append("50-59")
                            before_filter = len(self._cached_data)
                            self._cached_data = self._cached_data[self._cached_data['VesselType'].isin([int(st) for st in ship_types])]
                            logger.info(f"Filtered consolidated data by exact VesselType ({', '.join(range_desc)} ranges): {before_filter} -> {len(self._cached_data)} records")
                        elif 'MainVesselType' in self._cached_data.columns:
                            # Filter by MainVesselType if available (for non-30-39/50-59 types)
                            before_filter = len(self._cached_data)
                            self._cached_data = self._cached_data[self._cached_data['MainVesselType'].isin([int(st) for st in ship_types])]
                            logger.info(f"Filtered consolidated data by MainVesselType: {before_filter} -> {len(self._cached_data)} records")
                        elif 'VesselType' in self._cached_data.columns:
                            # Calculate MainVesselType and filter (for non-30-39/50-59 types)
                            self._cached_data['MainVesselType'] = (self._cached_data['VesselType'] // 10).astype(int) * 10
                            before_filter = len(self._cached_data)
                            self._cached_data = self._cached_data[self._cached_data['MainVesselType'].isin([int(st) for st in ship_types])]
                            logger.info(f"Filtered consolidated data by calculated MainVesselType: {before_filter} -> {len(self._cached_data)} records")
                
                # Only consider the consolidated data useful if it has records after filtering
                if not self._cached_data.empty:
                    consolidated_found = True
                    logger.info(f"Successfully loaded {len(self._cached_data)} records from consolidated dataframe")
                    return self._cached_data
                else:
                    # Check root cache directory
                    consolidated_path = os.path.join(cache_dir, "consolidated_data.parquet")
                    if os.path.exists(consolidated_path):
                        logger.info(f"Found consolidated dataframe in root cache: {consolidated_path}")
                        self._cached_data = pd.read_parquet(consolidated_path)
                        
                        # Filter by vessel type if needed
                        if ship_types and len(ship_types) > 0:
                            # Check if any selected type is in the 30-39 or 50-59 ranges (requires exact matching)
                            has_types_30_39 = any(30 <= int(st) <= 39 for st in ship_types)
                            has_types_50_59 = any(50 <= int(st) <= 59 for st in ship_types)
                            requires_exact_matching = has_types_30_39 or has_types_50_59
                            
                            if requires_exact_matching and 'VesselType' in self._cached_data.columns:
                                # For types 30-39 and 50-59, always use exact VesselType matching
                                range_desc = []
                                if has_types_30_39:
                                    range_desc.append("30-39")
                                if has_types_50_59:
                                    range_desc.append("50-59")
                                before_filter = len(self._cached_data)
                                self._cached_data = self._cached_data[self._cached_data['VesselType'].isin([int(st) for st in ship_types])]
                                logger.info(f"Filtered consolidated data by exact VesselType ({', '.join(range_desc)} ranges): {before_filter} -> {len(self._cached_data)} records")
                            elif 'MainVesselType' in self._cached_data.columns:
                                # Filter by MainVesselType if available (for non-30-39/50-59 types)
                                before_filter = len(self._cached_data)
                                self._cached_data = self._cached_data[self._cached_data['MainVesselType'].isin([int(st) for st in ship_types])]
                                logger.info(f"Filtered consolidated data by MainVesselType: {before_filter} -> {len(self._cached_data)} records")
                            elif 'VesselType' in self._cached_data.columns:
                                # Calculate MainVesselType and filter (for non-30-39/50-59 types)
                                self._cached_data['MainVesselType'] = (self._cached_data['VesselType'] // 10).astype(int) * 10
                                before_filter = len(self._cached_data)
                                self._cached_data = self._cached_data[self._cached_data['MainVesselType'].isin([int(st) for st in ship_types])]
                                logger.info(f"Filtered consolidated data by calculated MainVesselType: {before_filter} -> {len(self._cached_data)} records")
                        
                        if not self._cached_data.empty:
                            consolidated_found = True
                            logger.info(f"Successfully loaded {len(self._cached_data)} records from consolidated dataframe")
                            return self._cached_data
                    else:
                        logger.info("No consolidated dataframe found, will try individual cache files")
            except Exception as e:
                logger.warning(f"Error checking for consolidated dataframe: {e}")
                logger.warning("Falling back to individual cache files")
                
            # If we didn't find a consolidated dataframe or it was empty after filtering,
            # fall back to the original method of loading individual cache files
            if not consolidated_found:
                # Find matching cache files
                cache_files = find_cache_files_for_date_range(
                    start_date,
                    end_date,
                    ship_types,
                    cache_dir
                )
                
                if not cache_files:
                    logger.warning(f"No matching cache files found for date range {self.run_info['start_date']} to {self.run_info['end_date']}")
                    logger.info(f"The cache directory is: {cache_dir}")
                    logger.info("Please ensure data files exist for the specified date range and ship types")
                    self._cached_data = pd.DataFrame()
                    return self._cached_data
                
                # Load and concatenate all the cache files
                dataframes = []
                for file_path in cache_files:
                    try:
                        df = pd.read_parquet(file_path)
                        if df is not None and not df.empty:
                            dataframes.append(df)
                            logger.info(f"Loaded {len(df)} records from {os.path.basename(file_path)}")
                    except Exception as e:
                        logger.error(f"Error loading cache file {file_path}: {e}")
                
                if not dataframes:
                    logger.warning("No data could be loaded from cache files")
                    # Create an empty dataframe with expected columns to prevent downstream errors
                    columns = ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG', 'Heading', 'VesselType', 'Status',
                               'VesselName', 'IMO', 'CallSign', 'Length', 'Width', 'Draft', 'Flag']
                    self._cached_data = pd.DataFrame(columns=columns)
                    
                    # Show a warning message to the user
                    try:
                        if self.parent_window:
                            messagebox.showwarning("No Data Found", 
                                                 "No AIS data was found matching the search criteria.\n\n"
                                                 "This may occur if:\n"
                                                 "- The cache directory is empty\n"
                                                 "- No data exists for the selected date range\n"
                                                 "- No data exists for the selected vessel types\n\n"
                                                 "The application will continue with limited functionality.")
                    except Exception as e:
                        logger.error(f"Failed to show warning dialog: {e}")
                else:
                    # Combine all the dataframes
                    self._cached_data = pd.concat(dataframes, ignore_index=True)
                    logger.info(f"Combined {len(self._cached_data)} records from {len(dataframes)} cache files")
                    
                    # Remove duplicates if any
                    if 'MMSI' in self._cached_data.columns and 'BaseDateTime' in self._cached_data.columns:
                        before_dedup = len(self._cached_data)
                        self._cached_data = self._cached_data.drop_duplicates(subset=['MMSI', 'BaseDateTime'])
                        if before_dedup > len(self._cached_data):
                            logger.info(f"Removed {before_dedup - len(self._cached_data)} duplicate records")
                        
            return self._cached_data
        
        except Exception as e:
            logger.error(f"Error loading cached data: {e}")
            logger.error(traceback.format_exc())
            self._cached_data = pd.DataFrame()
            return self._cached_data

    def load_full_daily_datasets(self, start_date=None, end_date=None):
        """
        Load full daily datasets from cache directory for ML prediction.
        This loads the raw daily parquet files, not the filtered/consolidated data.
        
        Args:
            start_date: Start date string (YYYY-MM-DD). If None, uses run_info start_date.
            end_date: End date string (YYYY-MM-DD). If None, uses run_info end_date.
            
        Returns:
            DataFrame with all daily data for the date range
        """
        logger.info("Loading full daily datasets from cache for ML prediction...")
        
        try:
            cache_dir = get_cache_dir()
            if not cache_dir:
                logger.warning("Failed to get cache directory, using fallback")
                cache_dir = os.path.expanduser("~/.ais_data_cache")
                os.makedirs(cache_dir, exist_ok=True)
            
            # Get date range from run_info if not provided
            if start_date is None:
                start_date = self.run_info.get('start_date', '2024-10-01')
            if end_date is None:
                end_date = self.run_info.get('end_date', '2024-10-03')
            
            logger.info(f"Loading full daily datasets for date range: {start_date} to {end_date}")
            
            # Check date-specific subfolder first
            start_fmt = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')
            end_fmt = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d')
            date_subfolder = f"{start_fmt}-{end_fmt}"
            date_cache_dir = os.path.join(cache_dir, date_subfolder)
            
            dataframes = []
            
            if os.path.exists(date_cache_dir):
                # Load all parquet files from date subfolder, excluding consolidated_data.parquet
                logger.info(f"Loading daily datasets from: {date_cache_dir}")
                for filename in os.listdir(date_cache_dir):
                    if filename == "consolidated_data.parquet":
                        continue  # Skip consolidated data
                    
                    if filename.endswith('.parquet'):
                        file_path = os.path.join(date_cache_dir, filename)
                        try:
                            df = pd.read_parquet(file_path)
                            
                            # Filter by date range if BaseDateTime exists
                            if 'BaseDateTime' in df.columns:
                                df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], errors='coerce')
                                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                                end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
                                df = df[(df['BaseDateTime'] >= start_dt) & (df['BaseDateTime'] < end_dt)]
                            
                            if not df.empty:
                                dataframes.append(df)
                                logger.info(f"Loaded {len(df)} records from {filename}")
                        except Exception as e:
                            logger.warning(f"Error loading {filename}: {e}")
                            continue
            
            # If no data found in date subfolder, try root cache directory
            if not dataframes:
                logger.info(f"No data found in date subfolder, checking root cache directory: {cache_dir}")
                for filename in os.listdir(cache_dir):
                    if filename == "consolidated_data.parquet":
                        continue
                    
                    if filename.endswith('.parquet'):
                        file_path = os.path.join(cache_dir, filename)
                        try:
                            df = pd.read_parquet(file_path)
                            
                            # Filter by date range if BaseDateTime exists
                            if 'BaseDateTime' in df.columns:
                                df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], errors='coerce')
                                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                                end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
                                df = df[(df['BaseDateTime'] >= start_dt) & (df['BaseDateTime'] < end_dt)]
                            
                            if not df.empty:
                                dataframes.append(df)
                                logger.info(f"Loaded {len(df)} records from {filename}")
                        except Exception as e:
                            logger.warning(f"Error loading {filename}: {e}")
                            continue
            
            if not dataframes:
                logger.warning("No daily datasets found in cache directory")
                return pd.DataFrame()
            
            # Combine all dataframes
            combined_df = pd.concat(dataframes, ignore_index=True)
            
            # Remove duplicates if any
            if 'BaseDateTime' in combined_df.columns and 'MMSI' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['MMSI', 'BaseDateTime'], keep='first')
            
            logger.info(f"Total records loaded from full daily datasets: {len(combined_df)}")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error loading full daily datasets: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def load_anomaly_data(self, force_reload=False):
        """Load anomaly summary data."""
        if self._anomaly_data is not None and not force_reload:
            return self._anomaly_data
        
        logger.info("Loading anomaly data...")
        self._anomaly_data = load_anomaly_summary(self.output_directory)
        return self._anomaly_data
    
    # ========================================================================
    # TAB 1: ADDITIONAL OUTPUTS
    # ========================================================================
    
    def export_full_dataset(self, output_path=None, chunk_size=100000):
        """Export the complete analysis dataset to CSV format."""
        try:
            logger.info("Exporting full dataset to CSV...")
            log_memory_usage("before export")
            
            df = self.load_cached_data()
            
            if df.empty:
                logger.warning("No cached data available to export")
                
                # Try to generate a fallback export with date range info
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                if output_path is None:
                    output_path = os.path.join(self.output_directory, f"Dataset_Info_{timestamp}.csv")
                
                # Create a simple info file with the date range and settings used
                info_df = pd.DataFrame([
                    {"Parameter": "Start Date", "Value": self.run_info.get('start_date', 'N/A')},
                    {"Parameter": "End Date", "Value": self.run_info.get('end_date', 'N/A')},
                    {"Parameter": "Selected Ship Types", "Value": str(self.run_info.get('ship_types', []))},
                    {"Parameter": "Output Directory", "Value": self.output_directory},
                    {"Parameter": "Data Directory", "Value": self.run_info.get('data_directory', 'N/A')},
                    {"Parameter": "Export Time", "Value": datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                    {"Parameter": "Status", "Value": "No cached data found for specified date range"}
                ])
                
                info_df.to_csv(output_path, index=False)
                logger.info(f"Dataset info exported to: {output_path}")
                return output_path
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(self.output_directory, f"Full_Dataset_{timestamp}.csv")
            
            if len(df) > chunk_size:
                logger.info(f"Large dataset detected ({len(df)} records). Writing in chunks...")
                # Write header first
                df.head(0).to_csv(output_path, index=False, mode='w')
                # Write data in chunks
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i:i+chunk_size]
                    chunk.to_csv(output_path, index=False, mode='a', header=False)
                    logger.info(f"Written {min(i+chunk_size, len(df))}/{len(df)} records...")
            else:
                df.to_csv(output_path, index=False)
            
            file_size = os.path.getsize(output_path)
            logger.info(f"Full dataset exported to: {output_path} ({format_file_size(file_size)})")
            log_memory_usage("after export")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting full dataset: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def generate_summary_report(self, output_path=None):
        """Create a summary report with key findings and statistics."""
        try:
            logger.info("Generating summary report...")
            
            df = self.load_cached_data()
            anomaly_df = self.load_anomaly_data()
            
            if df.empty:
                logger.warning("No cached data available for detailed summary report")
                
                # Generate a basic report with available information
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                if output_path is None:
                    output_path = os.path.join(self.output_directory, f"Summary_Report_{timestamp}.html")
                
                report_lines = []
                report_lines.append("<html><head><title>AIS Analysis Summary Report</title>")
                report_lines.append("<style>body { font-family: Arial, sans-serif; margin: 20px; }")
                report_lines.append("h1 { color: #003366; } h2 { color: #0066cc; }")
                report_lines.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
                report_lines.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
                report_lines.append("th { background-color: #003366; color: white; }</style></head><body>")
                
                report_lines.append("<h1>AIS Shipping Fraud Detection - Summary Report</h1>")
                report_lines.append(f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
                report_lines.append(f"<p><strong>Analysis Period:</strong> {self.run_info['start_date']} to {self.run_info['end_date']}</p>")
                
                report_lines.append("<h2>Data Status</h2>")
                report_lines.append("<p><strong>Note:</strong> No cached data was found for the specified date range. ")
                report_lines.append("This report contains limited information based on available anomaly data.</p>")
                report_lines.append(f"<p>Cache directory checked: {get_cache_dir()}</p>")
                
                # Include anomaly data if available
                if not anomaly_df.empty:
                    report_lines.append("<h2>Anomaly Summary</h2>")
                    report_lines.append("<table>")
                    report_lines.append(f"<tr><th>Metric</th><th>Value</th></tr>")
                    report_lines.append(f"<tr><td>Total Anomalies Detected</td><td>{len(anomaly_df):,}</td></tr>")
                    
                    if 'AnomalyType' in anomaly_df.columns:
                        anomaly_counts = anomaly_df['AnomalyType'].value_counts()
                        report_lines.append("<tr><td colspan='2'><strong>Anomalies by Type:</strong></td></tr>")
                        for anomaly_type, count in anomaly_counts.items():
                            report_lines.append(f"<tr><td>&nbsp;&nbsp;{anomaly_type}</td><td>{count:,}</td></tr>")
                    
                    if 'MMSI' in anomaly_df.columns:
                        vessels_with_anomalies = anomaly_df['MMSI'].nunique()
                        report_lines.append(f"<tr><td>Vessels with Anomalies</td><td>{vessels_with_anomalies:,}</td></tr>")
                    
                    report_lines.append("</table>")
                    
                    if 'MMSI' in anomaly_df.columns:
                        report_lines.append("<h2>Top 10 Vessels by Total Anomaly Count</h2>")
                        top_vessels = anomaly_df['MMSI'].value_counts().head(10)
                        report_lines.append("<table>")
                        report_lines.append("<tr><th>MMSI</th><th>Anomaly Count</th></tr>")
                        for mmsi, count in top_vessels.items():
                            report_lines.append(f"<tr><td>{mmsi}</td><td>{count:,}</td></tr>")
                        report_lines.append("</table>")
                        
                        # Add top 10 vessels for each anomaly type
                        if 'AnomalyType' in anomaly_df.columns:
                            report_lines.append("<h2>Top 10 Vessels by Anomaly Type</h2>")
                            
                            for anomaly_type in anomaly_df['AnomalyType'].unique():
                                anomaly_type_df = anomaly_df[anomaly_df['AnomalyType'] == anomaly_type]
                                top_vessels_by_anomaly = anomaly_type_df['MMSI'].value_counts().head(10)
                                
                                if not top_vessels_by_anomaly.empty:
                                    report_lines.append(f"<h3>Top Vessels for {anomaly_type}</h3>")
                                    report_lines.append("<table>")
                                    report_lines.append("<tr><th>MMSI</th><th>Occurrence Count</th></tr>")
                                    for mmsi, count in top_vessels_by_anomaly.items():
                                        report_lines.append(f"<tr><td>{mmsi}</td><td>{count:,}</td></tr>")
                                    report_lines.append("</table>")
                        
                        # Add section for vessels with multiple anomalies (basic report version)
                        if 'AnomalyType' in anomaly_df.columns:
                            # Get vessels with multiple anomalies (more than 1)
                            vessel_anomaly_counts = anomaly_df['MMSI'].value_counts()
                            vessels_with_multiple = vessel_anomaly_counts[vessel_anomaly_counts > 1]
                            
                            if len(vessels_with_multiple) > 0:
                                report_lines.append("<h2>Vessels with Multiple Anomalies</h2>")
                                report_lines.append("<p>The following vessels have been detected with multiple anomalies during the analysis period:</p>")
                                report_lines.append("<table>")
                                report_lines.append("<tr><th>Vessel Name</th><th>MMSI</th><th>Vessel Type</th><th>Anomaly Counts by Type</th><th>Total Anomalies</th></tr>")
                                
                                # Format anomaly type helper function
                                def format_anomaly_type_for_report(atype):
                                    """Format anomaly type name for report display (shorter names)."""
                                    mapping = {
                                        "AIS_Beacon_Off": "AIS off",
                                        "AIS_Beacon_On": "AIS on",
                                        "Speed": "Speed",
                                        "Course": "Course",
                                        "Loitering": "Loitering",
                                        "Rendezvous": "Rendezvous",
                                        "Identity_Spoofing": "Identity Spoofing",
                                        "Zone_Violation": "Zone Violations"
                                    }
                                    return mapping.get(atype, atype)
                                
                                # Process each vessel with multiple anomalies
                                vessel_list = []
                                for mmsi, total_count in vessels_with_multiple.items():
                                    vessel_anomalies = anomaly_df[anomaly_df['MMSI'] == mmsi]
                                    
                                    # Get vessel name from anomaly data
                                    vessel_name = vessel_anomalies['VesselName'].mode().iloc[0] if 'VesselName' in vessel_anomalies.columns and not vessel_anomalies['VesselName'].mode().empty else None
                                    if pd.isna(vessel_name) or vessel_name == '':
                                        vessel_name = 'Unknown'
                                    
                                    # Get vessel type from anomaly data
                                    vessel_type = vessel_anomalies['VesselType'].mode().iloc[0] if 'VesselType' in vessel_anomalies.columns and not vessel_anomalies['VesselType'].mode().empty else None
                                    
                                    # Get anomaly counts by type
                                    anomaly_counts_by_type = vessel_anomalies['AnomalyType'].value_counts().to_dict()
                                    
                                    # Format anomaly counts string
                                    anomaly_counts_str = ", ".join([f"{format_anomaly_type_for_report(atype)}: {count}" 
                                                                    for atype, count in sorted(anomaly_counts_by_type.items(), key=lambda x: x[1], reverse=True)])
                                    
                                    # Format vessel type for display
                                    vessel_type_display = 'Unknown'
                                    if vessel_type is not None and pd.notna(vessel_type):
                                        try:
                                            vessel_type_int = int(vessel_type)
                                            vessel_type_name = get_vessel_type_name(vessel_type_int)
                                            vessel_type_display = f"{vessel_type_int} ({vessel_type_name})"
                                        except (ValueError, TypeError):
                                            vessel_type_display = str(vessel_type)
                                    
                                    vessel_list.append({
                                        'name': vessel_name,
                                        'mmsi': mmsi,
                                        'type': vessel_type_display,
                                        'anomaly_counts': anomaly_counts_str,
                                        'total': total_count
                                    })
                                
                                # Sort by total anomaly count (highest to lowest)
                                vessel_list.sort(key=lambda x: x['total'], reverse=True)
                                
                                # Add to report
                                for vessel in vessel_list:
                                    report_lines.append(f"<tr>")
                                    report_lines.append(f"<td>{vessel['name']}</td>")
                                    report_lines.append(f"<td>{vessel['mmsi']}</td>")
                                    report_lines.append(f"<td>{vessel['type']}</td>")
                                    report_lines.append(f"<td>{vessel['anomaly_counts']}</td>")
                                    report_lines.append(f"<td>{vessel['total']}</td>")
                                    report_lines.append(f"</tr>")
                                
                                report_lines.append("</table>")
                else:
                    report_lines.append("<h2>No Anomaly Data Available</h2>")
                    report_lines.append("<p>No anomaly data was found for analysis.</p>")
                
                report_lines.append("<h2>Recommendations</h2>")
                report_lines.append("<ul>")
                report_lines.append("<li>Check that AIS data files exist for the specified date range.</li>")
                report_lines.append("<li>Verify that the data has been processed and cached correctly.</li>")
                report_lines.append(f"<li>Ensure cache directory ({get_cache_dir()}) contains parquet files for the analysis period.</li>")
                report_lines.append("<li>Try running the analysis with a different date range if available.</li>")
                report_lines.append("</ul>")
                
                report_lines.append("</body></html>")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(report_lines))
                
                logger.info(f"Basic summary report generated: {output_path}")
                return output_path
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(self.output_directory, f"Summary_Report_{timestamp}.html")
            
            report_lines = []
            report_lines.append("<html><head><title>AIS Analysis Summary Report</title>")
            report_lines.append("<style>body { font-family: Arial, sans-serif; margin: 20px; }")
            report_lines.append("h1 { color: #003366; } h2 { color: #0066cc; }")
            report_lines.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
            report_lines.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
            report_lines.append("th { background-color: #003366; color: white; }</style></head><body>")
            
            report_lines.append("<h1>AIS Shipping Fraud Detection - Summary Report</h1>")
            report_lines.append(f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
            report_lines.append(f"<p><strong>Analysis Period:</strong> {self.run_info['start_date']} to {self.run_info['end_date']}</p>")
            
            report_lines.append("<h2>Data Overview</h2>")
            report_lines.append("<table>")
            report_lines.append(f"<tr><th>Metric</th><th>Value</th></tr>")
            report_lines.append(f"<tr><td>Total Records</td><td>{len(df):,}</td></tr>")
            if 'MMSI' in df.columns:
                report_lines.append(f"<tr><td>Unique Vessels (MMSI)</td><td>{df['MMSI'].nunique():,}</td></tr>")
            else:
                report_lines.append(f"<tr><td>Unique Vessels (MMSI)</td><td>N/A</td></tr>")
            
            if 'VesselType' in df.columns:
                report_lines.append(f"<tr><td>Vessel Types Analyzed</td><td>{df['VesselType'].nunique()}</td></tr>")
            
            if 'VesselName' in df.columns:
                named_vessels = df[df['VesselName'].notna()]['MMSI'].nunique() if 'MMSI' in df.columns else 0
                report_lines.append(f"<tr><td>Vessels with Names</td><td>{named_vessels:,}</td></tr>")
            
            report_lines.append("</table>")
            
            if not anomaly_df.empty:
                report_lines.append("<h2>Anomaly Summary</h2>")
                report_lines.append("<table>")
                report_lines.append(f"<tr><th>Metric</th><th>Value</th></tr>")
                report_lines.append(f"<tr><td>Total Anomalies Detected</td><td>{len(anomaly_df):,}</td></tr>")
                
                if 'AnomalyType' in anomaly_df.columns:
                    anomaly_counts = anomaly_df['AnomalyType'].value_counts()
                    report_lines.append("<tr><td colspan='2'><strong>Anomalies by Type:</strong></td></tr>")
                    for anomaly_type, count in anomaly_counts.items():
                        report_lines.append(f"<tr><td>&nbsp;&nbsp;{anomaly_type}</td><td>{count:,}</td></tr>")
                
                if 'MMSI' in anomaly_df.columns:
                    vessels_with_anomalies = anomaly_df['MMSI'].nunique()
                    report_lines.append(f"<tr><td>Vessels with Anomalies</td><td>{vessels_with_anomalies:,}</td></tr>")
                
                report_lines.append("</table>")
            
            # Key Findings section - moved to appear right after Anomaly Summary
            report_lines.append("<h2>Key Findings</h2>")
            report_lines.append("<ul>")
            
            if not anomaly_df.empty:
                report_lines.append(f"<li>Detected {len(anomaly_df):,} anomalies across {anomaly_df['MMSI'].nunique() if 'MMSI' in anomaly_df.columns else 'N/A'} vessels</li>")
            
            if 'VesselType' in df.columns:
                most_common_type = df['VesselType'].mode()[0] if not df['VesselType'].mode().empty else 'N/A'
                if most_common_type != 'N/A' and pd.notna(most_common_type):
                    try:
                        vessel_type_int = int(most_common_type)
                        vessel_type_name = get_vessel_type_name(vessel_type_int)
                        # Extract just the base name (before comma) for cleaner display
                        if ',' in vessel_type_name:
                            vessel_type_display = vessel_type_name.split(',')[0].strip()
                        else:
                            vessel_type_display = vessel_type_name
                        report_lines.append(f"<li>Most common vessel type: {vessel_type_display}</li>")
                    except (ValueError, TypeError):
                        report_lines.append(f"<li>Most common vessel type: {most_common_type}</li>")
                else:
                    report_lines.append(f"<li>Most common vessel type: {most_common_type}</li>")
            
            report_lines.append(f"<li>Analysis covered {len(df):,} total AIS records</li>")
            report_lines.append("</ul>")
            
            if 'VesselType' in df.columns:
                report_lines.append("<h2>Vessel Type Distribution</h2>")
                vessel_type_counts = df['VesselType'].value_counts().head(10)
                report_lines.append("<table>")
                report_lines.append("<tr><th>Vessel Type</th><th>Record Count</th></tr>")
                for vessel_type, count in vessel_type_counts.items():
                    report_lines.append(f"<tr><td>{vessel_type}</td><td>{count:,}</td></tr>")
                report_lines.append("</table>")
            
            if not anomaly_df.empty and 'MMSI' in anomaly_df.columns:
                report_lines.append("<h2>Top 10 Vessels by Total Anomaly Count</h2>")
                top_vessels = anomaly_df['MMSI'].value_counts().head(10)
                report_lines.append("<table>")
                report_lines.append("<tr><th>MMSI</th><th>Anomaly Count</th></tr>")
                for mmsi, count in top_vessels.items():
                    report_lines.append(f"<tr><td>{mmsi}</td><td>{count:,}</td></tr>")
                report_lines.append("</table>")
                
                # Add top 10 vessels for each anomaly type
                if 'AnomalyType' in anomaly_df.columns:
                    report_lines.append("<h2>Top 10 Vessels by Anomaly Type</h2>")
                    
                    for anomaly_type in anomaly_df['AnomalyType'].unique():
                        anomaly_type_df = anomaly_df[anomaly_df['AnomalyType'] == anomaly_type]
                        top_vessels_by_anomaly = anomaly_type_df['MMSI'].value_counts().head(10)
                        
                        if not top_vessels_by_anomaly.empty:
                            report_lines.append(f"<h3>Top Vessels for {anomaly_type}</h3>")
                            report_lines.append("<table>")
                            report_lines.append("<tr><th>MMSI</th><th>Occurrence Count</th></tr>")
                            for mmsi, count in top_vessels_by_anomaly.items():
                                report_lines.append(f"<tr><td>{mmsi}</td><td>{count:,}</td></tr>")
                            report_lines.append("</table>")
            
            # Add section for vessels with multiple anomalies
            if not anomaly_df.empty and 'MMSI' in anomaly_df.columns and 'AnomalyType' in anomaly_df.columns:
                # Get vessels with multiple anomalies (more than 1)
                vessel_anomaly_counts = anomaly_df['MMSI'].value_counts()
                vessels_with_multiple = vessel_anomaly_counts[vessel_anomaly_counts > 1]
                
                if len(vessels_with_multiple) > 0:
                    report_lines.append("<h2>Vessels with Multiple Anomalies</h2>")
                    report_lines.append("<p>The following vessels have been detected with multiple anomalies during the analysis period:</p>")
                    report_lines.append("<table>")
                    report_lines.append("<tr><th>Vessel Name</th><th>MMSI</th><th>Vessel Type</th><th>Anomaly Counts by Type</th><th>Total Anomalies</th></tr>")
                    
                    # Create lookup for vessel info from main dataframe
                    vessel_info_lookup = {}
                    if not df.empty and 'MMSI' in df.columns:
                        for mmsi in df['MMSI'].unique():
                            vessel_data = df[df['MMSI'] == mmsi]
                            if not vessel_data.empty:
                                vessel_name = vessel_data['VesselName'].mode().iloc[0] if 'VesselName' in vessel_data.columns and not vessel_data['VesselName'].mode().empty else None
                                vessel_type = vessel_data['VesselType'].mode().iloc[0] if 'VesselType' in vessel_data.columns and not vessel_data['VesselType'].mode().empty else None
                                vessel_info_lookup[mmsi] = {
                                    'name': vessel_name if pd.notna(vessel_name) else None,
                                    'type': vessel_type if pd.notna(vessel_type) else None
                                }
                    
                    # Process each vessel with multiple anomalies
                    vessel_list = []
                    for mmsi, total_count in vessels_with_multiple.items():
                        vessel_anomalies = anomaly_df[anomaly_df['MMSI'] == mmsi]
                        
                        # Get vessel name
                        vessel_name = vessel_anomalies['VesselName'].mode().iloc[0] if 'VesselName' in vessel_anomalies.columns and not vessel_anomalies['VesselName'].mode().empty else None
                        if pd.isna(vessel_name) or vessel_name == '':
                            vessel_name = vessel_info_lookup.get(mmsi, {}).get('name', None)
                        if pd.isna(vessel_name) or vessel_name == '':
                            vessel_name = 'Unknown'
                        
                        # Get vessel type
                        vessel_type = vessel_anomalies['VesselType'].mode().iloc[0] if 'VesselType' in vessel_anomalies.columns and not vessel_anomalies['VesselType'].mode().empty else None
                        if pd.isna(vessel_type) or vessel_type == '':
                            vessel_type = vessel_info_lookup.get(mmsi, {}).get('type', None)
                        
                        # Get anomaly counts by type
                        anomaly_counts_by_type = vessel_anomalies['AnomalyType'].value_counts().to_dict()
                        
                        # Format anomaly counts string (e.g., "AIS off: 2, AIS on: 1, Course: 1")
                        # Create shorter, cleaner names for the report
                        def format_anomaly_type_for_report(atype):
                            """Format anomaly type name for report display (shorter names)."""
                            mapping = {
                                "AIS_Beacon_Off": "AIS off",
                                "AIS_Beacon_On": "AIS on",
                                "Speed": "Speed",
                                "Course": "Course",
                                "Loitering": "Loitering",
                                "Rendezvous": "Rendezvous",
                                "Identity_Spoofing": "Identity Spoofing",
                                "Zone_Violation": "Zone Violations"
                            }
                            return mapping.get(atype, atype)
                        
                        anomaly_counts_str = ", ".join([f"{format_anomaly_type_for_report(atype)}: {count}" 
                                                        for atype, count in sorted(anomaly_counts_by_type.items(), key=lambda x: x[1], reverse=True)])
                        
                        # Format vessel type for display
                        vessel_type_display = 'Unknown'
                        if vessel_type is not None and pd.notna(vessel_type):
                            try:
                                vessel_type_int = int(vessel_type)
                                vessel_type_name = get_vessel_type_name(vessel_type_int)
                                vessel_type_display = f"{vessel_type_int} ({vessel_type_name})"
                            except (ValueError, TypeError):
                                vessel_type_display = str(vessel_type)
                        
                        vessel_list.append({
                            'name': vessel_name,
                            'mmsi': mmsi,
                            'type': vessel_type_display,
                            'anomaly_counts': anomaly_counts_str,
                            'total': total_count
                        })
                    
                    # Sort by total anomaly count (highest to lowest)
                    vessel_list.sort(key=lambda x: x['total'], reverse=True)
                    
                    # Add to report
                    for vessel in vessel_list:
                        report_lines.append(f"<tr>")
                        report_lines.append(f"<td>{vessel['name']}</td>")
                        report_lines.append(f"<td>{vessel['mmsi']}</td>")
                        report_lines.append(f"<td>{vessel['type']}</td>")
                        report_lines.append(f"<td>{vessel['anomaly_counts']}</td>")
                        report_lines.append(f"<td>{vessel['total']}</td>")
                        report_lines.append(f"</tr>")
                    
                    report_lines.append("</table>")
            
            report_lines.append("</body></html>")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            logger.info(f"Summary report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def export_vessel_statistics(self, output_path=None):
        """Export vessel-specific statistics to Excel format."""
        # Suppress warnings during export to prevent repeated warnings during vessel processing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                logger.info("Exporting vessel statistics...")
                
                df = self.load_cached_data()
                anomaly_df = self.load_anomaly_data()
                
                if df.empty:
                    logger.warning("No cached data available for vessel statistics")
                    
                    # Generate a basic stats file with available information
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    if output_path is None:
                        output_path = os.path.join(self.output_directory, f"Vessel_Statistics_{timestamp}.xlsx")
                    
                    # Create a basic statistics file with the anomaly data if available
                    if not anomaly_df.empty and 'MMSI' in anomaly_df.columns:
                        # We can still provide vessel statistics based on anomaly data alone
                        vessel_stats = []
                        
                        # Group by MMSI for efficient processing
                        anomaly_groups = anomaly_df.groupby('MMSI')
                        unique_mmsis = anomaly_df['MMSI'].unique()
                        
                        for mmsi in unique_mmsis:
                            try:
                                # Use groupby get_group() which is much faster than boolean indexing
                                vessel_anomalies = anomaly_groups.get_group(mmsi)
                                stats = {
                                    'MMSI': mmsi,
                                    'Anomaly_Count': len(vessel_anomalies),
                                }
                                
                                if 'VesselName' in vessel_anomalies.columns:
                                    names = vessel_anomalies['VesselName'].dropna().unique()
                                    stats['VesselName'] = names[0] if len(names) > 0 else 'Unknown'
                                else:
                                    stats['VesselName'] = 'Unknown'
                                
                                if 'AnomalyType' in vessel_anomalies.columns:
                                    anomaly_types = vessel_anomalies['AnomalyType'].value_counts().to_dict()
                                    for anomaly_type, count in anomaly_types.items():
                                        stats[f'Anomaly_{anomaly_type}'] = count
                                
                                vessel_stats.append(stats)
                            except Exception as e:
                                logger.warning(f"Error processing vessel {mmsi} from anomaly data: {e}")
                                continue
                            
                        if vessel_stats:
                            stats_df = pd.DataFrame(vessel_stats)
                            
                            try:
                                if OPENPYXL_AVAILABLE:
                                    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                                        stats_df.to_excel(writer, sheet_name='Vessel Statistics (Limited)', index=False)
                                        
                                        summary_data = {
                                            'Metric': ['Total Vessels with Anomalies', 'Total Anomalies', 'Note'],
                                            'Value': [
                                                len(stats_df),
                                                stats_df['Anomaly_Count'].sum() if 'Anomaly_Count' in stats_df.columns else 0,
                                                'Limited statistics due to no cached AIS data being available'
                                            ]
                                        }
                                        summary_df = pd.DataFrame(summary_data)
                                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                                    
                                    logger.info(f"Limited vessel statistics exported to: {output_path}")
                                    return output_path
                                else:
                                    raise ImportError("openpyxl not available")
                            except (ImportError, Exception) as e:
                                csv_path = output_path.replace('.xlsx', '.csv')
                                stats_df.to_csv(csv_path, index=False)
                                logger.warning(f"Excel export failed ({str(e)}). Exported to CSV instead: {csv_path}")
                                return csv_path
                    else:
                        logger.warning("No anomaly data available for vessel statistics either")
                    
                    # If no anomaly data or couldn't create stats from it, create an info file
                    info_file = output_path.replace('.xlsx', '_info.csv')
                    info_df = pd.DataFrame([
                        {"Parameter": "Start Date", "Value": self.run_info.get('start_date', 'N/A')},
                        {"Parameter": "End Date", "Value": self.run_info.get('end_date', 'N/A')},
                        {"Parameter": "Selected Ship Types", "Value": str(self.run_info.get('ship_types', []))},
                        {"Parameter": "Output Directory", "Value": self.output_directory},
                        {"Parameter": "Data Directory", "Value": self.run_info.get('data_directory', 'N/A')},
                        {"Parameter": "Export Time", "Value": datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                        {"Parameter": "Status", "Value": "No data found for vessel statistics"}
                    ])
                    
                    info_df.to_csv(info_file, index=False)
                    logger.info(f"Vessel statistics info file exported to: {info_file}")
                    return info_file
                
                if output_path is None:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_path = os.path.join(self.output_directory, f"Vessel_Statistics_{timestamp}.xlsx")
                
                if 'MMSI' not in df.columns:
                    logger.error("MMSI column not found in dataset")
                    return None
                
                vessel_stats = []
                
                # Pre-convert datetime once to avoid repeated conversion
                if 'BaseDateTime' in df.columns:
                    df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], errors='coerce')
                
                # Group by MMSI once for efficient processing (much faster than filtering in a loop)
                logger.info(f"Grouping data by MMSI for {len(df['MMSI'].unique())} vessels...")
                df_groups = df.groupby('MMSI')
                
                # Pre-group anomalies if available
                anomaly_groups = None
                if not anomaly_df.empty and 'MMSI' in anomaly_df.columns:
                    anomaly_groups = anomaly_df.groupby('MMSI')
                
                unique_mmsis = df['MMSI'].unique()
                logger.info(f"Processing statistics for {len(unique_mmsis)} vessels...")
                
                for idx, mmsi in enumerate(unique_mmsis):
                    if (idx + 1) % 100 == 0:
                        logger.info(f"Processing vessel {idx + 1}/{len(unique_mmsis)}...")
                    
                    try:
                        # Use groupby get_group() which is much faster than boolean indexing
                        vessel_data = df_groups.get_group(mmsi)
                        
                        # Get anomalies for this vessel if available
                        vessel_anomalies = pd.DataFrame()
                        if anomaly_groups is not None:
                            try:
                                vessel_anomalies = anomaly_groups.get_group(mmsi)
                            except KeyError:
                                # No anomalies for this vessel
                                pass
                        
                        stats = {
                            'MMSI': mmsi,
                            'Total_Records': len(vessel_data),
                            'Anomaly_Count': len(vessel_anomalies),
                        }
                        
                        if 'VesselName' in vessel_data.columns:
                            names = vessel_data['VesselName'].dropna().unique()
                            stats['VesselName'] = names[0] if len(names) > 0 else 'Unknown'
                        else:
                            stats['VesselName'] = 'Unknown'
                        
                        if 'VesselType' in vessel_data.columns:
                            types = vessel_data['VesselType'].unique()
                            stats['VesselType'] = types[0] if len(types) > 0 else 'Unknown'
                        else:
                            stats['VesselType'] = 'Unknown'
                        
                        if 'BaseDateTime' in vessel_data.columns:
                            stats['First_Seen'] = vessel_data['BaseDateTime'].min()
                            stats['Last_Seen'] = vessel_data['BaseDateTime'].max()
                            if pd.notna(stats['First_Seen']) and pd.notna(stats['Last_Seen']):
                                stats['Days_Active'] = (stats['Last_Seen'] - stats['First_Seen']).days + 1
                            else:
                                stats['Days_Active'] = 0
                        
                        if 'SOG' in vessel_data.columns:
                            speeds = vessel_data['SOG'].dropna()
                            if len(speeds) > 0:
                                stats['Avg_Speed'] = speeds.mean()
                                stats['Max_Speed'] = speeds.max()
                                stats['Min_Speed'] = speeds.min()
                        
                        if 'LAT' in vessel_data.columns and 'LON' in vessel_data.columns:
                            stats['Min_Latitude'] = vessel_data['LAT'].min()
                            stats['Max_Latitude'] = vessel_data['LAT'].max()
                            stats['Min_Longitude'] = vessel_data['LON'].min()
                            stats['Max_Longitude'] = vessel_data['LON'].max()
                        
                        if not vessel_anomalies.empty and 'AnomalyType' in vessel_anomalies.columns:
                            anomaly_types = vessel_anomalies['AnomalyType'].value_counts().to_dict()
                            for anomaly_type, count in anomaly_types.items():
                                stats[f'Anomaly_{anomaly_type}'] = count
                        
                        vessel_stats.append(stats)
                    except Exception as e:
                        logger.warning(f"Error processing vessel {mmsi}: {e}")
                        continue
                
                if not vessel_stats:
                    logger.error("No vessel statistics generated. Check that data contains valid vessel information.")
                    # Create an informative error file instead of returning None
                    error_file = output_path.replace('.xlsx', '_error.csv')
                    error_df = pd.DataFrame([
                        {"Error": "No vessel statistics could be generated"},
                        {"Possible Cause": "Data may not contain valid vessel information"},
                        {"Date Range": f"{self.run_info.get('start_date', 'N/A')} to {self.run_info.get('end_date', 'N/A')}"},
                        {"Total Vessels Found": len(unique_mmsis)},
                        {"Anomalies Available": "Yes" if not anomaly_df.empty else "No"}
                    ])
                    error_df.to_csv(error_file, index=False)
                    logger.info(f"Error information exported to: {error_file}")
                    return error_file
                
                stats_df = pd.DataFrame(vessel_stats)
                
                try:
                    if OPENPYXL_AVAILABLE:
                        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                            stats_df.to_excel(writer, sheet_name='Vessel Statistics', index=False)
                            
                            summary_data = {
                                'Metric': ['Total Vessels', 'Vessels with Anomalies', 'Total Anomalies', 'Total Records'],
                                'Value': [
                                    len(stats_df),
                                    len(stats_df[stats_df['Anomaly_Count'] > 0]) if 'Anomaly_Count' in stats_df.columns else 0,
                                    stats_df['Anomaly_Count'].sum() if 'Anomaly_Count' in stats_df.columns else 0,
                                    stats_df['Total_Records'].sum() if 'Total_Records' in stats_df.columns else 0
                                ]
                            }
                            summary_df = pd.DataFrame(summary_data)
                            summary_df.to_excel(writer, sheet_name='Summary', index=False)
                        
                        logger.info(f"Vessel statistics exported to: {output_path}")
                        return output_path
                    else:
                        raise ImportError("openpyxl not available")
                except (ImportError, Exception) as e:
                    csv_path = output_path.replace('.xlsx', '.csv')
                    stats_df.to_csv(csv_path, index=False)
                    logger.warning(f"Excel export failed ({str(e)}). Exported to CSV instead: {csv_path}")
                    return csv_path
            except Exception as e:
                logger.error(f"Error exporting vessel statistics: {e}")
                logger.error(traceback.format_exc())
                return None
    
    def generate_anomaly_timeline(self, output_path=None):
        """Create a timeline visualization of anomalies."""
        try:
            logger.info("Generating anomaly timeline...")
            
            anomaly_df = self.load_anomaly_data()
            
            if anomaly_df.empty:
                logger.error("No anomaly data available for timeline")
                return None
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(self.output_directory, f"Anomaly_Timeline_{timestamp}.html")
            
            if 'Date' in anomaly_df.columns:
                anomaly_df['DateTime'] = pd.to_datetime(anomaly_df['Date'])
            elif 'BaseDateTime' in anomaly_df.columns:
                anomaly_df['DateTime'] = pd.to_datetime(anomaly_df['BaseDateTime'])
            else:
                logger.error("No date column found in anomaly data")
                return None
            
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                
                if 'AnomalyType' in anomaly_df.columns:
                    for anomaly_type in anomaly_df['AnomalyType'].unique():
                        type_data = anomaly_df[anomaly_df['AnomalyType'] == anomaly_type]
                        timeline_data = type_data.groupby(type_data['DateTime'].dt.date).size()
                        
                        fig.add_trace(go.Scatter(
                            x=timeline_data.index,
                            y=timeline_data.values,
                            mode='lines+markers',
                            name=anomaly_type,
                            line=dict(width=2)
                        ))
                else:
                    timeline_data = anomaly_df.groupby(anomaly_df['DateTime'].dt.date).size()
                    fig.add_trace(go.Scatter(
                        x=timeline_data.index,
                        y=timeline_data.values,
                        mode='lines+markers',
                        name='All Anomalies',
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title='Anomaly Timeline',
                    xaxis_title='Date',
                    yaxis_title='Number of Anomalies',
                    hovermode='x unified',
                    template='plotly_white'
                )
                
                fig.write_html(output_path)
                logger.info(f"Anomaly timeline generated: {output_path}")
                return output_path
            else:
                logger.warning("Plotly not available. Creating simple timeline table.")
                timeline_data = anomaly_df.groupby(anomaly_df['DateTime'].dt.date).size().reset_index()
                timeline_data.columns = ['Date', 'Anomaly_Count']
                
                html_content = ["<html><head><title>Anomaly Timeline</title></head><body>"]
                html_content.append("<h1>Anomaly Timeline</h1>")
                html_content.append("<table border='1'><tr><th>Date</th><th>Anomaly Count</th></tr>")
                
                for _, row in timeline_data.iterrows():
                    html_content.append(f"<tr><td>{row['Date']}</td><td>{row['Anomaly_Count']}</td></tr>")
                
                html_content.append("</table></body></html>")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(html_content))
                
                logger.info(f"Anomaly timeline (table) generated: {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"Error generating anomaly timeline: {e}")
            logger.error(traceback.format_exc())
            return None


    # ========================================================================
    # TAB 2: FURTHER ANALYSIS  
    # ========================================================================
    
    def correlation_analysis(self, vessel_types, anomaly_types, output_path=None):
        """Perform correlation analysis between vessel types and anomaly types."""
        try:
            logger.info(f"Performing correlation analysis: Vessel Types={vessel_types}, Anomaly Types={anomaly_types}")
            
            df = self.load_cached_data()
            anomaly_df = self.load_anomaly_data()
            
            # Get the vessel types from run_info for reference, even if df is empty
            run_info_vessel_types = []
            if hasattr(self, 'run_info') and 'ship_types' in self.run_info:
                try:
                    run_info_vessel_types = [int(vt) for vt in self.run_info['ship_types'] if vt and pd.notna(vt)]
                    logger.info(f"Using vessel types from run_info for correlation: {run_info_vessel_types}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert run_info ship_types to int: {e}")
            
            # If we have vessel types in run_info that match the requested types, consider them valid
            vessel_types_match_run_info = any(vt in run_info_vessel_types for vt in vessel_types) if vessel_types and run_info_vessel_types else False
            
            # Special case: We have vessel types from run_info but no data
            if df.empty and vessel_types_match_run_info:
                logger.warning("No cached data available, but using vessel types from run_info for analysis")
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                if output_path is None:
                    output_path = os.path.join(self.output_directory, f"Correlation_Analysis_NoData_{timestamp}.html")
                
                # Create a special report for this case
                report_lines = []
                report_lines.append("<html><head><title>Correlation Analysis Report (No Data)</title>")
                report_lines.append("<style>body { font-family: Arial, sans-serif; margin: 20px; }")
                report_lines.append("h1 { color: #003366; } h2 { color: #0066cc; }")
                report_lines.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
                report_lines.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
                report_lines.append("th { background-color: #003366; color: white; }</style></head><body>")
                
                report_lines.append("<h1>AIS Anomaly Correlation Analysis (No Data Available)</h1>")
                report_lines.append(f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
                report_lines.append("<h2>Analysis Parameters</h2>")
                report_lines.append("<table>")
                report_lines.append("<tr><th>Parameter</th><th>Value</th></tr>")
                report_lines.append(f"<tr><td>Analysis Period</td><td>{self.run_info['start_date']} to {self.run_info['end_date']}</td></tr>")
                report_lines.append(f"<tr><td>Selected Vessel Types</td><td>{', '.join([f'{vt} ({get_vessel_type_name(vt)})' for vt in vessel_types])}</td></tr>")
                report_lines.append(f"<tr><td>Selected Anomaly Types</td><td>{', '.join(anomaly_types) if anomaly_types else 'None'}</td></tr>")
                report_lines.append(f"<tr><td>Data Status</td><td>No cached data available</td></tr>")
                report_lines.append("</table>")
                
                report_lines.append("<h2>Recommendations</h2>")
                report_lines.append("<ul>")
                report_lines.append("<li>Ensure AIS data is available for the specified date range.</li>")
                report_lines.append("<li>Verify that the selected vessel types exist in your dataset.</li>")
                report_lines.append("<li>Try analyzing a different time period where more data is available.</li>")
                report_lines.append("</ul>")
                
                report_lines.append("</body></html>")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(report_lines))
                
                logger.info(f"Correlation analysis report (no data) generated: {output_path}")
                return output_path
            
            elif (df.empty or anomaly_df.empty) and not vessel_types_match_run_info:
                logger.warning("Insufficient complete data for correlation analysis")
                
                # Create a basic report if we have at least anomaly data
                if not anomaly_df.empty:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    if output_path is None:
                        output_path = os.path.join(self.output_directory, f"Correlation_Analysis_{timestamp}.html")
                    
                    # Create simple HTML report
                    report_lines = []
                    report_lines.append("<html><head><title>Correlation Analysis Report</title>")
                    report_lines.append("<style>body { font-family: Arial, sans-serif; margin: 20px; }")
                    report_lines.append("h1 { color: #003366; } h2 { color: #0066cc; }")
                    report_lines.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
                    report_lines.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
                    report_lines.append("th { background-color: #003366; color: white; }</style></head><body>")
                    
                    report_lines.append("<h1>AIS Anomaly Correlation Analysis</h1>")
                    report_lines.append(f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
                    report_lines.append("<h2>Data Status</h2>")
                    report_lines.append("<p><strong>Note:</strong> Limited analysis possible. No AIS vessel data available, "
                                      "but anomaly data is present.</p>")
                    
                    # Basic anomaly statistics
                    if 'AnomalyType' in anomaly_df.columns:
                        anomaly_counts = anomaly_df['AnomalyType'].value_counts()
                        report_lines.append("<h2>Anomaly Distribution</h2>")
                        report_lines.append("<table>")
                        report_lines.append("<tr><th>Anomaly Type</th><th>Count</th><th>Percentage</th></tr>")
                        for anomaly_type, count in anomaly_counts.items():
                            percentage = 100 * count / len(anomaly_df)
                            report_lines.append(f"<tr><td>{anomaly_type}</td><td>{count:,}</td><td>{percentage:.2f}%</td></tr>")
                        report_lines.append("</table>")
                    
                    report_lines.append("<h2>Recommendations</h2>")
                    report_lines.append("<ul>")
                    report_lines.append("<li>Ensure AIS data is available for a complete correlation analysis.</li>")
                    report_lines.append("<li>For vessel type analysis, the AIS data must include vessel type information.</li>")
                    report_lines.append("</ul>")
                    
                    report_lines.append("</body></html>")
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(report_lines))
                    
                    logger.info(f"Basic correlation analysis report generated: {output_path}")
                    return output_path
                else:
                    logger.error("No data available for correlation analysis")
                    return None
            
            if vessel_types:
                # Convert vessel types to int for proper comparison
                try:
                    vessel_types_int = [int(vt) for vt in vessel_types]
                except (ValueError, TypeError) as e:
                    logger.error(f"Error converting vessel types to int: {e}")
                    logger.error(f"Vessel types: {vessel_types}")
                    return None
                
                if 'VesselType' in df.columns:
                    df['VesselType'] = pd.to_numeric(df['VesselType'], errors='coerce').astype('Int64')
                    df_filtered = df[df['VesselType'].isin(vessel_types_int)]
                    
                    # Check if any of the selected vessel types are present in the data
                    present_types = set(df_filtered['VesselType'].unique()).intersection(set(vessel_types_int))
                    missing_types = set(vessel_types_int) - present_types
                    
                    if missing_types:
                        logger.warning(f"Selected vessel types {missing_types} not found in dataset")
                    
                    if len(df_filtered) > 0:
                        vessel_mmsis = df_filtered['MMSI'].unique()
                        anomaly_df_filtered = anomaly_df[anomaly_df['MMSI'].isin(vessel_mmsis)]
                    else:
                        logger.warning("No matching vessels found for selected vessel types")
                        anomaly_df_filtered = anomaly_df  # Use all anomaly data if no matching vessels
                else:
                    logger.warning("VesselType column not found in dataset. Using all anomaly data.")
                    anomaly_df_filtered = anomaly_df
            else:
                # No vessel types specified, use all anomaly data
                anomaly_df_filtered = anomaly_df
            
            if anomaly_types and 'AnomalyType' in anomaly_df_filtered.columns:
                # Check if any of the selected anomaly types are present in the data
                present_anomalies = set(anomaly_df_filtered['AnomalyType'].unique()).intersection(set(anomaly_types))
                missing_anomalies = set(anomaly_types) - present_anomalies
                
                if missing_anomalies:
                    logger.warning(f"Selected anomaly types {missing_anomalies} not found in dataset")
                    
                # Filter for selected anomaly types
                anomaly_df_filtered = anomaly_df_filtered[anomaly_df_filtered['AnomalyType'].isin(anomaly_types)]
                
                if len(anomaly_df_filtered) == 0:
                    logger.warning("No data matches the selected anomaly types. Using all anomaly data.")
                    # Reset to all anomaly data if filtering results in empty set
                    if vessel_types and 'VesselType' in df.columns and len(df_filtered) > 0:
                        # Still respect vessel filtering if it was applied
                        vessel_mmsis = df_filtered['MMSI'].unique()
                        anomaly_df_filtered = anomaly_df[anomaly_df['MMSI'].isin(vessel_mmsis)]
                    else:
                        anomaly_df_filtered = anomaly_df
            
            results = {}
            
            if vessel_types and 'VesselType' in df.columns:
                # Ensure VesselType is int for proper comparison
                df['VesselType'] = pd.to_numeric(df['VesselType'], errors='coerce').astype('Int64')
                # vessel_types_int was already converted earlier in this method
                vessel_counts = df[df['VesselType'].isin(vessel_types_int)]['VesselType'].value_counts()
                results['vessel_type_counts'] = vessel_counts.to_dict()
            
            if 'AnomalyType' in anomaly_df_filtered.columns:
                anomaly_counts = anomaly_df_filtered['AnomalyType'].value_counts()
                results['anomaly_type_counts'] = anomaly_counts.to_dict()
            
            if 'VesselType' in df.columns and 'AnomalyType' in anomaly_df_filtered.columns:
                # Ensure VesselType is int for proper merging
                df['VesselType'] = pd.to_numeric(df['VesselType'], errors='coerce').astype('Int64')
                
                # Log some diagnostics about the data before merge
                logger.info(f"Unique MMSI in vessel data: {len(df['MMSI'].unique())}")
                logger.info(f"Unique MMSI in anomaly data: {len(anomaly_df_filtered['MMSI'].unique())}")
                
                # Create mapping from MMSI to VesselType
                mmsi_vessel_map = df[['MMSI', 'VesselType']].drop_duplicates()
                
                # Check if we have data to merge
                if mmsi_vessel_map.empty:
                    logger.warning("No MMSI-to-VesselType mapping available for merge operation")
                    results['merge_error'] = "No MMSI-to-VesselType mapping available"
                    anomaly_with_vessel = anomaly_df_filtered.copy()
                else:
                    # Add informative logging
                    common_mmsi = set(anomaly_df_filtered['MMSI']).intersection(set(mmsi_vessel_map['MMSI']))
                    logger.info(f"Common MMSI between vessel and anomaly data: {len(common_mmsi)}")
                    
                    # Perform the merge
                    anomaly_with_vessel = anomaly_df_filtered.merge(mmsi_vessel_map, on='MMSI', how='left')
                
                if not anomaly_with_vessel.empty:
                    # Check if VesselType column exists in the merged DataFrame
                    if 'VesselType' in anomaly_with_vessel.columns:
                        # Convert VesselType to int for crosstab
                        anomaly_with_vessel['VesselType'] = pd.to_numeric(anomaly_with_vessel['VesselType'], errors='coerce').astype('Int64')
                        crosstab = pd.crosstab(anomaly_with_vessel['VesselType'], 
                                              anomaly_with_vessel['AnomalyType'])
                        results['crosstab'] = crosstab
                    else:
                        logger.warning("VesselType column not found in merged anomaly data. This could indicate no matching MMSI values were found.")
                        logger.info(f"Available columns in merged data: {anomaly_with_vessel.columns.tolist()}")
                                # Create a placeholder result to indicate the issue
                        results['crosstab_error'] = "No VesselType column found in merged data"
                        
                        # Let's create a section for the report that explains the issue
                        results['error_info'] = {
                            'title': 'Data Correlation Issue Detected',
                            'message': 'The VesselType column was not found in the merged data. This typically happens when there are no matching MMSI values between the vessel and anomaly datasets.',
                            'recommendations': [
                                'Verify that the selected vessel types have associated anomalies in the time period',
                                'Check if the MMSI values in the anomaly data match those in the vessel data',
                                'Try selecting a broader range of vessel types or anomaly types'
                            ],
                            'available_columns': anomaly_with_vessel.columns.tolist()
                        }
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(self.output_directory, f"Correlation_Analysis_{timestamp}.html")
            
            if PLOTLY_AVAILABLE:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Vessel Type Distribution', 'Anomaly Type Distribution', 
                                  'Cross-Tabulation Heatmap', 'Summary Statistics'),
                    specs=[[{"type": "bar"}, {"type": "bar"}],
                          [{"type": "heatmap"}, {"type": "table"}]]
                )
                
                if 'vessel_type_counts' in results:
                    fig.add_trace(
                        go.Bar(x=list(results['vessel_type_counts'].keys()),
                              y=list(results['vessel_type_counts'].values()),
                              name='Vessel Types'),
                        row=1, col=1
                    )
                
                if 'anomaly_type_counts' in results:
                    fig.add_trace(
                        go.Bar(x=list(results['anomaly_type_counts'].keys()),
                              y=list(results['anomaly_type_counts'].values()),
                              name='Anomaly Types'),
                        row=1, col=2
                    )
                
                if 'crosstab' in results:
                    crosstab = results['crosstab']
                    fig.add_trace(
                        go.Heatmap(z=crosstab.values,
                                  x=crosstab.columns,
                                  y=crosstab.index,
                                  colorscale='Viridis'),
                        row=2, col=1
                    )
                
                fig.update_layout(height=800, title_text="Correlation Analysis Results")
                fig.write_html(output_path)
            else:
                html_content = ["<html><head><title>Correlation Analysis</title></head><body>"]
                html_content.append("<h1>Correlation Analysis Results</h1>")
                
                # Handle error_info if present
                if 'error_info' in results:
                    error_info = results['error_info']
                    html_content.append(f"<h2 style='color: #cc0000;'>{error_info['title']}</h2>")
                    html_content.append(f"<p>{error_info['message']}</p>")
                    html_content.append("<h3>Recommendations:</h3>")
                    html_content.append("<ul>")
                    for rec in error_info['recommendations']:
                        html_content.append(f"<li>{rec}</li>")
                    html_content.append("</ul>")
                    html_content.append("<h3>Available Data Columns:</h3>")
                    html_content.append("<p>" + ", ".join(error_info['available_columns']) + "</p>")
                
                if 'vessel_type_counts' in results:
                    html_content.append("<h2>Vessel Type Distribution</h2>")
                    vessel_counts_df = pd.DataFrame(list(results['vessel_type_counts'].items()), 
                                                 columns=['VesselType', 'Count'])
                    html_content.append(vessel_counts_df.to_html(index=False))
                
                if 'anomaly_type_counts' in results:
                    html_content.append("<h2>Anomaly Type Distribution</h2>")
                    anomaly_counts_df = pd.DataFrame(list(results['anomaly_type_counts'].items()), 
                                                  columns=['AnomalyType', 'Count'])
                    html_content.append(anomaly_counts_df.to_html(index=False))
                
                if 'crosstab' in results:
                    html_content.append("<h2>Cross-Tabulation</h2>")
                    html_content.append(results['crosstab'].to_html())
                
                html_content.append("</body></html>")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(html_content))
            
            logger.info(f"Correlation analysis completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def temporal_pattern_analysis(self, output_path=None):
        """Analyze temporal patterns including hourly/daily distributions."""
        try:
            logger.info("Performing temporal pattern analysis...")
            
            anomaly_df = self.load_anomaly_data()
            
            if anomaly_df.empty:
                logger.error("No anomaly data for temporal analysis")
                return None
            
            if 'BaseDateTime' in anomaly_df.columns:
                anomaly_df['DateTime'] = pd.to_datetime(anomaly_df['BaseDateTime'])
            elif 'Date' in anomaly_df.columns:
                anomaly_df['DateTime'] = pd.to_datetime(anomaly_df['Date'])
            else:
                logger.error("No datetime column found")
                return None
            
            anomaly_df['Hour'] = anomaly_df['DateTime'].dt.hour
            anomaly_df['DayOfWeek'] = anomaly_df['DateTime'].dt.day_name()
            anomaly_df['Date'] = anomaly_df['DateTime'].dt.date
            
            hourly_dist = anomaly_df['Hour'].value_counts().sort_index()
            daily_dist = anomaly_df.groupby('Date').size()
            dayofweek_dist = anomaly_df['DayOfWeek'].value_counts()
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(self.output_directory, f"Temporal_Patterns_{timestamp}.html")
            
            if PLOTLY_AVAILABLE:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Hourly Distribution', 'Daily Trend', 
                                  'Day of Week Distribution', 'Anomaly Type by Hour'),
                    specs=[[{"type": "bar"}, {"type": "scatter"}],
                          [{"type": "bar"}, {"type": "bar"}]]
                )
                
                fig.add_trace(
                    go.Bar(x=hourly_dist.index, y=hourly_dist.values, name='Hourly'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=daily_dist.index, y=daily_dist.values, 
                              mode='lines+markers', name='Daily'),
                    row=1, col=2
                )
                
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                dayofweek_ordered = pd.Series([dayofweek_dist.get(d, 0) for d in day_order], index=day_order)
                fig.add_trace(
                    go.Bar(x=dayofweek_ordered.index, y=dayofweek_ordered.values, name='Day of Week'),
                    row=2, col=1
                )
                
                if 'AnomalyType' in anomaly_df.columns:
                    anomaly_by_hour = pd.crosstab(anomaly_df['Hour'], anomaly_df['AnomalyType'])
                    for col in anomaly_by_hour.columns:
                        fig.add_trace(
                            go.Bar(x=anomaly_by_hour.index, y=anomaly_by_hour[col], 
                                  name=col),
                            row=2, col=2
                        )
                
                fig.update_layout(height=800, title_text="Temporal Pattern Analysis")
                fig.write_html(output_path)
            else:
                html_content = ["<html><head><title>Temporal Patterns</title></head><body>"]
                html_content.append("<h1>Temporal Pattern Analysis</h1>")
                html_content.append("<h2>Hourly Distribution</h2>")
                html_content.append(hourly_dist.to_frame('Count').to_html())
                html_content.append("</body></html>")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(html_content))
            
            logger.info(f"Temporal pattern analysis completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error in temporal pattern analysis: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def vessel_behavior_clustering(self, vessel_types=None, output_path=None, n_clusters=5):
        """Apply clustering algorithms to identify similar vessel behaviors.
        
        Args:
            vessel_types: List of vessel type integers to filter by. If None, uses all vessels.
            output_path: Output file path
            n_clusters: Number of clusters to create
        """
        try:
            logger.info("Performing vessel behavior clustering...")
            
            df = self.load_cached_data()
            
            if df.empty:
                logger.warning("No cached data available for vessel behavior clustering")
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                if output_path is None:
                    output_path = os.path.join(self.output_directory, f"Vessel_Clustering_NoData_{timestamp}.html")
                
                # Create a useful report instead of just returning None
                report_lines = []
                report_lines.append("<html><head><title>Vessel Behavior Clustering (No Data)</title>")
                report_lines.append("<style>body { font-family: Arial, sans-serif; margin: 20px; }")
                report_lines.append("h1 { color: #003366; } h2 { color: #0066cc; }")
                report_lines.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
                report_lines.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
                report_lines.append("th { background-color: #003366; color: white; }</style></head><body>")
                
                report_lines.append("<h1>Vessel Behavior Clustering (No Data Available)</h1>")
                report_lines.append(f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
                report_lines.append("<h2>Analysis Parameters</h2>")
                report_lines.append("<table>")
                report_lines.append("<tr><th>Parameter</th><th>Value</th></tr>")
                report_lines.append(f"<tr><td>Analysis Period</td><td>{self.run_info['start_date']} to {self.run_info['end_date']}</td></tr>")
                
                # Show selected vessel types if provided
                if vessel_types is not None and len(vessel_types) > 0:
                    vessel_type_names = [f"{vt} ({get_vessel_type_name(vt)})" for vt in vessel_types]
                    report_lines.append(f"<tr><td>Selected Vessel Types</td><td>{', '.join(vessel_type_names)}</td></tr>")
                else:
                    report_lines.append(f"<tr><td>Selected Vessel Types</td><td>All</td></tr>")
                
                report_lines.append(f"<tr><td>Number of Clusters</td><td>{n_clusters}</td></tr>")
                report_lines.append(f"<tr><td>Data Status</td><td>No cached data available</td></tr>")
                report_lines.append("</table>")
                
                report_lines.append("<h2>Recommendations</h2>")
                report_lines.append("<ul>")
                report_lines.append("<li>Ensure AIS data is available for the specified date range.</li>")
                report_lines.append("<li>Verify that the selected vessel types exist in your dataset.</li>")
                report_lines.append("<li>Try analyzing a different time period where more data is available.</li>")
                report_lines.append("</ul>")
                
                report_lines.append("</body></html>")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(report_lines))
                
                logger.info(f"Created vessel clustering report explaining no data: {output_path}")
                return output_path
            
            # Filter by vessel types if specified
            if vessel_types is not None and len(vessel_types) > 0:
                if 'VesselType' in df.columns:
                    # Convert VesselType to int for comparison, handling NaN values
                    df['VesselType'] = df['VesselType'].astype('Int64')
                    vessel_types_int = [int(vt) for vt in vessel_types]
                    df = df[df['VesselType'].isin(vessel_types_int)]
                    logger.info(f"Filtered to {len(vessel_types)} vessel type(s): {vessel_types}")
                    
                    if df.empty:
                        logger.warning(f"No data found for selected vessel types: {vessel_types}")
                        logger.info("This may be because these vessel types are not in the current dataset.")
                        logger.info("The analysis will continue with available data, or you may need to run analysis with these vessel types included.")
                else:
                    logger.warning("VesselType column not found, cannot filter by vessel types")
            
            if df.empty:
                logger.warning("No data available for clustering after filtering by vessel types")
                # Reuse the same report creation logic as above
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                if output_path is None:
                    output_path = os.path.join(self.output_directory, f"Vessel_Clustering_NoData_{timestamp}.html")
                
                report_lines = []
                report_lines.append("<html><head><title>Vessel Behavior Clustering (No Data)</title>")
                report_lines.append("<style>body { font-family: Arial, sans-serif; margin: 20px; }")
                report_lines.append("h1 { color: #003366; } h2 { color: #0066cc; }")
                report_lines.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
                report_lines.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
                report_lines.append("th { background-color: #003366; color: white; }</style></head><body>")
                
                report_lines.append("<h1>Vessel Behavior Clustering (No Matching Data)</h1>")
                report_lines.append(f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
                report_lines.append("<h2>Analysis Parameters</h2>")
                report_lines.append("<table>")
                report_lines.append("<tr><th>Parameter</th><th>Value</th></tr>")
                report_lines.append(f"<tr><td>Analysis Period</td><td>{self.run_info['start_date']} to {self.run_info['end_date']}</td></tr>")
                
                if vessel_types is not None and len(vessel_types) > 0:
                    vessel_type_names = [f"{vt} ({get_vessel_type_name(vt)})" for vt in vessel_types]
                    report_lines.append(f"<tr><td>Selected Vessel Types</td><td>{', '.join(vessel_type_names)}</td></tr>")
                else:
                    report_lines.append(f"<tr><td>Selected Vessel Types</td><td>All</td></tr>")
                
                report_lines.append(f"<tr><td>Number of Clusters</td><td>{n_clusters}</td></tr>")
                report_lines.append(f"<tr><td>Data Status</td><td>No data found matching selected vessel types</td></tr>")
                report_lines.append("</table>")
                
                report_lines.append("<h2>Recommendations</h2>")
                report_lines.append("<ul>")
                report_lines.append("<li>Try selecting different vessel types that are present in the dataset.</li>")
                report_lines.append("<li>Verify that the selected vessel types exist in your dataset.</li>")
                report_lines.append("<li>Try analyzing a different time period where these vessel types have data.</li>")
                report_lines.append("</ul>")
                
                report_lines.append("</body></html>")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(report_lines))
                
                logger.info(f"Created vessel clustering report explaining no matching data: {output_path}")
                return output_path
            
            vessel_features = []
            
            for mmsi in df['MMSI'].unique():
                vessel_data = df[df['MMSI'] == mmsi]
                
                features = {
                    'MMSI': mmsi,
                    'avg_speed': vessel_data['SOG'].mean() if 'SOG' in vessel_data.columns else 0,
                    'max_speed': vessel_data['SOG'].max() if 'SOG' in vessel_data.columns else 0,
                    'total_records': len(vessel_data),
                    'lat_range': vessel_data['LAT'].max() - vessel_data['LAT'].min() if 'LAT' in vessel_data.columns else 0,
                    'lon_range': vessel_data['LON'].max() - vessel_data['LON'].min() if 'LON' in vessel_data.columns else 0,
                }
                
                if 'VesselType' in vessel_data.columns:
                    features['vessel_type'] = vessel_data['VesselType'].mode()[0] if not vessel_data['VesselType'].mode().empty else 0
                
                vessel_features.append(features)
            
            features_df = pd.DataFrame(vessel_features)
            
            feature_cols = [col for col in features_df.columns if col != 'MMSI']
            X = features_df[feature_cols].fillna(0)
            
            if not SKLEARN_AVAILABLE:
                logger.error("scikit-learn not available for clustering")
                return None
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            features_df['Cluster'] = kmeans.fit_predict(X)
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(self.output_directory, f"Vessel_Clustering_{timestamp}.html")
            
            if PLOTLY_AVAILABLE and 'lat_range' in feature_cols and 'lon_range' in feature_cols:
                fig = px.scatter(features_df, x='lat_range', y='lon_range', 
                                color='Cluster', size='total_records',
                                hover_data=['MMSI', 'avg_speed'],
                                title='Vessel Behavior Clusters')
                fig.write_html(output_path)
            else:
                features_df.to_csv(output_path.replace('.html', '.csv'), index=False)
                output_path = output_path.replace('.html', '.csv')
            
            logger.info(f"Clustering completed: {output_path}")
            return output_path
                
        except Exception as e:
            logger.error(f"Error in vessel behavior clustering: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def anomaly_frequency_analysis(self, output_path=None):
        """Analyze frequency and distribution of different anomaly types."""
        try:
            logger.info("Performing anomaly frequency analysis...")
            
            anomaly_df = self.load_anomaly_data()
            
            if anomaly_df.empty:
                logger.error("No anomaly data for frequency analysis")
                return None
            
            if 'AnomalyType' in anomaly_df.columns:
                freq_dist = anomaly_df['AnomalyType'].value_counts()
                total = len(anomaly_df)
                rel_freq = freq_dist / total * 100
                
                if output_path is None:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_path = os.path.join(self.output_directory, f"Anomaly_Frequency_{timestamp}.html")
                
                if PLOTLY_AVAILABLE:
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Absolute Frequency', 'Relative Frequency (%)'),
                        specs=[[{"type": "bar"}, {"type": "bar"}]]
                    )
                    
                    fig.add_trace(
                        go.Bar(x=freq_dist.index, y=freq_dist.values, name='Count'),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Bar(x=rel_freq.index, y=rel_freq.values, name='Percentage'),
                        row=1, col=2
                    )
                    
                    fig.update_layout(height=500, title_text="Anomaly Frequency Analysis")
                    fig.write_html(output_path)
                else:
                    freq_df = pd.DataFrame({
                        'AnomalyType': freq_dist.index,
                        'Count': freq_dist.values,
                        'Percentage': rel_freq.values
                    })
                    csv_path = output_path.replace('.html', '.csv')
                    freq_df.to_csv(csv_path, index=False)
                    output_path = csv_path
                
                logger.info(f"Anomaly frequency analysis completed: {output_path}")
                return output_path
            else:
                logger.error("AnomalyType column not found")
                return None
                
        except Exception as e:
            logger.error(f"Error in anomaly frequency analysis: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def create_custom_chart(self, chart_type, x_column=None, y_column=None, color_column=None, 
                           group_by=None, aggregation='count', title=None, output_path=None):
        """Create a custom chart based on user selections.
        
        Args:
            chart_type: Type of chart ('bar', 'scatter', 'line', 'pie', 'stacked_bar', 'timeline', 'histogram', 'box')
            x_column: Column name for x-axis
            y_column: Column name for y-axis (if applicable)
            color_column: Column name for color grouping
            group_by: Column name to group by
            aggregation: Aggregation function ('count', 'sum', 'mean', 'max', 'min')
            title: Chart title
            output_path: Output file path
        """
        try:
            logger.info(f"Creating custom chart: type={chart_type}, x={x_column}, y={y_column}")
            
            df = self.load_cached_data()
            anomaly_df = self.load_anomaly_data()
            
            # Determine which dataset to use
            if chart_type in ['timeline'] or (x_column and x_column in anomaly_df.columns):
                data_df = anomaly_df.copy()
            else:
                data_df = df.copy()
            
            if data_df.empty:
                logger.error("No data available for chart creation")
                return None
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(self.output_directory, f"Custom_Chart_{chart_type}_{timestamp}.html")
            
            if not PLOTLY_AVAILABLE:
                logger.error("Plotly not available for chart creation")
                return None
            
            # Prepare data based on chart type
            if group_by and group_by in data_df.columns:
                if aggregation == 'count':
                    chart_data = data_df.groupby(group_by).size().reset_index(name='Count')
                    x_col = group_by
                    y_col = 'Count'
                elif aggregation == 'sum' and y_column:
                    chart_data = data_df.groupby(group_by)[y_column].sum().reset_index()
                    x_col = group_by
                    y_col = y_column
                elif aggregation == 'mean' and y_column:
                    chart_data = data_df.groupby(group_by)[y_column].mean().reset_index()
                    x_col = group_by
                    y_col = y_column
                elif aggregation == 'max' and y_column:
                    chart_data = data_df.groupby(group_by)[y_column].max().reset_index()
                    x_col = group_by
                    y_col = y_column
                elif aggregation == 'min' and y_column:
                    chart_data = data_df.groupby(group_by)[y_column].min().reset_index()
                    x_col = group_by
                    y_col = y_column
                else:
                    chart_data = data_df.groupby(group_by).size().reset_index(name='Count')
                    x_col = group_by
                    y_col = 'Count'
            else:
                chart_data = data_df.copy()
                x_col = x_column
                y_col = y_column
            
            # Create chart based on type
            fig = None
            
            if chart_type == 'bar':
                if y_col:
                    fig = px.bar(chart_data, x=x_col, y=y_col, color=color_column, 
                                title=title or f"Bar Chart: {x_col} vs {y_col}")
                else:
                    value_counts = chart_data[x_col].value_counts() if x_col else pd.Series()
                    fig = go.Figure(data=[go.Bar(x=value_counts.index, y=value_counts.values)])
                    fig.update_layout(title=title or f"Bar Chart: {x_col}", xaxis_title=x_col, yaxis_title='Count')
            
            elif chart_type == 'scatter':
                if x_col and y_col:
                    fig = px.scatter(chart_data, x=x_col, y=y_col, color=color_column,
                                    title=title or f"Scatter Plot: {x_col} vs {y_col}")
                else:
                    logger.error("Scatter plot requires both x and y columns")
                    return None
            
            elif chart_type == 'line':
                if x_col and y_col:
                    fig = px.line(chart_data, x=x_col, y=y_col, color=color_column,
                                title=title or f"Line Chart: {x_col} vs {y_col}")
                else:
                    logger.error("Line chart requires both x and y columns")
                    return None
            
            elif chart_type == 'pie':
                if y_col:
                    fig = px.pie(chart_data, names=x_col, values=y_col, 
                               title=title or f"Pie Chart: {x_col}")
                else:
                    value_counts = chart_data[x_col].value_counts() if x_col else pd.Series()
                    fig = px.pie(values=value_counts.values, names=value_counts.index,
                                title=title or f"Pie Chart: {x_col}")
            
            elif chart_type == 'stacked_bar':
                if x_col and y_col and color_column:
                    fig = px.bar(chart_data, x=x_col, y=y_col, color=color_column,
                                title=title or f"Stacked Bar Chart: {x_col} by {color_column}",
                                barmode='stack')
                else:
                    logger.error("Stacked bar chart requires x, y, and color columns")
                    return None
            
            elif chart_type == 'timeline':
                if 'BaseDateTime' in chart_data.columns or 'Date' in chart_data.columns:
                    date_col = 'BaseDateTime' if 'BaseDateTime' in chart_data.columns else 'Date'
                    chart_data[date_col] = pd.to_datetime(chart_data[date_col])
                    chart_data['Date'] = chart_data[date_col].dt.date
                    
                    if group_by and group_by in chart_data.columns:
                        timeline_data = chart_data.groupby(['Date', group_by]).size().reset_index(name='Count')
                        fig = px.bar(timeline_data, x='Date', y='Count', color=group_by,
                                    title=title or f"Timeline: {group_by} over time",
                                    barmode='stack')
                    else:
                        timeline_data = chart_data.groupby('Date').size().reset_index(name='Count')
                        fig = px.bar(timeline_data, x='Date', y='Count',
                                    title=title or f"Timeline Chart")
                else:
                    logger.error("Timeline chart requires a date column (BaseDateTime or Date)")
                    return None
            
            elif chart_type == 'histogram':
                if x_col:
                    fig = px.histogram(chart_data, x=x_col, color=color_column,
                                      title=title or f"Histogram: {x_col}")
                else:
                    logger.error("Histogram requires an x column")
                    return None
            
            elif chart_type == 'box':
                if x_col and y_col:
                    fig = px.box(chart_data, x=x_col, y=y_col, color=color_column,
                                title=title or f"Box Plot: {x_col} vs {y_col}")
                else:
                    logger.error("Box plot requires both x and y columns")
                    return None
            
            else:
                logger.error(f"Unsupported chart type: {chart_type}")
                return None
            
            if fig:
                fig.update_layout(height=600)
                fig.write_html(output_path)
                logger.info(f"Custom chart created: {output_path}")
                return output_path
            else:
                logger.error("Failed to create chart")
                return None
                
        except Exception as e:
            logger.error(f"Error creating custom chart: {e}")
            logger.error(traceback.format_exc())
            return None
    
    # ========================================================================
    # TAB 3: MAPPING TOOLS
    # ========================================================================
    
    def create_full_spectrum_map(self, show_pins=True, show_heatmap=True, output_path=None):
        """Create a comprehensive map showing all anomalies."""
        try:
            if not FOLIUM_AVAILABLE:
                logger.error("folium not available for map creation")
                return None
            
            logger.info("Creating full spectrum anomaly map...")
            
            df = self.load_cached_data()
            anomaly_df = self.load_anomaly_data()
            
            if anomaly_df.empty:
                logger.error("No anomaly data for map")
                return None
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                map_dir = self.get_map_output_directory()
                output_path = os.path.join(map_dir, f"Full_Spectrum_Map_{timestamp}.html")
            
            center_lat = anomaly_df['LAT'].mean() if 'LAT' in anomaly_df.columns else 0
            center_lon = anomaly_df['LON'].mean() if 'LON' in anomaly_df.columns else 0
            m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
            
            if show_heatmap:
                valid_anomalies = anomaly_df.dropna(subset=['LAT', 'LON'])
                if not valid_anomalies.empty:
                    heat_data = [[row['LAT'], row['LON'], 1] for _, row in valid_anomalies.iterrows()]
                    HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
            
            if show_pins:
                marker_cluster = MarkerCluster().add_to(m)
                
                valid_anomalies = anomaly_df.dropna(subset=['LAT', 'LON'])
                for _, row in valid_anomalies.iterrows():
                    popup_text = f"MMSI: {row.get('MMSI', 'N/A')}<br>"
                    if 'AnomalyType' in row:
                        popup_text += f"Type: {row['AnomalyType']}<br>"
                    if 'BaseDateTime' in row:
                        popup_text += f"Time: {row['BaseDateTime']}"
                    
                    folium.Marker(
                        location=[row['LAT'], row['LON']],
                        popup=folium.Popup(popup_text, max_width=200)
                    ).add_to(marker_cluster)
            
            folium.LayerControl().add_to(m)
            
            m.save(output_path)
            logger.info(f"Full spectrum map created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating full spectrum map: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def create_vessel_map(self, mmsi, map_type='path', output_path=None):
        """Create maps focused on specific vessels by MMSI."""
        try:
            if not FOLIUM_AVAILABLE:
                logger.error("folium not available for map creation")
                return None
            
            logger.info(f"Creating {map_type} map for vessel {mmsi} using full daily datasets...")
            
            # Use full daily datasets for vessel-specific analysis
            df = self.load_full_daily_datasets()
            anomaly_df = self.load_anomaly_data()
            
            if df.empty:
                logger.error("No data available for vessel map")
                return None
            
            # Convert MMSI to same type as in dataframe for proper comparison
            if 'MMSI' in df.columns:
                # Try to match the type in the dataframe
                sample_mmsi = df['MMSI'].iloc[0] if len(df) > 0 else None
                if sample_mmsi is not None:
                    if isinstance(sample_mmsi, (int, np.integer)):
                        mmsi = int(mmsi)
                    elif isinstance(sample_mmsi, (str, np.str_)):
                        mmsi = str(mmsi)
                # Also ensure dataframe MMSI is consistent type
                df['MMSI'] = df['MMSI'].astype(type(mmsi))
            
            vessel_data = df[df['MMSI'] == mmsi].copy()
            
            if vessel_data.empty:
                logger.error(f"No data found for vessel {mmsi}")
                return None
            
            # Filter anomalies for this vessel
            if not anomaly_df.empty and 'MMSI' in anomaly_df.columns:
                # Ensure MMSI types match
                anomaly_df['MMSI'] = anomaly_df['MMSI'].astype(type(mmsi))
                vessel_anomalies = anomaly_df[anomaly_df['MMSI'] == mmsi].copy()
            else:
                vessel_anomalies = pd.DataFrame()
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                map_dir = self.get_map_output_directory()
                output_path = os.path.join(map_dir, f"Vessel_{mmsi}_{map_type}_{timestamp}.html")
            
            center_lat = vessel_data['LAT'].mean()
            center_lon = vessel_data['LON'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
            
            if map_type == 'path':
                # Sort by datetime if available for proper path ordering
                if 'BaseDateTime' in vessel_data.columns:
                    vessel_data = vessel_data.sort_values('BaseDateTime')
                
                path_coords = [[row['LAT'], row['LON']] for _, row in vessel_data.iterrows() 
                              if pd.notna(row['LAT']) and pd.notna(row['LON'])]
                if path_coords:
                    folium.PolyLine(path_coords, color='blue', weight=3, opacity=0.7).add_to(m)
                    folium.Marker(path_coords[0], popup='Start', icon=folium.Icon(color='green')).add_to(m)
                    folium.Marker(path_coords[-1], popup='End', icon=folium.Icon(color='red')).add_to(m)
                    logger.info(f"Path map: Added {len(path_coords)} points for vessel {mmsi}")
                else:
                    logger.warning(f"No valid coordinates found for path map")
                
                # Add anomalies to path map
                if not vessel_anomalies.empty:
                    marker_count = 0
                    for _, row in vessel_anomalies.iterrows():
                        if pd.notna(row.get('LAT')) and pd.notna(row.get('LON')):
                            popup_text = f"<b>Anomaly Detected</b><br>"
                            popup_text += f"MMSI: {mmsi}<br>"
                            popup_text += f"Anomaly Type: {row.get('AnomalyType', 'Unknown')}<br>"
                            if 'BaseDateTime' in row and pd.notna(row.get('BaseDateTime')):
                                popup_text += f"Time: {row['BaseDateTime']}<br>"
                            if 'LAT' in row and pd.notna(row.get('LAT')):
                                popup_text += f"Latitude: {row['LAT']:.6f}°<br>"
                            if 'LON' in row and pd.notna(row.get('LON')):
                                popup_text += f"Longitude: {row['LON']:.6f}°"
                            
                            folium.Marker(
                                [row['LAT'], row['LON']],
                                popup=folium.Popup(popup_text, max_width=250),
                                icon=folium.Icon(color='red', icon='exclamation-sign', prefix='fa')
                            ).add_to(m)
                            marker_count += 1
                    logger.info(f"Path map: Added {marker_count} anomaly markers for vessel {mmsi}")
                else:
                    logger.info(f"Path map: No anomalies found for vessel {mmsi}")
            
            elif map_type == 'anomaly':
                if not vessel_anomalies.empty:
                    marker_count = 0
                    for _, row in vessel_anomalies.iterrows():
                        if pd.notna(row.get('LAT')) and pd.notna(row.get('LON')):
                            popup_text = f"MMSI: {mmsi}<br>Anomaly: {row.get('AnomalyType', 'Unknown')}"
                            if 'BaseDateTime' in row:
                                popup_text += f"<br>Time: {row['BaseDateTime']}"
                            folium.Marker(
                                [row['LAT'], row['LON']],
                                popup=folium.Popup(popup_text, max_width=200),
                                icon=folium.Icon(color='red', icon='exclamation-sign')
                            ).add_to(m)
                            marker_count += 1
                    logger.info(f"Anomaly map: Added {marker_count} anomaly markers for vessel {mmsi}")
                else:
                    logger.warning(f"No anomalies found for vessel {mmsi}")
            
            elif map_type == 'heatmap':
                heat_data = [[row['LAT'], row['LON'], 1] for _, row in vessel_data.iterrows()
                            if pd.notna(row['LAT']) and pd.notna(row['LON'])]
                if heat_data:
                    HeatMap(heat_data, radius=15, blur=10).add_to(m)
                    logger.info(f"Heatmap: Added {len(heat_data)} heat points for vessel {mmsi}")
                else:
                    logger.warning(f"No valid coordinates found for heatmap")
            
            m.save(output_path)
            logger.info(f"Vessel map created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating vessel map: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def get_top_vessels_by_anomaly(self, anomaly_type=None, limit=10):
        """Get top vessels by anomaly count for a specific anomaly type."""
        try:
            anomaly_df = self.load_anomaly_data()
            
            if anomaly_df.empty:
                return pd.DataFrame()
            
            if anomaly_type and 'AnomalyType' in anomaly_df.columns:
                filtered = anomaly_df[anomaly_df['AnomalyType'] == anomaly_type]
            else:
                filtered = anomaly_df
            
            if 'MMSI' in filtered.columns:
                top_vessels = filtered['MMSI'].value_counts().head(limit).reset_index()
                top_vessels.columns = ['MMSI', 'Anomaly_Count']
                return top_vessels
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting top vessels: {e}")
            return pd.DataFrame()
    
    def create_filtered_map(self, map_types=None, map_type=None, vessel_types=None, anomaly_types=None, vessel_mmsi=None, output_path=None, use_full_datasets=False):
        """Create filtered maps for specific anomaly types and/or vessel types, or specific vessel + anomaly type.
        
        Args:
            map_types: List of map types to create ('path', 'anomaly', 'heatmap') - multiple layers can be selected
            map_type: Single map type (deprecated, use map_types instead) - for backward compatibility
            vessel_types: List of vessel types to filter
            anomaly_types: List of anomaly types to filter
            vessel_mmsi: Specific vessel MMSI to filter
            output_path: Path to save the map
            use_full_datasets: If True, use full daily datasets instead of filtered/consolidated data
        """
        # Backward compatibility: if map_type is provided but map_types is not, convert it
        if map_types is None:
            if map_type is not None:
                map_types = [map_type]
            else:
                map_types = ['path']  # Default to path if neither is provided
        try:
            if not FOLIUM_AVAILABLE:
                logger.error("folium not available for map creation")
                return None
            
            map_types_str = ', '.join(map_types)
            logger.info(f"Creating filtered map with layers: {map_types_str}, vessel_types={vessel_types}, anomaly_types={anomaly_types}, vessel_mmsi={vessel_mmsi}, use_full_datasets={use_full_datasets}")
            
            # Use full datasets for vessel-specific analysis, filtered data for general analysis
            if use_full_datasets or vessel_mmsi is not None:
                logger.info("Using full daily datasets for vessel-specific analysis")
                df = self.load_full_daily_datasets()
            else:
                df = self.load_cached_data()
            anomaly_df = self.load_anomaly_data()
            
            # Filter by vessel MMSI if specified
            if vessel_mmsi:
                df = df[df['MMSI'] == vessel_mmsi].copy()
                anomaly_df = anomaly_df[anomaly_df['MMSI'] == vessel_mmsi].copy() if not anomaly_df.empty and 'MMSI' in anomaly_df.columns else pd.DataFrame()
            
            # Filter by vessel types if specified
            if vessel_types and 'VesselType' in df.columns:
                # Convert vessel types to int for proper comparison
                vessel_types_int = [int(vt) for vt in vessel_types]
                df['VesselType'] = pd.to_numeric(df['VesselType'], errors='coerce').astype('Int64')
                df = df[df['VesselType'].isin(vessel_types_int)].copy()
                vessel_mmsis = df['MMSI'].unique()
                anomaly_df = anomaly_df[anomaly_df['MMSI'].isin(vessel_mmsis)].copy() if not anomaly_df.empty and 'MMSI' in anomaly_df.columns else pd.DataFrame()
            
            # Filter by anomaly types if specified
            if anomaly_types and 'AnomalyType' in anomaly_df.columns:
                anomaly_df = anomaly_df[anomaly_df['AnomalyType'].isin(anomaly_types)].copy()
                if vessel_mmsi is None:  # Only filter df if not already filtered by MMSI
                    vessel_mmsis = anomaly_df['MMSI'].unique()
                    df = df[df['MMSI'].isin(vessel_mmsis)].copy()
            
            if df.empty:
                logger.error("No data found for filtered map")
                return None
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                map_dir = self.get_map_output_directory()
                map_types_str_file = '_'.join(map_types)
                filter_str = f"filtered_{map_types_str_file}_{timestamp}"
                if vessel_mmsi:
                    filter_str = f"vessel_{vessel_mmsi}_{map_types_str_file}_{timestamp}"
                output_path = os.path.join(map_dir, f"{filter_str}.html")
            
            center_lat = df['LAT'].mean() if 'LAT' in df.columns else 0
            center_lon = df['LON'].mean() if 'LON' in df.columns else 0
            m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
            
            # Create FeatureGroups for each layer type so they can be toggled
            path_group = folium.FeatureGroup(name='Path Map', show=True)
            anomaly_group = folium.FeatureGroup(name='Anomaly Map', show=True)
            heatmap_group = folium.FeatureGroup(name='Heatmap', show=True)
            
            # Add Path Map layer if selected
            if 'path' in map_types:
                path_count = 0
                for mmsi in df['MMSI'].unique():
                    vessel_data = df[df['MMSI'] == mmsi]
                    path_coords = [[row['LAT'], row['LON']] for _, row in vessel_data.iterrows() 
                                  if pd.notna(row['LAT']) and pd.notna(row['LON'])]
                    if path_coords:
                        folium.PolyLine(path_coords, color='blue', weight=2, opacity=0.7, 
                                       popup=f"Vessel {mmsi}").add_to(path_group)
                        path_count += 1
                path_group.add_to(m)
                logger.info(f"Path map: Added {path_count} vessel paths")
            
            # Add Anomaly Map layer if selected
            if 'anomaly' in map_types:
                if not anomaly_df.empty:
                    marker_cluster = MarkerCluster().add_to(anomaly_group)
                    anomaly_count = 0
                    
                    # Create a lookup dictionary for vessel info from main dataframe
                    vessel_info_lookup = {}
                    if not df.empty and 'MMSI' in df.columns:
                        for mmsi in df['MMSI'].unique():
                            vessel_data = df[df['MMSI'] == mmsi]
                            if not vessel_data.empty:
                                # Get most common vessel name and type
                                vessel_name = vessel_data['VesselName'].mode().iloc[0] if 'VesselName' in vessel_data.columns and not vessel_data['VesselName'].mode().empty else 'Unknown'
                                vessel_type = vessel_data['VesselType'].mode().iloc[0] if 'VesselType' in vessel_data.columns and not vessel_data['VesselType'].mode().empty else None
                                vessel_info_lookup[mmsi] = {
                                    'name': vessel_name if pd.notna(vessel_name) else 'Unknown',
                                    'type': vessel_type if pd.notna(vessel_type) else None
                                }
                    
                    for _, row in anomaly_df.iterrows():
                        # Get matching record from main dataframe if available for additional fields
                        mmsi = row.get('MMSI', None)
                        anomaly_time = row.get('BaseDateTime', None) if 'BaseDateTime' in row else None
                        matching_vessel_data = None
                        
                        if mmsi is not None and not df.empty:
                            vessel_records = df[df['MMSI'] == mmsi]
                            if not vessel_records.empty and anomaly_time is not None:
                                # Try to find the closest matching record by time
                                if 'BaseDateTime' in vessel_records.columns:
                                    vessel_records = vessel_records.copy()
                                    vessel_records['BaseDateTime'] = pd.to_datetime(vessel_records['BaseDateTime'], errors='coerce')
                                    anomaly_time_dt = pd.to_datetime(anomaly_time, errors='coerce')
                                    if pd.notna(anomaly_time_dt):
                                        # Find the closest record by time
                                        time_diffs = (vessel_records['BaseDateTime'] - anomaly_time_dt).abs()
                                        closest_idx = time_diffs.idxmin()
                                        matching_vessel_data = vessel_records.loc[closest_idx]
                            elif not vessel_records.empty:
                                # If no time match, use the first record
                                matching_vessel_data = vessel_records.iloc[0]
                        if pd.notna(row.get('LAT')) and pd.notna(row.get('LON')):
                            if mmsi is None:
                                mmsi = 'N/A'
                            
                            # Build comprehensive popup text
                            popup_text = f"<b>Anomaly Detected</b><br><br>"
                            
                            # Vessel Name - check anomaly row first, then lookup, then matching vessel data
                            vessel_name = row.get('VesselName', None)
                            if pd.isna(vessel_name) or vessel_name == '':
                                if matching_vessel_data is not None:
                                    try:
                                        if 'VesselName' in matching_vessel_data.index:
                                            vessel_name = matching_vessel_data['VesselName']
                                    except (KeyError, AttributeError):
                                        pass
                                if pd.isna(vessel_name) or vessel_name == '':
                                    vessel_name = vessel_info_lookup.get(mmsi, {}).get('name', 'Unknown')
                            if pd.isna(vessel_name) or vessel_name == '':
                                vessel_name = 'Unknown'
                            popup_text += f"<b>Vessel Name:</b> {vessel_name}<br>"
                            
                            # MMSI
                            popup_text += f"<b>MMSI:</b> {mmsi}<br>"
                            
                            # Vessel Type - check anomaly row first, then lookup, then matching vessel data
                            vessel_type = row.get('VesselType', None)
                            if pd.isna(vessel_type) or vessel_type == '':
                                if matching_vessel_data is not None:
                                    try:
                                        if 'VesselType' in matching_vessel_data.index:
                                            vessel_type = matching_vessel_data['VesselType']
                                    except (KeyError, AttributeError):
                                        pass
                                if pd.isna(vessel_type) or vessel_type == '':
                                    vessel_type = vessel_info_lookup.get(mmsi, {}).get('type', None)
                            
                            if vessel_type is not None and pd.notna(vessel_type):
                                try:
                                    vessel_type_int = int(vessel_type)
                                    vessel_type_name = get_vessel_type_name(vessel_type_int)
                                    popup_text += f"<b>Vessel Type:</b> {vessel_type_int} ({vessel_type_name})<br>"
                                except (ValueError, TypeError):
                                    popup_text += f"<b>Vessel Type:</b> {vessel_type}<br>"
                            else:
                                popup_text += f"<b>Vessel Type:</b> Unknown<br>"
                            
                            # Speed (SOG) - check anomaly row first, then matching vessel data
                            sog = row.get('SOG', None)
                            if pd.isna(sog) or sog == '':
                                if matching_vessel_data is not None:
                                    try:
                                        if 'SOG' in matching_vessel_data.index:
                                            sog = matching_vessel_data['SOG']
                                    except (KeyError, AttributeError):
                                        pass
                            
                            if pd.notna(sog) and sog != '':
                                try:
                                    popup_text += f"<b>Speed (SOG):</b> {float(sog):.2f} knots<br>"
                                except (ValueError, TypeError):
                                    popup_text += f"<b>Speed (SOG):</b> {sog} knots<br>"
                            else:
                                popup_text += f"<b>Speed (SOG):</b> N/A<br>"
                            
                            # Course (CoG) - check anomaly row first, then matching vessel data
                            cog = row.get('COG', None)
                            if pd.isna(cog) or cog == '':
                                if matching_vessel_data is not None:
                                    try:
                                        if 'COG' in matching_vessel_data.index:
                                            cog = matching_vessel_data['COG']
                                    except (KeyError, AttributeError):
                                        pass
                            
                            if pd.notna(cog) and cog != '':
                                try:
                                    popup_text += f"<b>Course (CoG):</b> {float(cog):.2f}°<br>"
                                except (ValueError, TypeError):
                                    popup_text += f"<b>Course (CoG):</b> {cog}°<br>"
                            else:
                                popup_text += f"<b>Course (CoG):</b> N/A<br>"
                            
                            # Heading - check anomaly row first, then matching vessel data
                            heading = row.get('Heading', None)
                            if pd.isna(heading) or heading == '':
                                if matching_vessel_data is not None:
                                    try:
                                        if 'Heading' in matching_vessel_data.index:
                                            heading = matching_vessel_data['Heading']
                                    except (KeyError, AttributeError):
                                        pass
                            
                            if pd.notna(heading) and heading != '':
                                try:
                                    popup_text += f"<b>Heading:</b> {float(heading):.2f}°<br>"
                                except (ValueError, TypeError):
                                    popup_text += f"<b>Heading:</b> {heading}°<br>"
                            else:
                                popup_text += f"<b>Heading:</b> N/A<br>"
                            
                            # Anomaly Type
                            anomaly_type = row.get('AnomalyType', 'Unknown')
                            popup_text += f"<b>Anomaly Type:</b> {anomaly_type}<br>"
                            
                            # Time
                            if 'BaseDateTime' in row and pd.notna(row.get('BaseDateTime')):
                                time_val = row['BaseDateTime']
                                if isinstance(time_val, pd.Timestamp):
                                    time_str = time_val.strftime('%Y-%m-%d %H:%M:%S')
                                else:
                                    time_str = str(time_val)
                                popup_text += f"<b>Time:</b> {time_str}"
                            else:
                                popup_text += f"<b>Time:</b> N/A"
                            
                            folium.Marker(
                                [row['LAT'], row['LON']],
                                popup=folium.Popup(popup_text, max_width=300),
                                icon=folium.Icon(color='red', icon='exclamation-sign', prefix='fa')
                            ).add_to(marker_cluster)
                            anomaly_count += 1
                    anomaly_group.add_to(m)
                    logger.info(f"Anomaly map: Added {anomaly_count} anomaly markers")
                else:
                    logger.info(f"Anomaly map: No anomalies found")
                    anomaly_group.add_to(m)  # Add empty group so it appears in layer control
            
            # Add Heatmap layer if selected
            if 'heatmap' in map_types:
                if not anomaly_df.empty:
                    heat_data = [[row['LAT'], row['LON'], 1] for _, row in anomaly_df.iterrows()
                                if pd.notna(row.get('LAT')) and pd.notna(row.get('LON'))]
                else:
                    heat_data = [[row['LAT'], row['LON'], 1] for _, row in df.iterrows()
                                if pd.notna(row['LAT']) and pd.notna(row['LON'])]
                if heat_data:
                    HeatMap(heat_data, radius=15, blur=10).add_to(heatmap_group)
                    heatmap_group.add_to(m)
                    logger.info(f"Heatmap: Added {len(heat_data)} heat points")
                else:
                    logger.info(f"Heatmap: No data points found")
                    heatmap_group.add_to(m)  # Add empty group so it appears in layer control
            
            # Add layer control to allow toggling layers on/off
            folium.LayerControl().add_to(m)
            
            m.save(output_path)
            logger.info(f"Filtered map created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating filtered map: {e}")
            logger.error(traceback.format_exc())
            return None
    
    # ========================================================================
    # TAB 4: VESSEL-SPECIFIC ANALYSIS
    # ========================================================================
    
    def extended_time_analysis(self, mmsi, additional_days_start, additional_days_end, output_path=None):
        """Analyze additional days of data for a specific vessel."""
        try:
            logger.info(f"Performing extended time analysis for vessel {mmsi}...")
            
            is_valid, error_msg = validate_date_range(additional_days_start, additional_days_end)
            if not is_valid:
                logger.error(f"Invalid date range: {error_msg}")
                return None
            
            # Use full daily datasets for original period (not filtered/consolidated data)
            logger.info("Loading full daily datasets for original period...")
            original_df = self.load_full_daily_datasets()
            original_vessel = original_df[original_df['MMSI'] == mmsi] if not original_df.empty else pd.DataFrame()
            
            # Use full daily datasets for extended period (not filtered by vessel types)
            logger.info(f"Loading full daily datasets for extended period: {additional_days_start} to {additional_days_end}...")
            extended_df = self.load_full_daily_datasets(
                start_date=additional_days_start,
                end_date=additional_days_end
            )
            extended_vessel = extended_df[extended_df['MMSI'] == mmsi] if not extended_df.empty else pd.DataFrame()
            
            if original_vessel.empty and extended_vessel.empty:
                logger.error(f"No data found for vessel {mmsi}")
                return None
            
            combined_vessel = pd.concat([original_vessel, extended_vessel], ignore_index=True) if not extended_vessel.empty else original_vessel
            
            stats = {
                'MMSI': mmsi,
                'Original_Period_Records': len(original_vessel),
                'Extended_Period_Records': len(extended_vessel),
                'Total_Records': len(combined_vessel),
            }
            
            if 'BaseDateTime' in combined_vessel.columns:
                combined_vessel['BaseDateTime'] = pd.to_datetime(combined_vessel['BaseDateTime'])
                stats['First_Seen'] = combined_vessel['BaseDateTime'].min()
                stats['Last_Seen'] = combined_vessel['BaseDateTime'].max()
                stats['Total_Days'] = (stats['Last_Seen'] - stats['First_Seen']).days + 1
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                map_dir = self.get_map_output_directory()
                output_path = os.path.join(map_dir, f"Extended_Analysis_{mmsi}_{timestamp}.html")
            
            html_content = ["<html><head><title>Extended Time Analysis</title></head><body>"]
            html_content.append(f"<h1>Extended Time Analysis for Vessel {mmsi}</h1>")
            html_content.append("<h2>Statistics</h2>")
            html_content.append("<table border='1'><tr><th>Metric</th><th>Value</th></tr>")
            
            for key, value in stats.items():
                html_content.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
            
            html_content.append("</table></body></html>")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(html_content))
            
            logger.info(f"Extended time analysis completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error in extended time analysis: {e}")
            logger.error(traceback.format_exc())
            return None


# ============================================================================
# GUI INTERFACE CLASS
# ============================================================================

class ProgressDialog:
    """Progress dialog for long-running operations."""
    
    def __init__(self, parent, title="Processing", message="Please wait..."):
        self.parent = parent
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x150")
        self.dialog.transient(parent)
        self.dialog.resizable(False, False)
        
        self.message_var = tk.StringVar(value=message)
        ttk.Label(self.dialog, textvariable=self.message_var, 
                 font=("Arial", 10)).pack(pady=10)
        
        self.progress = ttk.Progressbar(self.dialog, mode="indeterminate", length=300)
        self.progress.pack(pady=10, padx=20, fill=tk.X)
        self.progress.start()
        
        self.status_var = tk.StringVar(value="")
        ttk.Label(self.dialog, textvariable=self.status_var, 
                 font=("Arial", 9)).pack(pady=5)
        
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        self.dialog.grab_set()
        self.dialog.focus_set()
    
    def update_message(self, message):
        self.message_var.set(message)
        self.dialog.update()
    
    def update_status(self, status):
        self.status_var.set(status)
        self.dialog.update()
    
    def close(self):
        self.progress.stop()
        self.dialog.destroy()


class AdvancedAnalysisGUI:
    """GUI interface for Advanced Analysis features."""
    
    def __init__(self, parent_window, output_directory=None, config_path='config.ini'):
        self.parent_window = parent_window
        self.analysis = None  # Initialize to None for safety
        
        # Window will be created after successful initialization of analysis
        self.window = None
        
        try:
            # Resolve config path relative to script directory
            logger.info(f"Initializing advanced analysis with output_directory={output_directory}, config_path={config_path}")
            resolved_config_path = get_config_path(config_path)
            logger.info(f"Resolved config path: {resolved_config_path}")
            self.analysis = AdvancedAnalysis(parent_window, output_directory, resolved_config_path)
        except Exception as e:
            error_msg = f"Failed to initialize Advanced Analysis:\n{str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", error_msg)
            
            # Create an error window
            self.window = tk.Toplevel(parent_window)
            self.window.title("Advanced Analysis Error")
            self.window.geometry("900x800")
            
            # Create a basic frame with error info
            main_frame = ttk.Frame(self.window, padding=10)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            ttk.Label(main_frame, text="Advanced Analysis Initialization Error", 
                     font=("Arial", 16, "bold"), foreground="red").pack(pady=20)
            ttk.Label(main_frame, text=error_msg, wraplength=700).pack(pady=20)
            ttk.Button(main_frame, text="Close", command=self.window.destroy).pack(pady=20)
            
            # Create empty status bar
            self.status_var = tk.StringVar(value="Error: Failed to initialize")
            ttk.Label(self.window, textvariable=self.status_var, relief="sunken", anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)
            return

        # Variables for anomaly types tab
        self.anomaly_types = {
            "AIS Beacon Off": tk.BooleanVar(value=True),
            "AIS Beacon On": tk.BooleanVar(value=True),
            "Excessive Travel Distance (Fast)": tk.BooleanVar(value=True),
            "Excessive Travel Distance (Slow)": tk.BooleanVar(value=True),
            "Course over Ground-Heading Inconsistency": tk.BooleanVar(value=True),
            "Loitering": tk.BooleanVar(value=True),
            "Rendezvous": tk.BooleanVar(value=True),
            "Identity Spoofing": tk.BooleanVar(value=True),
            "Zone Violations": tk.BooleanVar(value=True)
        }
        self.min_travel_nm = tk.DoubleVar(value=200)
        self.max_travel_nm = tk.DoubleVar(value=550)
        self.cog_heading_max_diff = tk.DoubleVar(value=45)
        self.min_speed_for_cog_check = tk.DoubleVar(value=10)
        
        # Variables for analysis filters tab
        self.analysis_filters = {
            # Geographic boundaries
            'min_latitude': tk.DoubleVar(value=-90.0),
            'max_latitude': tk.DoubleVar(value=90.0),
            'min_longitude': tk.DoubleVar(value=-180.0),
            'max_longitude': tk.DoubleVar(value=180.0),
            # Time filters
            'time_start_hour': tk.IntVar(value=0),
            'time_end_hour': tk.IntVar(value=24),
            # Confidence and anomaly limits
            'min_confidence': tk.IntVar(value=75),
            'max_anomalies_per_vessel': tk.IntVar(value=10),
            # MMSI filter list (as string)
            'filter_mmsi_list': tk.StringVar(value='')
        }
        
        # Variables for zone violations tab
        self.zone_violations = []  # List of dicts: {'name': str, 'lat_min': float, 'lat_max': float, 'lon_min': float, 'lon_max': float, 'is_selected': bool}
        self.zone_checkboxes = {}  # Maps zone name to {'selected': BooleanVar, 'frame': Frame}
        self.zone_frames = {}  # Maps zone name to frame widget
        
        # Ship type variables for vessel selection tab
        self.ship_types = {}
        # Initialize ship types by categories
        self._init_ship_types()
        
        # Get default date range from analysis run info (with fallback)
        self.default_start_date = self.analysis.run_info.get('start_date', '2024-10-01')
        self.default_end_date = self.analysis.run_info.get('end_date', '2024-10-03')
        
        # Create the GUI
        self.window = tk.Toplevel(parent_window)
        self.window.title("Advanced Analytical Tools GUI")
        self.window.geometry("900x700")
        self.window.minsize(800, 600)
        
        # Ensure this window appears in front of other windows
        self.window.lift()
        self.window.attributes('-topmost', True)
        self.window.after(10, lambda: self.window.attributes('-topmost', False))
        self.window.focus_force()
        
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Advanced Analytical Tools", 
                 font=("Arial", 16, "bold")).pack(anchor=tk.W)
        ttk.Label(main_frame, 
                 text="Perform additional analysis on the previously generated dataset",
                 font=("Arial", 11)).pack(anchor=tk.W, pady=(0, 15))
        
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self._create_tab1_additional_outputs()
        self._create_tab2_further_analysis()
        self._create_tab3_mapping_tools()
        self._create_tab4_vessel_analysis()
        self._create_tab5_anomaly_types()
        self._create_tab6_analysis_filters()
        self._create_tab7_zone_violations()
        self._create_tab8_vessel_selection()
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.window, textvariable=self.status_var, relief="sunken", anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)
        
    def select_all_anomalies(self):
        """Select all anomaly types"""
        for var in self.anomaly_types.values():
            var.set(True)
    
    def deselect_all_anomalies(self):
        """Deselect all anomaly types"""
        for var in self.anomaly_types.values():
            var.set(False)
            
    def select_all_ships(self):
        """Select all ship types"""
        for ship_type in self.ship_types:
            self.ship_types[ship_type]['var'].set(True)
    
    def deselect_all_ships(self):
        """Deselect all ship types"""
        for ship_type in self.ship_types:
            self.ship_types[ship_type]['var'].set(False)
            
    def select_category(self, category):
        """Select all ships of a specific category"""
        for ship_type, details in self.ship_types.items():
            if details.get('category', '') == category:
                self.ship_types[ship_type]['var'].set(True)
                
    def deselect_category(self, category):
        """Deselect all ships of a specific category"""
        for ship_type, details in self.ship_types.items():
            if details.get('category', '') == category:
                self.ship_types[ship_type]['var'].set(False)
            
    def _init_ship_types(self):
        """Initialize the ship types dictionary"""
        # Wing in ground (WIG)
        for i in range(20, 25):
            category = 'WIG'
            if i == 20:
                name = 'Wing in ground (WIG), all ships of this type'
            else:
                name = f'Wing in ground (WIG), Hazardous category {chr(64 + (i-20))}'
            self.ship_types[i] = {'name': name, 'var': tk.BooleanVar(value=True), 'category': category}
        
        # Special craft
        special_craft = [
            (30, 'Fishing'),
            (31, 'Towing'),
            (32, 'Towing: length exceeds 200m or breadth exceeds 25m'),
            (33, 'Dredging or underwater ops'),
            (34, 'Diving ops'),
            (35, 'Military ops'),
            (36, 'Sailing'),
            (37, 'Pleasure Craft'),
            (38, 'Reserved'),
            (39, 'Reserved')
        ]
        for code, name in special_craft:
            self.ship_types[code] = {'name': name, 'var': tk.BooleanVar(value=True), 'category': 'Special'}
        
        # High speed craft (HSC)
        for i in range(40, 45):
            category = 'HSC'
            if i == 40:
                name = 'High speed craft (HSC), all ships of this type'
            else:
                name = f'High speed craft (HSC), Hazardous category {chr(64 + (i-40))}'
            self.ship_types[i] = {'name': name, 'var': tk.BooleanVar(value=True), 'category': category}
        self.ship_types[49] = {'name': 'High speed craft (HSC), No additional information', 'var': tk.BooleanVar(value=True), 'category': 'HSC'}
        
        # Special purpose
        special_purpose = [
            (50, 'Pilot Vessel'),
            (51, 'Search and Rescue vessel'),
            (52, 'Tug'),
            (53, 'Port Tender'),
            (54, 'Anti-pollution equipment'),
            (55, 'Law Enforcement'),
            (56, 'Spare - Local Vessel'),
            (57, 'Spare - Local Vessel'),
            (58, 'Medical Transport'),
            (59, 'Noncombatant ship')
        ]
        for code, name in special_purpose:
            self.ship_types[code] = {'name': name, 'var': tk.BooleanVar(value=True), 'category': 'Special Purpose'}
        
        # Passenger
        for i in range(60, 65):
            category = 'Passenger'
            if i == 60:
                name = 'Passenger, all ships of this type'
            else:
                name = f'Passenger, Hazardous category {chr(64 + (i-60))}'
            self.ship_types[i] = {'name': name, 'var': tk.BooleanVar(value=True), 'category': category}
        self.ship_types[69] = {'name': 'Passenger, No additional information', 'var': tk.BooleanVar(value=True), 'category': 'Passenger'}
        
        # Cargo
        for i in range(70, 75):
            category = 'Cargo'
            if i == 70:
                name = 'Cargo, all ships of this type'
            else:
                name = f'Cargo, Hazardous category {chr(64 + (i-70))}'
            self.ship_types[i] = {'name': name, 'var': tk.BooleanVar(value=True), 'category': category}
        self.ship_types[79] = {'name': 'Cargo, No additional information', 'var': tk.BooleanVar(value=True), 'category': 'Cargo'}
        
        # Tanker
        for i in range(80, 85):
            category = 'Tanker'
            if i == 80:
                name = 'Tanker, all ships of this type'
            else:
                name = f'Tanker, Hazardous category {chr(64 + (i-80))}'
            self.ship_types[i] = {'name': name, 'var': tk.BooleanVar(value=True), 'category': category}
        self.ship_types[89] = {'name': 'Tanker, No additional information', 'var': tk.BooleanVar(value=True), 'category': 'Tanker'}
        
        # Other
        for i in range(90, 95):
            category = 'Other'
            if i == 90:
                name = 'Other Type, all ships of this type'
            else:
                name = f'Other Type, Hazardous category {chr(64 + (i-90))}'
            self.ship_types[i] = {'name': name, 'var': tk.BooleanVar(value=True), 'category': category}
            
    def add_zone(self):
        """Add a new zone violation area - matches SFD_GUI implementation"""
        self._show_zone_dialog()
    
    def select_all_zones(self):
        """Select all zone violations"""
        for zone in self.zone_violations:
            zone['is_selected'] = True
            if zone['name'] in self.zone_checkboxes:
                self.zone_checkboxes[zone['name']]['selected'].set(True)
        self._save_zones_to_config()
    
    def deselect_all_zones(self):
        """Deselect all zone violations"""
        for zone in self.zone_violations:
            zone['is_selected'] = False
            if zone['name'] in self.zone_checkboxes:
                self.zone_checkboxes[zone['name']]['selected'].set(False)
        self._save_zones_to_config()
                    
    def _load_zones_from_config(self):
        """Load zone definitions from config.ini"""
        config = configparser.ConfigParser()
        try:
            config.read(self.analysis.config_path)
            
            # Reset the zones list
            self.zone_violations = []
            
            if 'ZONE_VIOLATIONS' in config:
                zone_indices = set()
                # Find all zone indices
                for key in config['ZONE_VIOLATIONS']:
                    if key.startswith('zone_') and '_name' in key:
                        try:
                            zone_idx = int(key.split('_')[1])
                            zone_indices.add(zone_idx)
                        except (ValueError, IndexError):
                            continue
                
                # Load each zone
                import json
                for i in sorted(zone_indices):
                    zone_key = f'zone_{i}'
                    try:
                        geometry_type = config.get('ZONE_VIOLATIONS', f'{zone_key}_geometry_type', fallback='rectangle')
                        zone = {
                            'name': config['ZONE_VIOLATIONS'][f'{zone_key}_name'],
                            'geometry_type': geometry_type,
                            'is_selected': config.getboolean('ZONE_VIOLATIONS', f'{zone_key}_is_selected', fallback=True)
                        }
                        
                        # Load coordinates based on geometry type
                        if geometry_type == 'rectangle':
                            # Legacy format for backward compatibility
                            zone['lat_min'] = float(config['ZONE_VIOLATIONS'][f'{zone_key}_lat_min'])
                            zone['lat_max'] = float(config['ZONE_VIOLATIONS'][f'{zone_key}_lat_max'])
                            zone['lon_min'] = float(config['ZONE_VIOLATIONS'][f'{zone_key}_lon_min'])
                            zone['lon_max'] = float(config['ZONE_VIOLATIONS'][f'{zone_key}_lon_max'])
                        elif geometry_type == 'circle':
                            zone['center_lat'] = float(config['ZONE_VIOLATIONS'][f'{zone_key}_center_lat'])
                            zone['center_lon'] = float(config['ZONE_VIOLATIONS'][f'{zone_key}_center_lon'])
                            zone['radius_meters'] = float(config['ZONE_VIOLATIONS'][f'{zone_key}_radius_meters'])
                        else:  # polygon or polyline
                            try:
                                coords_str = config.get('ZONE_VIOLATIONS', f'{zone_key}_coordinates', fallback='[]')
                                if not coords_str or coords_str.strip() == '':
                                    logger.warning(f"Zone {zone.get('name', 'Unknown')} has empty coordinates string in config, skipping")
                                    continue
                                # Handle case where configparser might have wrapped the JSON in quotes or escaped it
                                coords_str = coords_str.strip()
                                # Remove surrounding quotes if present
                                if (coords_str.startswith('"') and coords_str.endswith('"')) or \
                                   (coords_str.startswith("'") and coords_str.endswith("'")):
                                    coords_str = coords_str[1:-1]
                                # Try to parse JSON
                                try:
                                    coordinates = json.loads(coords_str)
                                except json.JSONDecodeError as parse_error:
                                    # Try unescaping if ConfigParser escaped the string
                                    try:
                                        # ConfigParser might escape quotes - try to handle that
                                        coords_str_unescaped = coords_str.replace('\\"', '"').replace("\\'", "'")
                                        coordinates = json.loads(coords_str_unescaped)
                                    except json.JSONDecodeError:
                                        logger.error(f"Error parsing coordinates JSON for zone {zone.get('name', 'Unknown')}: {parse_error}")
                                        logger.error(f"JSON string (first 200 chars): {coords_str[:200]}...")
                                        logger.error(f"Full JSON string length: {len(coords_str)} chars")
                                        continue
                                
                                if not isinstance(coordinates, list):
                                    logger.warning(f"Zone {zone.get('name', 'Unknown')} has invalid coordinates format (not a list), skipping")
                                    continue
                                if len(coordinates) < 2:
                                    logger.warning(f"Zone {zone.get('name', 'Unknown')} has fewer than 2 coordinate points, skipping")
                                    continue
                                zone['coordinates'] = coordinates
                                logger.debug(f"Loaded coordinates for zone '{zone.get('name', 'Unknown')}': {len(coordinates)} points")
                            except configparser.NoOptionError:
                                logger.warning(f"Zone {zone.get('name', 'Unknown')} is missing coordinates in config, skipping")
                                continue
                            if geometry_type == 'polyline':
                                zone['tolerance_meters'] = float(config.get('ZONE_VIOLATIONS', f'{zone_key}_tolerance_meters', fallback='100'))
                        
                        self.zone_violations.append(zone)
                    except (ValueError, KeyError, configparser.NoOptionError, json.JSONDecodeError):
                        continue
                        
            # If no zones were loaded, initialize with default zones
            if not self.zone_violations:
                self.zone_violations = [
                    {'name': 'Strait of Hormuz', 'lat_min': 25.0, 'lat_max': 27.0, 'lon_min': 55.0, 'lon_max': 57.5, 'is_selected': True},
                    {'name': 'South China Sea', 'lat_min': 5.0, 'lat_max': 25.0, 'lon_min': 105.0, 'lon_max': 120.0, 'is_selected': True}
                ]
        except Exception as e:
            logger.error(f"Error loading zones from config: {e}")
            # Default zones if loading fails
            self.zone_violations = [
                {'name': 'Strait of Hormuz', 'lat_min': 25.0, 'lat_max': 27.0, 'lon_min': 55.0, 'lon_max': 57.5, 'is_selected': True},
                {'name': 'South China Sea', 'lat_min': 5.0, 'lat_max': 25.0, 'lon_min': 105.0, 'lon_max': 120.0, 'is_selected': True}
            ]
            
    def _save_zones_to_config(self):
        """Save zone definitions to config.ini"""
        try:
            config = configparser.ConfigParser()
            config.read(self.analysis.config_path)
            
            # Ensure ZONE_VIOLATIONS section exists
            if 'ZONE_VIOLATIONS' not in config:
                config['ZONE_VIOLATIONS'] = {}
            
            # Remove existing zone entries
            for key in list(config['ZONE_VIOLATIONS'].keys()):
                if key.startswith('zone_'):
                    del config['ZONE_VIOLATIONS'][key]
            
            # Add current zones
            import json
            for i, zone in enumerate(self.zone_violations):
                zone_key = f'zone_{i}'
                geometry_type = zone.get('geometry_type', 'rectangle')
                
                try:
                    config['ZONE_VIOLATIONS'][f'{zone_key}_name'] = str(zone.get('name', ''))
                    config['ZONE_VIOLATIONS'][f'{zone_key}_geometry_type'] = geometry_type
                    config['ZONE_VIOLATIONS'][f'{zone_key}_is_selected'] = str(zone.get('is_selected', True))
                    
                    # Save coordinates based on geometry type
                    if geometry_type == 'rectangle':
                        # Legacy format for backward compatibility
                        if 'lat_min' not in zone or 'lat_max' not in zone or 'lon_min' not in zone or 'lon_max' not in zone:
                            logger.warning(f"Zone {zone.get('name', 'Unknown')} is missing rectangle coordinates, skipping")
                            continue
                        config['ZONE_VIOLATIONS'][f'{zone_key}_lat_min'] = str(zone.get('lat_min', 0.0))
                        config['ZONE_VIOLATIONS'][f'{zone_key}_lat_max'] = str(zone.get('lat_max', 0.0))
                        config['ZONE_VIOLATIONS'][f'{zone_key}_lon_min'] = str(zone.get('lon_min', 0.0))
                        config['ZONE_VIOLATIONS'][f'{zone_key}_lon_max'] = str(zone.get('lon_max', 0.0))
                    elif geometry_type == 'circle':
                        if 'center_lat' not in zone or 'center_lon' not in zone or 'radius_meters' not in zone:
                            logger.warning(f"Zone {zone.get('name', 'Unknown')} is missing circle parameters, skipping")
                            continue
                        config['ZONE_VIOLATIONS'][f'{zone_key}_center_lat'] = str(zone.get('center_lat', 0.0))
                        config['ZONE_VIOLATIONS'][f'{zone_key}_center_lon'] = str(zone.get('center_lon', 0.0))
                        config['ZONE_VIOLATIONS'][f'{zone_key}_radius_meters'] = str(zone.get('radius_meters', 0))
                    else:  # polygon or polyline
                        # Store coordinates as JSON string
                        coords = zone.get('coordinates', [])
                        if not coords:
                            error_msg = f"Zone '{zone.get('name', 'Unknown')}' has empty coordinates. Please draw the zone or enter coordinates before saving."
                            logger.error(error_msg)
                            messagebox.showerror("Error Saving Zone", error_msg)
                            continue
                        if not isinstance(coords, list):
                            error_msg = f"Zone '{zone.get('name', 'Unknown')}' has invalid coordinates type: {type(coords)}. Expected a list."
                            logger.error(error_msg)
                            messagebox.showerror("Error Saving Zone", error_msg)
                            continue
                        # Ensure coordinates are serializable and properly formatted
                        try:
                            # Validate coordinates format
                            if len(coords) < 2:
                                error_msg = f"Zone '{zone.get('name', 'Unknown')}' has fewer than 2 coordinate points. A {geometry_type} requires at least 2 points."
                                logger.error(error_msg)
                                messagebox.showerror("Error Saving Zone", error_msg)
                                continue
                            # Validate each coordinate point
                            for idx, coord in enumerate(coords):
                                if not isinstance(coord, (dict, list)):
                                    error_msg = f"Zone '{zone.get('name', 'Unknown')}' has invalid coordinate at index {idx}: {coord}"
                                    logger.error(error_msg)
                                    messagebox.showerror("Error Saving Zone", error_msg)
                                    continue
                                if isinstance(coord, dict):
                                    if 'lat' not in coord or 'lon' not in coord:
                                        error_msg = f"Zone '{zone.get('name', 'Unknown')}' coordinate at index {idx} is missing 'lat' or 'lon'"
                                        logger.error(error_msg)
                                        messagebox.showerror("Error Saving Zone", error_msg)
                                        continue
                            # Serialize coordinates to compact JSON (no extra whitespace)
                            coords_json = json.dumps(coords, ensure_ascii=False, separators=(',', ':'))
                            # Validate JSON length isn't too long (ConfigParser can handle long values, but let's be safe)
                            if len(coords_json) > 100000:  # 100KB limit
                                logger.warning(f"Zone {zone.get('name', 'Unknown')} has very large coordinates JSON ({len(coords_json)} chars)")
                            config['ZONE_VIOLATIONS'][f'{zone_key}_coordinates'] = coords_json
                            logger.info(f"Saved coordinates for zone '{zone.get('name', 'Unknown')}': {len(coords)} points, JSON length: {len(coords_json)} chars")
                        except (TypeError, ValueError) as e:
                            error_msg = f"Failed to save zone '{zone.get('name', 'Unknown')}': Invalid coordinates format - {e}"
                            logger.error(error_msg)
                            logger.error(f"Coordinates data (first 500 chars): {str(coords)[:500]}")
                            messagebox.showerror("Error Saving Zone", error_msg)
                            continue
                        if geometry_type == 'polyline':
                            config['ZONE_VIOLATIONS'][f'{zone_key}_tolerance_meters'] = str(zone.get('tolerance_meters', 100))
                except Exception as zone_error:
                    logger.error(f"Error saving zone {zone.get('name', 'Unknown')}: {zone_error}")
                    logger.error(traceback.format_exc())
                    messagebox.showerror("Error", f"Failed to save zone '{zone.get('name', 'Unknown')}': {zone_error}")
                    continue
            
            # Write back to file with UTF-8 encoding to handle JSON properly
            try:
                # Ensure the directory exists
                config_dir = os.path.dirname(self.analysis.config_path)
                if config_dir and not os.path.exists(config_dir):
                    os.makedirs(config_dir, exist_ok=True)
                
                with open(self.analysis.config_path, 'w', encoding='utf-8') as configfile:
                    config.write(configfile)
                logger.info(f"Successfully saved {len(self.zone_violations)} zones to config file")
            except IOError as io_err:
                logger.error(f"IO error writing config file: {io_err}")
                messagebox.showerror("Error", f"Failed to write config file: {io_err}")
                raise
            except Exception as write_err:
                logger.error(f"Error writing config file: {write_err}")
                logger.error(traceback.format_exc())
                messagebox.showerror("Error", f"Failed to write config file: {write_err}")
                raise
                
        except Exception as e:
            logger.error(f"Error saving zones to config: {e}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to save zones to config file: {e}")
            
    def _refresh_zone_list(self):
        """Refresh the zone list display with checkboxes"""
        # Ensure zone_violations is initialized
        if not hasattr(self, 'zone_violations'):
            self.zone_violations = []
        
        # Ensure zone_list_container exists
        if not hasattr(self, 'zone_list_container'):
            return
        
        # Clear existing zone frames
        if hasattr(self, 'zone_checkboxes'):
            for zone_name, zone_data in self.zone_checkboxes.items():
                if 'frame' in zone_data and zone_data['frame'].winfo_exists():
                    zone_data['frame'].destroy()
            self.zone_checkboxes.clear()
        else:
            self.zone_checkboxes = {}
        
        if hasattr(self, 'zone_frames'):
            self.zone_frames.clear()
        else:
            self.zone_frames = {}
        
        # Create checkbox for each zone
        for i, zone in enumerate(self.zone_violations):
            zone_name = zone['name']
            
            # Create frame for this zone
            zone_item_frame = ttk.Frame(self.zone_list_container)
            zone_item_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
            self.zone_frames[zone_name] = zone_item_frame
            
            # Create BooleanVar for checkbox
            selected_var = tk.BooleanVar(value=zone.get('is_selected', True))
            
            # Store checkbox
            self.zone_checkboxes[zone_name] = {
                'selected': selected_var,
                'frame': zone_item_frame
            }
            
            # Create checkbox
            ttk.Checkbutton(zone_item_frame, text="Selected", variable=selected_var,
                           command=lambda z=zone, v=selected_var: self._on_zone_selected_changed(z, v)).pack(side=tk.LEFT, padx=5)
            
            # Zone name and coordinates
            zone_info = f"{zone_name} - Lat: [{zone.get('lat_min', 0.0):.2f}, {zone.get('lat_max', 0.0):.2f}], Lon: [{zone.get('lon_min', 0.0):.2f}, {zone.get('lon_max', 0.0):.2f}]"
            ttk.Label(zone_item_frame, text=zone_info).pack(side=tk.LEFT, padx=10)
            
            # Edit button
            ttk.Button(zone_item_frame, text="Edit", 
                      command=lambda z=zone: self._edit_zone_by_name(z['name'])).pack(side=tk.RIGHT, padx=5)
    
    def _on_zone_selected_changed(self, zone, var):
        """Handle selected checkbox change"""
        zone['is_selected'] = var.get()
        self._save_zones_to_config()
        
    def _edit_zone_by_name(self, zone_name):
        """Edit zone by name"""
        for zone in self.zone_violations:
            if zone['name'] == zone_name:
                self._show_zone_dialog(
                    zone_index=self.zone_violations.index(zone),
                    zone_name=zone['name'],
                    lat_min=zone.get('lat_min', 0.0),
                    lat_max=zone.get('lat_max', 0.0),
                    lon_min=zone.get('lon_min', 0.0),
                    lon_max=zone.get('lon_max', 0.0),
                    is_selected=zone.get('is_selected', True),
                    zone_data=zone  # Pass full zone data for proper loading of all geometry types
                )
                break
                
    def _update_zone_selection_in_memory(self, zone_name, is_selected):
        """Update the selection status of a zone in memory"""
        for zone in self.zone_violations:
            if zone['name'] == zone_name:
                zone['is_selected'] = is_selected
                break
        self._save_zones_to_config()
        
    def _draw_zone_dialog(self):
        """Open a map for drawing a zone and capture coordinates"""
        # Check if folium is available
        try:
            import folium
            from folium.plugins import Draw
            import webbrowser
            import tempfile
        except ImportError as e:
            messagebox.showerror("Error", f"Required library not found: {e}\n\nPlease install folium: pip install folium")
            return
            
        # Create a dialog to get the zone name first
        pre_dialog = tk.Toplevel(self.window)
        pre_dialog.title("New Zone")
        pre_dialog.geometry("400x150")
        pre_dialog.transient(self.window)
        pre_dialog.grab_set()
        
        ttk.Label(pre_dialog, text="Enter a name for the new zone:").pack(pady=10)
        name_var = tk.StringVar()
        ttk.Entry(pre_dialog, textvariable=name_var, width=40).pack(pady=5)
        
        def proceed_to_map():
            zone_name = name_var.get().strip()
            if not zone_name:
                messagebox.showerror("Error", "Please enter a zone name")
                return
                
            # Close the name dialog
            pre_dialog.destroy()
            
            # Now launch the map
            self._launch_map_for_zone(zone_name)
        
        ttk.Button(pre_dialog, text="Continue", command=proceed_to_map).pack(side=tk.LEFT, padx=10, pady=10)
        ttk.Button(pre_dialog, text="Cancel", command=pre_dialog.destroy).pack(side=tk.RIGHT, padx=10, pady=10)
    
    def _launch_map_for_zone(self, zone_name):
        """Launch a map for drawing the zone with the given name"""
        try:
            import folium
            from folium.plugins import Draw
            import webbrowser
            import tempfile
        except ImportError:
            return
            
        # Create map
        m = folium.Map(location=[20, 0], zoom_start=2)
        
        # Add drawing tools
        Draw(
            export=True,
            position='topleft',
            draw_options={
                'polyline': False,
                'polygon': False,
                'rectangle': True,
                'circle': False,
                'marker': False,
                'circlemarker': False
            }
        ).add_to(m)
        
        # Add simple instructions
        folium.Element("""
        <div style="position:fixed;top:10px;right:10px;width:250px;background:white;padding:10px;z-index:9999;border:1px solid black;">
            <h4>Instructions</h4>
            <ol><li>Use rectangle tool to draw</li><li>Copy coordinates</li><li>Enter in the next dialog</li></ol>
        </div>
        <script>
        setTimeout(function() {
            var map = null;
            for (var k in window) { if (window[k] && window[k]._container) { map = window[k]; break; } }
            if (map) {
                map.on('draw:created', function(e) {
                    var b = e.layer.getBounds(), s = b.getSouth().toFixed(6), n = b.getNorth().toFixed(6),
                        w = b.getWest().toFixed(6), e = b.getEast().toFixed(6),
                        txt = s+','+n+','+w+','+e;
                    alert('Coordinates: ' + txt + '\n\nPlease copy these values for the next dialog.');
                });
            }
        }, 1000);
        </script>
        """).add_to(m)
        
        # Save and open
        temp_file = os.path.join(tempfile.gettempdir(), 'zone_map.html')
        m.save(temp_file)
        webbrowser.open('file://' + temp_file)
        
        # Show dialog to enter coordinates
        self._show_coord_entry_dialog(zone_name)
    
    def _show_coord_entry_dialog(self, zone_name):
        """Show dialog to enter coordinates for the zone"""
        dialog = tk.Toplevel(self.window)
        dialog.title(f"Enter Coordinates for {zone_name}")
        dialog.geometry("400x200")
        dialog.transient(self.window)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Enter coordinates (lat_min,lat_max,lon_min,lon_max):").pack(pady=10)
        coord_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=coord_var, width=40).pack(pady=5)
        
        def save_zone():
            coords = coord_var.get().strip()
            if not coords:
                messagebox.showerror("Error", "Please enter coordinates")
                return
                
            try:
                parts = coords.split(',')
                if len(parts) != 4:
                    messagebox.showerror("Error", "Invalid format")
                    return
                    
                lat_min = float(parts[0])
                lat_max = float(parts[1])
                lon_min = float(parts[2])
                lon_max = float(parts[3])
                
                # Create the zone
                zone_data = {
                    'name': zone_name,
                    'is_selected': True,
                    'lat_min': lat_min,
                    'lat_max': lat_max,
                    'lon_min': lon_min,
                    'lon_max': lon_max
                }
                self.zone_violations.append(zone_data)
                self._refresh_zone_list()
                self._save_zones_to_config()
                dialog.destroy()
            except (ValueError, IndexError):
                messagebox.showerror("Error", "Invalid coordinate values")
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(button_frame, text="Save", command=save_zone).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
    def _delete_selected_zone(self):
        """Delete the selected zone from the list"""
        if not self.zone_checkboxes:
            messagebox.showinfo("Info", "No zones available to delete.")
            return
            
        # Find which zones are selected for deletion
        zones_to_delete = []
        for zone_name, checkbox_data in self.zone_checkboxes.items():
            if checkbox_data.get('selected').get():
                zones_to_delete.append(zone_name)
        
        if not zones_to_delete:
            messagebox.showinfo("Info", "Please select at least one zone to delete.")
            return
        
        # Confirm deletion
        if len(zones_to_delete) == 1:
            confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete the zone '{zones_to_delete[0]}'?")
        else:
            confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete the selected {len(zones_to_delete)} zones?")
        
        if confirm:
            # Remove the zones
            self.zone_violations = [zone for zone in self.zone_violations if zone['name'] not in zones_to_delete]
            self._refresh_zone_list()
            self._save_zones_to_config()
        
    def _show_zone_dialog(self, zone_index=None, zone_name='', lat_min=0.0, lat_max=0.0, lon_min=0.0, lon_max=0.0, is_selected=True, zone_data=None):
        """Show dialog to add or edit a zone - supports all geometry types"""
        dialog = tk.Toplevel(self.window)
        dialog.title("Add Zone" if zone_index is None else "Edit Zone")
        dialog.geometry("500x600")
        dialog.transient(self.window)
        dialog.grab_set()
        
        # Store zone data from drawing if provided
        dialog.drawn_zone_data = zone_data
        
        # If editing existing zone, load its geometry type
        existing_zone = None
        if zone_index is not None and zone_index < len(self.zone_violations):
            existing_zone = self.zone_violations[zone_index]
        
        current_geometry_type = 'rectangle'
        if existing_zone:
            current_geometry_type = existing_zone.get('geometry_type', 'rectangle')
        elif zone_data:
            current_geometry_type = zone_data.get('geometry_type', 'rectangle')
        
        # Create form
        form_frame = ttk.Frame(dialog, padding=10)
        form_frame.pack(fill=tk.BOTH, expand=True)
        
        # Zone name
        ttk.Label(form_frame, text="Zone Name:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        name_var = tk.StringVar(value=zone_name)
        ttk.Entry(form_frame, textvariable=name_var, width=30).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Geometry type selection
        ttk.Label(form_frame, text="Geometry Type:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        geometry_type_var = tk.StringVar(value=current_geometry_type)
        geometry_frame = ttk.Frame(form_frame)
        geometry_frame.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Radiobutton(geometry_frame, text="Rectangle", variable=geometry_type_var, value="rectangle").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(geometry_frame, text="Circle", variable=geometry_type_var, value="circle").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(geometry_frame, text="Polygon", variable=geometry_type_var, value="polygon").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(geometry_frame, text="Polyline", variable=geometry_type_var, value="polyline").pack(side=tk.LEFT, padx=5)
        
        # Container for coordinate fields (will be updated based on geometry type)
        coord_frame = ttk.LabelFrame(form_frame, text="Coordinates")
        coord_frame.grid(row=2, column=0, columnspan=2, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Rectangle fields
        rect_frame = ttk.Frame(coord_frame)
        ttk.Label(rect_frame, text="Latitude Min:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        lat_min_var = tk.DoubleVar(value=lat_min)
        lat_min_entry = ttk.Entry(rect_frame, textvariable=lat_min_var, width=20)
        lat_min_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(rect_frame, text="Latitude Max:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        lat_max_var = tk.DoubleVar(value=lat_max)
        lat_max_entry = ttk.Entry(rect_frame, textvariable=lat_max_var, width=20)
        lat_max_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(rect_frame, text="Longitude Min:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        lon_min_var = tk.DoubleVar(value=lon_min)
        lon_min_entry = ttk.Entry(rect_frame, textvariable=lon_min_var, width=20)
        lon_min_entry.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(rect_frame, text="Longitude Max:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        lon_max_var = tk.DoubleVar(value=lon_max)
        lon_max_entry = ttk.Entry(rect_frame, textvariable=lon_max_var, width=20)
        lon_max_entry.grid(row=3, column=1, padx=5, pady=5)
        
        # Circle fields
        circle_frame = ttk.Frame(coord_frame)
        ttk.Label(circle_frame, text="Center Latitude:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        center_lat_var = tk.DoubleVar(value=existing_zone.get('center_lat', 0.0) if existing_zone and existing_zone.get('geometry_type') == 'circle' else 0.0)
        center_lat_entry = ttk.Entry(circle_frame, textvariable=center_lat_var, width=20)
        center_lat_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(circle_frame, text="Center Longitude:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        center_lon_var = tk.DoubleVar(value=existing_zone.get('center_lon', 0.0) if existing_zone and existing_zone.get('geometry_type') == 'circle' else 0.0)
        center_lon_entry = ttk.Entry(circle_frame, textvariable=center_lon_var, width=20)
        center_lon_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(circle_frame, text="Radius (meters):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        radius_var = tk.DoubleVar(value=existing_zone.get('radius_meters', 0.0) if existing_zone and existing_zone.get('geometry_type') == 'circle' else 0.0)
        radius_entry = ttk.Entry(circle_frame, textvariable=radius_var, width=20)
        radius_entry.grid(row=2, column=1, padx=5, pady=5)
        
        # Polygon/Polyline fields (JSON coordinates)
        poly_frame = ttk.Frame(coord_frame)
        ttk.Label(poly_frame, text="Coordinates (JSON):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        import json
        coords_text = ""
        if existing_zone and existing_zone.get('geometry_type') in ['polygon', 'polyline']:
            coords_text = json.dumps(existing_zone.get('coordinates', []), indent=2)
        elif zone_data and zone_data.get('geometry_type') in ['polygon', 'polyline']:
            coords_text = json.dumps(zone_data.get('coordinates', []), indent=2)
        
        coords_var = tk.StringVar(value=coords_text)
        coords_text_widget = tk.Text(poly_frame, width=50, height=8)
        coords_text_widget.insert('1.0', coords_text)
        coords_text_widget.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        
        # Polyline tolerance field
        ttk.Label(poly_frame, text="Tolerance (meters, for polyline):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        tolerance_var = tk.DoubleVar(value=existing_zone.get('tolerance_meters', 100) if existing_zone and existing_zone.get('geometry_type') == 'polyline' else 100)
        tolerance_entry = ttk.Entry(poly_frame, textvariable=tolerance_var, width=20)
        tolerance_entry.grid(row=2, column=1, padx=5, pady=5)
        
        # Function to show/hide coordinate frames based on geometry type
        def update_coord_fields():
            geometry_type = geometry_type_var.get()
            # Hide all frames
            rect_frame.grid_remove()
            circle_frame.grid_remove()
            poly_frame.grid_remove()
            
            # Show appropriate frame
            if geometry_type == 'rectangle':
                rect_frame.grid(row=0, column=0, sticky=tk.W+tk.E, padx=5, pady=5)
            elif geometry_type == 'circle':
                circle_frame.grid(row=0, column=0, sticky=tk.W+tk.E, padx=5, pady=5)
            else:  # polygon or polyline
                poly_frame.grid(row=0, column=0, sticky=tk.W+tk.E, padx=5, pady=5)
                if geometry_type == 'polyline':
                    tolerance_entry.grid()
                    ttk.Label(poly_frame, text="Tolerance (meters, for polyline):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
                else:
                    tolerance_entry.grid_remove()
        
        geometry_type_var.trace('w', lambda *args: update_coord_fields())
        update_coord_fields()  # Initial update
        
        # Selected checkbox (for use in analysis)
        is_selected_var = tk.BooleanVar(value=is_selected)
        ttk.Checkbutton(form_frame, text="Selected (use in analysis)", variable=is_selected_var).grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        
        # Draw Zone button
        def draw_zone_from_dialog():
            """Draw zone on map and populate coordinate fields"""
            # Call the drawing method with parent dialog reference
            def draw_and_update():
                self._draw_zone_for_dialog(dialog, None, None, None, None, parent_dialog_ref=dialog)
                # Wait a bit for the coordinate dialog to process and close, then update
                dialog.after(500, lambda: self._update_dialog_from_drawn_coords(dialog, geometry_type_var, lat_min_var, lat_max_var, 
                                                                                  lon_min_var, lon_max_var, center_lat_var, center_lon_var,
                                                                                  radius_var, coords_text_widget, tolerance_var, update_coord_fields))
            draw_and_update()
        
        draw_button_frame = ttk.Frame(form_frame)
        draw_button_frame.grid(row=4, column=0, columnspan=2, pady=5)
        ttk.Button(draw_button_frame, text="Draw Zone on Map", command=draw_zone_from_dialog).pack(side=tk.LEFT, padx=5)
        
        # Save function
        def save_zone():
            name = name_var.get().strip()
            if not name:
                messagebox.showerror("Error", "Zone name is required.")
                return
            
            geometry_type = geometry_type_var.get()
            zone_data = {
                'name': name,
                'geometry_type': geometry_type,
                'is_selected': is_selected_var.get()
            }
            
            # Validate and populate coordinates based on geometry type
            import json
            if geometry_type == 'rectangle':
                try:
                    lat_min_val = float(lat_min_var.get())
                    lat_max_val = float(lat_max_var.get())
                    lon_min_val = float(lon_min_var.get())
                    lon_max_val = float(lon_max_var.get())
                except ValueError:
                    messagebox.showerror("Error", "All coordinates must be valid numbers.")
                    return
                
                # Validate ranges
                if lat_min_val >= lat_max_val:
                    messagebox.showerror("Error", "Latitude Min must be less than Latitude Max.")
                    return
                
                if lon_min_val >= lon_max_val:
                    messagebox.showerror("Error", "Longitude Min must be less than Longitude Max.")
                    return
                
                zone_data['lat_min'] = lat_min_val
                zone_data['lat_max'] = lat_max_val
                zone_data['lon_min'] = lon_min_val
                zone_data['lon_max'] = lon_max_val
                
            elif geometry_type == 'circle':
                try:
                    center_lat_val = float(center_lat_var.get())
                    center_lon_val = float(center_lon_var.get())
                    radius_val = float(radius_var.get())
                except ValueError:
                    messagebox.showerror("Error", "All circle parameters must be valid numbers.")
                    return
                
                if radius_val <= 0:
                    messagebox.showerror("Error", "Radius must be greater than 0.")
                    return
                
                zone_data['center_lat'] = center_lat_val
                zone_data['center_lon'] = center_lon_val
                zone_data['radius_meters'] = radius_val
                
            else:  # polygon or polyline
                coords_text = coords_text_widget.get('1.0', tk.END).strip()
                if not coords_text:
                    messagebox.showerror("Error", "Coordinates are required. Please paste JSON coordinates from the map.")
                    return
                
                try:
                    coordinates = json.loads(coords_text)
                    if not isinstance(coordinates, list) or len(coordinates) < 2:
                        messagebox.showerror("Error", "Coordinates must be a JSON array with at least 2 points.")
                        return
                    
                    zone_data['coordinates'] = coordinates
                    if geometry_type == 'polyline':
                        try:
                            tolerance_val = float(tolerance_var.get())
                            if tolerance_val <= 0:
                                messagebox.showerror("Error", "Tolerance must be greater than 0.")
                                return
                            zone_data['tolerance_meters'] = tolerance_val
                        except ValueError:
                            messagebox.showerror("Error", "Tolerance must be a valid number.")
                            return
                except json.JSONDecodeError as e:
                    messagebox.showerror("Error", f"Invalid JSON format: {e}")
                    return
            
            # Check for duplicate name (if editing, allow same name)
            if zone_index is None:
                for zone in self.zone_violations:
                    if zone['name'] == name:
                        messagebox.showerror("Error", f"Zone name '{name}' already exists.")
                        return
            
            # Create or update zone
            if zone_index is None:
                # Add new zone
                self.zone_violations.append(zone_data)
            else:
                # Update existing zone
                self.zone_violations[zone_index] = zone_data
            
            # Refresh display
            self._refresh_zone_list()
            # Save to config.ini
            self._save_zones_to_config()
            dialog.destroy()
        
        # Buttons
        button_frame = ttk.Frame(dialog, padding=10)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        ttk.Button(button_frame, text="Save", command=save_zone).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    def _update_dialog_from_drawn_coords(self, dialog, geometry_type_var, lat_min_var, lat_max_var, 
                                         lon_min_var, lon_max_var, center_lat_var, center_lon_var,
                                         radius_var, coords_text_widget, tolerance_var, update_coord_fields_func):
        """Update dialog fields from drawn zone coordinates"""
        if hasattr(dialog, 'drawn_zone_data') and dialog.drawn_zone_data:
            zone_data_from_drawing = dialog.drawn_zone_data
            import json
            geometry_type_var.set(zone_data_from_drawing.get('geometry_type', 'rectangle'))
            if zone_data_from_drawing['geometry_type'] == 'rectangle':
                lat_min_var.set(float(zone_data_from_drawing.get('lat_min', 0)))
                lat_max_var.set(float(zone_data_from_drawing.get('lat_max', 0)))
                lon_min_var.set(float(zone_data_from_drawing.get('lon_min', 0)))
                lon_max_var.set(float(zone_data_from_drawing.get('lon_max', 0)))
            elif zone_data_from_drawing['geometry_type'] == 'circle':
                center_lat_var.set(float(zone_data_from_drawing.get('center_lat', 0)))
                center_lon_var.set(float(zone_data_from_drawing.get('center_lon', 0)))
                radius_var.set(float(zone_data_from_drawing.get('radius_meters', 0)))
            else:  # polygon or polyline
                coords_text_widget.delete('1.0', tk.END)
                coords_text_widget.insert('1.0', json.dumps(zone_data_from_drawing.get('coordinates', []), indent=2))
                if zone_data_from_drawing['geometry_type'] == 'polyline':
                    tolerance_var.set(float(zone_data_from_drawing.get('tolerance_meters', 100)))
            update_coord_fields_func()
            # Clear the drawn zone data after using it
            dialog.drawn_zone_data = None
    
    def _draw_zone_for_dialog(self, parent_dialog, lat_min_var=None, lat_max_var=None, lon_min_var=None, lon_max_var=None, parent_dialog_ref=None):
        """Draw a zone on a map and populate coordinate fields in the dialog
        
        Args:
            parent_dialog: The coordinate input dialog
            lat_min_var, lat_max_var, lon_min_var, lon_max_var: Optional variables for rectangle coordinates (for backward compatibility)
            parent_dialog_ref: Optional reference to the main zone dialog to update directly
        """
        try:
            import folium
            from folium.plugins import Draw
            import webbrowser
            import tempfile
            import json
        except ImportError as e:
            messagebox.showerror("Error", f"Required library not found: {e}\n\nPlease install folium: pip install folium")
            return
        
        # Create a map centered on a default location (middle of world)
        m = folium.Map(location=[20, 0], zoom_start=2)
        
        # Add drawing tools - enable all geometry types
        draw = Draw(
            export=True,
            filename='zone_draw_data.geojson',
            position='topleft',
            draw_options={
                'polyline': True,   # Enable polyline drawing
                'polygon': True,    # Enable polygon drawing
                'rectangle': True,  # Enable rectangle drawing
                'circle': True,     # Enable circle drawing
                'marker': False,
                'circlemarker': False
            },
            edit_options={'edit': True, 'remove': True}
        )
        draw.add_to(m)
        
        # Add instructions and coordinate display to the map
        instructions_html = """
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 300px; height: 180px; 
                    background-color: white; z-index:9999; 
                    border: 2px solid grey; padding: 10px; border-radius: 5px;">
            <h4 style="margin-top:0;">Draw Zone Instructions</h4>
            <ol style="margin: 0; padding-left: 20px; font-size: 12px;">
                <li>Select a drawing tool from the toolbar (top-left):</li>
                <ul style="margin: 5px 0; padding-left: 20px; font-size: 11px;">
                    <li><strong>Rectangle:</strong> Click and drag to draw</li>
                    <li><strong>Circle:</strong> Click center, drag to set radius</li>
                    <li><strong>Polygon:</strong> Click points, double-click to finish</li>
                    <li><strong>Polyline:</strong> Click points, double-click to finish</li>
                </ul>
                <li>Coordinates will appear in the bottom-left box</li>
                <li>Click "Copy Coordinates" to copy them</li>
                <li>Return to the application and paste/enter them</li>
            </ol>
        </div>
        <div id="coords-display" style="position: fixed; bottom: 10px; left: 10px; width: 400px; 
            background-color: white; z-index:9999; border: 2px solid #007bff; 
            padding: 15px; border-radius: 5px; font-family: Arial, sans-serif; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);">
            <h4 style="margin-top:0;">Zone Coordinates</h4>
            <p id="coords-text" style="font-size: 12px; color: #666;">Draw a rectangle on the map...</p>
            <button id="copy-coords" style="margin-top: 10px; padding: 5px 10px; 
                background-color: #007bff; color: white; border: none; border-radius: 3px; 
                cursor: pointer;">Copy Coordinates</button>
        </div>
        """
        m.get_root().html.add_child(folium.Element(instructions_html))
        
        # Add JavaScript to extract coordinates from drawn rectangles
        extract_coords_js = folium.Element("""
        <script>
        // Wait for map and draw plugin to be initialized
        setTimeout(function() {
            // Find the map object (folium stores it in the window)
            var mapObj = null;
            for (var key in window) {
                if (window[key] && window[key].hasOwnProperty && window[key].hasOwnProperty('_container')) {
                    mapObj = window[key];
                    break;
                }
            }
            
            // Alternative: try to get from Leaflet's global map registry
            if (!mapObj && typeof L !== 'undefined') {
                L.eachLayer = L.eachLayer || function(callback) {
                    for (var id in this._layers) {
                        callback(this._layers[id]);
                    }
                };
                // Get the first map instance
                for (var id in L._layers) {
                    var layer = L._layers[id];
                    if (layer instanceof L.Map) {
                        mapObj = layer;
                        break;
                    }
                }
            }
            
            if (mapObj) {
                // Listen for draw events
                mapObj.on('draw:created', function(e) {
                    var layer = e.layer;
                    var geometryType = e.layerType;
                    var coords = {};
                    var coordsText = '';
                    
                    if (geometryType === 'rectangle') {
                        var bounds = layer.getBounds();
                        var sw = bounds.getSouthWest();
                        var ne = bounds.getNorthEast();
                        coords = {
                            geometry_type: 'rectangle',
                            lat_min: sw.lat.toFixed(6),
                            lat_max: ne.lat.toFixed(6),
                            lon_min: sw.lng.toFixed(6),
                            lon_max: ne.lng.toFixed(6)
                        };
                        coordsText = '<strong>Rectangle:</strong><br>' +
                                   'Lat Min: ' + coords.lat_min + '<br>' +
                                   'Lat Max: ' + coords.lat_max + '<br>' +
                                   'Lon Min: ' + coords.lon_min + '<br>' +
                                   'Lon Max: ' + coords.lon_max;
                    } else if (geometryType === 'circle') {
                        var center = layer.getLatLng();
                        var radius = layer.getRadius(); // in meters
                        coords = {
                            geometry_type: 'circle',
                            center_lat: center.lat.toFixed(6),
                            center_lon: center.lng.toFixed(6),
                            radius_meters: Math.round(radius)
                        };
                        coordsText = '<strong>Circle:</strong><br>' +
                                   'Center Lat: ' + coords.center_lat + '<br>' +
                                   'Center Lon: ' + coords.center_lon + '<br>' +
                                   'Radius: ' + coords.radius_meters + ' meters';
                    } else if (geometryType === 'polygon') {
                        var latlngs = layer.getLatLngs();
                        var coordinates = [];
                        if (latlngs && latlngs[0]) {
                            // Handle nested array structure
                            var points = Array.isArray(latlngs[0]) ? latlngs[0] : latlngs;
                            for (var i = 0; i < points.length; i++) {
                                coordinates.push({
                                    lat: points[i].lat.toFixed(6),
                                    lon: points[i].lng.toFixed(6)
                                });
                            }
                        }
                        coords = {
                            geometry_type: 'polygon',
                            coordinates: coordinates
                        };
                        coordsText = '<strong>Polygon:</strong><br>' +
                                   'Points: ' + coordinates.length + '<br>' +
                                   'Coordinates: ' + JSON.stringify(coordinates).substring(0, 100) + '...';
                    } else if (geometryType === 'polyline') {
                        var latlngs = layer.getLatLngs();
                        var coordinates = [];
                        if (latlngs && latlngs[0]) {
                            var points = Array.isArray(latlngs[0]) ? latlngs[0] : latlngs;
                            for (var i = 0; i < points.length; i++) {
                                coordinates.push({
                                    lat: points[i].lat.toFixed(6),
                                    lon: points[i].lng.toFixed(6)
                                });
                            }
                        }
                        coords = {
                            geometry_type: 'polyline',
                            coordinates: coordinates,
                            tolerance_meters: 100 // default tolerance
                        };
                        coordsText = '<strong>Polyline:</strong><br>' +
                                   'Points: ' + coordinates.length + '<br>' +
                                   'Tolerance: ' + coords.tolerance_meters + ' meters<br>' +
                                   'Coordinates: ' + JSON.stringify(coordinates).substring(0, 100) + '...';
                    }
                    
                    var coordsTextEl = document.getElementById('coords-text');
                    if (coordsTextEl) {
                        coordsTextEl.innerHTML = coordsText;
                    }
                    
                    // Store coordinates in a global variable for copying
                    window.zoneCoords = coords;
                });
            }
            
            // Copy button functionality
            var copyBtn = document.getElementById('copy-coords');
            if (copyBtn) {
                copyBtn.addEventListener('click', function() {
                    if (window.zoneCoords) {
                        // Convert coordinates to JSON string for copying
                        var text = JSON.stringify(window.zoneCoords);
                        if (navigator.clipboard && navigator.clipboard.writeText) {
                            navigator.clipboard.writeText(text).then(function() {
                                alert('Coordinates copied to clipboard!');
                                // Close the browser window/tab after user clicks OK
                                window.close();
                            }).catch(function() {
                                // Fallback for browsers without clipboard API
                                var textarea = document.createElement('textarea');
                                textarea.value = text;
                                document.body.appendChild(textarea);
                                textarea.select();
                                document.execCommand('copy');
                                document.body.removeChild(textarea);
                                alert('Coordinates copied to clipboard!');
                                // Close the browser window/tab after user clicks OK
                                window.close();
                            });
                        } else {
                            // Fallback for older browsers
                            var textarea = document.createElement('textarea');
                            textarea.value = text;
                            document.body.appendChild(textarea);
                            textarea.select();
                            document.execCommand('copy');
                            document.body.removeChild(textarea);
                            alert('Coordinates copied to clipboard!');
                            // Close the browser window/tab after user clicks OK
                            window.close();
                        }
                    }
                });
            }
        }, 2000);
        </script>
        """)
        m.get_root().html.add_child(extract_coords_js)
        
        # Save map to temporary file
        temp_dir = tempfile.gettempdir()
        temp_map_file = os.path.join(temp_dir, 'zone_draw_map.html')
        
        # Open map in browser
        webbrowser.open(f'file://{temp_map_file}')
        
        # Show dialog to get coordinates
        coord_dialog = tk.Toplevel(parent_dialog)
        coord_dialog.title("Enter Zone Coordinates from Map")
        coord_dialog.geometry("500x300")
        coord_dialog.transient(parent_dialog)
        coord_dialog.grab_set()
        
        # Store reference to parent dialog if provided
        coord_dialog.parent_dialog_ref = parent_dialog_ref if 'parent_dialog_ref' in locals() else None
        
        # Instructions
        instructions = tk.Text(coord_dialog, height=8, wrap=tk.WORD, font=("Arial", 9))
        instructions.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        instructions.insert('1.0', 
            "Instructions:\n\n"
            "1. A map has been opened in your browser\n"
            "2. Select a drawing tool from the toolbar (top-left):\n"
            "   - Rectangle: Click and drag\n"
            "   - Circle: Click center, drag for radius\n"
            "   - Polygon: Click points, double-click to finish\n"
            "   - Polyline: Click points, double-click to finish\n"
            "3. After drawing, coordinates will appear in the bottom-left box on the map\n"
            "4. Click 'Copy Coordinates' button on the map to copy them (JSON format)\n"
            "5. Paste the JSON coordinates below, or manually enter coordinates\n"
            "   For rectangles: lat_min,lat_max,lon_min,lon_max"
        )
        instructions.config(state=tk.DISABLED)
        
        # Paste coordinates frame
        paste_frame = ttk.LabelFrame(coord_dialog, text="Paste Coordinates (JSON or comma-separated)")
        paste_frame.pack(fill=tk.X, padx=10, pady=5)
        
        paste_var = tk.StringVar()
        paste_entry = ttk.Entry(paste_frame, textvariable=paste_var, width=50)
        paste_entry.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Enable right-click context menu for paste entry
        def create_context_menu(event):
            """Create right-click context menu for the entry field"""
            context_menu = tk.Menu(coord_dialog, tearoff=0)
            context_menu.add_command(label="Cut", command=lambda: paste_entry.event_generate("<<Cut>>"))
            context_menu.add_command(label="Copy", command=lambda: paste_entry.event_generate("<<Copy>>"))
            context_menu.add_command(label="Paste", command=lambda: paste_entry.event_generate("<<Paste>>"))
            context_menu.add_separator()
            context_menu.add_command(label="Select All", command=lambda: paste_entry.select_range(0, tk.END))
            try:
                context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                context_menu.grab_release()
        
        paste_entry.bind("<Button-3>", create_context_menu)  # Right-click on Windows/Linux
        paste_entry.bind("<Button-2>", create_context_menu)  # Right-click on macOS
        paste_entry.bind("<Control-Button-1>", create_context_menu)  # Control+Click on macOS
        
        def parse_pasted_coords():
            """Parse pasted coordinates and fill in the fields - supports JSON and legacy format"""
            paste_text = paste_var.get().strip()
            if not paste_text:
                return
            
            try:
                # Try to parse as JSON first
                import json
                try:
                    coords_data = json.loads(paste_text)
                    geometry_type = coords_data.get('geometry_type', 'rectangle')
                    
                    # Store zone data in parent dialog if available
                    if coord_dialog.parent_dialog_ref:
                        coord_dialog.parent_dialog_ref.drawn_zone_data = coords_data
                    
                    if geometry_type == 'rectangle' and lat_min_var is not None:
                        # Update rectangle fields if available (backward compatibility)
                        lat_min_var.set(float(coords_data['lat_min']))
                        lat_max_var.set(float(coords_data['lat_max']))
                        lon_min_var.set(float(coords_data['lon_min']))
                        lon_max_var.set(float(coords_data['lon_max']))
                        messagebox.showinfo("Success", "Rectangle coordinates parsed and filled in!")
                        coord_dialog.destroy()
                    else:
                        # For other geometry types or when parent dialog is provided
                        messagebox.showinfo("Success", f"{geometry_type.capitalize()} coordinates parsed! Return to the zone dialog and the coordinates will be filled in.")
                        coord_dialog.zone_data = coords_data
                        coord_dialog.destroy()
                        return coords_data
                except json.JSONDecodeError:
                    # Not JSON, try legacy comma-separated format
                    coords = [float(x.strip()) for x in paste_text.split(',')]
                    if len(coords) == 4:
                        if lat_min_var is not None:
                            lat_min_var.set(coords[0])
                            lat_max_var.set(coords[1])
                            lon_min_var.set(coords[2])
                            lon_max_var.set(coords[3])
                            messagebox.showinfo("Success", "Rectangle coordinates parsed and filled in!")
                            coord_dialog.destroy()
                        else:
                            messagebox.showerror("Error", "Rectangle coordinates require the zone dialog to be open.")
                    else:
                        messagebox.showerror("Error", "Please enter 4 comma-separated values: lat_min,lat_max,lon_min,lon_max\nOr paste JSON format from the map")
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid format: {e}\n\nPlease enter:\n- JSON format from map, or\n- 4 comma-separated values: lat_min,lat_max,lon_min,lon_max")
        
        ttk.Button(paste_frame, text="Parse & Fill", command=parse_pasted_coords).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(coord_dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(button_frame, text="Close", command=coord_dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def reset_analysis_filters_to_defaults(self):
        """Reset analysis filters to default values"""
        self.analysis_filters['min_latitude'].set(-90.0)
        self.analysis_filters['max_latitude'].set(90.0)
        self.analysis_filters['min_longitude'].set(-180.0)
        self.analysis_filters['max_longitude'].set(180.0)
        self.analysis_filters['time_start_hour'].set(0)
        self.analysis_filters['time_end_hour'].set(24)
        self.analysis_filters['min_confidence'].set(75)
        self.analysis_filters['max_anomalies_per_vessel'].set(10)
        self.analysis_filters['filter_mmsi_list'].set('')
    
    def _create_tab1_additional_outputs(self):
        """Create Tab 1: Additional Outputs"""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Additional Outputs")
        
        ttk.Label(tab, text="Generate Additional Outputs from Dataset", 
                 font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        frame1 = ttk.Frame(tab)
        frame1.pack(fill=tk.X, pady=5)
        ttk.Button(frame1, text="Export Full Dataset to CSV", width=30,
                  command=self._export_full_dataset).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame1, text="Export the complete analysis dataset to CSV format", 
                 wraplength=500).pack(side=tk.LEFT, padx=5)
        
        frame2 = ttk.Frame(tab)
        frame2.pack(fill=tk.X, pady=5)
        ttk.Button(frame2, text="Generate Summary Report", width=30,
                  command=self._generate_summary_report).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame2, text="Create a summary report with key findings and statistics", 
                 wraplength=500).pack(side=tk.LEFT, padx=5)
        
        frame3 = ttk.Frame(tab)
        frame3.pack(fill=tk.X, pady=5)
        ttk.Button(frame3, text="Export Vessel Statistics", width=30,
                  command=self._export_vessel_statistics).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame3, text="Export vessel-specific statistics to Excel format", 
                 wraplength=500).pack(side=tk.LEFT, padx=5)
        
        frame4 = ttk.Frame(tab)
        frame4.pack(fill=tk.X, pady=5)
        ttk.Button(frame4, text="Generate Anomaly Timeline", width=30,
                  command=self._generate_anomaly_timeline).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame4, text="Create a timeline visualization of anomalies", 
                 wraplength=500).pack(side=tk.LEFT, padx=5)
    
    def _create_tab2_further_analysis(self):
        """Create Tab 2: Further Analysis"""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Further Analysis")
        
        ttk.Label(tab, text="Further Analysis Tools", 
                 font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        frame1 = ttk.Frame(tab)
        frame1.pack(fill=tk.X, pady=5)
        ttk.Button(frame1, text="Anomaly Correlation Analysis", width=30,
                  command=self._correlation_analysis_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame1, text="Analyze correlations between vessel types and anomaly types", 
                 wraplength=500).pack(side=tk.LEFT, padx=5)
        
        frame2 = ttk.Frame(tab)
        frame2.pack(fill=tk.X, pady=5)
        ttk.Button(frame2, text="Temporal Pattern Analysis", width=30,
                  command=self._temporal_pattern_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame2, text="Analyze patterns over time, including hourly/daily distributions", 
                 wraplength=500).pack(side=tk.LEFT, padx=5)
        
        frame3 = ttk.Frame(tab)
        frame3.pack(fill=tk.X, pady=5)
        ttk.Button(frame3, text="Vessel Behavior Clustering", width=30,
                  command=self._vessel_behavior_clustering).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame3, text="Apply clustering algorithms to identify similar vessel behaviors", 
                 wraplength=500).pack(side=tk.LEFT, padx=5)
        
        frame4 = ttk.Frame(tab)
        frame4.pack(fill=tk.X, pady=5)
        ttk.Button(frame4, text="Anomaly Frequency Analysis", width=30,
                  command=self._anomaly_frequency_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame4, text="Analyze frequency and distribution of different anomaly types", 
                 wraplength=500).pack(side=tk.LEFT, padx=5)
        
        frame5 = ttk.Frame(tab)
        frame5.pack(fill=tk.X, pady=5)
        ttk.Button(frame5, text="Create Custom Chart", width=30,
                  command=self._create_custom_chart_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame5, text="Create custom charts with various types (bar, scatter, pie, etc.)", 
                 wraplength=500).pack(side=tk.LEFT, padx=5)
    
    def _create_tab3_mapping_tools(self):
        """Create Tab 3: Mapping Tools"""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Mapping Tools")
        
        ttk.Label(tab, text="Advanced Mapping Tools", 
                 font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        frame1 = ttk.Frame(tab)
        frame1.pack(fill=tk.X, pady=5)
        ttk.Button(frame1, text="Full Spectrum Anomaly Map", width=30,
                  command=self._full_spectrum_map_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame1, text="Create a comprehensive map showing all anomalies", 
                 wraplength=500).pack(side=tk.LEFT, padx=5)
        
        frame2 = ttk.Frame(tab)
        frame2.pack(fill=tk.X, pady=5)
        ttk.Label(frame2, text="MMSI:").pack(side=tk.LEFT, padx=5)
        self.vessel_mmsi_var = tk.StringVar()
        ttk.Entry(frame2, textvariable=self.vessel_mmsi_var, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame2, text="Create Vessel Map", width=25,
                  command=self._create_vessel_map_dialog).pack(side=tk.LEFT, padx=5)
        
        frame3 = ttk.Frame(tab)
        frame3.pack(fill=tk.X, pady=5)
        ttk.Label(frame3, text="Top 10 Vessels by Anomaly Type:").pack(anchor=tk.W)
        listbox_frame = ttk.Frame(frame3)
        listbox_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.top_vessels_listbox = tk.Listbox(listbox_frame, height=10)
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.top_vessels_listbox.yview)
        self.top_vessels_listbox.configure(yscrollcommand=scrollbar.set)
        self.top_vessels_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add right-click context menu for copy
        self.top_vessels_menu = tk.Menu(self.window, tearoff=0)
        self.top_vessels_menu.add_command(label="Copy MMSI", command=self._copy_mmsi_from_listbox)
        self.top_vessels_listbox.bind("<Button-3>", self._show_listbox_menu)
        # Add double-click handler to populate MMSI fields
        self.top_vessels_listbox.bind("<Double-Button-1>", self._populate_mmsi_from_listbox)
        self._populate_top_vessels()
        
        # Create Filtered Map section with inline filters
        filtered_map_frame = ttk.LabelFrame(tab, text="Create Filtered Map", padding=10)
        filtered_map_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Map type selection (checkboxes - can select multiple) and Create Map button on same line
        map_type_frame = ttk.Frame(filtered_map_frame)
        map_type_frame.pack(fill=tk.X, pady=5)
        ttk.Label(map_type_frame, text="Map Layers (select all that apply):", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.filtered_map_type_path_var = tk.BooleanVar(value=True)
        self.filtered_map_type_anomaly_var = tk.BooleanVar(value=True)
        self.filtered_map_type_heatmap_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(map_type_frame, text="Path Map", variable=self.filtered_map_type_path_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(map_type_frame, text="Anomaly Map", variable=self.filtered_map_type_anomaly_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(map_type_frame, text="Heatmap", variable=self.filtered_map_type_heatmap_var).pack(side=tk.LEFT, padx=5)
        
        # Create Map button on the same line
        ttk.Button(map_type_frame, text="Create Map", width=20,
                  command=self._create_filtered_map_from_tab).pack(side=tk.RIGHT, padx=10)
        
        # Create a container for filters side by side
        filters_container = ttk.Frame(filtered_map_frame)
        filters_container.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Vessel Type Filters (left side)
        vessel_filter_frame = ttk.LabelFrame(filters_container, text="Filter by Vessel Type", padding=5)
        vessel_filter_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Create scrollable frame for vessel types
        vessel_canvas = tk.Canvas(vessel_filter_frame, height=150)
        vessel_scrollbar = ttk.Scrollbar(vessel_filter_frame, orient=tk.VERTICAL, command=vessel_canvas.yview)
        vessel_scrollable_frame = ttk.Frame(vessel_canvas)
        
        vessel_scrollable_frame.bind(
            "<Configure>",
            lambda e: vessel_canvas.configure(scrollregion=vessel_canvas.bbox("all"))
        )
        
        vessel_canvas.create_window((0, 0), window=vessel_scrollable_frame, anchor="nw")
        vessel_canvas.configure(yscrollcommand=vessel_scrollbar.set)
        
        # Store vessel type filter variables
        self.filtered_vessel_vars = {}
        try:
            df = self.analysis.load_cached_data()
            if 'VesselType' in df.columns:
                available_vessel_types_raw = df['VesselType'].dropna().unique()
                available_vessel_types = sorted([int(vt) for vt in available_vessel_types_raw if pd.notna(vt)])
                for vtype in available_vessel_types:
                    var = tk.BooleanVar()
                    self.filtered_vessel_vars[vtype] = var
                    vessel_name = get_vessel_type_name(vtype)
                    display_text = f"Type {vtype}: {vessel_name}"
                    cb = ttk.Checkbutton(vessel_scrollable_frame, text=display_text, variable=var)
                    cb.pack(anchor=tk.W)
        except Exception as e:
            logger.warning(f"Could not load vessel types for filtering: {e}")
        
        # Bind mouse wheel to canvas scrolling when hovering over the canvas
        def _on_mousewheel_vessel(event):
            if vessel_canvas.winfo_containing(event.x_root, event.y_root) == vessel_canvas:
                vessel_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        vessel_canvas.bind("<MouseWheel>", _on_mousewheel_vessel)
        
        vessel_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vessel_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Anomaly Type Filters (right side)
        anomaly_filter_frame = ttk.LabelFrame(filters_container, text="Filter by Anomaly Type", padding=5)
        anomaly_filter_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Create scrollable frame for anomaly types
        anomaly_canvas = tk.Canvas(anomaly_filter_frame, height=150)
        anomaly_scrollbar = ttk.Scrollbar(anomaly_filter_frame, orient=tk.VERTICAL, command=anomaly_canvas.yview)
        anomaly_scrollable_frame = ttk.Frame(anomaly_canvas)
        
        anomaly_scrollable_frame.bind(
            "<Configure>",
            lambda e: anomaly_canvas.configure(scrollregion=anomaly_canvas.bbox("all"))
        )
        
        anomaly_canvas.create_window((0, 0), window=anomaly_scrollable_frame, anchor="nw")
        anomaly_canvas.configure(yscrollcommand=anomaly_scrollbar.set)
        
        # Store anomaly type filter variables
        self.filtered_anomaly_vars = {}
        try:
            anomaly_df = self.analysis.load_anomaly_data()
            if not anomaly_df.empty and 'AnomalyType' in anomaly_df.columns:
                available_anomaly_types_data = sorted(anomaly_df['AnomalyType'].unique().tolist())
                available_anomaly_types_gui = [map_anomaly_type_data_to_gui(at) for at in available_anomaly_types_data]
                # Remove duplicates while preserving order
                seen = set()
                unique_anomaly_types_gui = []
                for at in available_anomaly_types_gui:
                    if at not in seen:
                        seen.add(at)
                        unique_anomaly_types_gui.append(at)
                
                for atype in unique_anomaly_types_gui:
                    var = tk.BooleanVar()
                    self.filtered_anomaly_vars[atype] = var
                    cb = ttk.Checkbutton(anomaly_scrollable_frame, text=atype, variable=var)
                    cb.pack(anchor=tk.W)
        except Exception as e:
            logger.warning(f"Could not load anomaly types for filtering: {e}")
        
        # Bind mouse wheel to canvas scrolling when hovering over the canvas
        def _on_mousewheel_anomaly(event):
            if anomaly_canvas.winfo_containing(event.x_root, event.y_root) == anomaly_canvas:
                anomaly_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        anomaly_canvas.bind("<MouseWheel>", _on_mousewheel_anomaly)
        
        anomaly_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        anomaly_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_tab4_vessel_analysis(self):
        """Create Tab 4: Vessel-Specific Analysis"""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Vessel-Specific Analysis")
        
        ttk.Label(tab, text="Vessel-Specific Analysis Tools", 
                 font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        frame1 = ttk.Frame(tab)
        frame1.pack(fill=tk.X, pady=5)
        ttk.Label(frame1, text="MMSI:").pack(side=tk.LEFT, padx=5)
        self.extended_mmsi_var = tk.StringVar()
        ttk.Entry(frame1, textvariable=self.extended_mmsi_var, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame1, text="Additional Date Range:").pack(side=tk.LEFT, padx=5)
        
        # Use DateEntry if available, otherwise use text entry
        if TKCALENDAR_AVAILABLE:
            ttk.Label(frame1, text="Start:").pack(side=tk.LEFT, padx=2)
            # Use state='normal' to allow manual input; month/year dropdowns work by default in calendar popup
            # Parse default start date
            try:
                default_start_dt = datetime.strptime(self.default_start_date, '%Y-%m-%d').date()
            except (ValueError, AttributeError):
                default_start_dt = datetime(2024, 10, 1).date()
            self.extended_start_picker = DateEntry(frame1, width=12, background='darkblue',
                                                   foreground='white', borderwidth=2, date_pattern='y-mm-dd',
                                                   state='normal')
            self.extended_start_picker.set_date(default_start_dt)
            self.extended_start_picker.pack(side=tk.LEFT, padx=2)
            
            ttk.Label(frame1, text="End:").pack(side=tk.LEFT, padx=2)
            # Parse default end date
            try:
                default_end_dt = datetime.strptime(self.default_end_date, '%Y-%m-%d').date()
            except (ValueError, AttributeError):
                default_end_dt = datetime(2024, 10, 3).date()
            self.extended_end_picker = DateEntry(frame1, width=12, background='darkblue',
                                                 foreground='white', borderwidth=2, date_pattern='y-mm-dd',
                                                 state='normal')
            self.extended_end_picker.set_date(default_end_dt)
            self.extended_end_picker.pack(side=tk.LEFT, padx=2)
            self.extended_start_var = None
            self.extended_end_var = None
        else:
            self.extended_start_var = tk.StringVar(value=self.default_start_date)
            self.extended_end_var = tk.StringVar(value=self.default_end_date)
            ttk.Entry(frame1, textvariable=self.extended_start_var, width=12).pack(side=tk.LEFT, padx=2)
            ttk.Label(frame1, text="to").pack(side=tk.LEFT)
            ttk.Entry(frame1, textvariable=self.extended_end_var, width=12).pack(side=tk.LEFT, padx=2)
            self.extended_start_picker = None
            self.extended_end_picker = None
        
        ttk.Button(frame1, text="Analyze", width=20,
                  command=self._extended_time_analysis).pack(side=tk.LEFT, padx=5)

        frame2 = ttk.Frame(tab)
        frame2.pack(fill=tk.X, pady=5)
        if ML_PREDICTION_AVAILABLE:
            ttk.Button(frame2, text="AI Predicted Path", width=30,
                      command=self._ml_course_prediction).pack(side=tk.LEFT, padx=5)
        else:
            ttk.Button(frame2, text="AI Predicted Path (ML Module Not Available)", width=50,
                      command=lambda: messagebox.showwarning("Not Available", 
                      "ML Course Prediction module is not available. Please ensure PyTorch and ml_course_prediction module are installed.")).pack(side=tk.LEFT, padx=5)
    
    def _export_full_dataset(self):
        progress = ProgressDialog(self.window, "Exporting Dataset", "Exporting full dataset to CSV...")
        try:
            self.status_var.set("Exporting full dataset...")
            result = self.analysis.export_full_dataset()
            progress.close()
            if result:
                self.status_var.set(f"Exported to: {result}")
                file_size = os.path.getsize(result)
                messagebox.showinfo("Success", 
                                  f"Full dataset exported successfully!\n\n"
                                  f"Location: {result}\n"
                                  f"Size: {format_file_size(file_size)}")
            else:
                self.status_var.set("Export failed")
                messagebox.showerror("Error", "Failed to export full dataset. Check logs for details.")
        except Exception as e:
            progress.close()
            self.status_var.set("Export failed")
            logger.error(f"Error in export: {e}")
            messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def _generate_summary_report(self):
        progress = ProgressDialog(self.window, "Generating Report", "Generating summary report...")
        try:
            self.status_var.set("Generating summary report...")
            result = self.analysis.generate_summary_report()
            progress.close()
            if result:
                self.status_var.set(f"Report generated: {result}")
                messagebox.showinfo("Success", f"Summary report generated:\n{result}")
            else:
                self.status_var.set("Report generation failed")
                messagebox.showerror("Error", "Failed to generate summary report. Check logs for details.")
        except Exception as e:
            progress.close()
            self.status_var.set("Report generation failed")
            logger.error(f"Error generating report: {e}")
            messagebox.showerror("Error", f"Report generation failed: {str(e)}")
    
    def _export_vessel_statistics(self):
        progress = ProgressDialog(self.window, "Exporting Statistics", "Exporting vessel statistics...")
        try:
            self.status_var.set("Exporting vessel statistics...")
            result = self.analysis.export_vessel_statistics()
            progress.close()
            if result:
                self.status_var.set(f"Statistics exported: {result}")
                messagebox.showinfo("Success", f"Vessel statistics exported to:\n{result}")
            else:
                self.status_var.set("Export failed")
                messagebox.showerror("Error", "Failed to export vessel statistics. Check logs for details.")
        except Exception as e:
            progress.close()
            self.status_var.set("Export failed")
            logger.error(f"Error exporting statistics: {e}")
            messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def _generate_anomaly_timeline(self):
        progress = ProgressDialog(self.window, "Generating Timeline", "Generating anomaly timeline...")
        try:
            self.status_var.set("Generating anomaly timeline...")
            result = self.analysis.generate_anomaly_timeline()
            progress.close()
            if result:
                self.status_var.set(f"Timeline generated: {result}")
                messagebox.showinfo("Success", f"Anomaly timeline generated:\n{result}")
            else:
                self.status_var.set("Timeline generation failed")
                messagebox.showerror("Error", "Failed to generate anomaly timeline. Check logs for details.")
        except Exception as e:
            progress.close()
            self.status_var.set("Timeline generation failed")
            logger.error(f"Error generating timeline: {e}")
            messagebox.showerror("Error", f"Timeline generation failed: {str(e)}")
    
    def _correlation_analysis_dialog(self):
        """Create dialog for correlation analysis selection"""
        dialog = tk.Toplevel(self.window)
        dialog.title("Anomaly Correlation Analysis")
        dialog.geometry("700x600")  # Increased size for better visibility
        
        try:
            df = self.analysis.load_cached_data()
            anomaly_df = self.analysis.load_anomaly_data()
            
            # Get available types from data - convert to int for proper comparison
            if 'VesselType' in df.columns:
                # Convert to int, handling NaN values
                available_vessel_types_raw = df['VesselType'].dropna().unique()
                available_vessel_types = sorted([int(vt) for vt in available_vessel_types_raw if pd.notna(vt)])
            else:
                available_vessel_types = []
                
            # Get vessel types from run_info (these should be considered available)
            run_info_vessel_types = []
            if hasattr(self.analysis, 'run_info') and 'ship_types' in self.analysis.run_info:
                try:
                    # Ensure we get integers for the ship types
                    run_info_vessel_types = [int(vt) for vt in self.analysis.run_info['ship_types'] if vt and pd.notna(vt)]
                    logger.info(f"Using vessel types from run_info: {run_info_vessel_types}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert run_info ship_types to int: {e}")
            
            # If no cached data is available, ensure we still use run_info vessel types
            if df.empty and run_info_vessel_types:
                logger.info("No cached data found but using vessel types from run_info")
                # Create warning at top of dialog - don't reference canvas yet
                warning_frame = ttk.Frame(dialog)
                warning_frame.pack(fill=tk.X, pady=5, side=tk.TOP)
                warning_label = ttk.Label(warning_frame, 
                                        text="WARNING: No cached data found for the current date range.\n" +
                                             "Analysis will use vessel types from configuration but may have limited results.", 
                                        foreground="red", font=("Arial", 10, "bold"))
                warning_label.pack(pady=5)
                ttk.Separator(warning_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
                
            # Combine available types from data and run_info
            available_vessel_types = sorted(list(set(available_vessel_types + run_info_vessel_types)))
            
            available_anomaly_types_data = sorted(anomaly_df['AnomalyType'].unique().tolist()) if not anomaly_df.empty and 'AnomalyType' in anomaly_df.columns else []
            
            # Map data anomaly types to GUI names
            available_anomaly_types_gui = [map_anomaly_type_data_to_gui(at) for at in available_anomaly_types_data]
            
            # Get all possible types
            all_vessel_types = get_all_vessel_types()
            all_anomaly_types = get_all_anomaly_types()
            
            # Create scrollable frames
            canvas = tk.Canvas(dialog)
            scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            ttk.Label(scrollable_frame, text="Select Vessel Types (up to 2):", font=("Arial", 10, "bold")).pack(pady=5)
            ttk.Label(scrollable_frame, text="[In Data] indicates vessel types present in the dataset", font=("Arial", 8, "italic")).pack()
            vessel_frame = ttk.Frame(scrollable_frame)
            vessel_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            vessel_vars = {}
            vessel_checkboxes = []
            for vtype in all_vessel_types:
                # Auto-select vessel types from run_info
                is_in_run_info = vtype in run_info_vessel_types
                var = tk.BooleanVar(value=is_in_run_info)  # Pre-select if in run_info
                vessel_vars[vtype] = var
                is_available = vtype in available_vessel_types
                vessel_name = get_vessel_type_name(vtype)
                
                # Use the same format as vessel_behavior_clustering
                if is_available:
                    display_text = f"Type {vtype}: {vessel_name} [In Data]"
                else:
                    display_text = f"Type {vtype}: {vessel_name} [Not in Current Data]"
                
                # Add a special indicator for run_info vessel types
                if is_in_run_info:
                    display_text += " [Selected in Initial Analysis]"
                
                # All vessel types are selectable
                cb = ttk.Checkbutton(vessel_frame, text=display_text, variable=var)
                cb.pack(anchor=tk.W)
                vessel_checkboxes.append((vtype, var, True))  # Set all as available
            
            ttk.Label(scrollable_frame, text="Select Anomaly Types (up to 2):", font=("Arial", 10, "bold")).pack(pady=5)
            ttk.Label(scrollable_frame, text="[In Data] indicates anomaly types present in the dataset", font=("Arial", 8, "italic")).pack()
            anomaly_frame = ttk.Frame(scrollable_frame)
            anomaly_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            anomaly_vars = {}
            anomaly_checkboxes = []
            for atype in all_anomaly_types:
                var = tk.BooleanVar()
                anomaly_vars[atype] = var
                is_available = atype in available_anomaly_types_gui
                
                # Match the vessel type format
                if is_available:
                    display_text = f"{atype} [In Data]"
                else:
                    display_text = f"{atype} [Not in Current Data]"
                    
                cb = ttk.Checkbutton(anomaly_frame, text=display_text, variable=var)
                cb.pack(anchor=tk.W)
                anomaly_checkboxes.append((atype, var, True))  # Set all as available
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            def run_analysis():
                # Select up to 2 vessel types, ensuring they are integers (critical for the analysis)
                selected_vessel_types = []
                for vtype, var, available in vessel_checkboxes:
                    if var.get():
                        try:
                            # Ensure vessel type is an integer
                            selected_vessel_types.append(int(vtype))
                            if len(selected_vessel_types) >= 2:
                                break
                        except (ValueError, TypeError):
                            messagebox.showerror("Error", f"Invalid vessel type: {vtype}")
                            return
                
                # Select up to 2 anomaly types
                selected_anomaly_types_gui = [atype for atype, var, available in anomaly_checkboxes if var.get()][:2]
                
                # Map GUI names back to data names
                selected_anomaly_types = [map_anomaly_type_gui_to_data(at) for at in selected_anomaly_types_gui]
                
                if not selected_vessel_types and not selected_anomaly_types:
                    messagebox.showerror("Error", "Please select at least one vessel type or anomaly type")
                    return
                
                dialog.destroy()
                self.status_var.set("Performing correlation analysis...")
                
                # Log what we're passing to the analysis method
                logger.info(f"Running correlation analysis with vessel types: {selected_vessel_types}")
                logger.info(f"Running correlation analysis with anomaly types: {selected_anomaly_types}")
                
                try:
                    logger.info(f"Starting correlation analysis with vessel types: {selected_vessel_types}")
                    result = self.analysis.correlation_analysis(selected_vessel_types, selected_anomaly_types)
                    if result:
                        self.status_var.set(f"Analysis complete: {result}")
                        messagebox.showinfo("Success", f"Correlation analysis complete:\n{result}")
                    else:
                        # Even if result is None, don't immediately show error
                        self.status_var.set("Analysis completed with limited data")
                        messagebox.showinfo("Limited Results", "Correlation analysis completed, but may have limited results due to missing data")
                except Exception as e:
                    self.status_var.set("Analysis failed")
                    logger.error(f"Correlation analysis error: {e}")
                    messagebox.showerror("Error", f"Failed to perform correlation analysis: {str(e)}")
                    logger.error(traceback.format_exc())
            
            button_frame = ttk.Frame(dialog)
            button_frame.pack(pady=10)
            ttk.Button(button_frame, text="Run Analysis", command=run_analysis).pack()
            
        except Exception as e:
            logger.error(f"Error creating correlation dialog: {e}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Error creating correlation analysis dialog: {e}")
            dialog.destroy()
    
    def _temporal_pattern_analysis(self):
        progress = ProgressDialog(self.window, "Analyzing Patterns", "Analyzing temporal patterns...")
        try:
            self.status_var.set("Analyzing temporal patterns...")
            result = self.analysis.temporal_pattern_analysis()
            progress.close()
            if result:
                self.status_var.set(f"Analysis complete: {result}")
                messagebox.showinfo("Success", f"Temporal pattern analysis complete:\n{result}")
            else:
                self.status_var.set("Analysis failed")
                messagebox.showerror("Error", "Failed to perform temporal pattern analysis. Check logs for details.")
        except Exception as e:
            progress.close()
            self.status_var.set("Analysis failed")
            logger.error(f"Error in temporal analysis: {e}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    def _vessel_behavior_clustering(self):
        """Create dialog for vessel behavior clustering with vessel type selection"""
        dialog = tk.Toplevel(self.window)
        dialog.title("Vessel Behavior Clustering")
        dialog.geometry("600x600")
        
        try:
            df = self.analysis.load_cached_data()
            
            # Get available types from data - convert to int for proper comparison
            if 'VesselType' in df.columns:
                # Convert to int, handling NaN values
                available_vessel_types_raw = df['VesselType'].dropna().unique()
                available_vessel_types = sorted([int(vt) for vt in available_vessel_types_raw if pd.notna(vt)])
            else:
                available_vessel_types = []
            
            # Get all possible types
            all_vessel_types = get_all_vessel_types()
            
            # Create scrollable frames
            canvas = tk.Canvas(dialog)
            scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            ttk.Label(scrollable_frame, text="Select Vessel Types for Clustering", 
                     font=("Arial", 12, "bold")).pack(pady=10)
            ttk.Label(scrollable_frame, 
                     text="Select one or more vessel types to analyze. All selected types will be included.",
                     wraplength=550).pack(pady=5)
            
            # Number of clusters
            ttk.Label(scrollable_frame, text="Number of Clusters:", font=("Arial", 10, "bold")).pack(pady=(10, 5))
            n_clusters_var = tk.StringVar(value='5')
            clusters_frame = ttk.Frame(scrollable_frame)
            clusters_frame.pack(pady=5)
            ttk.Label(clusters_frame, text="Clusters:").pack(side=tk.LEFT, padx=5)
            clusters_entry = ttk.Entry(clusters_frame, textvariable=n_clusters_var, width=10)
            clusters_entry.pack(side=tk.LEFT, padx=5)
            
            # Vessel type selection
            ttk.Label(scrollable_frame, text="Vessel Types:", font=("Arial", 10, "bold")).pack(pady=(10, 5))
            
            vessel_vars = {}
            vessel_checkboxes = []
            
            for vtype in all_vessel_types:
                var = tk.BooleanVar()
                vessel_vars[vtype] = var
                vessel_name = get_vessel_type_name(vtype)
                
                # Check if this type is in the current data
                is_in_data = vtype in available_vessel_types
                
                # Make all types selectable, but indicate which are in data
                if is_in_data:
                    display_text = f"Type {vtype}: {vessel_name} [In Data]"
                else:
                    display_text = f"Type {vtype}: {vessel_name} [Not in Current Data]"
                
                # All types are enabled for selection
                cb = ttk.Checkbutton(scrollable_frame, text=display_text, variable=var, 
                                   state='normal')
                cb.pack(anchor=tk.W, padx=20, pady=2)
                vessel_checkboxes.append((vtype, var, True))  # All are available for selection
            
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            def perform_clustering():
                selected_types = [vtype for vtype, var, is_available in vessel_checkboxes 
                                if var.get()]
                
                if not selected_types:
                    messagebox.showwarning("Warning", "No vessel types selected. Please select at least one vessel type.")
                    return
                
                try:
                    n_clusters = int(n_clusters_var.get())
                    if n_clusters < 2:
                        messagebox.showerror("Error", "Number of clusters must be at least 2")
                        return
                except ValueError:
                    messagebox.showerror("Error", "Number of clusters must be a valid integer")
                    return
                
                dialog.destroy()
                progress = ProgressDialog(self.window, "Clustering", "Performing vessel clustering...")
                try:
                    self.status_var.set("Performing vessel clustering...")
                    result = self.analysis.vessel_behavior_clustering(
                        vessel_types=selected_types if selected_types else None,
                        n_clusters=n_clusters
                    )
                    progress.close()
                    if result:
                        self.status_var.set(f"Clustering complete: {result}")
                        messagebox.showinfo("Success", f"Vessel clustering complete:\n{result}")
                    else:
                        self.status_var.set("Clustering failed")
                        messagebox.showerror("Error", "Failed to perform vessel clustering. Check logs for details.")
                except Exception as e:
                    progress.close()
                    self.status_var.set("Clustering failed")
                    logger.error(f"Error in clustering: {e}")
                    logger.error(traceback.format_exc())
                    messagebox.showerror("Error", f"Clustering failed: {str(e)}")
            
            button_frame = ttk.Frame(dialog)
            button_frame.pack(pady=10)
            ttk.Button(button_frame, text="Perform Clustering", command=perform_clustering).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            logger.error(f"Error creating clustering dialog: {e}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Error creating clustering dialog: {e}")
            dialog.destroy()
    
    def _anomaly_frequency_analysis(self):
        progress = ProgressDialog(self.window, "Analyzing Frequency", "Analyzing anomaly frequency...")
        try:
            self.status_var.set("Analyzing anomaly frequency...")
            result = self.analysis.anomaly_frequency_analysis()
            progress.close()
            if result:
                self.status_var.set(f"Analysis complete: {result}")
                messagebox.showinfo("Success", f"Anomaly frequency analysis complete:\n{result}")
            else:
                self.status_var.set("Analysis failed")
                messagebox.showerror("Error", "Failed to perform anomaly frequency analysis. Check logs for details.")
        except Exception as e:
            progress.close()
            self.status_var.set("Analysis failed")
            logger.error(f"Error in frequency analysis: {e}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    def _create_custom_chart_dialog(self):
        """Create dialog for custom chart creation"""
        dialog = tk.Toplevel(self.window)
        dialog.title("Create Custom Chart")
        dialog.geometry("700x650")
        
        try:
            df = self.analysis.load_cached_data()
            anomaly_df = self.analysis.load_anomaly_data()
            
            # Get available columns from both datasets
            df_columns = sorted([col for col in df.columns if col not in ['MMSI', 'LAT', 'LON']]) if not df.empty else []
            anomaly_columns = sorted([col for col in anomaly_df.columns if col not in ['MMSI', 'LAT', 'LON']]) if not anomaly_df.empty else []
            all_columns = sorted(list(set(df_columns + anomaly_columns)))
            
            # Chart type selection
            ttk.Label(dialog, text="Chart Type:", font=("Arial", 10, "bold")).pack(pady=5, anchor=tk.W, padx=10)
            chart_type_var = tk.StringVar(value='bar')
            chart_types = [
                ('Bar Chart', 'bar'),
                ('Scatter Plot', 'scatter'),
                ('Line Chart', 'line'),
                ('Pie Chart', 'pie'),
                ('Stacked Bar Chart', 'stacked_bar'),
                ('Timeline', 'timeline'),
                ('Histogram', 'histogram'),
                ('Box Plot', 'box')
            ]
            
            chart_frame = ttk.Frame(dialog)
            chart_frame.pack(fill=tk.X, padx=10, pady=5)
            # Arrange chart types in a grid for better layout
            row = 0
            col = 0
            for text, value in chart_types:
                ttk.Radiobutton(chart_frame, text=text, variable=chart_type_var, value=value).grid(row=row, column=col, padx=5, pady=2, sticky=tk.W)
                col += 1
                if col > 3:  # 4 columns per row
                    col = 0
                    row += 1
            
            # X-axis column
            ttk.Label(dialog, text="X-Axis Column:", font=("Arial", 10, "bold")).pack(pady=(10, 5), anchor=tk.W, padx=10)
            x_column_var = tk.StringVar()
            x_combo = ttk.Combobox(dialog, textvariable=x_column_var, values=all_columns, width=40, state='readonly')
            x_combo.pack(padx=10, pady=5, anchor=tk.W)
            
            # Y-axis column (for charts that need it)
            ttk.Label(dialog, text="Y-Axis Column (optional):", font=("Arial", 10, "bold")).pack(pady=(10, 5), anchor=tk.W, padx=10)
            y_column_var = tk.StringVar()
            y_combo = ttk.Combobox(dialog, textvariable=y_column_var, values=all_columns, width=40, state='readonly')
            y_combo.pack(padx=10, pady=5, anchor=tk.W)
            
            # Color/Group by column
            ttk.Label(dialog, text="Color/Group By Column (optional):", font=("Arial", 10, "bold")).pack(pady=(10, 5), anchor=tk.W, padx=10)
            color_column_var = tk.StringVar()
            color_combo = ttk.Combobox(dialog, textvariable=color_column_var, values=all_columns, width=40, state='readonly')
            color_combo.pack(padx=10, pady=5, anchor=tk.W)
            
            # Group by column
            ttk.Label(dialog, text="Group By Column (optional):", font=("Arial", 10, "bold")).pack(pady=(10, 5), anchor=tk.W, padx=10)
            group_by_var = tk.StringVar()
            group_combo = ttk.Combobox(dialog, textvariable=group_by_var, values=all_columns, width=40, state='readonly')
            group_combo.pack(padx=10, pady=5, anchor=tk.W)
            
            # Aggregation method
            ttk.Label(dialog, text="Aggregation (if grouping):", font=("Arial", 10, "bold")).pack(pady=(10, 5), anchor=tk.W, padx=10)
            aggregation_var = tk.StringVar(value='count')
            agg_frame = ttk.Frame(dialog)
            agg_frame.pack(fill=tk.X, padx=10, pady=5)
            for agg in ['count', 'sum', 'mean', 'max', 'min']:
                ttk.Radiobutton(agg_frame, text=agg.capitalize(), variable=aggregation_var, value=agg).pack(side=tk.LEFT, padx=5)
            
            # Chart title
            ttk.Label(dialog, text="Chart Title (optional):", font=("Arial", 10, "bold")).pack(pady=(10, 5), anchor=tk.W, padx=10)
            title_var = tk.StringVar()
            title_entry = ttk.Entry(dialog, textvariable=title_var, width=50)
            title_entry.pack(padx=10, pady=5, anchor=tk.W)
            
            def update_requirements():
                """Update UI based on selected chart type"""
                chart_type = chart_type_var.get()
                
                # Enable/disable fields based on chart type
                if chart_type in ['scatter', 'line', 'box', 'stacked_bar']:
                    y_combo.config(state='readonly')
                else:
                    y_combo.config(state='disabled')
                    y_column_var.set('')
                
                if chart_type == 'timeline':
                    x_combo.config(state='disabled')
                    x_column_var.set('')
                else:
                    x_combo.config(state='readonly')
            
            chart_type_var.trace('w', lambda *args: update_requirements())
            update_requirements()
            
            def create_chart():
                chart_type = chart_type_var.get()
                x_column = x_column_var.get() if x_column_var.get() else None
                y_column = y_column_var.get() if y_column_var.get() else None
                color_column = color_column_var.get() if color_column_var.get() else None
                group_by = group_by_var.get() if group_by_var.get() else None
                aggregation = aggregation_var.get()
                title = title_var.get() if title_var.get() else None
                
                # Validation
                if chart_type in ['scatter', 'line', 'box', 'stacked_bar'] and not y_column:
                    messagebox.showerror("Error", f"{chart_type.capitalize()} chart requires a Y-axis column")
                    return
                
                if chart_type != 'timeline' and not x_column:
                    messagebox.showerror("Error", "Please select an X-axis column")
                    return
                
                if chart_type == 'stacked_bar' and not color_column:
                    messagebox.showerror("Error", "Stacked bar chart requires a color/group by column")
                    return
                
                dialog.destroy()
                progress = ProgressDialog(self.window, "Creating Chart", f"Creating {chart_type} chart...")
                try:
                    self.status_var.set(f"Creating {chart_type} chart...")
                    result = self.analysis.create_custom_chart(
                        chart_type=chart_type,
                        x_column=x_column,
                        y_column=y_column,
                        color_column=color_column,
                        group_by=group_by,
                        aggregation=aggregation,
                        title=title
                    )
                    progress.close()
                    if result:
                        self.status_var.set(f"Chart created: {result}")
                        messagebox.showinfo("Success", f"Chart created:\n{result}")
                    else:
                        self.status_var.set("Chart creation failed")
                        messagebox.showerror("Error", "Failed to create chart. Check logs for details.")
                except Exception as e:
                    progress.close()
                    self.status_var.set("Chart creation failed")
                    logger.error(f"Error creating chart: {e}")
                    logger.error(traceback.format_exc())
                    messagebox.showerror("Error", f"Chart creation failed: {str(e)}")
            
            button_frame = ttk.Frame(dialog)
            button_frame.pack(pady=20)
            ttk.Button(button_frame, text="Create Chart", command=create_chart).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
            
        except Exception as e:
            logger.error(f"Error creating chart dialog: {e}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Error creating chart dialog: {e}")
            dialog.destroy()
    
    def _full_spectrum_map_dialog(self):
        dialog = tk.Toplevel(self.window)
        dialog.title("Full Spectrum Map Options")
        
        show_pins_var = tk.BooleanVar(value=True)
        show_heatmap_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(dialog, text="Show Anomaly Pins", variable=show_pins_var).pack(pady=5)
        ttk.Checkbutton(dialog, text="Show Heatmap Overlay", variable=show_heatmap_var).pack(pady=5)
        
        def create_map():
            dialog.destroy()
            progress = ProgressDialog(self.window, "Creating Map", "Creating full spectrum map...")
            try:
                self.status_var.set("Creating full spectrum map...")
                result = self.analysis.create_full_spectrum_map(
                    show_pins=show_pins_var.get(),
                    show_heatmap=show_heatmap_var.get()
                )
                progress.close()
                if result:
                    self.status_var.set(f"Map created: {result}")
                    messagebox.showinfo("Success", f"Map created:\n{result}")
                else:
                    self.status_var.set("Map creation failed")
                    messagebox.showerror("Error", "Failed to create map. Check logs for details.")
            except Exception as e:
                progress.close()
                self.status_var.set("Map creation failed")
                logger.error(f"Error creating map: {e}")
                messagebox.showerror("Error", f"Map creation failed: {str(e)}")
        
        ttk.Button(dialog, text="Create Map", command=create_map).pack(pady=10)
    
    def _generate_map(self):
        """Generate a map with the current dataset"""
        dialog = tk.Toplevel(self.window)
        dialog.title("Generate Map")
        dialog.geometry("400x250")
        dialog.transient(self.window)
        dialog.grab_set()
        
        # Map type selection
        type_frame = ttk.Frame(dialog, padding=5)
        type_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(type_frame, text="Map Type:").pack(side=tk.LEFT, padx=5)
        
        map_type_var = tk.StringVar(value='path')
        ttk.Radiobutton(type_frame, text="Path Map", variable=map_type_var, value='path').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(type_frame, text="Anomaly Map", variable=map_type_var, value='anomaly').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(type_frame, text="Heatmap", variable=map_type_var, value='heatmap').pack(side=tk.LEFT, padx=5)
        
        # Generate map function
        def do_generate_map():
            dialog.destroy()
            progress = ProgressDialog(self.window, "Creating Map", f"Generating {map_type_var.get()} map...")
            try:
                self.status_var.set("Generating map...")
                result = self.analysis.create_filtered_map(
                    map_type=map_type_var.get(),
                    vessel_types=None,  # Use all vessel types
                    anomaly_types=None,  # Use all anomaly types
                    vessel_mmsi=None,    # No specific vessel
                    output_path=None     # Use default output path
                )
                progress.close()
                if result:
                    self.status_var.set(f"Map created: {result}")
                    messagebox.showinfo("Success", f"Map created successfully:\n{result}")
                    # Open the map file
                    open_file(result)
                else:
                    self.status_var.set("Map creation failed")
                    messagebox.showerror("Error", "Failed to create map. Check logs for details.")
            except Exception as e:
                progress.close()
                self.status_var.set("Map creation failed")
                logger.error(f"Error creating map: {e}")
                messagebox.showerror("Error", f"Map creation failed: {str(e)}")
        
        # Add button to generate
        button_frame = ttk.Frame(dialog, padding=10)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Button(button_frame, text="Generate Map", command=do_generate_map).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    def _create_vessel_map_dialog(self):
        mmsi_str = self.vessel_mmsi_var.get()
        if not mmsi_str:
            messagebox.showerror("Error", "Please enter an MMSI number")
            return
        
        try:
            mmsi = int(mmsi_str)
        except ValueError:
            messagebox.showerror("Error", "MMSI must be a number")
            return
        
        dialog = tk.Toplevel(self.window)
        dialog.title("Vessel Map Type")
        
        map_type_var = tk.StringVar(value='path')
        ttk.Radiobutton(dialog, text="Path Map", variable=map_type_var, value='path').pack(pady=5)
        ttk.Radiobutton(dialog, text="Anomaly Map", variable=map_type_var, value='anomaly').pack(pady=5)
        ttk.Radiobutton(dialog, text="Heatmap", variable=map_type_var, value='heatmap').pack(pady=5)
        
        def create_map():
            dialog.destroy()
            progress = ProgressDialog(self.window, "Creating Map", f"Creating {map_type_var.get()} map...")
            try:
                self.status_var.set(f"Creating {map_type_var.get()} map for vessel {mmsi}...")
                result = self.analysis.create_vessel_map(mmsi, map_type_var.get())
                progress.close()
                if result:
                    self.status_var.set(f"Map created: {result}")
                    messagebox.showinfo("Success", f"Map created:\n{result}")
                else:
                    self.status_var.set("Map creation failed")
                    messagebox.showerror("Error", "Failed to create map. Check logs for details.")
            except Exception as e:
                progress.close()
                self.status_var.set("Map creation failed")
                logger.error(f"Error creating map: {e}")
                messagebox.showerror("Error", f"Map creation failed: {str(e)}")
        
        ttk.Button(dialog, text="Create Map", command=create_map).pack(pady=10)
    
    def _populate_top_vessels(self):
        """Populate the top vessels listbox"""
        try:
            anomaly_df = self.analysis.load_anomaly_data()
            if not anomaly_df.empty and 'AnomalyType' in anomaly_df.columns:
                for anomaly_type in sorted(anomaly_df['AnomalyType'].unique()):
                    top_vessels = self.analysis.get_top_vessels_by_anomaly(anomaly_type, 10)
                    if not top_vessels.empty:
                        # Map data name to GUI name for display
                        gui_name = map_anomaly_type_data_to_gui(anomaly_type)
                        self.top_vessels_listbox.insert(tk.END, f"{gui_name}:")
                        for _, row in top_vessels.iterrows():
                            self.top_vessels_listbox.insert(tk.END, f"  MMSI {row['MMSI']}: {row['Anomaly_Count']} anomalies")
        except Exception as e:
            logger.error(f"Error populating top vessels: {e}")
    
    def _show_listbox_menu(self, event):
        """Show context menu for listbox"""
        try:
            selection = self.top_vessels_listbox.curselection()
            if selection:
                self.top_vessels_menu.post(event.x_root, event.y_root)
        except Exception as e:
            logger.error(f"Error showing listbox menu: {e}")
    
    def _copy_mmsi_from_listbox(self):
        """Copy MMSI from selected listbox item to clipboard"""
        try:
            selection = self.top_vessels_listbox.curselection()
            if selection:
                item_text = self.top_vessels_listbox.get(selection[0])
                # Extract MMSI from text like "  MMSI 123456789: 5 anomalies"
                match = re.search(r'MMSI\s+(\d+)', item_text)
                if match:
                    mmsi = match.group(1)
                    self.window.clipboard_clear()
                    self.window.clipboard_append(mmsi)
                    self.status_var.set(f"Copied MMSI {mmsi} to clipboard")
                else:
                    messagebox.showwarning("Warning", "Could not extract MMSI from selected item")
        except Exception as e:
            logger.error(f"Error copying MMSI: {e}")
            messagebox.showerror("Error", f"Failed to copy MMSI: {e}")
    
    def _populate_mmsi_from_listbox(self, event=None):
        """Populate MMSI input fields from double-clicked listbox item"""
        try:
            selection = self.top_vessels_listbox.curselection()
            if selection:
                item_text = self.top_vessels_listbox.get(selection[0])
                # Skip category headers (lines ending with ":")
                if item_text.strip().endswith(':'):
                    return
                # Extract MMSI from text like "  MMSI 123456789: 5 anomalies"
                match = re.search(r'MMSI\s+(\d+)', item_text)
                if match:
                    mmsi = match.group(1)
                    # Populate MMSI field on Mapping Tools tab
                    if hasattr(self, 'vessel_mmsi_var'):
                        self.vessel_mmsi_var.set(mmsi)
                    # Populate MMSI field on Vessel-Specific Analysis tab
                    if hasattr(self, 'extended_mmsi_var'):
                        self.extended_mmsi_var.set(mmsi)
                    self.status_var.set(f"Populated MMSI {mmsi} in input fields")
                else:
                    # Silently ignore if it's not an MMSI entry (e.g., category header)
                    pass
            else:
                messagebox.showwarning("Warning", "Please select a vessel from the list")
        except Exception as e:
            logger.error(f"Error populating MMSI: {e}")
            messagebox.showerror("Error", f"Failed to populate MMSI: {e}")
    
    def _create_filtered_map_from_tab(self):
        """Create filtered map using filters selected on tab 3"""
        try:
            # Check if filter variables exist (they should be initialized when tab 3 is created)
            if not hasattr(self, 'filtered_vessel_vars'):
                self.filtered_vessel_vars = {}
            if not hasattr(self, 'filtered_anomaly_vars'):
                self.filtered_anomaly_vars = {}
            if not hasattr(self, 'filtered_map_type_path_var'):
                messagebox.showerror("Error", "Map type variables not initialized. Please restart the application.")
                return
            
            # Get selected vessel types
            selected_vessel_types = [vtype for vtype, var in self.filtered_vessel_vars.items() if var.get()]
            
            # Get selected anomaly types (GUI names) and convert to data names
            selected_anomaly_types_gui = [atype for atype, var in self.filtered_anomaly_vars.items() if var.get()]
            selected_anomaly_types = [map_anomaly_type_gui_to_data(at) for at in selected_anomaly_types_gui]
            # Remove duplicates
            selected_anomaly_types = list(set(selected_anomaly_types))
            
            # Get selected map types (can be multiple)
            selected_map_types = []
            if self.filtered_map_type_path_var.get():
                selected_map_types.append('path')
            if self.filtered_map_type_anomaly_var.get():
                selected_map_types.append('anomaly')
            if self.filtered_map_type_heatmap_var.get():
                selected_map_types.append('heatmap')
            
            # Check if at least one map type is selected
            if not selected_map_types:
                messagebox.showwarning("No Map Layers Selected", 
                    "Please select at least one map layer (Path Map, Anomaly Map, or Heatmap) before creating the map.")
                return
            
            # Check if at least one filter is selected
            if not selected_vessel_types and not selected_anomaly_types:
                messagebox.showwarning("No Filters Selected", 
                    "Please select at least one vessel type or anomaly type filter before creating the map.")
                return
            
            # Create the map
            progress = ProgressDialog(self.window, "Creating Map", "Creating filtered map...")
            try:
                self.status_var.set("Creating filtered map...")
                result = self.analysis.create_filtered_map(
                    map_types=selected_map_types,
                    vessel_types=selected_vessel_types if selected_vessel_types else None,
                    anomaly_types=selected_anomaly_types if selected_anomaly_types else None,
                    vessel_mmsi=None
                )
                progress.close()
                if result:
                    self.status_var.set(f"Map created: {result}")
                    messagebox.showinfo("Success", f"Filtered map created:\n{result}")
                    # Open the map file
                    open_file(result)
                else:
                    self.status_var.set("Map creation failed")
                    messagebox.showerror("Error", "Failed to create filtered map. Check logs for details.")
            except Exception as e:
                progress.close()
                self.status_var.set("Map creation failed")
                logger.error(f"Error creating filtered map: {e}")
                logger.error(traceback.format_exc())
                messagebox.showerror("Error", f"Map creation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Error creating filtered map from tab: {e}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Error creating filtered map: {e}")
    
    def _create_filtered_map_dialog(self):
        """Create dialog for filtered map creation"""
        dialog = tk.Toplevel(self.window)
        dialog.title("Create Filtered Map")
        dialog.geometry("500x600")
        
        try:
            df = self.analysis.load_cached_data()
            anomaly_df = self.analysis.load_anomaly_data()
            
            # Get available types from data - convert to int for proper comparison
            if 'VesselType' in df.columns:
                # Convert to int, handling NaN values
                available_vessel_types_raw = df['VesselType'].dropna().unique()
                available_vessel_types = sorted([int(vt) for vt in available_vessel_types_raw if pd.notna(vt)])
            else:
                available_vessel_types = []
            
            available_anomaly_types_data = sorted(anomaly_df['AnomalyType'].unique().tolist()) if not anomaly_df.empty and 'AnomalyType' in anomaly_df.columns else []
            available_anomaly_types_gui = [map_anomaly_type_data_to_gui(at) for at in available_anomaly_types_data]
            
            # Map type selection (checkboxes - can select multiple)
            ttk.Label(dialog, text="Map Layers (select all that apply):", font=("Arial", 10, "bold")).pack(pady=5)
            map_type_path_var = tk.BooleanVar(value=True)
            map_type_anomaly_var = tk.BooleanVar(value=True)
            map_type_heatmap_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(dialog, text="Path Map", variable=map_type_path_var).pack(anchor=tk.W, padx=20)
            ttk.Checkbutton(dialog, text="Anomaly Map", variable=map_type_anomaly_var).pack(anchor=tk.W, padx=20)
            ttk.Checkbutton(dialog, text="Heatmap", variable=map_type_heatmap_var).pack(anchor=tk.W, padx=20)
            
            # Vessel MMSI (optional)
            ttk.Label(dialog, text="Vessel MMSI (optional):", font=("Arial", 10, "bold")).pack(pady=5)
            vessel_mmsi_var = tk.StringVar()
            ttk.Entry(dialog, textvariable=vessel_mmsi_var, width=20).pack(pady=5)
            
            # Vessel types selection
            ttk.Label(dialog, text="Vessel Types (optional):", font=("Arial", 10, "bold")).pack(pady=5)
            vessel_frame = ttk.Frame(dialog)
            vessel_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            vessel_vars = {}
            vessel_checkboxes = []
            for vtype in available_vessel_types:
                var = tk.BooleanVar()
                vessel_vars[vtype] = var
                vessel_name = get_vessel_type_name(vtype)
                display_text = f"Type {vtype}: {vessel_name}"
                cb = ttk.Checkbutton(vessel_frame, text=display_text, variable=var)
                cb.pack(anchor=tk.W)
                vessel_checkboxes.append((vtype, var))
            
            # Anomaly types selection
            ttk.Label(dialog, text="Anomaly Types (optional):", font=("Arial", 10, "bold")).pack(pady=5)
            anomaly_frame = ttk.Frame(dialog)
            anomaly_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            anomaly_vars = {}
            anomaly_checkboxes = []
            for atype in available_anomaly_types_gui:
                var = tk.BooleanVar()
                anomaly_vars[atype] = var
                cb = ttk.Checkbutton(anomaly_frame, text=atype, variable=var)
                cb.pack(anchor=tk.W)
                anomaly_checkboxes.append((atype, var))
            
            def create_map():
                vessel_mmsi = None
                mmsi_str = vessel_mmsi_var.get().strip()
                if mmsi_str:
                    try:
                        vessel_mmsi = int(mmsi_str)
                    except ValueError:
                        messagebox.showerror("Error", "MMSI must be a number")
                        return
                
                selected_vessel_types = [vtype for vtype, var in vessel_checkboxes if var.get()]
                selected_anomaly_types_gui = [atype for atype, var in anomaly_checkboxes if var.get()]
                selected_anomaly_types = [map_anomaly_type_gui_to_data(at) for at in selected_anomaly_types_gui]
                
                # Get selected map types (can be multiple)
                selected_map_types = []
                if map_type_path_var.get():
                    selected_map_types.append('path')
                if map_type_anomaly_var.get():
                    selected_map_types.append('anomaly')
                if map_type_heatmap_var.get():
                    selected_map_types.append('heatmap')
                
                if not selected_map_types:
                    messagebox.showerror("Error", "Please select at least one map layer (Path Map, Anomaly Map, or Heatmap)")
                    return
                
                if not vessel_mmsi and not selected_vessel_types and not selected_anomaly_types:
                    messagebox.showerror("Error", "Please select at least one filter (vessel MMSI, vessel types, or anomaly types)")
                    return
                
                dialog.destroy()
                progress = ProgressDialog(self.window, "Creating Map", "Creating filtered map...")
                try:
                    self.status_var.set("Creating filtered map...")
                    result = self.analysis.create_filtered_map(
                        map_types=selected_map_types,
                        vessel_types=selected_vessel_types if selected_vessel_types else None,
                        anomaly_types=selected_anomaly_types if selected_anomaly_types else None,
                        vessel_mmsi=vessel_mmsi
                    )
                    progress.close()
                    if result:
                        self.status_var.set(f"Map created: {result}")
                        messagebox.showinfo("Success", f"Filtered map created:\n{result}")
                    else:
                        self.status_var.set("Map creation failed")
                        messagebox.showerror("Error", "Failed to create filtered map. Check logs for details.")
                except Exception as e:
                    progress.close()
                    self.status_var.set("Map creation failed")
                    logger.error(f"Error creating filtered map: {e}")
                    messagebox.showerror("Error", f"Map creation failed: {str(e)}")
            
            ttk.Button(dialog, text="Create Map", command=create_map).pack(pady=10)
            
        except Exception as e:
            logger.error(f"Error creating filtered map dialog: {e}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Error creating filtered map dialog: {e}")
            dialog.destroy()
    
    def draw_geographic_box(self):
        """Draw a geographic box on a map and update lat/long bounds"""
        self._draw_box_for_bounds(
            self.window,
            self.analysis_filters['min_latitude'],
            self.analysis_filters['max_latitude'],
            self.analysis_filters['min_longitude'],
            self.analysis_filters['max_longitude']
        )
    
    def _draw_box_for_bounds(self, parent_window, min_lat_var, max_lat_var, min_lon_var, max_lon_var):
        """Draw a box on a map and populate lat/long bound fields"""
        try:
            import folium
            from folium.plugins import Draw
            import webbrowser
            import tempfile
        except ImportError as e:
            messagebox.showerror("Error", f"Required library not found: {e}\n\nPlease install folium: pip install folium")
            return
        
        # Create a map centered on a default location (middle of world)
        m = folium.Map(location=[20, 0], zoom_start=2)
        
        # Add drawing tools
        draw = Draw(
            export=True,
            filename='bounds_draw_data.geojson',
            position='topleft',
            draw_options={
                'polyline': False,
                'polygon': False,
                'rectangle': True,  # Enable rectangle drawing
                'circle': False,
                'marker': False,
                'circlemarker': False
            },
            edit_options={'edit': True, 'remove': True}
        )
        draw.add_to(m)
        
        # Add instructions and coordinate display to the map
        instructions_html = """
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 300px; height: 180px; 
                    background-color: white; z-index:9999; 
                    border: 2px solid grey; padding: 10px; border-radius: 5px;">
            <h4 style="margin-top:0;">Draw Geographic Box Instructions</h4>
            <ol style="margin: 0; padding-left: 20px; font-size: 12px;">
                <li>Click the rectangle tool in the toolbar (top-left)</li>
                <li>Click and drag on the map to draw a rectangle</li>
                <li>Coordinates will appear in the bottom-left box</li>
                <li>Click "Copy Coordinates" to copy them</li>
                <li>Return to the application and paste/enter them</li>
            </ol>
        </div>
        <div id="coords-display" style="position: fixed; bottom: 10px; left: 10px; width: 400px; 
            background-color: white; z-index:9999; border: 2px solid #007bff; 
            padding: 15px; border-radius: 5px; font-family: Arial, sans-serif; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);">
            <h4 style="margin-top:0;">Geographic Box Coordinates</h4>
            <p id="coords-text" style="font-size: 12px; color: #666;">Draw a rectangle on the map...</p>
            <button id="copy-coords" style="margin-top: 10px; padding: 5px 10px; 
                background-color: #007bff; color: white; border: none; border-radius: 3px; 
                cursor: pointer;">Copy Coordinates</button>
        </div>
        """
        m.get_root().html.add_child(folium.Element(instructions_html))
        
        # Add JavaScript to extract coordinates from drawn rectangles
        extract_coords_js = folium.Element("""
        <script>
        // Wait for map and draw plugin to be initialized
        setTimeout(function() {
            // Find the map object (folium stores it in the window)
            var mapObj = null;
            for (var key in window) {
                if (window[key] && window[key].hasOwnProperty && window[key].hasOwnProperty('_container')) {
                    mapObj = window[key];
                    break;
                }
            }
            
            // Alternative: try to get from Leaflet's global map registry
            if (!mapObj && typeof L !== 'undefined') {
                L.eachLayer = L.eachLayer || function(callback) {
                    for (var id in this._layers) {
                        callback(this._layers[id]);
                    }
                };
                // Get the first map instance
                for (var id in L._layers) {
                    var layer = L._layers[id];
                    if (layer instanceof L.Map) {
                        mapObj = layer;
                        break;
                    }
                }
            }
            
            if (mapObj) {
                // Listen for draw events
                mapObj.on('draw:created', function(e) {
                    var layer = e.layer;
                    var bounds = layer.getBounds();
                    var sw = bounds.getSouthWest();
                    var ne = bounds.getNorthEast();
                    
                    var coords = {
                        lat_min: sw.lat.toFixed(6),
                        lat_max: ne.lat.toFixed(6),
                        lon_min: sw.lng.toFixed(6),
                        lon_max: ne.lng.toFixed(6)
                    };
                    
                    var coordsText = 'Lat Min: ' + coords.lat_min + '<br>' +
                                   'Lat Max: ' + coords.lat_max + '<br>' +
                                   'Lon Min: ' + coords.lon_min + '<br>' +
                                   'Lon Max: ' + coords.lon_max;
                    
                    var coordsTextEl = document.getElementById('coords-text');
                    if (coordsTextEl) {
                        coordsTextEl.innerHTML = coordsText;
                    }
                    
                    // Store coordinates in a global variable for copying
                    window.boxCoords = coords;
                });
            }
            
            // Copy button functionality
            var copyBtn = document.getElementById('copy-coords');
            if (copyBtn) {
                copyBtn.addEventListener('click', function() {
                    if (window.boxCoords) {
                        var text = window.boxCoords.lat_min + ',' + window.boxCoords.lat_max + ',' +
                                  window.boxCoords.lon_min + ',' + window.boxCoords.lon_max;
                        if (navigator.clipboard && navigator.clipboard.writeText) {
                            navigator.clipboard.writeText(text).then(function() {
                                alert('Coordinates copied to clipboard!');
                                // Close the browser window/tab after user clicks OK
                                window.close();
                            }).catch(function() {
                                // Fallback for browsers without clipboard API
                                var textarea = document.createElement('textarea');
                                textarea.value = text;
                                document.body.appendChild(textarea);
                                textarea.select();
                                document.execCommand('copy');
                                document.body.removeChild(textarea);
                                alert('Coordinates copied to clipboard!');
                                // Close the browser window/tab after user clicks OK
                                window.close();
                            });
                        } else {
                            // Fallback for older browsers
                            var textarea = document.createElement('textarea');
                            textarea.value = text;
                            document.body.appendChild(textarea);
                            textarea.select();
                            document.execCommand('copy');
                            document.body.removeChild(textarea);
                            alert('Coordinates copied to clipboard!');
                            // Close the browser window/tab after user clicks OK
                            window.close();
                        }
                    }
                });
            }
        }, 2000);
        </script>
        """)
        m.get_root().html.add_child(extract_coords_js)
        
        # Save map to temporary file
        temp_dir = tempfile.gettempdir()
        temp_map_file = os.path.join(temp_dir, 'bounds_draw_map.html')
        m.save(temp_map_file)
        
        # Open map in browser
        webbrowser.open(f'file://{temp_map_file}')
        
        # Show dialog to get coordinates
        coord_dialog = tk.Toplevel(parent_window)
        coord_dialog.title("Enter Geographic Box Coordinates from Map")
        coord_dialog.geometry("500x300")
        coord_dialog.transient(parent_window)
        coord_dialog.grab_set()
        
        # Instructions
        instructions = tk.Text(coord_dialog, height=6, wrap=tk.WORD, font=("Arial", 9))
        instructions.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        instructions.insert('1.0', 
            "Instructions:\n\n"
            "1. A map has been opened in your browser\n"
            "2. Use the rectangle tool in the map toolbar (top-left) to draw a geographic box\n"
            "3. After drawing, coordinates will appear in the bottom-left box on the map\n"
            "4. Click 'Copy Coordinates' button on the map to copy them\n"
            "5. Paste the coordinates below (comma-separated: lat_min,lat_max,lon_min,lon_max)\n"
            "   Or manually enter the coordinates in the fields below"
        )
        instructions.config(state=tk.DISABLED)
        
        # Paste coordinates frame
        paste_frame = ttk.LabelFrame(coord_dialog, text="Paste Coordinates (comma-separated)")
        paste_frame.pack(fill=tk.X, padx=10, pady=5)
        
        paste_var = tk.StringVar()
        paste_entry = ttk.Entry(paste_frame, textvariable=paste_var, width=50)
        paste_entry.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Enable right-click context menu for paste entry
        def create_context_menu(event):
            """Create right-click context menu for the entry field"""
            context_menu = tk.Menu(coord_dialog, tearoff=0)
            context_menu.add_command(label="Cut", command=lambda: paste_entry.event_generate("<<Cut>>"))
            context_menu.add_command(label="Copy", command=lambda: paste_entry.event_generate("<<Copy>>"))
            context_menu.add_command(label="Paste", command=lambda: paste_entry.event_generate("<<Paste>>"))
            context_menu.add_separator()
            context_menu.add_command(label="Select All", command=lambda: paste_entry.select_range(0, tk.END))
            try:
                context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                context_menu.grab_release()
        
        paste_entry.bind("<Button-3>", create_context_menu)  # Right-click on Windows/Linux
        paste_entry.bind("<Button-2>", create_context_menu)  # Right-click on macOS
        paste_entry.bind("<Control-Button-1>", create_context_menu)  # Control+Click on macOS
        
        def parse_pasted_coords():
            """Parse pasted coordinates and fill in the fields"""
            paste_text = paste_var.get().strip()
            if not paste_text:
                return
            
            try:
                coords = [float(x.strip()) for x in paste_text.split(',')]
                if len(coords) == 4:
                    min_lat_var.set(coords[0])
                    max_lat_var.set(coords[1])
                    min_lon_var.set(coords[2])
                    max_lon_var.set(coords[3])
                    messagebox.showinfo("Success", "Coordinates parsed and filled in!")
                    coord_dialog.destroy()
                else:
                    messagebox.showerror("Error", "Please enter 4 comma-separated values: lat_min,lat_max,lon_min,lon_max")
            except ValueError:
                messagebox.showerror("Error", "Invalid format. Please enter: lat_min,lat_max,lon_min,lon_max")
        
        ttk.Button(paste_frame, text="Parse & Fill", command=parse_pasted_coords).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(coord_dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(button_frame, text="Close", command=coord_dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def _extended_time_analysis(self):
        mmsi_str = self.extended_mmsi_var.get()
        
        # Get dates from DateEntry if available, otherwise from text entry
        if TKCALENDAR_AVAILABLE and self.extended_start_picker is not None:
            start_date = self.extended_start_picker.get_date().strftime('%Y-%m-%d')
            end_date = self.extended_end_picker.get_date().strftime('%Y-%m-%d')
        else:
            start_date = self.extended_start_var.get() if self.extended_start_var else None
            end_date = self.extended_end_var.get() if self.extended_end_var else None
        
        if not mmsi_str:
            messagebox.showerror("Error", "Please enter an MMSI number")
            return
        
        if not start_date or not end_date:
            messagebox.showerror("Error", "Please select both start and end dates")
            return
        
        try:
            mmsi = int(mmsi_str)
        except ValueError:
            messagebox.showerror("Error", "MMSI must be a number")
            return
        
        progress = ProgressDialog(self.window, "Extended Analysis", f"Analyzing vessel {mmsi}...")
        try:
            self.status_var.set(f"Performing extended analysis for vessel {mmsi}...")
            result = self.analysis.extended_time_analysis(mmsi, start_date, end_date)
            progress.close()
            if result:
                self.status_var.set(f"Analysis complete: {result}")
                messagebox.showinfo("Success", f"Extended analysis complete:\n{result}")
            else:
                self.status_var.set("Analysis failed")
                messagebox.showerror("Error", "Failed to perform extended analysis. Check logs for details.")
        except Exception as e:
            progress.close()
            self.status_var.set("Analysis failed")
            logger.error(f"Error in extended analysis: {e}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

    def _ml_course_prediction(self):
        """Perform ML-based course prediction for a vessel"""
        mmsi_str = self.extended_mmsi_var.get()
        
        if not mmsi_str:
            messagebox.showerror("Error", "Please enter an MMSI number in the field above")
            return
        
        try:
            mmsi = int(mmsi_str)
        except ValueError:
            messagebox.showerror("Error", "MMSI must be a number")
            return
        
        progress = ProgressDialog(self.window, "ML Course Prediction", 
                                 f"Predicting course for vessel {mmsi}...")
        try:
            self.status_var.set(f"Loading full daily datasets for vessel {mmsi}...")
            
            # Load full daily datasets from cache (not filtered/consolidated data)
            df = self.analysis.load_full_daily_datasets()
            if df.empty:
                progress.close()
                messagebox.showerror("Error", 
                    "No daily datasets found in cache directory.\n\n"
                    "Please ensure that:\n"
                    "1. AIS data has been downloaded and cached\n"
                    "2. Cache directory contains parquet files for the date range\n"
                    "3. Files are in the correct date subfolder (YYYYMMDD-YYYYMMDD)")
                return
            
            self.status_var.set(f"Initializing ML prediction...")
            
            # Create integrator
            integrator = MLPredictionIntegrator()
            
            self.status_var.set(f"Generating predictions for vessel {mmsi}...")
            
            # Run prediction pipeline
            result = integrator.predict_vessel_course(df, mmsi, hours_back=24)
            
            progress.close()
            
            # Display results
            self._display_prediction_results(result)
            
            self.status_var.set(f"Prediction complete for vessel {mmsi}")
            
        except MLPredictionError as e:
            progress.close()
            self.status_var.set("Prediction failed")
            logger.error(f"ML Prediction Error: {e}")
            messagebox.showerror("Prediction Error", str(e))
        except Exception as e:
            progress.close()
            self.status_var.set("Prediction failed")
            logger.error(f"Error in ML course prediction: {e}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Prediction failed: {str(e)}\n\nCheck logs for details.")
    
    def _display_prediction_results(self, result: dict):
        """Display prediction results on a map"""
        mmsi = result['mmsi']
        predictions = result['predictions']
        trajectory = result['trajectory']
        last_lat, last_lon = result['last_position']
        last_time = result['last_time']
        
        position_mean = predictions['position_mean']
        position_lower = predictions['position_lower']
        position_upper = predictions['position_upper']
        # position_std is optional but used for uncertainty display
        position_std = predictions.get('position_std', None)
        
        # Log initial shapes for debugging
        logger.info(f"Initial position_mean type: {type(position_mean)}, shape: {getattr(position_mean, 'shape', 'N/A')}")
        logger.info(f"Expected shape: (8, 2) for 48 hours at 6-hour intervals")
        
        # Ensure position arrays are 2D (time_steps, 2)
        # Convert to numpy if needed and handle different shapes
        if not isinstance(position_mean, np.ndarray):
            position_mean = np.array(position_mean)
        if not isinstance(position_lower, np.ndarray):
            position_lower = np.array(position_lower)
        if not isinstance(position_upper, np.ndarray):
            position_upper = np.array(position_upper)
        
        # Always generate list of predicted coordinates (for logging and display)
        predicted_coords = []
        try:
            if position_mean.ndim == 2 and position_mean.shape[1] == 2:
                for i in range(position_mean.shape[0]):
                    lat = float(position_mean[i, 0])
                    lon = float(position_mean[i, 1])
                    # Check for valid coordinates
                    if not (np.isnan(lat) or np.isnan(lon) or np.isinf(lat) or np.isinf(lon)):
                        hours_ahead = (i + 1) * 6  # Assuming 6-hour intervals
                        predicted_coords.append({
                            'hours_ahead': hours_ahead,
                            'lat': lat,
                            'lon': lon
                        })
            logger.info(f"Generated {len(predicted_coords)} predicted coordinates for vessel {mmsi}")
        except Exception as e:
            logger.error(f"Error generating predicted coordinates list: {e}")
            predicted_coords = []
        
        # Handle shape issues - be very explicit
        logger.debug(f"After conversion - position_mean shape: {position_mean.shape}, ndim: {position_mean.ndim}")
        
        if position_mean.ndim == 0:
            # Scalar - this shouldn't happen but handle it
            raise ValueError(f"position_mean is a scalar (value: {position_mean}), expected 2D array")
        elif position_mean.ndim == 1:
            # Single prediction point (2,) -> reshape to (1, 2)
            if position_mean.shape[0] == 2:
                position_mean = position_mean.reshape(1, 2)
                if position_lower.ndim == 1 and position_lower.shape[0] == 2:
                    position_lower = position_lower.reshape(1, 2)
                if position_upper.ndim == 1 and position_upper.shape[0] == 2:
                    position_upper = position_upper.reshape(1, 2)
            else:
                raise ValueError(f"Unexpected 1D position_mean shape: {position_mean.shape}, expected (2,)")
        elif position_mean.ndim == 2:
            # Already 2D - verify second dimension is 2
            if position_mean.shape[1] != 2:
                raise ValueError(f"position_mean has 2D shape but second dim is {position_mean.shape[1]}, expected 2")
        else:
            # 3D or higher - try to squeeze or reshape
            logger.warning(f"position_mean has {position_mean.ndim} dimensions, shape: {position_mean.shape}, attempting to fix")
            if position_mean.ndim == 3:
                # (batch, time_steps, 2) -> (time_steps, 2)
                if position_mean.shape[0] == 1:
                    position_mean = position_mean[0]
                    position_lower = position_lower[0] if position_lower.ndim == 3 else position_lower
                    position_upper = position_upper[0] if position_upper.ndim == 3 else position_upper
                else:
                    raise ValueError(f"position_mean has 3D shape with batch size > 1: {position_mean.shape}")
            else:
                raise ValueError(f"position_mean has unexpected number of dimensions: {position_mean.ndim}, shape: {position_mean.shape}")
        
        # Final verification - must be 2D with shape (time_steps, 2)
        if position_mean.ndim != 2:
            raise ValueError(f"After processing, position_mean still has {position_mean.ndim} dimensions, shape: {position_mean.shape}")
        if position_mean.shape[1] != 2:
            raise ValueError(f"After processing, position_mean second dimension is {position_mean.shape[1]}, expected 2. Shape: {position_mean.shape}")
        
        logger.debug(f"Final position_mean shape: {position_mean.shape}")
        
        if not FOLIUM_AVAILABLE:
            # Fallback to text display - use the predicted_coords list we already generated
            result_text = f"Prediction Results for Vessel {mmsi}\n\n"
            result_text += f"Last Known Position: ({last_lat:.4f}, {last_lon:.4f})\n"
            if last_time:
                result_text += f"Last Known Time: {last_time}\n"
            result_text += f"\nPredicted Positions (48 hours ahead):\n"
            if predicted_coords:
                for coord in predicted_coords:
                    result_text += f"  +{coord['hours_ahead']}h: ({coord['lat']:.4f}, {coord['lon']:.4f})\n"
            else:
                result_text += "  No valid predicted coordinates generated\n"
            messagebox.showinfo("Prediction Results", result_text)
            return
        
        try:
            # Collect all points for bounds calculation
            all_lats = [last_lat]
            all_lons = [last_lon]
            
            # Add historical trajectory points
            traj_points = []
            for lat, lon in zip(trajectory['LAT'].values, trajectory['LON'].values):
                if not (np.isnan(lat) or np.isnan(lon) or np.isinf(lat) or np.isinf(lon)):
                    traj_points.append([lat, lon])
                    all_lats.append(lat)
                    all_lons.append(lon)
            
            # Prepare predicted points (filter invalid coordinates)
            pred_points = []
            if position_mean.ndim == 2 and position_mean.shape[1] == 2:
                for i in range(position_mean.shape[0]):
                    lat = float(position_mean[i, 0])
                    lon = float(position_mean[i, 1])
                    # Only add valid coordinates
                    if not (np.isnan(lat) or np.isnan(lon) or np.isinf(lat) or np.isinf(lon)):
                        pred_points.append([lat, lon])
                        all_lats.append(lat)
                        all_lons.append(lon)
            
            logger.info(f"Historical trajectory points: {len(traj_points)}, Predicted points: {len(pred_points)}")
            
            # Calculate map center and bounds to include both historical and predicted paths
            if all_lats and all_lons:
                center_lat = np.mean(all_lats)
                center_lon = np.mean(all_lons)
                # Add padding to bounds
                lat_range = max(all_lats) - min(all_lats)
                lon_range = max(all_lons) - min(all_lons)
                padding = max(lat_range, lon_range) * 0.2  # 20% padding
                bounds = [
                    [min(all_lats) - padding, min(all_lons) - padding],
                    [max(all_lats) + padding, max(all_lons) + padding]
                ]
            else:
                center_lat = last_lat
                center_lon = last_lon
                bounds = None
            
            # Create map centered on combined data
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
            
            # Fit bounds if we have valid bounds
            if bounds:
                m.fit_bounds(bounds)
            
            # Add historical trajectory
            if traj_points:
                folium.PolyLine(traj_points, color='blue', weight=3, opacity=0.7,
                              popup=f"Historical Path - Vessel {mmsi}").add_to(m)
            
            # Get first position from trajectory
            first_lat = None
            first_lon = None
            first_time = None
            if not trajectory.empty:
                first_lat = float(trajectory['LAT'].iloc[0])
                first_lon = float(trajectory['LON'].iloc[0])
                if 'BaseDateTime' in trajectory.columns:
                    first_time = trajectory['BaseDateTime'].iloc[0]
            
            # Add first position marker (green)
            if first_lat is not None and first_lon is not None:
                first_popup_text = f"<b>First Position</b><br>"
                first_popup_text += f"MMSI: {mmsi}<br>"
                if 'VesselName' in trajectory.columns and pd.notna(trajectory['VesselName'].iloc[0]):
                    first_popup_text += f"Vessel: {trajectory['VesselName'].iloc[0]}<br>"
                if 'COG' in trajectory.columns and pd.notna(trajectory['COG'].iloc[0]):
                    first_popup_text += f"COG: {float(trajectory['COG'].iloc[0]):.1f}°<br>"
                if 'Heading' in trajectory.columns and pd.notna(trajectory['Heading'].iloc[0]):
                    first_popup_text += f"Heading: {float(trajectory['Heading'].iloc[0]):.1f}°<br>"
                if 'SOG' in trajectory.columns and pd.notna(trajectory['SOG'].iloc[0]):
                    first_popup_text += f"Speed (SOG): {float(trajectory['SOG'].iloc[0]):.2f} knots<br>"
                if first_time:
                    first_popup_text += f"Date/Time: {first_time}"
                
                folium.Marker(
                    [first_lat, first_lon],
                    popup=folium.Popup(first_popup_text, max_width=300),
                    tooltip=folium.Tooltip(first_popup_text, sticky=False),
                    icon=folium.Icon(color='green', icon='info-sign')
                ).add_to(m)
            
            # Add blue markers for each known position between start and finish
            if not trajectory.empty and len(trajectory) > 2:
                # Iterate through all trajectory points (excluding first and last)
                for idx in range(1, len(trajectory) - 1):
                    row = trajectory.iloc[idx]
                    lat = float(row['LAT'])
                    lon = float(row['LON'])
                    
                    # Skip invalid coordinates
                    if np.isnan(lat) or np.isnan(lon) or np.isinf(lat) or np.isinf(lon):
                        continue
                    
                    # Build tooltip/popup text
                    tooltip_text = f"<b>Position Report</b><br>"
                    tooltip_text += f"MMSI: {mmsi}<br>"
                    if 'VesselName' in row and pd.notna(row.get('VesselName')):
                        tooltip_text += f"Vessel: {row['VesselName']}<br>"
                    if 'COG' in row and pd.notna(row.get('COG')):
                        tooltip_text += f"COG: {float(row['COG']):.1f}°<br>"
                    if 'Heading' in row and pd.notna(row.get('Heading')):
                        tooltip_text += f"Heading: {float(row['Heading']):.1f}°<br>"
                    if 'SOG' in row and pd.notna(row.get('SOG')):
                        tooltip_text += f"Speed (SOG): {float(row['SOG']):.2f} knots<br>"
                    if 'BaseDateTime' in row and pd.notna(row.get('BaseDateTime')):
                        tooltip_text += f"Date/Time: {row['BaseDateTime']}"
                    
                    folium.CircleMarker(
                        [lat, lon],
                        radius=4,
                        popup=folium.Popup(tooltip_text, max_width=300),
                        tooltip=folium.Tooltip(tooltip_text, sticky=False),
                        color='blue',
                        fill=True,
                        fillColor='blue',
                        fillOpacity=0.7
                    ).add_to(m)
            
            # Add last known position marker (red)
            last_popup_text = f"<b>Last Known Position</b><br>"
            last_popup_text += f"MMSI: {mmsi}<br>"
            if not trajectory.empty:
                last_row = trajectory.iloc[-1]
                if 'VesselName' in last_row and pd.notna(last_row.get('VesselName')):
                    last_popup_text += f"Vessel: {last_row['VesselName']}<br>"
                if 'COG' in last_row and pd.notna(last_row.get('COG')):
                    last_popup_text += f"COG: {float(last_row['COG']):.1f}°<br>"
                if 'Heading' in last_row and pd.notna(last_row.get('Heading')):
                    last_popup_text += f"Heading: {float(last_row['Heading']):.1f}°<br>"
                if 'SOG' in last_row and pd.notna(last_row.get('SOG')):
                    last_popup_text += f"Speed (SOG): {float(last_row['SOG']):.2f} knots<br>"
            if last_time:
                last_popup_text += f"Date/Time: {last_time}"
            
            folium.Marker(
                [last_lat, last_lon],
                popup=folium.Popup(last_popup_text, max_width=300),
                tooltip=folium.Tooltip(last_popup_text, sticky=False),
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            
            # Add predicted path with uncertainty
            # Use explicit indexing to avoid unpacking errors
            try:
                if position_mean.ndim == 2 and position_mean.shape[1] == 2:
                    # Add predicted path (already filtered for valid coordinates)
                    # Connect predicted path to last known position for continuity
                    if pred_points:
                        # Start from last known position
                        connected_pred_points = [[last_lat, last_lon]] + pred_points
                        folium.PolyLine(connected_pred_points, color='green', weight=3, opacity=0.8,
                                      popup="Predicted Path (48 hours)",
                                      tooltip="Predicted Path (48 hours)").add_to(m)
                        logger.info(f"Added predicted path with {len(connected_pred_points)} points (including start)")
                    else:
                        logger.warning("No valid predicted points to display")
                    
                    # Calculate perpendicular confidence intervals (cone shape)
                    # Uncertainty should be perpendicular to the predicted course (heading), forming an expanding cone
                    # Uses predicted course and speed from the model if available
                    def calculate_bearing(lat1, lon1, lat2, lon2):
                        """Calculate bearing (azimuth) from point 1 to point 2 in degrees"""
                        lat1_rad = math.radians(lat1)
                        lat2_rad = math.radians(lat2)
                        dlon_rad = math.radians(lon2 - lon1)
                        
                        y = math.sin(dlon_rad) * math.cos(lat2_rad)
                        x = math.cos(lat1_rad) * math.sin(lat2_rad) - \
                            math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
                        
                        bearing = math.degrees(math.atan2(y, x))
                        return (bearing + 360) % 360  # Normalize to 0-360
                    
                    def offset_point_by_bearing(lat, lon, bearing_deg, distance_nm):
                        """Offset a point in a given bearing direction by a distance in nautical miles"""
                        # Convert distance to degrees
                        # 1 degree latitude ≈ 60 nautical miles
                        # 1 degree longitude ≈ 60 * cos(latitude) nautical miles
                        lat_offset = (distance_nm / 60.0) * math.cos(math.radians(bearing_deg))
                        lon_offset = (distance_nm / (60.0 * math.cos(math.radians(lat)))) * math.sin(math.radians(bearing_deg))
                        return lat + lat_offset, lon + lon_offset
                    
                    def haversine_distance_meters(lat1, lon1, lat2, lon2):
                        """Calculate the great-circle distance between two points in meters using Haversine formula"""
                        # Earth radius in meters
                        R = 6371000.0  # meters
                        
                        # Convert to radians
                        lat1_rad = math.radians(lat1)
                        lat2_rad = math.radians(lat2)
                        dlat_rad = math.radians(lat2 - lat1)
                        dlon_rad = math.radians(lon2 - lon1)
                        
                        # Haversine formula
                        a = math.sin(dlat_rad / 2)**2 + \
                            math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon_rad / 2)**2
                        c = 2 * math.asin(math.sqrt(a))
                        
                        distance_meters = R * c
                        return distance_meters
                    
                    # Get predicted course and speed arrays if available
                    pred_courses = None
                    pred_speeds = None
                    if 'course' in predictions and isinstance(predictions['course'], np.ndarray):
                        pred_courses = predictions['course']
                    if 'speed' in predictions and isinstance(predictions['speed'], np.ndarray):
                        pred_speeds = predictions['speed']
                    
                    # Calculate perpendicular confidence intervals
                    if position_mean.ndim == 2 and position_mean.shape[1] == 2 and position_std is not None:
                        lower_points = []
                        upper_points = []
                        
                        # Start from last known position
                        lower_points.append([last_lat, last_lon])
                        upper_points.append([last_lat, last_lon])
                        
                        # Build predicted path points
                        path_points = [[last_lat, last_lon]]
                        for i in range(position_mean.shape[0]):
                            lat = float(position_mean[i, 0])
                            lon = float(position_mean[i, 1])
                            if not (np.isnan(lat) or np.isnan(lon) or np.isinf(lat) or np.isinf(lon)):
                                path_points.append([lat, lon])
                        
                        # Get last known COG (Course Over Ground) for first prediction if needed
                        # Always prioritize COG over Heading - COG represents actual movement direction
                        last_known_cog = None
                        if not trajectory.empty:
                            last_row = trajectory.iloc[-1]
                            if 'COG' in last_row and pd.notna(last_row.get('COG')):
                                last_known_cog = float(last_row['COG'])
                            # Only use Heading as fallback if COG is not available
                            elif 'Heading' in last_row and pd.notna(last_row.get('Heading')):
                                last_known_cog = float(last_row['Heading'])
                        
                        # Calculate confidence intervals using angular offset from predicted course (COG)
                        for i in range(len(path_points) - 1):
                            # Current point on predicted path
                            curr_lat, curr_lon = path_points[i + 1]
                            
                            # Determine predicted course (COG - Course Over Ground) for this point
                            # The model predicts COG, not Heading, as COG represents actual movement direction
                            predicted_course = None
                            
                            # Priority 1: Use predicted COG from model if available
                            # Note: The model predicts COG (Course Over Ground), not Heading
                            if pred_courses is not None:
                                if pred_courses.ndim == 1 and i < len(pred_courses):
                                    predicted_course = float(pred_courses[i])
                                elif pred_courses.ndim == 0:
                                    predicted_course = float(pred_courses)
                            
                            # Priority 2: Calculate COG from position (bearing from previous point)
                            # Bearing between points represents the actual COG (course over ground)
                            if predicted_course is None:
                                if i == 0:
                                    # From last known position to first predicted point
                                    # Use last known COG if available, otherwise calculate bearing (which is COG)
                                    if last_known_cog is not None:
                                        predicted_course = last_known_cog
                                    else:
                                        predicted_course = calculate_bearing(path_points[i][0], path_points[i][1], 
                                                                           curr_lat, curr_lon)
                                else:
                                    # From previous predicted point to current predicted point
                                    # Bearing represents the COG (actual movement direction)
                                    predicted_course = calculate_bearing(path_points[i][0], path_points[i][1], 
                                                                       curr_lat, curr_lon)
                            
                            # Priority 3: Fallback to last known COG (not Heading)
                            if predicted_course is None and last_known_cog is not None:
                                predicted_course = last_known_cog
                            
                            # Get predicted speed for distance calculation
                            predicted_speed = None
                            if pred_speeds is not None:
                                if pred_speeds.ndim == 1 and i < len(pred_speeds):
                                    predicted_speed = float(pred_speeds[i])
                                elif pred_speeds.ndim == 0:
                                    predicted_speed = float(pred_speeds)
                            
                            # Fallback to last known speed if predicted speed not available
                            if predicted_speed is None and not trajectory.empty:
                                last_row = trajectory.iloc[-1]
                                if 'SOG' in last_row and pd.notna(last_row.get('SOG')):
                                    predicted_speed = float(last_row['SOG'])
                            
                            # Default speed if still not available
                            if predicted_speed is None:
                                predicted_speed = 10.0  # Default 10 knots
                            
                            # Get uncertainty for this point
                            # Uncertainty is an angular offset from the predicted course: C ± (U × 0.5)
                            # Point number is i+1 (since i is 0-indexed: first point is i=0, point number 1)
                            point_number = i + 1
                            
                            if position_std.ndim == 2 and i < position_std.shape[0]:
                                # Get uncertainty in degrees (mean of lat/lon std)
                                std_lat = float(position_std[i, 0])
                                std_lon = float(position_std[i, 1])
                                uncertainty_degrees = (std_lat + std_lon) / 2.0
                                
                                # Angular offset from predicted course: U × 0.5
                                base_angular_offset = uncertainty_degrees * 0.5
                                
                                # Apply same multiplier as uncertainty circles: (point_number) × (1/8)
                                uncertainty_multiplier = point_number * (1.0 / 8.0)
                                angular_offset = base_angular_offset * uncertainty_multiplier
                            else:
                                # Fallback: use a default angular uncertainty with multiplier
                                base_angular_offset = 2.0  # 2° base
                                uncertainty_multiplier = point_number * (1.0 / 8.0)
                                angular_offset = base_angular_offset * uncertainty_multiplier
                            
                            # Calculate distance to project (based on speed and time interval)
                            # Time interval is 6 hours per prediction step
                            time_hours = 6.0
                            projection_distance_nm = predicted_speed * time_hours
                            
                            # Validate projection distance is reasonable (max 500 nm for 6 hours at reasonable speed)
                            max_projection_nm = 500.0
                            if projection_distance_nm > max_projection_nm:
                                logger.warning(f"Projection distance too large ({projection_distance_nm:.1f} nm) for point {i}, capping at {max_projection_nm} nm")
                                projection_distance_nm = max_projection_nm
                            
                            # Validate angular offset is reasonable (max 45 degrees)
                            max_angular_offset = 45.0
                            if angular_offset > max_angular_offset:
                                logger.warning(f"Angular offset too large ({angular_offset:.1f}°) for point {i}, capping at {max_angular_offset}°")
                                angular_offset = max_angular_offset
                            
                            # Calculate left and right bound courses (angular offset from predicted course)
                            # Left bound course: C - (U × 0.5)
                            left_bound_course = (predicted_course - angular_offset) % 360
                            # Project from previous bound point (or last known position for first point)
                            if i == 0:
                                # For first point, project from last known position
                                prev_left_lat, prev_left_lon = last_lat, last_lon
                            else:
                                # For subsequent points, project from previous bound point
                                prev_left_lat, prev_left_lon = lower_points[-1]
                            
                            # Log confidence interval bounds calculation
                            logger.info(f"[Predicted Point {i}] Confidence Interval Bounds Calculation:")
                            logger.info(f"  Input Variables:")
                            logger.info(f"    predicted_course (COG) = {predicted_course:.3f}°")
                            if position_std.ndim == 2 and i < position_std.shape[0]:
                                logger.info(f"    uncertainty_degrees = {uncertainty_degrees:.6f}°")
                                logger.info(f"    base_angular_offset = uncertainty_degrees × 0.5 = {base_angular_offset:.6f}°")
                            else:
                                logger.info(f"    base_angular_offset = 2.0° (fallback)")
                            logger.info(f"    point_number = {point_number}")
                            logger.info(f"    uncertainty_multiplier ({point_number} × 1/8) = {uncertainty_multiplier:.6f}")
                            logger.info(f"    angular_offset = base_angular_offset × uncertainty_multiplier = {angular_offset:.6f}°")
                            logger.info(f"    predicted_speed = {predicted_speed:.2f} knots")
                            logger.info(f"    time_hours = {time_hours:.1f} hours")
                            logger.info(f"  Calculated Variables:")
                            logger.info(f"    projection_distance_nm = predicted_speed × time_hours = {projection_distance_nm:.2f} nm")
                            
                            left_lat, left_lon = offset_point_by_bearing(prev_left_lat, prev_left_lon, left_bound_course, projection_distance_nm)
                            
                            logger.info(f"  Left Bound (Port):")
                            logger.info(f"    left_bound_course = (predicted_course - angular_offset) % 360 = ({predicted_course:.3f} - {angular_offset:.6f}) % 360 = {left_bound_course:.3f}°")
                            logger.info(f"    prev_left_position = ({prev_left_lat:.6f}°, {prev_left_lon:.6f}°)")
                            
                            # Validate coordinates are reasonable (within valid lat/lon range)
                            if -90 <= left_lat <= 90 and -180 <= left_lon <= 180:
                                lower_points.append([left_lat, left_lon])
                                logger.info(f"    left_bound_position = ({left_lat:.6f}°, {left_lon:.6f}°) [VALID]")
                            else:
                                logger.warning(f"    Invalid left bound coordinates for point {i}: lat={left_lat}, lon={left_lon}")
                                # Use predicted point as fallback
                                lower_points.append([curr_lat, curr_lon])
                                logger.info(f"    left_bound_position = ({curr_lat:.6f}°, {curr_lon:.6f}°) [FALLBACK]")
                            
                            # Right bound course: C + (U × 0.5)
                            right_bound_course = (predicted_course + angular_offset) % 360
                            # Project from previous bound point (or last known position for first point)
                            if i == 0:
                                # For first point, project from last known position
                                prev_right_lat, prev_right_lon = last_lat, last_lon
                            else:
                                # For subsequent points, project from previous bound point
                                prev_right_lat, prev_right_lon = upper_points[-1]
                            
                            right_lat, right_lon = offset_point_by_bearing(prev_right_lat, prev_right_lon, right_bound_course, projection_distance_nm)
                            
                            logger.info(f"  Right Bound (Starboard):")
                            logger.info(f"    right_bound_course = (predicted_course + angular_offset) % 360 = ({predicted_course:.3f} + {angular_offset:.6f}) % 360 = {right_bound_course:.3f}°")
                            logger.info(f"    prev_right_position = ({prev_right_lat:.6f}°, {prev_right_lon:.6f}°)")
                            
                            # Validate coordinates are reasonable (within valid lat/lon range)
                            if -90 <= right_lat <= 90 and -180 <= right_lon <= 180:
                                upper_points.append([right_lat, right_lon])
                                logger.info(f"    right_bound_position = ({right_lat:.6f}°, {right_lon:.6f}°) [VALID]")
                            else:
                                logger.warning(f"    Invalid right bound coordinates for point {i}: lat={right_lat}, lon={right_lon}")
                                # Use predicted point as fallback
                                upper_points.append([curr_lat, curr_lon])
                                logger.info(f"    right_bound_position = ({curr_lat:.6f}°, {curr_lon:.6f}°) [FALLBACK]")
                        
                        # Add confidence interval lines (solid, same weight as predicted path)
                        if len(lower_points) > 1 and len(upper_points) > 1:
                            # Smooth the polygon edges using Chaikin's corner cutting algorithm
                            def smooth_polyline(points, iterations=2):
                                """
                                Smooth a polyline using Chaikin's corner cutting algorithm.
                                Each iteration adds more points and smooths the curve.
                                """
                                if len(points) < 2:
                                    return points
                                
                                smoothed = points.copy()
                                for _ in range(iterations):
                                    new_points = []
                                    for i in range(len(smoothed) - 1):
                                        # Get two consecutive points
                                        p1 = smoothed[i]
                                        p2 = smoothed[i + 1]
                                        
                                        # Calculate 1/4 and 3/4 points (Chaikin's algorithm)
                                        q1_lat = p1[0] * 0.75 + p2[0] * 0.25
                                        q1_lon = p1[1] * 0.75 + p2[1] * 0.25
                                        q2_lat = p1[0] * 0.25 + p2[0] * 0.75
                                        q2_lon = p1[1] * 0.25 + p2[1] * 0.75
                                        
                                        new_points.append([q1_lat, q1_lon])
                                        new_points.append([q2_lat, q2_lon])
                                    
                                    # Keep the last point
                                    new_points.append(smoothed[-1])
                                    smoothed = new_points
                                
                                return smoothed
                            
                            # Apply smoothing to boundary points
                            smoothed_lower_points = smooth_polyline(lower_points, iterations=2)
                            smoothed_upper_points = smooth_polyline(upper_points, iterations=2)
                            
                            # Add smoothed boundary lines (solid, same weight as predicted path)
                            # Left bound line (smoothed)
                            folium.PolyLine(smoothed_lower_points, color='orange', weight=3, opacity=0.8,
                                          popup="Lower 68% Confidence (Port)",
                                          tooltip="Lower 68% Confidence Bound (Port Side)").add_to(m)
                            logger.info(f"Added smoothed lower confidence bound (perpendicular to predicted course) with {len(smoothed_lower_points)} points")
                            
                            # Right bound line (smoothed)
                            folium.PolyLine(smoothed_upper_points, color='orange', weight=3, opacity=0.8,
                                          popup="Upper 68% Confidence (Starboard)",
                                          tooltip="Upper 68% Confidence Bound (Starboard Side)").add_to(m)
                            logger.info(f"Added smoothed upper confidence bound (perpendicular to predicted course) with {len(smoothed_upper_points)} points")
                        elif len(lower_points) > 1:
                            # Only left bound available
                            folium.PolyLine(lower_points, color='orange', weight=3, opacity=0.8,
                                          popup="Lower 68% Confidence (Port)",
                                          tooltip="Lower 68% Confidence Bound (Port Side)").add_to(m)
                            logger.info(f"Added lower confidence bound (perpendicular to predicted course) with {len(lower_points)} points")
                        elif len(upper_points) > 1:
                            # Only right bound available
                            folium.PolyLine(upper_points, color='orange', weight=3, opacity=0.8,
                                          popup="Upper 68% Confidence (Starboard)",
                                          tooltip="Upper 68% Confidence Bound (Starboard Side)").add_to(m)
                            logger.info(f"Added upper confidence bound (perpendicular to predicted course) with {len(upper_points)} points")
                    else:
                        logger.warning(f"Cannot calculate perpendicular confidence intervals: position_mean shape={position_mean.shape if hasattr(position_mean, 'shape') else 'N/A'}, position_std={position_std is not None}")
                    
                    # Add predicted position markers (only for valid coordinates)
                    # Get vessel information from trajectory for popup
                    vessel_name = None
                    if not trajectory.empty:
                        if 'VesselName' in trajectory.columns:
                            vessel_name = trajectory['VesselName'].iloc[-1] if pd.notna(trajectory['VesselName'].iloc[-1]) else None
                    
                    # Get last known values for reference
                    last_cog = None
                    last_heading = None
                    last_sog = None
                    last_datetime = last_time
                    if not trajectory.empty:
                        last_row = trajectory.iloc[-1]
                        if 'COG' in last_row and pd.notna(last_row.get('COG')):
                            last_cog = float(last_row['COG'])
                        if 'Heading' in last_row and pd.notna(last_row.get('Heading')):
                            last_heading = float(last_row['Heading'])
                        if 'SOG' in last_row and pd.notna(last_row.get('SOG')):
                            last_sog = float(last_row['SOG'])
                        if 'BaseDateTime' in last_row and pd.notna(last_row.get('BaseDateTime')):
                            last_datetime = last_row['BaseDateTime']
                    
                    for i in range(position_mean.shape[0]):
                        lat = float(position_mean[i, 0])
                        lon = float(position_mean[i, 1])
                        # Only add markers for valid coordinates
                        if not (np.isnan(lat) or np.isnan(lon) or np.isinf(lat) or np.isinf(lon)):
                            hours_ahead = (i + 1) * 6  # Assuming 6-hour intervals
                            
                            # Calculate predicted datetime
                            predicted_datetime = None
                            if last_datetime:
                                try:
                                    if isinstance(last_datetime, pd.Timestamp):
                                        predicted_datetime = last_datetime + pd.Timedelta(hours=hours_ahead)
                                    elif isinstance(last_datetime, str):
                                        last_dt = pd.to_datetime(last_datetime)
                                        predicted_datetime = last_dt + pd.Timedelta(hours=hours_ahead)
                                    else:
                                        predicted_datetime = last_datetime + pd.Timedelta(hours=hours_ahead)
                                except Exception as e:
                                    logger.debug(f"Could not calculate predicted datetime: {e}")
                            
                            # Get uncertainty/std information
                            if position_std is not None and isinstance(position_std, np.ndarray):
                                if position_std.ndim == 2:
                                    std_val = position_std[i].mean()
                                else:
                                    std_val = position_std.mean() if position_std.size > 0 else 0.0
                                std_text = f"±{std_val:.3f}°"
                            else:
                                std_text = "N/A"
                            
                            # Get predicted speed and course if available
                            pred_speed = None
                            pred_course = None
                            if 'speed' in predictions and isinstance(predictions['speed'], np.ndarray):
                                speed_array = predictions['speed']
                                if speed_array.ndim == 1 and i < len(speed_array):
                                    pred_speed = float(speed_array[i])
                                elif speed_array.ndim == 0:
                                    pred_speed = float(speed_array)
                            if 'course' in predictions and isinstance(predictions['course'], np.ndarray):
                                course_array = predictions['course']
                                if course_array.ndim == 1 and i < len(course_array):
                                    pred_course = float(course_array[i])
                                elif course_array.ndim == 0:
                                    pred_course = float(course_array)
                            
                            # Build comprehensive popup text
                            popup_text = f"<b>Predicted Position</b><br>"
                            popup_text += f"MMSI: {mmsi}<br>"
                            if vessel_name:
                                popup_text += f"Vessel: {vessel_name}<br>"
                            popup_text += f"<b>Prediction:</b><br>"
                            popup_text += f"Hours Ahead: +{hours_ahead}h<br>"
                            if predicted_datetime:
                                popup_text += f"Predicted Date/Time: {predicted_datetime}<br>"
                            
                            # Always show predicted position (lat/lon) prominently
                            popup_text += f"<b>Predicted Position:</b><br>"
                            popup_text += f"Latitude: {lat:.6f}°<br>"
                            popup_text += f"Longitude: {lon:.6f}°<br>"
                            
                            # Always show predicted SOG and COG prominently
                            popup_text += f"<b>Predicted Motion:</b><br>"
                            if pred_speed is not None:
                                popup_text += f"Predicted SOG: {pred_speed:.2f} knots<br>"
                            else:
                                popup_text += f"Predicted SOG: N/A"
                                if last_sog is not None:
                                    popup_text += f" (Last Known: {last_sog:.2f} knots)<br>"
                                else:
                                    popup_text += f"<br>"
                            
                            if pred_course is not None:
                                popup_text += f"Predicted COG: {pred_course:.1f}°<br>"
                            else:
                                popup_text += f"Predicted COG: N/A"
                                if last_cog is not None:
                                    popup_text += f" (Last Known: {last_cog:.1f}°)<br>"
                                else:
                                    popup_text += f"<br>"
                            
                            # Additional information
                            if last_heading is not None:
                                popup_text += f"<br>Last Known Heading: {last_heading:.1f}°<br>"
                            popup_text += f"<br>Uncertainty: {std_text}"
                            
                            # Build tooltip text (shorter version for hover)
                            tooltip_text = f"<b>Predicted Position</b><br>"
                            tooltip_text += f"Vessel: {mmsi}<br>"
                            if vessel_name:
                                tooltip_text += f"{vessel_name}<br>"
                            tooltip_text += f"+{hours_ahead}h"
                            
                            folium.CircleMarker(
                                [lat, lon],
                                radius=5,
                                popup=folium.Popup(popup_text, max_width=300),
                                tooltip=folium.Tooltip(tooltip_text, sticky=False),
                                color='green',
                                fill=True,
                                fillColor='green',
                                fillOpacity=0.7
                            ).add_to(m)
                            
                            # Add orange circle around predicted point to represent uncertainty
                            # Calculate uncertainty radius as 1/2 the distance between upper and lower bounds
                            uncertainty_meters = None
                            uncertainty_degrees = None
                            
                            # Check if we have upper and lower bounds for this point
                            # Note: lower_points[0] and upper_points[0] are the last known position
                            # So for predicted point i, we need lower_points[i+1] and upper_points[i+1]
                            try:
                                if len(lower_points) > i + 1 and len(upper_points) > i + 1:
                                    lower_bound_point = lower_points[i + 1]
                                    upper_bound_point = upper_points[i + 1]
                                    
                                    lower_lat, lower_lon = lower_bound_point[0], lower_bound_point[1]
                                    upper_lat, upper_lon = upper_bound_point[0], upper_bound_point[1]
                                    
                                    # Calculate distance between upper and lower bounds
                                    distance_between_bounds_meters = haversine_distance_meters(
                                        lower_lat, lower_lon, upper_lat, upper_lon
                                    )
                                    
                                    # If distance is 0 or very small (e.g., due to zero projection distance),
                                    # calculate it based on angular separation at the predicted point
                                    if distance_between_bounds_meters < 1.0:  # Less than 1 meter
                                        # Get the angular offset that was used for this point
                                        # We need to recalculate it or get it from the bounds calculation
                                        # For now, use the uncertainty from position_std if available
                                        if position_std is not None and isinstance(position_std, np.ndarray):
                                            if position_std.ndim == 2 and i < position_std.shape[0]:
                                                std_lat = float(position_std[i, 0])
                                                std_lon = float(position_std[i, 1])
                                                uncertainty_degrees_for_calc = (std_lat + std_lon) / 2.0
                                                # Angular offset: U × 0.5
                                                base_angular_offset_for_calc = uncertainty_degrees_for_calc * 0.5
                                                # Apply same multiplier as uncertainty circles: (point_number) × (1/8)
                                                point_number_for_calc = i + 1
                                                uncertainty_multiplier_for_calc = point_number_for_calc * (1.0 / 8.0)
                                                angular_offset_for_calc = base_angular_offset_for_calc * uncertainty_multiplier_for_calc
                                                # Cap at 45 degrees
                                                angular_offset_for_calc = min(angular_offset_for_calc, 45.0)
                                            else:
                                                # Fallback: use a default angular offset with multiplier
                                                base_angular_offset_for_calc = 2.0  # 2° base
                                                point_number_for_calc = i + 1
                                                uncertainty_multiplier_for_calc = point_number_for_calc * (1.0 / 8.0)
                                                angular_offset_for_calc = base_angular_offset_for_calc * uncertainty_multiplier_for_calc
                                                angular_offset_for_calc = min(angular_offset_for_calc, 45.0)
                                        else:
                                            # Fallback: use a default angular offset with multiplier
                                            base_angular_offset_for_calc = 2.0  # 2° base
                                            point_number_for_calc = i + 1
                                            uncertainty_multiplier_for_calc = point_number_for_calc * (1.0 / 8.0)
                                            angular_offset_for_calc = base_angular_offset_for_calc * uncertainty_multiplier_for_calc
                                            angular_offset_for_calc = min(angular_offset_for_calc, 45.0)
                                        
                                        # Calculate the distance between bounds based on angular separation
                                        # Use the predicted point as center and project bounds at angular offset
                                        # Use a reasonable base distance (e.g., 1 nm) for the calculation
                                        base_distance_nm = 1.0  # 1 nautical mile base distance
                                        
                                        # Get predicted course for this point (use pred_course if available, or calculate from positions)
                                        predicted_course_for_calc = None
                                        if pred_course is not None:
                                            predicted_course_for_calc = pred_course
                                        else:
                                            # Calculate course from position_mean array
                                            if i == 0:
                                                # From last known position to first predicted point
                                                predicted_course_for_calc = calculate_bearing(last_lat, last_lon, lat, lon)
                                            elif i > 0:
                                                # From previous predicted point to current predicted point
                                                prev_lat = float(position_mean[i - 1, 0])
                                                prev_lon = float(position_mean[i - 1, 1])
                                                if not (np.isnan(prev_lat) or np.isnan(prev_lon)):
                                                    predicted_course_for_calc = calculate_bearing(prev_lat, prev_lon, lat, lon)
                                            # Fallback to last known COG if available
                                            if predicted_course_for_calc is None and last_cog is not None:
                                                predicted_course_for_calc = last_cog
                                        
                                        # Project lower and upper bounds from predicted point
                                        if predicted_course_for_calc is not None:
                                            left_bound_course_calc = (predicted_course_for_calc - angular_offset_for_calc) % 360
                                            right_bound_course_calc = (predicted_course_for_calc + angular_offset_for_calc) % 360
                                        else:
                                            # Fallback: use 0° and angular_offset
                                            left_bound_course_calc = (360.0 - angular_offset_for_calc) % 360
                                            right_bound_course_calc = angular_offset_for_calc % 360
                                        
                                        # Calculate bounds positions from predicted point
                                        calc_lower_lat, calc_lower_lon = offset_point_by_bearing(lat, lon, left_bound_course_calc, base_distance_nm)
                                        calc_upper_lat, calc_upper_lon = offset_point_by_bearing(lat, lon, right_bound_course_calc, base_distance_nm)
                                        
                                        # Calculate distance between these calculated bounds
                                        distance_between_bounds_meters = haversine_distance_meters(
                                            calc_lower_lat, calc_lower_lon, calc_upper_lat, calc_upper_lon
                                        )
                                        
                                        logger.info(f"  [Distance Calculation] Bounds were at same location, calculated from angular separation:")
                                        logger.info(f"    Angular offset: {angular_offset_for_calc:.3f}°")
                                        logger.info(f"    Base distance: {base_distance_nm:.2f} nm")
                                        logger.info(f"    Calculated distance: {distance_between_bounds_meters:.2f} m ({distance_between_bounds_meters/1000:.3f} km)")
                                    
                                    # Radius is 1/2 the distance between bounds
                                    base_uncertainty_meters = distance_between_bounds_meters / 2.0
                                    
                                    # Multiply uncertainty by (point number) × (1/8)
                                    # Point number is i+1 (since i is 0-indexed: first point is i=0, point number 1)
                                    # This makes uncertainty increase with point number (further predictions have more uncertainty)
                                    point_number = i + 1
                                    uncertainty_multiplier = point_number * (1.0 / 8.0)
                                    uncertainty_meters_unclamped = base_uncertainty_meters * uncertainty_multiplier
                                    
                                    # Convert to degrees for display (approximate)
                                    # 1 degree latitude ≈ 111,000 meters
                                    lat_rad = math.radians(lat)
                                    meters_per_degree_lat = 111000.0
                                    meters_per_degree_lon = 111000.0 * math.cos(lat_rad)
                                    avg_meters_per_degree = (meters_per_degree_lat + meters_per_degree_lon) / 2.0
                                    uncertainty_degrees = uncertainty_meters_unclamped / avg_meters_per_degree
                                    
                                    # Set reasonable bounds for circle radius (min 100m, max 50km)
                                    min_radius_meters = 100.0
                                    max_radius_meters = 50000.0  # 50 km max
                                    uncertainty_meters = max(min_radius_meters, min(uncertainty_meters_unclamped, max_radius_meters))
                                    
                                    # Comprehensive logging for uncertainty circle calculation
                                    logger.info(f"[Predicted Point {i}] Uncertainty Circle Calculation:")
                                    logger.info(f"  Position: lat={lat:.6f}°, lon={lon:.6f}°")
                                    logger.info(f"  Hours Ahead: {hours_ahead}h")
                                    logger.info(f"  Point Number: {point_number}")
                                    logger.info(f"  Lower Bound: lat={lower_lat:.6f}°, lon={lower_lon:.6f}°")
                                    logger.info(f"  Upper Bound: lat={upper_lat:.6f}°, lon={upper_lon:.6f}°")
                                    logger.info(f"  Distance Between Bounds: {distance_between_bounds_meters:.2f} m ({distance_between_bounds_meters/1000:.3f} km)")
                                    logger.info(f"  Base Radius (1/2 distance): {base_uncertainty_meters:.2f} m ({base_uncertainty_meters/1000:.3f} km)")
                                    logger.info(f"  Uncertainty Multiplier ({point_number} × 1/8): {uncertainty_multiplier:.6f}")
                                    logger.info(f"  Adjusted Radius: {uncertainty_meters:.2f} m ({uncertainty_meters/1000:.3f} km)")
                                    logger.info(f"  Uncertainty (degrees): {uncertainty_degrees:.6f}°")
                                    
                                    # Only add circle if radius is reasonable
                                    if uncertainty_meters <= max_radius_meters:
                                        # Add orange circle to represent uncertainty
                                        folium.Circle(
                                            [lat, lon],
                                            radius=uncertainty_meters,
                                            popup=f"Uncertainty: ±{uncertainty_degrees:.4f}° ({uncertainty_meters/1000:.2f} km)",
                                            tooltip=f"Uncertainty Radius: {uncertainty_meters/1000:.2f} km",
                                            color='orange',
                                            fill=True,
                                            fillColor='orange',
                                            fillOpacity=0.2,
                                            weight=2,
                                            opacity=0.5
                                        ).add_to(m)
                                        logger.info(f"  [Circle Added] Radius: {uncertainty_meters:.2f} m ({uncertainty_meters/1000:.3f} km)")
                                    else:
                                        logger.warning(f"  [Circle Skipped] Uncertainty radius too large ({uncertainty_meters/1000:.2f} km) for point {i}")
                                else:
                                    logger.debug(f"[Point {i}] Upper/lower bounds not available for uncertainty circle calculation")
                            except (NameError, UnboundLocalError):
                                logger.debug(f"[Point {i}] Upper/lower bounds not calculated (confidence intervals not available)")
                else:
                    raise ValueError(f"position_mean has invalid shape: {position_mean.shape}, expected (time_steps, 2)")
                
                # Calculate diameter of uncertainty circle for last predicted point at 1 order of magnitude error
                magnitude_info = ""
                if position_mean.ndim == 2 and position_mean.shape[0] > 0:
                    last_point_idx = position_mean.shape[0] - 1  # Last predicted point (48 hours ahead)
                    
                    if position_std is not None and isinstance(position_std, np.ndarray):
                        if position_std.ndim == 2 and last_point_idx < position_std.shape[0]:
                            # Get uncertainty for last point
                            std_lat = float(position_std[last_point_idx, 0])
                            std_lon = float(position_std[last_point_idx, 1])
                            
                            # Validate uncertainty values
                            max_reasonable_std = 1.0
                            if std_lat <= max_reasonable_std and std_lon <= max_reasonable_std:
                                # Get last predicted point coordinates
                                last_pred_lat = float(position_mean[last_point_idx, 0])
                                last_pred_lon = float(position_mean[last_point_idx, 1])
                                
                                # Use the same calculation method as uncertainty circles:
                                # 1. Get distance between bounds for the last point
                                # 2. Base uncertainty = 1/2 the distance between bounds
                                # 3. Apply multiplier (point_number × 1/8)
                                # 4. Multiply by 10 for 1 order of magnitude error
                                
                                # Check if we have upper and lower bounds for the last point
                                if len(lower_points) > last_point_idx + 1 and len(upper_points) > last_point_idx + 1:
                                    lower_bound_point = lower_points[last_point_idx + 1]
                                    upper_bound_point = upper_points[last_point_idx + 1]
                                    
                                    lower_lat = lower_bound_point[0]
                                    lower_lon = lower_bound_point[1]
                                    upper_lat = upper_bound_point[0]
                                    upper_lon = upper_bound_point[1]
                                    
                                    # Calculate distance between upper and lower bounds
                                    distance_between_bounds_meters = haversine_distance_meters(
                                        lower_lat, lower_lon, upper_lat, upper_lon
                                    )
                                    
                                    # If distance is 0 or very small, calculate from angular separation
                                    if distance_between_bounds_meters < 1.0:
                                        # Use same fallback calculation as uncertainty circles
                                        uncertainty_degrees_for_calc = (std_lat + std_lon) / 2.0
                                        base_angular_offset_for_calc = uncertainty_degrees_for_calc * 0.5
                                        point_number_for_calc = last_point_idx + 1
                                        uncertainty_multiplier_for_calc = point_number_for_calc * (1.0 / 8.0)
                                        angular_offset_for_calc = base_angular_offset_for_calc * uncertainty_multiplier_for_calc
                                        angular_offset_for_calc = min(angular_offset_for_calc, 45.0)
                                        
                                        # Calculate bounds from predicted point
                                        base_distance_nm = 1.0
                                        # Get predicted course for calculation
                                        if last_point_idx > 0:
                                            prev_lat = float(position_mean[last_point_idx - 1, 0])
                                            prev_lon = float(position_mean[last_point_idx - 1, 1])
                                            predicted_course_for_calc = calculate_bearing(prev_lat, prev_lon, last_pred_lat, last_pred_lon)
                                        else:
                                            predicted_course_for_calc = 0.0
                                        
                                        left_bound_course_calc = (predicted_course_for_calc - angular_offset_for_calc) % 360
                                        right_bound_course_calc = (predicted_course_for_calc + angular_offset_for_calc) % 360
                                        
                                        calc_lower_lat, calc_lower_lon = offset_point_by_bearing(last_pred_lat, last_pred_lon, left_bound_course_calc, base_distance_nm)
                                        calc_upper_lat, calc_upper_lon = offset_point_by_bearing(last_pred_lat, last_pred_lon, right_bound_course_calc, base_distance_nm)
                                        
                                        distance_between_bounds_meters = haversine_distance_meters(
                                            calc_lower_lat, calc_lower_lon, calc_upper_lat, calc_upper_lon
                                        )
                                    
                                    # Base uncertainty = 1/2 the distance between bounds (same as uncertainty circles)
                                    base_uncertainty_meters = distance_between_bounds_meters / 2.0
                                    
                                    # Apply same multiplier as uncertainty circles: (point_number) × (1/8)
                                    point_number = last_point_idx + 1
                                    uncertainty_multiplier = point_number * (1.0 / 8.0)
                                    adjusted_uncertainty_meters = base_uncertainty_meters * uncertainty_multiplier
                                    
                                    # Calculate for 1 order of magnitude error (10x the adjusted uncertainty)
                                    magnitude_error_factor = 10.0
                                    radius_1magnitude_meters = adjusted_uncertainty_meters * magnitude_error_factor
                                    
                                    # Diameter = 2 * radius
                                    diameter_1magnitude_meters = 2.0 * radius_1magnitude_meters
                                    diameter_1magnitude_km = diameter_1magnitude_meters / 1000.0
                                    diameter_1magnitude_nm = diameter_1magnitude_meters / 1852.0  # 1 nm = 1852 m
                                    
                                    # Log the calculation
                                    logger.info(f"[Last Predicted Point] Uncertainty Circle Diameter Calculation (1 Order of Magnitude Error):")
                                    logger.info(f"  Point Index: {last_point_idx} (48 hours ahead)")
                                    logger.info(f"  Position: lat={last_pred_lat:.6f}°, lon={last_pred_lon:.6f}°")
                                    logger.info(f"  Input Variables:")
                                    logger.info(f"    std_lat = {std_lat:.6f}°")
                                    logger.info(f"    std_lon = {std_lon:.6f}°")
                                    logger.info(f"  Calculated Variables:")
                                    logger.info(f"    Distance Between Bounds: {distance_between_bounds_meters:.2f} m ({distance_between_bounds_meters/1000:.3f} km)")
                                    logger.info(f"    Base Uncertainty (1/2 distance): {base_uncertainty_meters:.2f} m ({base_uncertainty_meters/1000:.3f} km)")
                                    logger.info(f"    Point Number: {point_number}")
                                    logger.info(f"    Uncertainty Multiplier ({point_number} × 1/8): {uncertainty_multiplier:.6f}")
                                    logger.info(f"    Adjusted Uncertainty: {adjusted_uncertainty_meters:.2f} m ({adjusted_uncertainty_meters/1000:.3f} km)")
                                    logger.info(f"  For 1 Order of Magnitude Error (10× adjusted uncertainty):")
                                    logger.info(f"    magnitude_error_factor = {magnitude_error_factor}")
                                    logger.info(f"    radius_1magnitude_meters = adjusted_uncertainty × {magnitude_error_factor} = {radius_1magnitude_meters:.2f} m ({radius_1magnitude_meters/1000:.3f} km)")
                                    logger.info(f"    diameter_1magnitude_meters = 2 × radius = {diameter_1magnitude_meters:.2f} m")
                                    logger.info(f"    diameter_1magnitude_km = {diameter_1magnitude_km:.3f} km")
                                    logger.info(f"    diameter_1magnitude_nm = {diameter_1magnitude_nm:.3f} nautical miles")
                                    
                                    # Display in message box or add to popup
                                    magnitude_info = (f"\n\nUncertainty Circle (1 Order of Magnitude Error) for Last Point:\n"
                                                    f"Diameter: {diameter_1magnitude_km:.3f} km ({diameter_1magnitude_nm:.3f} nm)\n"
                                                    f"Radius: {radius_1magnitude_meters/1000:.3f} km ({radius_1magnitude_meters/1852:.3f} nm)")
                                else:
                                    logger.warning(f"Cannot calculate 1 magnitude error diameter: bounds not available for last point")
                                    magnitude_info = "\n\nUncertainty Circle (1 Order of Magnitude Error): Calculation skipped (bounds not available)"
                            else:
                                logger.warning(f"Cannot calculate 1 magnitude error diameter: uncertainty values too large (std_lat={std_lat}, std_lon={std_lon})")
                                magnitude_info = "\n\nUncertainty Circle (1 Order of Magnitude Error): Calculation skipped (invalid uncertainty values)"
                        else:
                            logger.warning(f"Cannot calculate 1 magnitude error diameter: position_std shape invalid")
                            magnitude_info = "\n\nUncertainty Circle (1 Order of Magnitude Error): Calculation skipped (no uncertainty data)"
                    else:
                        logger.warning(f"Cannot calculate 1 magnitude error diameter: position_std not available")
                        magnitude_info = "\n\nUncertainty Circle (1 Order of Magnitude Error): Calculation skipped (no uncertainty data)"
            except Exception as e:
                logger.error(f"Error creating prediction visualization: {e}")
                logger.error(f"position_mean shape: {position_mean.shape if hasattr(position_mean, 'shape') else 'N/A'}")
                raise
            
            # Save map to Path_Maps subdirectory (use self.analysis.get_map_output_directory())
            # Ensure we always use the Path_Maps directory, never the project folder
            if not hasattr(self, 'analysis') or not hasattr(self.analysis, 'get_map_output_directory'):
                raise ValueError("Analysis object not properly initialized")
            
            # Get map output directory (Path_Maps subdirectory)
            map_dir = self.analysis.get_map_output_directory()
            
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            map_path = map_dir / f"ml_prediction_vessel_{mmsi}_{timestamp}.html"
            
            # Verify we're not saving to project folder (current working directory)
            project_folder = Path.cwd()
            if map_path.parent == project_folder:
                logger.warning(f"Map path is in project folder, redirecting to Path_Maps directory")
                map_path = map_dir / f"ml_prediction_vessel_{mmsi}_{timestamp}.html"
            
            logger.info(f"Saving prediction map to Path_Maps directory: {map_path}")
            logger.info(f"Map directory: {map_dir}, Project folder: {project_folder}")
            m.save(str(map_path))
            
            # Show success message with predicted coordinates list
            coord_text = "\n\nPredicted Coordinates:\n"
            if predicted_coords:
                for coord in predicted_coords:
                    coord_text += f"  +{coord['hours_ahead']}h: ({coord['lat']:.4f}, {coord['lon']:.4f})\n"
            else:
                coord_text += "  No valid predicted coordinates generated\n"
            
            messagebox.showinfo("Prediction Complete", 
                              f"ML Course Prediction complete for vessel {mmsi}!\n\n"
                              f"Map saved to: {map_path}\n\n"
                              f"The map shows:\n"
                              f"- Blue line: Historical trajectory\n"
                              f"- Red marker: Last known position\n"
                              f"- Green line: Predicted path (48 hours)\n"
                              f"- Orange lines: 68% confidence intervals"
                              + coord_text
                              + magnitude_info)
            
            open_file(str(map_path))
            
        except Exception as e:
            logger.error(f"Error displaying prediction results: {e}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to create visualization: {str(e)}")    
    
    def _run_anomaly_analysis(self):
        """Run analysis based on selected anomaly types and vessel types"""
        # Get selected anomaly types from Anomaly Types tab (GUI names)
        selected_anomalies_gui = [name for name, var in self.anomaly_types.items() if var.get()]
        
        if not selected_anomalies_gui:
            messagebox.showwarning("No Selection", "Please select at least one anomaly type.")
            return
        
        # Convert GUI anomaly type names to data format
        selected_anomaly_types = [map_anomaly_type_gui_to_data(gui_name) for gui_name in selected_anomalies_gui]
        # Remove duplicates (e.g., "Excessive Travel Distance (Fast)" and "Excessive Travel Distance (Slow)" both map to "Speed")
        selected_anomaly_types = list(set(selected_anomaly_types))
        
        # Get selected vessel types from Vessel Selection tab
        selected_vessel_types = [vtype for vtype, details in self.ship_types.items() if details['var'].get()]
        
        if not selected_vessel_types:
            messagebox.showwarning("No Selection", "Please select at least one vessel type in the Vessel Selection tab.")
            return
        
        # Limit to 2 vessel types and 2 anomaly types (same as correlation analysis dialog)
        if len(selected_vessel_types) > 2:
            messagebox.showwarning("Too Many Selections", 
                                 f"You have selected {len(selected_vessel_types)} vessel types. "
                                 "Correlation analysis supports up to 2 vessel types. "
                                 "Using the first 2 selected types.")
            selected_vessel_types = selected_vessel_types[:2]
        
        if len(selected_anomaly_types) > 2:
            messagebox.showwarning("Too Many Selections", 
                                 f"You have selected {len(selected_anomalies_gui)} anomaly types. "
                                 "Correlation analysis supports up to 2 anomaly types. "
                                 "Using the first 2 selected types.")
            selected_anomaly_types = selected_anomaly_types[:2]
        
        # Show a confirmation dialog
        vessel_names = [get_vessel_type_name(vt) for vt in selected_vessel_types]
        confirm_msg = (f"Run correlation analysis with:\n\n"
                      f"Vessel Types ({len(selected_vessel_types)}): {', '.join([f'Type {vt} ({name})' for vt, name in zip(selected_vessel_types, vessel_names)])}\n\n"
                      f"Anomaly Types ({len(selected_anomaly_types)}): {', '.join(selected_anomaly_types)}")
        confirm = messagebox.askyesno("Confirm Analysis", confirm_msg)
        if not confirm:
            return
        
        progress = ProgressDialog(self.window, "Anomaly Analysis", "Running anomaly analysis...")
        try:
            self.status_var.set("Running anomaly analysis...")
            result = self.analysis.correlation_analysis(selected_vessel_types, selected_anomaly_types)
            progress.close()
            if result:
                self.status_var.set(f"Analysis complete: {result}")
                messagebox.showinfo("Success", f"Analysis complete. Report saved to:\n{result}")
                # Try to open the report if it exists
                if os.path.exists(result):
                    try:
                        import webbrowser
                        webbrowser.open(f"file://{result}")
                    except Exception as e:
                        logger.error(f"Error opening report: {e}")
            else:
                self.status_var.set("Analysis failed")
                messagebox.showerror("Error", "Failed to perform analysis. Check logs for details.")
        except Exception as e:
            progress.close()
            self.status_var.set("Analysis failed")
            logger.error(f"Error in anomaly analysis: {e}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    def _create_tab5_anomaly_types(self):
        """Create the Correlation Analysis tab with checkboxes for each type and thresholds"""
        anomaly_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(anomaly_frame, text="Correlation Analysis")
        
        # Left panel - anomaly types selection
        types_frame = ttk.LabelFrame(anomaly_frame, text="Select Anomaly Types to Detect")
        types_frame.grid(row=0, column=0, sticky=tk.N+tk.W, padx=10, pady=10)
        
        # Create checkboxes for each anomaly type
        for i, (anomaly_type, var) in enumerate(self.anomaly_types.items()):
            ttk.Checkbutton(types_frame, text=anomaly_type, variable=var).grid(row=i, column=0, sticky=tk.W, padx=20, pady=5)
            
        # Add "Select All" buttons
        button_frame = ttk.Frame(types_frame)
        button_frame.grid(row=len(self.anomaly_types), column=0, sticky=tk.W, padx=5, pady=10, columnspan=2)
        ttk.Button(button_frame, text="Select All", command=self.select_all_anomalies).grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Button(button_frame, text="Deselect All", command=self.deselect_all_anomalies).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Right panel - thresholds
        thresholds_frame = ttk.LabelFrame(anomaly_frame, text="Anomaly Detection Thresholds")
        thresholds_frame.grid(row=0, column=1, sticky=tk.N+tk.W, padx=10, pady=10)
        
        # Travel distance thresholds
        ttk.Label(thresholds_frame, text="Travel Distance Thresholds (nautical miles):", font=("", 12)).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=10)
        
        ttk.Label(thresholds_frame, text="Minimum (below this is 'Slow'):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(thresholds_frame, textvariable=self.min_travel_nm, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(thresholds_frame, text="Maximum (above this is 'Fast'):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(thresholds_frame, textvariable=self.max_travel_nm, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # COG-Heading thresholds
        ttk.Label(thresholds_frame, text="COG-Heading Inconsistency Thresholds:", font=("", 12)).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=10)
        
        ttk.Label(thresholds_frame, text="Maximum difference (degrees):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(thresholds_frame, textvariable=self.cog_heading_max_diff, width=10).grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(thresholds_frame, text="Minimum speed for check (knots):").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(thresholds_frame, textvariable=self.min_speed_for_cog_check, width=10).grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Correlation Analysis section with Run Analysis button
        correlation_frame = ttk.LabelFrame(anomaly_frame, text="Correlation Analysis")
        correlation_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W+tk.E, padx=10, pady=20)
        
        run_btn = tk.Button(correlation_frame, text="Run Analysis", command=self._run_anomaly_analysis, 
                          bg="green", fg="white", font=("Arial", 10, "bold"))
        run_btn.pack(side=tk.RIGHT, padx=10, pady=10)
    
    def _create_tab6_analysis_filters(self):
        """Create the Analysis Filters tab with settings for filtering analysis"""
        filters_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(filters_frame, text="Analysis Filters")
        
        # Create a canvas with scrollbar
        canvas = tk.Canvas(filters_frame, width=850)
        scrollbar = ttk.Scrollbar(filters_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Geographic Filter section
        geo_frame = ttk.LabelFrame(scrollable_frame, text="Geographic Filter")
        geo_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(geo_frame, text="Latitude Range:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        latitude_frame = ttk.Frame(geo_frame)
        latitude_frame.grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)
        ttk.Label(latitude_frame, text="Min:").pack(side=tk.LEFT)
        ttk.Entry(latitude_frame, textvariable=self.analysis_filters['min_latitude'], width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(latitude_frame, text="Max:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(latitude_frame, textvariable=self.analysis_filters['max_latitude'], width=10).pack(side=tk.LEFT)
        
        ttk.Label(geo_frame, text="Longitude Range:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        longitude_frame = ttk.Frame(geo_frame)
        longitude_frame.grid(row=1, column=1, sticky=tk.W, padx=10, pady=5)
        ttk.Label(longitude_frame, text="Min:").pack(side=tk.LEFT)
        ttk.Entry(longitude_frame, textvariable=self.analysis_filters['min_longitude'], width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(longitude_frame, text="Max:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(longitude_frame, textvariable=self.analysis_filters['max_longitude'], width=10).pack(side=tk.LEFT)
        
        # Draw Box button
        draw_box_frame = ttk.Frame(geo_frame)
        draw_box_frame.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=10, pady=5)
        ttk.Button(draw_box_frame, text="Draw Box on Map", 
                   command=self.draw_geographic_box).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Time Filter section
        time_frame = ttk.LabelFrame(scrollable_frame, text="Time Filter")
        time_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(time_frame, text="Hour Range (0-24):").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        time_range_frame = ttk.Frame(time_frame)
        time_range_frame.grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)
        ttk.Label(time_range_frame, text="Start:").pack(side=tk.LEFT)
        ttk.Entry(time_range_frame, textvariable=self.analysis_filters['time_start_hour'], width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(time_range_frame, text="End:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(time_range_frame, textvariable=self.analysis_filters['time_end_hour'], width=5).pack(side=tk.LEFT)
        
        # Confidence and Anomaly Filters
        conf_frame = ttk.LabelFrame(scrollable_frame, text="Confidence and Anomaly Limits")
        conf_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(conf_frame, text="Minimum Confidence Score (0-100):").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        ttk.Entry(conf_frame, textvariable=self.analysis_filters['min_confidence'], width=10).grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)
        
        ttk.Label(conf_frame, text="Max Anomalies Per Vessel:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        ttk.Entry(conf_frame, textvariable=self.analysis_filters['max_anomalies_per_vessel'], width=10).grid(row=1, column=1, sticky=tk.W, padx=10, pady=5)
        
        # MMSI Filter section
        mmsi_frame = ttk.LabelFrame(scrollable_frame, text="MMSI Filter")
        mmsi_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(mmsi_frame, text="Filter by MMSI List (comma-separated):").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        ttk.Entry(mmsi_frame, textvariable=self.analysis_filters['filter_mmsi_list'], width=30).grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)
        
        # Reset button
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_analysis_filters_to_defaults).pack(side=tk.RIGHT, padx=10)
        
        # Help text
        help_frame = ttk.LabelFrame(scrollable_frame, text="Help")
        help_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(help_frame, text="These filters apply to all analyses performed in the Advanced Analysis module.", 
                wraplength=800, justify=tk.LEFT).pack(padx=10, pady=5, anchor=tk.W)
    
    def _create_tab7_zone_violations(self):
        """Create the Zone Violations tab for defining restricted zones"""
        zones_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(zones_frame, text="Zone Violations")
        
        # Top section: Buttons
        button_frame = ttk.Frame(zones_frame)
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=10)
        
        # Add zone button - exactly like SFD_GUI.py
        ttk.Button(button_frame, text="Add Zone", command=self.add_zone).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Middle section: Zone list with checkboxes
        list_frame = ttk.LabelFrame(zones_frame, text="Restricted Zones")
        list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a canvas with scrollbar for the zone list
        canvas = tk.Canvas(list_frame, width=750)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Store zone checkboxes and frames
        self.zone_checkboxes = {}
        self.zone_frames = {}
        
        # Zone list container
        self.zone_list_container = scrollable_frame
        
        # Add "Select All" buttons at the top
        select_button_frame = ttk.Frame(scrollable_frame)
        select_button_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        ttk.Button(select_button_frame, text="Select All", command=self.select_all_zones).pack(side=tk.LEFT, padx=5)
        ttk.Button(select_button_frame, text="Deselect All", command=self.deselect_all_zones).pack(side=tk.LEFT, padx=5)
        
        # Load zones from config.ini
        self._load_zones_from_config()
        
        # Refresh zone list display
        self._refresh_zone_list()
        
        # Description
        desc_label = ttk.Label(zones_frame, 
                            text="Manage restricted zones for Zone Violation detection. Check 'Selected' to include in analysis.",
                            font=("Arial", 9))
        desc_label.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
    def _create_tab8_vessel_selection(self):
        """Create the Vessel Selection tab with checkboxes for each ship type"""
        ship_types_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(ship_types_frame, text="Vessel Selection")
        
        # Group the ship types by category (needed for category buttons)
        categories = {}
        for ship_type, details in self.ship_types.items():
            category = details.get('category', 'Other')
            if category not in categories:
                categories[category] = []
            categories[category].append(ship_type)
        
        # Add buttons frame at the top
        button_frame = ttk.Frame(ship_types_frame)
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=10)
        
        # First row: Select All and Deselect All buttons
        first_row_frame = ttk.Frame(button_frame)
        first_row_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        ttk.Button(first_row_frame, text="Select All", command=self.select_all_ships).pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
        ttk.Button(first_row_frame, text="Deselect All", command=self.deselect_all_ships).pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
        
        # Prepare category-specific buttons
        category_buttons = []
        for category in sorted(categories.keys()):
            select_cmd = lambda c=category: self.select_category(c)
            deselect_cmd = lambda c=category: self.deselect_category(c)
            category_buttons.append((f"Select All {category}", select_cmd))
            category_buttons.append((f"Deselect All {category}", deselect_cmd))
        
        # Calculate rows for category buttons: if odd, top row gets one more
        num_category_buttons = len(category_buttons)
        if num_category_buttons > 0:
            top_row_count = (num_category_buttons + 1) // 2  # Top row gets one more if odd
            bottom_row_count = num_category_buttons // 2
            
            # Create second row frame (for category buttons)
            second_row_frame = ttk.Frame(button_frame)
            second_row_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
            
            # Create third row frame (for category buttons)
            third_row_frame = ttk.Frame(button_frame)
            third_row_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
            
            # Add category buttons to rows
            for i, (text, command) in enumerate(category_buttons):
                if i < top_row_count:
                    ttk.Button(second_row_frame, text=text, command=command).pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
                else:
                    ttk.Button(third_row_frame, text=text, command=command).pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
        
        # Create a canvas with scrollbar for the ship types
        canvas = tk.Canvas(ship_types_frame, width=750)
        scrollbar = ttk.Scrollbar(ship_types_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack the canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Instructions and title
        ttk.Label(scrollable_frame, text="Select vessel types to include in analysis:", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky=tk.W, padx=5, pady=10, columnspan=3)
            
        # Create frames for each category
        row = 1
        col = 0
        max_columns = 3  # Maximum columns in grid
        
        for category, ship_types_list in sorted(categories.items()):
            # Create a frame for this category
            category_frame = ttk.LabelFrame(scrollable_frame, text=f"{category} Vessels")
            category_frame.grid(row=row, column=col, sticky=tk.W, padx=10, pady=5)
            
            # Add checkboxes for each ship type in this category
            frame_row = 0
            for ship_type in sorted(ship_types_list):
                ttk.Checkbutton(
                    category_frame, 
                    text=f"{ship_type}: {self.ship_types[ship_type]['name']}",
                    variable=self.ship_types[ship_type]['var']
                ).grid(row=frame_row, column=0, sticky=tk.W, padx=5, pady=3)
                frame_row += 1
                
            # Move to next column or row
            col += 1
            if col >= max_columns:
                col = 0
                row += 1
    


# ============================================================================
# CLI INTERFACE FUNCTIONS
# ============================================================================

def cli_export_full_dataset(output_dir=None, config_path='config.ini'):
    """CLI function to export full dataset"""
    resolved_config_path = get_config_path(config_path)
    analysis = AdvancedAnalysis(None, output_dir, resolved_config_path)
    return analysis.export_full_dataset()

def cli_generate_summary_report(output_dir=None, config_path='config.ini'):
    """CLI function to generate summary report"""
    resolved_config_path = get_config_path(config_path)
    analysis = AdvancedAnalysis(None, output_dir, resolved_config_path)
    return analysis.generate_summary_report()
