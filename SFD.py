#!/usr/bin/env python3
"""
AIS Shipping Fraud Detection System (Integrated Version)

This module implements an anomaly detection system for shipping vessels using AIS data.
It identifies potential fraud patterns by analyzing daily ship movement data.

[VERSION}
Team = Dreadnaught
Alex Giacomello, Christopher Matherne, Rupert Rigg, Zachary Zhao
version = 2.1 Beta
 
"""

import os
import glob
import argparse
import configparser
import logging
import re
import platform
import sys
import traceback
from datetime import datetime, timedelta
import subprocess
import importlib
import hashlib
import json
import shutil
import boto3
import botocore.exceptions
from urllib.parse import urlparse
import pandas as pd
import dask.dataframe as dd
import numpy as np
from geopy.distance import great_circle
import folium
from folium.plugins import MarkerCluster, HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import tkinter as tk
from tkinter import messagebox
import threading
import subprocess
from branca.element import Element
import math


# Import local utility modules
from utils import get_cache_dir, clear_cache, check_dependencies, format_file_size
from utils import log_memory_usage, suppress_warnings, validate_config, generate_cache_key
from map_utils import MapCoordinateManager, add_lat_lon_grid_lines

# Global variables for tracking background processes
statistics_thread = None
statistics_requested = False
statistics_completed = False

# Try to import importlib.metadata (Python 3.8+), fallback to pkg_resources for older Python
# Note: Python 3.14 is fully supported
try:
    import importlib.metadata  # Modern replacement for pkg_resources
except ImportError:
    # Fallback for Python < 3.8
    try:
        import pkg_resources  # type: ignore
        # Create a compatibility wrapper
        class MetadataWrapper:
            @staticmethod
            def version(package_name):
                try:
                    return pkg_resources.get_distribution(package_name).version
                except pkg_resources.DistributionNotFound:
                    raise MetadataWrapper.PackageNotFoundError(f"No package named '{package_name}'")
            
            class PackageNotFoundError(Exception):
                pass
        
        importlib.metadata = MetadataWrapper()
    except ImportError:
        importlib.metadata = None  # Will need to handle this case in code


# def clear_cache():
#     """
#     Clear the data cache directory.
    
#     Returns:
#         bool: True if cache was cleared successfully, False otherwise
#     """
#     cache_dir = get_cache_dir()
#     if not cache_dir or not os.path.exists(cache_dir):
#         logger.info("No cache directory found")
#         return True
    
#     try:
#         # Remove all files in cache directory
#         for filename in os.listdir(cache_dir):
#             file_path = os.path.join(cache_dir, filename)
#             if os.path.isfile(file_path):
#                 os.unlink(file_path)
#         logger.info(f"Cache cleared: {cache_dir}")
#         return True
#     except Exception as e:
#         logger.error(f"Error clearing cache: {e}")
#         return False


# def check_dependencies():
#     """
#     Check if all required dependencies are installed.
#     If not, offer to install them from requirements.txt.
    
#     Returns:
#         bool: True if all dependencies are satisfied, False otherwise
#     """
#     logger = logging.getLogger(__name__)
#     logger.info("Checking dependencies...")
    
#     # Log Python version information
#     logger.info(f"Python version: {sys.version}")
#     logger.info(f"Python version info: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
#     logger.info(f"Python executable: {sys.executable}")
    
#     # Find the requirements.txt file (look in script directory first)
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     req_file = os.path.join(script_dir, "requirements.txt")
    
#     # If not found in script directory, check current working directory
#     if not os.path.exists(req_file):
#         req_file = os.path.join(os.getcwd(), "requirements.txt")
#         if not os.path.exists(req_file):
#             logger.warning("requirements.txt not found in script directory or current working directory")
#             return True  # Continue without checking
    
#     try:
#         # Parse requirements file
#         requirements = []
#         with open(req_file, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 if line and not line.startswith('#') and not line.startswith('-e'):
#                     # Strip off version specifiers
#                     if ';' in line:
#                         # Handle platform-specific requirements
#                         req, condition = line.split(';', 1)
#                         req = req.strip()
#                         condition = condition.strip()
                        
#                         # Check if this requirement applies to current platform
#                         if 'platform_system==' in condition:
#                             platform_name = condition.split('==')[1].strip().strip("'").strip('"')
#                             if platform.system() != platform_name:
#                                 continue  # Skip this requirement if platform doesn't match
#                     else:
#                         req = line
                    
#                     if '>=' in req:
#                         req = req.split('>=')[0]
#                     elif '==' in req:
#                         req = req.split('==')[0]
#                     elif '<=' in req:
#                         req = req.split('<=')[0]
                    
#                     req = req.strip()
#                     if req:
#                         requirements.append(req)
        
#         # Check installed packages
#         missing = []
#         for req in requirements:
#             try:
#                 importlib.import_module(req.lower().replace('-', '_'))
#             except ImportError:
#                 try:
#                     # Double-check with importlib.metadata as some package names don't match import names
#                     if importlib.metadata is not None:
#                         importlib.metadata.version(req)
#                     else:
#                         # If importlib.metadata is not available, skip this check
#                         missing.append(req)
#                 except (importlib.metadata.PackageNotFoundError, AttributeError, TypeError):
#                     missing.append(req)
        
#         if missing:
#             logger.warning(f"Missing dependencies: {', '.join(missing)}")
            
#             # On Windows, check for admin rights since some installs may require it
#             admin_needed = False
#             if platform.system() == "Windows" and any(pkg in ["pywin32"] for pkg in missing):
#                 admin_needed = True
#                 logger.warning("Some dependencies may require administrator privileges to install.")
            
#             # Ask to install missing dependencies
#             if admin_needed:
#                 print("\nWARNING: Some dependencies require administrator privileges to install.")
#                 print("Please run this script with administrator privileges or install manually:")
#                 print(f"pip install {' '.join(missing)}")
#                 return False
#             else:
#                 print(f"\nMissing dependencies: {', '.join(missing)}")
#                 resp = input("Would you like to install them now? (y/n): ").strip().lower()
                
#                 if resp in ('y', 'yes'):
#                     logger.info("Installing missing dependencies...")
#                     try:
#                         subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
#                         logger.info("Dependencies installed successfully!")
#                         return True
#                     except subprocess.CalledProcessError:
#                         logger.error("Failed to install dependencies. Please install them manually.")
#                         print(f"Please install the following dependencies manually: {', '.join(missing)}")
#                         return False
#                 else:
#                     logger.warning("Missing dependencies will not be installed. Some features may not work.")
#                     print("You can install dependencies manually with:")
#                     print(f"pip install {' '.join(missing)}")
#                     return False
        
#         logger.info("All dependencies are satisfied.")
#         return True
    
#     except Exception as e:
#         logger.error(f"Error checking dependencies: {e}")
#         return True  # Continue anyway
# import boto3
# import botocore.exceptions
# from urllib.parse import urlparse
# import pandas as pd
# import dask.dataframe as dd
# import numpy as np
# from geopy.distance import great_circle
# import folium
# from folium.plugins import MarkerCluster, HeatMap
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.cm as cm
# import matplotlib.colors as mcolors
# import tkinter as tk
# from tkinter import messagebox
# import subprocess
# from branca.element import Element
# import math

# Try to import GPU libraries (NVIDIA CUDA, AMD ROCm/HIP, or cupy)
GPU_AVAILABLE = False
GPU_TYPE = None  # 'NVIDIA', 'AMD', or None
GPU_BACKEND = None  # 'CUDA', 'ROCm', 'HIP', or None
cudf = None
cp = None  # cupy (CUDA) or cupy-rocm (ROCm/HIP)
hip = None  # PyHIP for direct HIP access

# Function to check if conda is available
def is_conda_available():
    try:
        # Check if conda is in PATH
        result = subprocess.run(['conda', '--version'], 
                               capture_output=True, text=True, check=False)
        return result.returncode == 0
    except:
        return False

# Try to import GPU libraries - check for NVIDIA (CUDA), AMD (ROCm/HIP), and cupy
# Note: ROCm/cupy-rocm/HIP support on Windows is limited but we'll attempt detection
is_windows = platform.system() == 'Windows'

# First try NVIDIA CUDA (cudf + cupy) - works on both Windows and Linux
try:
    import cudf  # type: ignore
    import cupy as cp  # type: ignore
    # Verify CUDA is actually available
    if cp.cuda.is_available():
        GPU_AVAILABLE = True
        GPU_TYPE = 'NVIDIA'
        GPU_BACKEND = 'CUDA'
    else:
        raise ImportError("CUDA not available")
except ImportError:
    # If NVIDIA libraries not available, try AMD ROCm/HIP
    # Note: ROCm/cupy-rocm/HIP support on Windows is limited but we'll attempt detection
    try:
        # Try cupy-rocm for AMD GPUs (cupy-rocm is installed as 'cupy' but uses ROCm/HIP backend)
        # First check if we can import cupy (might be cupy-rocm)
        import cupy as cp  # type: ignore
        # Check if this is ROCm version by trying to access device info
        # cupy-rocm should work similarly to cupy but for AMD GPUs using HIP
        try:
            # Try to get device info - ROCm version should work
            if hasattr(cp, 'cuda') and cp.cuda.is_available():
                GPU_AVAILABLE = True
                GPU_TYPE = 'AMD'
                GPU_BACKEND = 'ROCm'  # cupy-rocm uses ROCm which is built on HIP
            else:
                raise ImportError("ROCm not available")
        except:
            # If cupy is installed but not working, it might be regular cupy without GPU
            raise ImportError("GPU not available")
    except ImportError:
        # Try direct HIP support via PyHIP (if available)
        # Prioritize pyhip import - this is the standard PyHIP package
        try:
            # First try pyhip (the standard PyHIP package name)
            try:
                import pyhip as hip  # type: ignore
                # Check if HIP is available and working
                # PyHIP provides direct access to HIP runtime
                # Test if HIP devices are available
                try:
                    if hasattr(hip, 'is_available') and hip.is_available():
                        GPU_AVAILABLE = True
                        GPU_TYPE = 'AMD'
                        GPU_BACKEND = 'HIP'
                    elif hasattr(hip, 'getDeviceCount'):
                        # Alternative check: try to get device count
                        device_count = hip.getDeviceCount()
                        if device_count > 0:
                            GPU_AVAILABLE = True
                            GPU_TYPE = 'AMD'
                            GPU_BACKEND = 'HIP'
                        else:
                            raise ImportError("PyHIP: No HIP devices available")
                    else:
                        # If no availability check, assume it's available if imported
                        GPU_AVAILABLE = True
                        GPU_TYPE = 'AMD'
                        GPU_BACKEND = 'HIP'
                except Exception as e:
                    raise ImportError(f"PyHIP not available: {e}")
            except ImportError:
                # Try alternative HIP import name
                try:
                    import hip  # type: ignore
                    if hasattr(hip, 'is_available') and hip.is_available():
                        GPU_AVAILABLE = True
                        GPU_TYPE = 'AMD'
                        GPU_BACKEND = 'HIP'
                    elif hasattr(hip, 'getDeviceCount'):
                        device_count = hip.getDeviceCount()
                        if device_count > 0:
                            GPU_AVAILABLE = True
                            GPU_TYPE = 'AMD'
                            GPU_BACKEND = 'HIP'
                        else:
                            raise ImportError("HIP: No devices available")
                    else:
                        GPU_AVAILABLE = True
                        GPU_TYPE = 'AMD'
                        GPU_BACKEND = 'HIP'
                except ImportError:
                    raise ImportError("PyHIP libraries not found")
        except ImportError:
            # No GPU libraries available
            GPU_AVAILABLE = False
            GPU_TYPE = None
            GPU_BACKEND = None
            cudf = None
            cp = None
            hip = None

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sfd.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log Python version information
logger.info(f"Python version: {sys.version}")
logger.info(f"Python version info: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
logger.info(f"Python executable: {sys.executable}")

# The application should still run without GPU support, so we don't attempt installation
# The GPU installation is handled separately by the GUI when the user explicitly requests it

# Advanced Analysis import (logger is already initialized at line 343)
try:
    from advanced_analysis import AdvancedAnalysis
    ADVANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    ADVANCED_ANALYSIS_AVAILABLE = False
    logger.warning("Advanced analysis module not available for CLI operations")
        
if not GPU_AVAILABLE:
    logger.info("Running without GPU acceleration")


# def suppress_warnings(enabled=True):
#     """
#     Suppress Python warnings if enabled.
    
#     Args:
#         enabled (bool): Whether to suppress warnings
#     """
#     if enabled:
#         import warnings
#         warnings.filterwarnings("ignore")
#         logger.info("Warnings have been suppressed based on configuration")
#     else:
#         import warnings
#         warnings.resetwarnings()
#         logger.info("Warnings are enabled based on configuration")




# def add_lat_lon_grid_lines(m_or_fg, lat_start=-90, lat_end=90, lon_start=-180, lon_end=180, lat_step=10, lon_step=10, label_step=10):
#     """
#     Add latitude and longitude grid lines to a Folium map or FeatureGroup.
    
#     Args:
#         m_or_fg (folium.Map or folium.FeatureGroup): Folium map or feature group to add grid lines to
#         lat_start (float): Starting latitude
#         lat_end (float): Ending latitude
#         lon_start (float): Starting longitude
#         lon_end (float): Ending longitude
#         lat_step (float): Step size for latitude lines
#         lon_step (float): Step size for longitude lines
#         label_step (int): Add labels every N degrees
        
#     Returns:
#         folium.Map: The map with grid lines added
#     """
#     # For backward compatibility, use m internally
#     m = m_or_fg
#     try:
#         # Set safety limits for grid lines to prevent hanging
#         logger.info(f"Adding grid lines from lat {lat_start} to {lat_end}, lon {lon_start} to {lon_end}")
        
#         # Use fixed step sizes for better performance
#         # Set reasonable limits for grid boundaries to avoid poles and edge cases
#         lat_range = [max(lat_start, -85), min(lat_end, 85)]  # Avoid poles
#         lon_range = [max(lon_start, -175), min(lon_end, 175)]  # Limit range
        
#         # Using fixed step sizes (lat_step=10, lon_step=10) for better performance
#         # No need to calculate or adjust step sizes dynamically
            
#         # Add custom CSS to style the grid lines
#         css = '''
#             .lat-line {
#                 stroke: #ff6b6b;
#                 stroke-width: 1.5;
#                 stroke-dasharray: 5, 5;
#                 opacity: 0.6;
#             }
#             .lon-line {
#                 stroke: #4ecdc4;
#                 stroke-width: 1.5;
#                 stroke-dasharray: 5, 5;
#                 opacity: 0.6;
#             }
#             .coordinate-label {
#                 background: rgba(255, 255, 255, 0.7);
#                 border-radius: 3px;
#                 padding: 2px;
#                 font-size: 10px;
#                 font-weight: bold;
#                 box-shadow: 1px 1px 2px rgba(0,0,0,0.3);
#                 color: #333;
#             }
#         '''
#         # Only add CSS if this is a map, not a feature group
#         if hasattr(m, 'get_root'):
#             element = Element(f"<style>{css}</style>")
#             m.get_root().html.add_child(element)
        
#         # Add latitude lines with error handling
#         for lat in range(math.ceil(lat_range[0]), math.floor(lat_range[1])+1, lat_step):
#             try:
#                 # Add grid line
#                 points = [[lat, lon_range[0]], [lat, lon_range[1]]]
#                 folium.PolyLine(
#                     points, 
#                     tooltip=f"Latitude: {lat} degrees", 
#                     color='#ff6b6b', 
#                     weight=1.5, 
#                     opacity=0.6, 
#                     dash_array='5,5',
#                     className='lat-line'
#                 ).add_to(m)
                
#                 # Add label every label_step degrees (only at select intervals)
#                 if lat % label_step == 0:
#                     # Both left and right side labels
#                     html_left = f'<div class="coordinate-label">{lat} deg N</div>'
#                     folium.Marker(
#                         [lat, lon_range[0]],
#                         icon=folium.DivIcon(html=html_left)
#                     ).add_to(m)
                    
#                     # Right side label
#                     html_right = f'<div class="coordinate-label">{lat} deg N</div>'
#                     folium.Marker(
#                         [lat, lon_range[1]],
#                         icon=folium.DivIcon(html=html_right)
#                     ).add_to(m)
#             except Exception as e:
#                 logger.warning(f"Error adding latitude line at {lat}: {e}")
#                 continue  # Skip this line and continue
        
#         # Add longitude lines with error handling
#         for lon in range(math.ceil(lon_range[0]), math.floor(lon_range[1])+1, lon_step):
#             try:
#                 # Add grid line
#                 points = [[lat_range[0], lon], [lat_range[1], lon]]
#                 folium.PolyLine(
#                     points, 
#                     tooltip=f"Longitude: {lon} degrees", 
#                     color='#4ecdc4', 
#                     weight=1.5, 
#                     opacity=0.6, 
#                     dash_array='5,5',
#                     className='lon-line'
#                 ).add_to(m)
                
#                 # Add label every label_step degrees (only at select intervals)
#                 if lon % label_step == 0:
#                     # Both bottom and top labels
#                     html_bottom = f'<div class="coordinate-label">{lon} deg E</div>'
#                     folium.Marker(
#                         [lat_range[0], lon],
#                         icon=folium.DivIcon(html=html_bottom)
#                     ).add_to(m)
                    
#                     # Top label
#                     html_top = f'<div class="coordinate-label">{lon} deg E</div>'
#                     folium.Marker(
#                         [lat_range[1], lon],
#                         icon=folium.DivIcon(html=html_top)
#                     ).add_to(m)
#             except Exception as e:
#                 logger.warning(f"Error adding longitude line at {lon}: {e}")
#                 continue  # Skip this line and continue
        
#         logger.info("Successfully added grid lines to map")
#         return m
    
#     except Exception as e:
#         # If anything goes wrong, log the error and return the map without grid lines
#         logger.error(f"Failed to add grid lines to map: {e}")
#         return m

# Log GPU availability status
if GPU_AVAILABLE:
    if GPU_TYPE == 'NVIDIA':
        logger.info("GPU support detected and enabled (NVIDIA CUDA with RAPIDS libraries)")
    elif GPU_TYPE == 'AMD':
        if GPU_BACKEND == 'HIP':
            logger.info("GPU support detected and enabled (AMD HIP via PyHIP)")
            if hip is not None:
                try:
                    if hasattr(hip, 'getDeviceCount'):
                        device_count = hip.getDeviceCount()
                        logger.info(f"PyHIP: {device_count} HIP device(s) detected")
                except:
                    pass
        elif GPU_BACKEND == 'ROCm':
            logger.info("GPU support detected and enabled (AMD ROCm/HIP via cupy-rocm)")
        else:
            logger.info("GPU support detected and enabled (AMD)")
        if is_windows:
            logger.info("AMD GPU detected on Windows - functionality may be limited")
    else:
        logger.info("GPU support detected and enabled")
else:
    if is_windows:
        logger.info("GPU support not available, using CPU-based processing")
        logger.info("Note: For AMD GPUs, install cupy-rocm or PyHIP. ROCm/HIP support on Windows may be limited.")
    else:
        logger.info("GPU support not available, using CPU-based processing")

# Required columns for AIS data
REQUIRED_COLUMNS = [
    'MMSI', 'VesselName', 'VesselType', 'LAT', 'LON',
    'BaseDateTime', 'SOG', 'Heading', 'COG'
]

# Using MapCoordinateManager to store calculated lat/lon range for all maps
map_manager = MapCoordinateManager()

def calculate_global_boundaries(anomalies_df):
    """
    Calculate global lat/lon boundaries from anomalies data for consistent map display.
    
    Args:
        anomalies_df (DataFrame): DataFrame containing anomalies with LAT and LON columns
    """
    global map_manager
    map_manager.calculate_boundaries(anomalies_df)

# def calculate_global_boundaries(anomalies_df):
#     """
#     Calculate global lat/lon boundaries from anomalies data for consistent map display.
    
#     Args:
#         anomalies_df (DataFrame): DataFrame containing anomalies with LAT and LON columns
#     """
#     global GLOBAL_MIN_LAT, GLOBAL_MAX_LAT, GLOBAL_MIN_LON, GLOBAL_MAX_LON
    
#     if anomalies_df is None or anomalies_df.empty or 'LAT' not in anomalies_df.columns or 'LON' not in anomalies_df.columns:
#         logger.warning("Cannot calculate global boundaries from empty or invalid anomalies dataframe")
#         return
    
#     # Add padding to ensure maps have some margin
#     padding = 5
    
#     # Calculate boundaries from the anomalies dataframe
#     GLOBAL_MIN_LAT = anomalies_df['LAT'].min() - padding
#     GLOBAL_MAX_LAT = anomalies_df['LAT'].max() + padding
#     GLOBAL_MIN_LON = anomalies_df['LON'].min() - padding
#     GLOBAL_MAX_LON = anomalies_df['LON'].max() + padding
    
#     logger.info(f"Global map boundaries calculated: LAT [{GLOBAL_MIN_LAT} to {GLOBAL_MAX_LAT}], LON [{GLOBAL_MIN_LON} to {GLOBAL_MAX_LON}]")



def normalize_angle_difference(angle_diff):
    """
    Normalize angle difference to be between -180 and 180 degrees.
    
    Args:
        angle_diff (float): The angle difference in degrees
        
    Returns:
        float: The normalized angle difference
    """
    while angle_diff > 180:
        angle_diff -= 360
    while angle_diff < -180:
        angle_diff += 360
    return angle_diff


def is_s3_uri(path):
    """
    Check if the given path is an S3 URI.
    
    Args:
        path (str): The path to check
        
    Returns:
        bool: True if the path is an S3 URI, False otherwise
    """
    return path.startswith('s3://')


def parse_s3_uri(uri):
    """
    Parse an S3 URI into bucket name and prefix.
    
    Args:
        uri (str): The S3 URI to parse
        
    Returns:
        tuple: (bucket_name, prefix)
    """
    parsed = urlparse(uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip('/')
    return bucket, prefix


def list_s3_files_by_date_range(s3_uri, start_date, end_date, config=None):
    """
    Get AIS data files from S3 for a specific date range based on their filenames.
    Supports both .csv and .parquet files with filenames in YYYY-MM-DD format.
    
    Args:
        s3_uri (str): S3 URI in the format s3://bucket-name/prefix/
        start_date (date): Start date for file selection
        end_date (date): End date for file selection
        config (dict, optional): Configuration dictionary containing AWS credentials
        
    Returns:
        tuple: (selected_files, selected_dates)
    """
    try:
        bucket_name, prefix = parse_s3_uri(s3_uri)
        
        # Create an S3 client with credentials from config - using keys method only
        if config and 'AWS' in config:
            region = config['AWS'].get('s3_region', '')
            
            # Get AWS credentials from config
            access_key = config['AWS'].get('s3_access_key', '')
            secret_key = config['AWS'].get('s3_secret_key', '')
            session_token = config['AWS'].get('s3_session_token', '')
            
            if access_key and secret_key:
                # Debug output (masking most of the key for security)
                access_key_masked = access_key[:4] + "..." + access_key[-4:] if len(access_key) > 8 else "[empty]"
                session_token_len = len(session_token) if session_token else 0
                logger.info(f"Creating S3 client with access keys. Key: {access_key_masked}, Token length: {session_token_len}")
                
                # Clean up credentials - remove any whitespace
                access_key = access_key.strip()
                secret_key = secret_key.strip()
                if session_token:
                    session_token = session_token.strip()
                
                # Create boto3 session with explicit credentials
                try:
                    if session_token:
                        session = boto3.Session(
                            aws_access_key_id=access_key,
                            aws_secret_access_key=secret_key,
                            aws_session_token=session_token,
                            region_name=region if region else None
                        )
                    else:
                        session = boto3.Session(
                            aws_access_key_id=access_key,
                            aws_secret_access_key=secret_key,
                            region_name=region if region else None
                        )
                    s3 = session.client('s3')
                except Exception as e:
                    logger.error(f"Error creating boto3 session: {e}")
                    raise
            else:
                logger.error("AWS access keys not provided in config")
                raise ValueError("AWS access keys are required for S3 access")
        else:
            logger.error("No AWS configuration section found in config")
            raise ValueError("AWS configuration is required for S3 access")
        
        if prefix and not prefix.endswith('/'):
            prefix += '/'
        
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        
        if 'Contents' not in response:
            logger.warning(f"No files found in S3 bucket {bucket_name} with prefix {prefix}")
            return [], []
        
        dated_files = {}
        
        for item in response['Contents']:
            key = item['Key']
            filename = os.path.basename(key)
            
            if '.' not in filename:
                continue
                
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in ['.csv', '.parquet']:
                continue
                
            file_name_without_ext = filename.replace(file_ext, '')
            
            if file_name_without_ext.startswith('ais-'):
                date_str = file_name_without_ext[4:]
            else:
                date_str = file_name_without_ext
                
            try:
                file_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                dated_files[file_date] = f"s3://{bucket_name}/{key}"
            except ValueError:
                logger.warning(f"Skipping malformed filename {filename}. Expected ais-YYYY-MM-DD{file_ext} or YYYY-MM-DD{file_ext}")
                continue
        
        if not dated_files:
            logger.error(f"No valid data files found in S3 bucket {bucket_name} with prefix {prefix}")
            return [], []
        
        selected_files = []
        selected_dates = []
        
        for file_date, file_path in sorted(dated_files.items()):
            if start_date <= file_date <= end_date:
                selected_files.append(file_path)
                selected_dates.append(file_date)
        
        if not selected_files:
            logger.warning(f"No files found in date range {start_date} to {end_date}")
            
        return selected_files, selected_dates
            
    except botocore.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"AWS S3 error: {error_code} - {error_message}")
        
        # Special handling for common S3 errors
        if error_code == 'InvalidAccessKeyId':
            logger.error("Invalid AWS access key provided in config.ini")
        elif error_code == 'SignatureDoesNotMatch':
            logger.error("Invalid AWS secret key provided in config.ini")
        elif error_code == 'ExpiredToken':
            logger.error("AWS session token has expired. Please obtain a new token.")
        elif error_code == 'AccessDenied':
            logger.error("Access denied to S3 bucket. Check your permissions.")
            
        return [], []
    except Exception as e:
        logger.error(f"Error finding files in S3 bucket: {str(e)}")
        logger.error(f"Check your AWS credentials and S3 bucket configuration in config.ini")
        return [], []


def get_files_for_date_range(data_dir, start_date, end_date, config=None):
    """
    Get AIS data files for a specific date range based on their filenames.
    Supports both .csv and .parquet files with filenames in YYYY-MM-DD format.
    
    Args:
        data_dir (str): Directory containing the AIS data files
        start_date (date): Start date for file selection
        end_date (date): End date for file selection
        config (dict, optional): Configuration dictionary containing AWS credentials
        
    Returns:
        tuple: (selected_files, selected_dates)
    """
    try:
        if is_s3_uri(data_dir):
            return list_s3_files_by_date_range(data_dir, start_date, end_date, config)
            
        # Check if directory exists
        if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
            # Try to use the .ais_data_cache directory as a fallback
            cache_dir = get_cache_dir()
            
            # Create date-specific subfolder path if dates are available
            try:
                start_fmt = start_date.strftime('%Y%m%d')
                end_fmt = end_date.strftime('%Y%m%d')
                date_subfolder = f"{start_fmt}-{end_fmt}"
                cache_subdir = os.path.join(cache_dir, date_subfolder)
                
                if os.path.exists(cache_subdir):
                    logger.info(f"Data directory {data_dir} not found, using cache directory: {cache_subdir}")
                    data_dir = cache_subdir
                else:
                    logger.info(f"Data directory {data_dir} not found, using main cache directory: {cache_dir}")
                    data_dir = cache_dir
            except Exception as e:
                logger.warning(f"Error creating cache subfolder path: {e}, using main cache directory")
                logger.info(f"Data directory {data_dir} not found, using main cache directory: {cache_dir}")
                data_dir = cache_dir
                
            # If even the cache directory doesn't exist, return empty
            if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
                logger.error(f"Neither specified data directory nor cache directory exists")
                return [], []
            
        # Get all CSV and Parquet files in the directory
        all_files = glob.glob(os.path.join(data_dir, "*.csv")) + \
                    glob.glob(os.path.join(data_dir, "*.parquet"))
        
        dated_files = {}
        
        for file_path in all_files:
            filename = os.path.basename(file_path)
            file_ext = os.path.splitext(filename)[1].lower()
            file_name_without_ext = filename.replace(file_ext, '')
            
            if file_name_without_ext.startswith('ais-'):
                date_str = file_name_without_ext[4:]
            else:
                date_str = file_name_without_ext
                
            try:
                file_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                dated_files[file_date] = file_path
            except ValueError:
                logger.warning(f"Skipping malformed filename {filename}. Expected ais-YYYY-MM-DD{file_ext} or YYYY-MM-DD{file_ext}")
                continue
        
        if not dated_files:
            logger.error(f"No valid data files found in {data_dir}")
            return [], []
        
        selected_files = []
        selected_dates = []
        
        for file_date, file_path in sorted(dated_files.items()):
            if start_date <= file_date <= end_date:
                selected_files.append(file_path)
                selected_dates.append(file_date)
                
        if not selected_files:
            logger.warning(f"No files found in date range {start_date} to {end_date}")
            
        return selected_files, selected_dates
                
    except Exception as e:
        logger.error(f"Error finding files in {data_dir}: {e}")
        return [], []

# The commented-out duplicate version of get_files_for_date_range has been removed
# as part of code cleanup. The active implementation is maintained above.


def load_config(config_file='config.ini'):
    """
    Load configuration from config.ini file.
    
    Args:
        config_file (str, optional): Path to the configuration file. Defaults to 'config.ini'.
        
    Returns:
        dict: Configuration parameters
    """
    if not os.path.exists(config_file):
        logger.warning(f"Config file {config_file} not found. Using default values.")
        return {
            'COG_HEADING_MAX_DIFF': 45,
            'MIN_SPEED_FOR_COG_CHECK': 10,
            'SPEED_THRESHOLD': 102,  # Max theoretical speed in knots (117 mph / 189 kph)
            'USE_DASK': True,
            'USE_GPU': GPU_AVAILABLE,  # Use GPU if available
            'DATA_DIRECTORY': 'data',
            'OUTPUT_DIRECTORY': 'C:\\AIS_Data\\Reports',  # Proper Windows path with double backslashes
            'SELECTED_SHIP_TYPES': [20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 69, 70, 71, 72, 73, 74, 79, 80, 81, 82, 83, 84, 89, 90, 91, 92, 93, 94],  # All vessel types
            'USE_S3': False,
            'S3_DATA_URI': '',
            'TIME_DIFF_THRESHOLD_MIN': 240,  # Default 4 hours (240 min) for time difference detection
            'BEACON_TIME_THRESHOLD_HOURS': 6,  # Default 6 hours for AIS beacon anomaly detection
            'start_date': '2024-10-15',  # Default start date
            'end_date': '2024-10-17'   # Default end date
        }
        
    config = configparser.ConfigParser()
    try:
        config.read(config_file)
        
        # Helper function to get config values case-insensitively from specific sections
        def get_config_value(section, key, fallback=None, value_type='str'):
            # Try with original case first
            try:
                if value_type == 'float':
                    return config.getfloat(section, key, fallback=fallback)
                elif value_type == 'int':
                    return config.getint(section, key, fallback=fallback)
                elif value_type == 'boolean':
                    return config.getboolean(section, key, fallback=fallback)
                else:  # default to string
                    return config.get(section, key, fallback=fallback)
            except (configparser.NoSectionError, configparser.NoOptionError):
                # Try lowercase variant
                try:
                    if value_type == 'float':
                        return config.getfloat(section, key.lower(), fallback=fallback)
                    elif value_type == 'int':
                        return config.getint(section, key.lower(), fallback=fallback)
                    elif value_type == 'boolean':
                        return config.getboolean(section, key.lower(), fallback=fallback)
                    else:  # default to string
                        return config.get(section, key.lower(), fallback=fallback)
                except (configparser.NoSectionError, configparser.NoOptionError):
                    # Try uppercase variant
                    try:
                        if value_type == 'float':
                            return config.getfloat(section, key.upper(), fallback=fallback)
                        elif value_type == 'int':
                            return config.getint(section, key.upper(), fallback=fallback)
                        elif value_type == 'boolean':
                            return config.getboolean(section, key.upper(), fallback=fallback)
                        else:  # default to string
                            return config.get(section, key.upper(), fallback=fallback)
                    except (configparser.NoSectionError, configparser.NoOptionError):
                        return fallback
            
        result = {
            'COG_HEADING_MAX_DIFF': get_config_value('Parameters', 'COG_HEADING_MAX_DIFF', fallback=45, value_type='float'),
            'MIN_SPEED_FOR_COG_CHECK': get_config_value('Parameters', 'MIN_SPEED_FOR_COG_CHECK', fallback=10, value_type='float'),
            'SPEED_THRESHOLD': get_config_value('Parameters', 'SPEED_THRESHOLD', fallback=102, value_type='float'),
            'USE_DASK': get_config_value('Processing', 'USE_DASK', fallback=True, value_type='boolean'),
            'USE_GPU': get_config_value('Processing', 'USE_GPU', fallback=GPU_AVAILABLE, value_type='boolean'),
            
            # Get directory paths checking both Paths and DEFAULT sections
            'DATA_DIRECTORY': get_config_value('Paths', 'DATA_DIRECTORY', 
                                   fallback=get_config_value('DEFAULT', 'DATA_DIRECTORY', fallback='data')),
            'OUTPUT_DIRECTORY': get_config_value('Paths', 'OUTPUT_DIRECTORY',
                                     fallback=get_config_value('DEFAULT', 'OUTPUT_DIRECTORY', fallback='output')),
                                     
            'USE_S3': get_config_value('AWS', 'USE_S3', fallback=False, value_type='boolean'),
            'S3_DATA_URI': get_config_value('AWS', 'S3_DATA_URI', fallback=''),
            'TIME_DIFF_THRESHOLD_MIN': get_config_value('Parameters', 'TIME_DIFF_THRESHOLD_MIN', fallback=240, value_type='float'),
            'BEACON_TIME_THRESHOLD_HOURS': get_config_value('ANOMALY_THRESHOLDS', 'BEACON_TIME_THRESHOLD_HOURS', fallback=6, value_type='float'),
            
            # Anomaly types (all now enabled by default)
            'ais_beacon_on': get_config_value('ANOMALY_TYPES', 'ais_beacon_on', fallback=True, value_type='boolean'),
            'ais_beacon_off': get_config_value('ANOMALY_TYPES', 'ais_beacon_off', fallback=True, value_type='boolean'),
            'excessive_travel_distance_fast': get_config_value('ANOMALY_TYPES', 'excessive_travel_distance_fast', fallback=True, value_type='boolean'),
            'cog-heading_inconsistency': get_config_value('ANOMALY_TYPES', 'cog-heading_inconsistency', fallback=True, value_type='boolean'),
            'loitering': get_config_value('ANOMALY_TYPES', 'loitering', fallback=True, value_type='boolean'),
            'rendezvous': get_config_value('ANOMALY_TYPES', 'rendezvous', fallback=True, value_type='boolean'),
            'identity_spoofing': get_config_value('ANOMALY_TYPES', 'identity_spoofing', fallback=True, value_type='boolean'),
            'zone_violations': get_config_value('ANOMALY_TYPES', 'zone_violations', fallback=True, value_type='boolean'),
            
            # New anomaly thresholds
            'LOITERING_RADIUS_NM': get_config_value('ANOMALY_THRESHOLDS', 'LOITERING_RADIUS_NM', fallback=5.0, value_type='float'),
            'LOITERING_DURATION_HOURS': get_config_value('ANOMALY_THRESHOLDS', 'LOITERING_DURATION_HOURS', fallback=24.0, value_type='float'),
            'RENDEZVOUS_PROXIMITY_NM': get_config_value('ANOMALY_THRESHOLDS', 'RENDEZVOUS_PROXIMITY_NM', fallback=0.5, value_type='float'),
            'RENDEZVOUS_DURATION_MINUTES': get_config_value('ANOMALY_THRESHOLDS', 'RENDEZVOUS_DURATION_MINUTES', fallback=30, value_type='int'),
            
            # Date range settings
            'start_date': get_config_value('DEFAULT', 'start_date', fallback=None),
            'end_date': get_config_value('DEFAULT', 'end_date', fallback=None),
            
            # Add AWS section with all credentials
            'AWS': {
                's3_access_key': get_config_value('AWS', 's3_access_key', fallback=''),
                's3_secret_key': get_config_value('AWS', 's3_secret_key', fallback=''),
                's3_session_token': get_config_value('AWS', 's3_session_token', fallback=''),
                's3_auth_method': get_config_value('AWS', 's3_auth_method', fallback='keys'),
                's3_region': get_config_value('AWS', 's3_region', fallback='us-east-1'),
                's3_bucket_name': get_config_value('AWS', 's3_bucket_name', fallback=''),
                's3_prefix': get_config_value('AWS', 's3_prefix', fallback=''),
            },
            
            # OUTPUT_CONTROLS settings
            'generate_anomaly_summary': get_config_value('OUTPUT_CONTROLS', 'generate_anomaly_summary', fallback=True, value_type='boolean'),
            'generate_statistics_excel': get_config_value('OUTPUT_CONTROLS', 'generate_statistics_excel', fallback=True, value_type='boolean'),
            'generate_statistics_csv': get_config_value('OUTPUT_CONTROLS', 'generate_statistics_csv', fallback=True, value_type='boolean'),
            'generate_overall_map': get_config_value('OUTPUT_CONTROLS', 'generate_overall_map', fallback=True, value_type='boolean'),
            'generate_vessel_path_maps': get_config_value('OUTPUT_CONTROLS', 'generate_vessel_path_maps', fallback=True, value_type='boolean'),
            'generate_charts': get_config_value('OUTPUT_CONTROLS', 'generate_charts', fallback=True, value_type='boolean'),
            'generate_anomaly_type_chart': get_config_value('OUTPUT_CONTROLS', 'generate_anomaly_type_chart', fallback=True, value_type='boolean'),
            'generate_vessel_anomaly_chart': get_config_value('OUTPUT_CONTROLS', 'generate_vessel_anomaly_chart', fallback=True, value_type='boolean'),
            'generate_date_anomaly_chart': get_config_value('OUTPUT_CONTROLS', 'generate_date_anomaly_chart', fallback=True, value_type='boolean'),
            'filter_to_anomaly_vessels_only': get_config_value('OUTPUT_CONTROLS', 'filter_to_anomaly_vessels_only', fallback=False, value_type='boolean'),
            'show_lat_long_grid': get_config_value('OUTPUT_CONTROLS', 'show_lat_long_grid', fallback=True, value_type='boolean'),
            'show_anomaly_heatmap': get_config_value('OUTPUT_CONTROLS', 'show_anomaly_heatmap', fallback=True, value_type='boolean'),
            
            # LOGGING settings
            'suppress_warnings': get_config_value('LOGGING', 'suppress_warnings', fallback=True, value_type='boolean'),
            
            # ANALYSIS_FILTERS settings
            'min_latitude': get_config_value('ANALYSIS_FILTERS', 'min_latitude', fallback=-90.0, value_type='float'),
            'max_latitude': get_config_value('ANALYSIS_FILTERS', 'max_latitude', fallback=90.0, value_type='float'),
            'min_longitude': get_config_value('ANALYSIS_FILTERS', 'min_longitude', fallback=-180.0, value_type='float'),
            'max_longitude': get_config_value('ANALYSIS_FILTERS', 'max_longitude', fallback=180.0, value_type='float'),
            'time_start_hour': get_config_value('ANALYSIS_FILTERS', 'time_start_hour', fallback=0, value_type='int'),
            'time_end_hour': get_config_value('ANALYSIS_FILTERS', 'time_end_hour', fallback=24, value_type='int'),
            'min_confidence': get_config_value('ANALYSIS_FILTERS', 'min_confidence', fallback=75, value_type='int'),
            'max_anomalies_per_vessel': get_config_value('ANALYSIS_FILTERS', 'max_anomalies_per_vessel', fallback=10, value_type='int')
        }
        
        # Load Zone Violations from config
        restricted_zones = []
        if 'ZONE_VIOLATIONS' in config:
            # Find all zone indices
            zone_indices = set()
            for key in config['ZONE_VIOLATIONS']:
                if key.startswith('zone_') and '_name' in key:
                    try:
                        zone_index = int(key.split('_')[1])
                        zone_indices.add(zone_index)
                    except (ValueError, IndexError):
                        continue
            
            # Load each zone
            import json
            for i in sorted(zone_indices):
                zone_key = f'zone_{i}'
                try:
                    is_selected = config.getboolean('ZONE_VIOLATIONS', f'{zone_key}_is_selected', fallback=True)
                    # Only include selected zones
                    if is_selected:
                        geometry_type = config.get('ZONE_VIOLATIONS', f'{zone_key}_geometry_type', fallback='rectangle')
                        zone = {
                            'name': config.get('ZONE_VIOLATIONS', f'{zone_key}_name'),
                            'geometry_type': geometry_type
                        }
                        
                        # Load coordinates based on geometry type
                        if geometry_type == 'rectangle':
                            # Legacy format for backward compatibility
                            zone['lat_min'] = config.getfloat('ZONE_VIOLATIONS', f'{zone_key}_lat_min')
                            zone['lat_max'] = config.getfloat('ZONE_VIOLATIONS', f'{zone_key}_lat_max')
                            zone['lon_min'] = config.getfloat('ZONE_VIOLATIONS', f'{zone_key}_lon_min')
                            zone['lon_max'] = config.getfloat('ZONE_VIOLATIONS', f'{zone_key}_lon_max')
                        elif geometry_type == 'circle':
                            zone['center_lat'] = config.getfloat('ZONE_VIOLATIONS', f'{zone_key}_center_lat')
                            zone['center_lon'] = config.getfloat('ZONE_VIOLATIONS', f'{zone_key}_center_lon')
                            zone['radius_meters'] = config.getfloat('ZONE_VIOLATIONS', f'{zone_key}_radius_meters')
                        else:  # polygon or polyline
                            coords_str = config.get('ZONE_VIOLATIONS', f'{zone_key}_coordinates', fallback='[]')
                            zone['coordinates'] = json.loads(coords_str)
                            if geometry_type == 'polyline':
                                zone['tolerance_meters'] = config.getfloat('ZONE_VIOLATIONS', f'{zone_key}_tolerance_meters', fallback=100)
                        
                        restricted_zones.append(zone)
                except (configparser.NoOptionError, ValueError, json.JSONDecodeError):
                    continue
        
        # If no zones found, use defaults
        if not restricted_zones:
            restricted_zones = [
                {'name': 'Strait of Hormuz', 'lat_min': 25.0, 'lat_max': 27.0, 'lon_min': 55.0, 'lon_max': 57.5},
                {'name': 'South China Sea', 'lat_min': 5.0, 'lat_max': 25.0, 'lon_min': 105.0, 'lon_max': 120.0},
            ]
        
        result['RESTRICTED_ZONES'] = restricted_zones
        
        # Handle MMSI list filter (special case as it's a comma-separated string)
        try:
            mmsi_list_str = get_config_value('ANALYSIS_FILTERS', 'filter_mmsi_list', fallback='')
            if mmsi_list_str and mmsi_list_str.strip():
                # Parse comma-separated list of MMSIs
                result['filter_mmsi_list'] = [int(mmsi.strip()) for mmsi in mmsi_list_str.split(',') if mmsi.strip()]
            else:
                result['filter_mmsi_list'] = []
        except Exception as e:
            logger.warning(f"Error parsing MMSI filter list: {e}. Using empty list.")
            result['filter_mmsi_list'] = []
        
        # Check if the output directory exists, create if not
        if not os.path.exists(result['OUTPUT_DIRECTORY']):
            os.makedirs(result['OUTPUT_DIRECTORY'], exist_ok=True)
            logger.info(f"Created output directory: {result['OUTPUT_DIRECTORY']}")
        
        # Parse selected ship types
        try:
            # Try to get ship types from different possible sections
            selected_types_str = get_config_value('Parameters', 'SELECTED_SHIP_TYPES', 
                                       fallback=get_config_value('SHIP_FILTERS', 'SELECTED_SHIP_TYPES', fallback=None))
            
            if selected_types_str:
                # Parse comma-separated ship types
                result['SELECTED_SHIP_TYPES'] = [int(t.strip()) for t in selected_types_str.split(',') if t.strip()]
            else:
                # For backward compatibility with older config files
                result['SELECTED_SHIP_TYPES'] = [20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 69, 70, 71, 72, 73, 74, 79, 80, 81, 82, 83, 84, 89, 90, 91, 92, 93, 94]  # All vessel types
        except ValueError:
            logger.warning("Invalid ship type values in config. Using all vessel types.")
            result['SELECTED_SHIP_TYPES'] = [20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 69, 70, 71, 72, 73, 74, 79, 80, 81, 82, 83, 84, 89, 90, 91, 92, 93, 94]  # All vessel types
        except Exception as e:
            logger.warning(f"Error parsing ship types: {e}. Using all vessel types.")
            result['SELECTED_SHIP_TYPES'] = [20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 69, 70, 71, 72, 73, 74, 79, 80, 81, 82, 83, 84, 89, 90, 91, 92, 93, 94]  # All vessel types
        
        # Check if AWS S3 is configured
        if result['USE_S3'] and not result['S3_DATA_URI']:
            logger.warning("USE_S3 is set to True but S3_DATA_URI is not specified. Disabling S3.")
            result['USE_S3'] = False
        
        return result
    
    except configparser.Error as e:
        logger.warning(f"Error parsing config file: {e}. Using default values.")
        return {
            'COG_HEADING_MAX_DIFF': 45,
            'MIN_SPEED_FOR_COG_CHECK': 10,
            'SPEED_THRESHOLD': 102,
            'USE_DASK': True,
            'USE_GPU': GPU_AVAILABLE,
            'DATA_DIRECTORY': 'data',
            'OUTPUT_DIRECTORY': 'C:\\AIS_Data\\Reports',  # Proper Windows path format
            'SELECTED_SHIP_TYPES': [70, 80],
            'USE_S3': False,
            'S3_DATA_URI': '',
            'TIME_DIFF_THRESHOLD_MIN': 240,
            'BEACON_TIME_THRESHOLD_HOURS': 6,  # Default 6 hours for AIS beacon anomaly detection
            'ais_beacon_on': True,
            'ais_beacon_off': True,
            'excessive_travel_distance_fast': True,
            'cog-heading_inconsistency': True,
            'loitering': True,
            'rendezvous': True,
            'identity_spoofing': True,
            'zone_violations': True,
            
            # New anomaly thresholds
            'LOITERING_RADIUS_NM': 5.0,
            'LOITERING_DURATION_HOURS': 24.0,
            'RENDEZVOUS_PROXIMITY_NM': 0.5,
            'RENDEZVOUS_DURATION_MINUTES': 30,
            
            'start_date': '2024-10-15',  # Default start date
            'end_date': '2024-10-17',   # Default end date
            
            # Default OUTPUT_CONTROLS settings
            'generate_anomaly_summary': True,
            'generate_statistics_excel': True,
            'generate_statistics_csv': True,
            'generate_overall_map': True,
            'generate_vessel_path_maps': True,
            'generate_charts': True,
            'generate_anomaly_type_chart': True,
            'generate_vessel_anomaly_chart': True,
            'generate_date_anomaly_chart': True,
            'filter_to_anomaly_vessels_only': False,
            'show_lat_long_grid': True,
            'show_anomaly_heatmap': True,
            
            # Default LOGGING settings
            'suppress_warnings': True,
            
            # Default ANALYSIS_FILTERS settings
            'min_latitude': -90.0,
            'max_latitude': 90.0,
            'min_longitude': -180.0,
            'max_longitude': 180.0,
            'time_start_hour': 0,
            'time_end_hour': 24,
            'min_confidence': 75,
            'max_anomalies_per_vessel': 10,
            'filter_mmsi_list': []
        }


def haversine_distance_nm(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth
    (specified in decimal degrees) in nautical miles.
    
    Args:
        lat1 (float): Latitude of first point
        lon1 (float): Longitude of first point
        lat2 (float): Latitude of second point
        lon2 (float): Longitude of second point
        
    Returns:
        float: Distance in nautical miles
    """
    try:
        # Check for NaN values
        if (pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2)):
            return None
            
        return great_circle((lat1, lon1), (lat2, lon2)).nm
    except Exception as e:
        logger.warning(f"Error calculating haversine distance: {e}")
        return None


def haversine_vectorized(df, use_gpu=None):
    """
    Vectorized implementation of the Haversine formula with GPU support when available.
    Supports NVIDIA CUDA (with cudf), AMD ROCm (with cupy-rocm), and AMD HIP (with PyHIP).
    
    Args:
        df (DataFrame): DataFrame containing LAT1, LON1, LAT2, LON2 columns
        use_gpu (bool, optional): Whether to use GPU acceleration. If None, uses GPU_AVAILABLE.
        
    Returns:
        Series: Distances in nautical miles
    """
    # Earth radius in nautical miles
    r = 3440.1
    
    # Determine if we should use GPU
    # If use_gpu is explicitly False, don't use GPU even if available
    # If use_gpu is None, use GPU if available
    # Can use cupy (CUDA/ROCm) or HIP directly
    should_use_gpu = (use_gpu is not False) and GPU_AVAILABLE and (cp is not None or hip is not None)
    
    # Check if GPU is available and we can use it
    if should_use_gpu:
        # For NVIDIA: check if this is a cuDF DataFrame
        if GPU_TYPE == 'NVIDIA' and cudf is not None and isinstance(df, cudf.DataFrame):
            # Use cuDF with cupy (CUDA)
            lat1_rad = cp.radians(df['LAT1'].values)
            lon1_rad = cp.radians(df['LON1'].values)
            lat2_rad = cp.radians(df['LAT2'].values)
            lon2_rad = cp.radians(df['LON2'].values)
        elif GPU_TYPE == 'AMD' and cp is not None:
            # For AMD ROCm/HIP via cupy-rocm, use cupy with pandas DataFrame
            # Convert pandas DataFrame columns to cupy arrays
            lat1_rad = cp.radians(cp.asarray(df['LAT1'].values))
            lon1_rad = cp.radians(cp.asarray(df['LON1'].values))
            lat2_rad = cp.radians(cp.asarray(df['LAT2'].values))
            lon2_rad = cp.radians(cp.asarray(df['LON2'].values))
        elif GPU_TYPE == 'AMD' and hip is not None and cp is None:
            # Direct HIP support (PyHIP) - use numpy arrays and convert
            # Note: HIP is lower-level, so we'll use numpy for computation
            # In practice, cupy-rocm is preferred for HIP access
            lat1_rad = np.radians(df['LAT1'].values)
            lon1_rad = np.radians(df['LON1'].values)
            lat2_rad = np.radians(df['LAT2'].values)
            lon2_rad = np.radians(df['LON2'].values)
            # For now, fall back to CPU computation with HIP detected
            # Future: could add direct HIP kernel calls if needed
        elif GPU_TYPE == 'NVIDIA' and (cudf is None or not isinstance(df, cudf.DataFrame)) and cp is not None:
            # NVIDIA without cudf, use cupy with pandas DataFrame
            lat1_rad = cp.radians(cp.asarray(df['LAT1'].values))
            lon1_rad = cp.radians(cp.asarray(df['LON1'].values))
            lat2_rad = cp.radians(cp.asarray(df['LAT2'].values))
            lon2_rad = cp.radians(cp.asarray(df['LON2'].values))
        else:
            # Fall back to CPU
            lat1_rad = np.radians(df['LAT1'])
            lon1_rad = np.radians(df['LON1'])
            lat2_rad = np.radians(df['LAT2'])
            lon2_rad = np.radians(df['LON2'])
            
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            
            a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            # Return as pandas Series to match expected return type
            return pd.Series(c * r, index=df.index)
        
        # Haversine formula with cupy (works for both CUDA and ROCm/HIP)
        if cp is not None:
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            
            a = cp.sin(dlat/2)**2 + cp.cos(lat1_rad) * cp.cos(lat2_rad) * cp.sin(dlon/2)**2
            c = 2 * cp.arcsin(cp.sqrt(a))
            
            # Convert back to numpy array for return
            result = cp.asnumpy(c * r)
            return pd.Series(result, index=df.index)
        else:
            # HIP detected but no cupy - fall back to CPU computation
            # (HIP is lower-level and would require custom kernels)
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            
            a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            # Return as pandas Series to match expected return type
            return pd.Series(c * r, index=df.index)
    else:
        # CPU implementation with numpy
        lat1_rad = np.radians(df['LAT1'])
        lon1_rad = np.radians(df['LON1'])
        lat2_rad = np.radians(df['LAT2'])
        lon2_rad = np.radians(df['LON2'])
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Return as pandas Series to match expected return type
        return pd.Series(c * r, index=df.index)


# def get_cache_dir():
#     """
#     Get the directory to use for caching data.
#     Creates the directory if it doesn't exist.
    
#     Returns:
#         str: Path to the cache directory
#     """
#     # Get user's home directory
#     home_dir = os.path.expanduser("~")
#     cache_dir = os.path.join(home_dir, ".ais_data_cache")
    
#     # Create cache directory if it doesn't exist
#     if not os.path.exists(cache_dir):
#         try:
#             os.makedirs(cache_dir)
#             logger.info(f"Created cache directory at {cache_dir}")
#         except Exception as e:
#             logger.error(f"Failed to create cache directory: {e}")
#             return None
    
#     return cache_dir


# Function generate_cache_key is now imported from utils.py
# See top of file for import statement


def check_cached_data(file_path, config):
    """
    Check if data for a file path is already cached.
    
    Args:
        file_path (str): Path to the data file to check
        config (dict): Configuration dictionary
    
    Returns:
        tuple: (DataFrame or None, cache_path or None)
    """
    # Skip cache if disabled
    if config.get('DISABLE_CACHE', False):
        logger.debug("Cache disabled by configuration")
        return None, None
    
    # Get the base cache directory
    base_cache_dir = get_cache_dir()
    if not base_cache_dir:
        return None, None
    
    # Create date-specific subfolder path if dates are available
    start_date = config.get('START_DATE', '')
    end_date = config.get('END_DATE', '')
    
    if start_date and end_date:
        # Format the dates for folder name (YYYYMMDD-YYYYMMDD)
        try:
            start_fmt = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')
            end_fmt = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d')
            date_subfolder = f"{start_fmt}-{end_fmt}"
            
            # Create subfolder for the date range
            cache_dir = os.path.join(base_cache_dir, date_subfolder)
            if not os.path.exists(cache_dir):
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    logger.info(f"Created cache subfolder: {cache_dir}")
                except Exception as e:
                    logger.warning(f"Failed to create cache subfolder: {e}")
                    cache_dir = base_cache_dir  # Fall back to base directory
        except Exception as e:
            logger.warning(f"Error formatting dates for cache subfolder: {e}")
            cache_dir = base_cache_dir
    else:
        cache_dir = base_cache_dir
    
    cache_key = generate_cache_key(file_path, config)
    cache_path = os.path.join(cache_dir, f"{cache_key}.parquet")
    
    # Check if the cache file exists
    if os.path.exists(cache_path):
        try:
            logger.info(f"CACHE: Using cached data for {os.path.basename(file_path)}")
            if config.get('USE_DASK', True):
                df = dd.read_parquet(cache_path).compute()
            else:
                df = pd.read_parquet(cache_path)
            return df, cache_path
        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}")
    
    return None, cache_path


def save_to_cache(df, cache_path):
    """
    Save processed data to cache.
    
    Args:
        df (DataFrame): The processed DataFrame to cache
        cache_path (str): Path where the cached data should be saved
    
    Returns:
        bool: True if successful, False otherwise
    """
    if df is None or df.empty or cache_path is None:
        return False
    
    try:
        # Create a temporary file then rename to avoid partial writes
        temp_path = cache_path + ".tmp"
        df.to_parquet(temp_path, index=False)
        shutil.move(temp_path, cache_path)
        logger.info(f"CACHE: Data saved to cache: {os.path.basename(cache_path)}")
        return True
    except Exception as e:
        logger.warning(f"Failed to save data to cache: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False


def load_and_preprocess_day(file_path, config, use_dask=True):
    """
    Load a single daily CSV or Parquet file, handle errors, and perform initial preprocessing.
    Checks for cached data first before loading from original source.
    
    Args:
        file_path (str): Path to the CSV or Parquet file
        config (dict): Configuration dictionary
        use_dask (bool, optional): Whether to use Dask for processing. Defaults to True.
        
    Returns:
        DataFrame: Preprocessed DataFrame or None if errors occurred
    """
    # Check for cached data first
    df, cache_path = check_cached_data(file_path, config)
    if df is not None:
        return df
    try:
        logger.info(f"Loading data from {file_path}")
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Try using GPU first if enabled (only for NVIDIA with cudf)
        # AMD ROCm doesn't have cudf, so we'll use CPU/Dask for loading but GPU for computations
        if config.get('USE_GPU', False) and GPU_AVAILABLE and GPU_TYPE == 'NVIDIA' and cudf is not None:
            try:
                logger.info("Using GPU acceleration for data loading (NVIDIA CUDA)")
                if file_ext == '.csv':
                    df = cudf.read_csv(file_path, dtype={'MMSI': 'float64'})
                elif file_ext == '.parquet':
                    df = cudf.read_parquet(file_path)
                else:
                    logger.error(f"Unsupported file extension: {file_ext}")
                    return None
                
                # If we successfully loaded with GPU, return the dataframe
                logger.info("Successfully loaded data with GPU acceleration")
                return df
            except Exception as e:
                logger.warning(f"Error loading file with GPU: {e}. Falling back to CPU processing.")
        elif config.get('USE_GPU', False) and GPU_AVAILABLE and GPU_TYPE == 'AMD':
            logger.info("AMD GPU detected - using CPU/Dask for data loading, GPU for computations")
        
        if use_dask and config.get('USE_DASK', True):
            try:
                if file_ext == '.csv':
                    df = dd.read_csv(file_path, dtype={'MMSI': 'float64'})
                elif file_ext == '.parquet':
                    df = dd.read_parquet(file_path)
                else:
                    logger.error(f"Unsupported file extension: {file_ext}")
                    return None
                    
                # Convert to pandas DataFrame
                df = df.compute()
            except Exception as e:
                logger.warning(f"Error loading file with Dask: {e}. Falling back to pandas.")
                use_dask = False
                
        # If not using Dask or Dask failed, use pandas
        if not use_dask or not config.get('USE_DASK', True):
            if file_ext == '.csv':
                df = pd.read_csv(file_path, dtype={'MMSI': 'float64'})
            elif file_ext == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                logger.error(f"Unsupported file extension: {file_ext}")
                return None
                
        # Check for required columns
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None
            
        # Convert BaseDateTime to datetime
        try:
            df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
        except Exception as e:
            logger.error(f"Error converting BaseDateTime: {e}")
            return None
            
        # Filter by ship type if specified
        selected_types = config.get('SELECTED_SHIP_TYPES', [])
        if selected_types:
            if 'VesselType' in df.columns:
                # Check if selected_types contains any types in the 30-39 or 50-59 ranges
                # Types 30-39 and 50-59 should always use exact matching (not main types grouping)
                has_types_30_39 = any(30 <= t <= 39 for t in selected_types)
                has_types_50_59 = any(50 <= t <= 59 for t in selected_types)
                requires_exact_matching = has_types_30_39 or has_types_50_59
                
                # Check if selected_types contains 2-digit main types (e.g., 70)
                # or 3-digit specific types (e.g., 701, 702, etc.)
                # But exclude 30-39 and 50-59 from main types grouping
                has_main_types = any(t < 100 and not (30 <= t <= 39) and not (50 <= t <= 59) for t in selected_types)
                
                # Convert VesselType to numeric before operations
                try:
                    # First convert VesselType to numeric
                    df['VesselType'] = pd.to_numeric(df['VesselType'], errors='coerce')
                    logger.info(f"VesselType conversion done, found {df['VesselType'].notnull().sum()} valid vessel types out of {len(df)} records")
                    
                    # Filter out rows with null VesselType after conversion
                    original_len = len(df)
                    df = df.dropna(subset=['VesselType'])
                    logger.info(f"After dropna: {len(df)} records remaining from {original_len}")
                    
                    if requires_exact_matching:
                        # For types 30-39 and 50-59, always use exact matching
                        range_desc = []
                        if has_types_30_39:
                            range_desc.append("30-39")
                        if has_types_50_59:
                            range_desc.append("50-59")
                        logger.info(f"Filtering for specific vessel types ({', '.join(range_desc)} ranges require exact matching): {selected_types}")
                        before_filter_count = len(df)
                        df = df[df['VesselType'].isin(selected_types)]
                        logger.info(f"After filtering: {len(df)} records match exact vessel types {selected_types} (was {before_filter_count})")
                    elif has_main_types:
                        # Extract the main vessel type by integer division by 10
                        df['MainVesselType'] = (df['VesselType'] // 10).astype(int) * 10
                        logger.info(f"Filtering for main vessel types: {selected_types}")
                        
                        # Save the count before filtering
                        before_filter_count = len(df)
                        df = df[df['MainVesselType'].isin(selected_types)]
                        logger.info(f"After filtering: {len(df)} records match vessel types {selected_types} (was {before_filter_count})")
                    else:
                        # Use specific vessel types (for types >= 100 or other specific types)
                        logger.info(f"Filtering for specific vessel types: {selected_types}")
                        
                        # Save the count before filtering
                        before_filter_count = len(df)
                        df = df[df['VesselType'].isin(selected_types)]
                        logger.info(f"After filtering: {len(df)} records match vessel types {selected_types} (was {before_filter_count})")
                        
                    # Check if DataFrame is empty after filtering
                    if df.empty:
                        logger.warning(f"DataFrame is empty after vessel type filtering - no matching vessels found")
                        return df  # Return empty DataFrame instead of None
                except Exception as e:
                    logger.error(f"Error converting or filtering VesselType: {e}")
                    # Continue without filtering if conversion fails
                    pass
            else:
                logger.warning("VesselType column not found, skipping vessel type filtering")
                
        # Convert coordinate and speed columns to numeric if needed
        for col in ['LAT', 'LON', 'SOG', 'COG', 'Heading']:
            if col in df.columns and df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Basic data cleaning
        # Remove rows with missing position data
        df = df.dropna(subset=['LAT', 'LON'])
        
        # Remove invalid coordinates
        df = df[(df['LAT'] >= -90) & (df['LAT'] <= 90) & 
                (df['LON'] >= -180) & (df['LON'] <= 180)]
                
        # Remove unrealistic speeds if configured
        speed_threshold = config.get('SPEED_THRESHOLD', 102)  # Default to 102 knots (max realistic speed)
        if 'SOG' in df.columns:
            df = df[df['SOG'] <= speed_threshold]
        
        # Save successfully loaded data to cache before returning
        if not df.empty and cache_path:
            save_to_cache(df, cache_path)
            
        # Return the DataFrame (empty or not)
        return df
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()  # Return empty DataFrame instead of None


def create_map_visualization(anomalies_df, output_path, config=None):
    """
    Create an interactive map visualization of detected anomalies.
    
    Args:
        anomalies_df (DataFrame): DataFrame containing anomalies
        output_path (str): Path to save the map HTML file
        config (dict, optional): Configuration dictionary with output controls
    """
    global map_manager
    # Check if map generation is enabled in config
    if isinstance(config, dict) and not config.get('generate_overall_map', True):
        logger.info("Map visualization is disabled in configuration")
        return
        
    if anomalies_df.empty:
        logger.warning("No anomalies to visualize")
        return
    
    try:
        # Create a map centered on the mean of anomaly coordinates
        m = folium.Map(location=[anomalies_df['LAT'].mean(), anomalies_df['LON'].mean()],
                      zoom_start=5)
        
        # Check if grid lines should be shown based on config
        if config is None or config.get('show_lat_long_grid', True):
            # Use global boundaries if available, otherwise calculate from current data
            if map_manager.is_valid():
                # Use pre-calculated global boundaries
                min_lat, max_lat, min_lon, max_lon = map_manager.get_boundaries()
                logger.info("Using global boundaries for map")
            else:
                # Calculate from current data as fallback
                min_lat = anomalies_df['LAT'].min() - 5
                max_lat = anomalies_df['LAT'].max() + 5
                min_lon = anomalies_df['LON'].min() - 5
                max_lon = anomalies_df['LON'].max() + 5
                logger.info("Using local boundaries for map (global not available)")
            
            # Add latitude and longitude grid lines
            add_lat_lon_grid_lines(m, 
                                  lat_start=min_lat, 
                                  lat_end=max_lat, 
                                  lon_start=min_lon, 
                                  lon_end=max_lon, 
                                  lat_step=10, 
                                  lon_step=10, 
                                  label_step=10)
        else:
            logger.info("Latitude/longitude grid lines disabled by configuration")
        
        ''' # Add heatmap layers if enabled in config
        if config is None or config.get('show_no_anomaly_vessels_heatmap', True) or config.get('show_anomaly_heatmap', True):
            try:
                # Get all data from global scope (using function parameter would be better design)
                all_data_exists = 'all_daily_data' in globals() or 'all_daily_data' in locals()
                if all_data_exists:
                    # Get all data
                    if 'all_daily_data' in globals():
                        all_data = pd.concat(globals()['all_daily_data'].values())
                    else:
                        all_data = pd.concat(all_daily_data.values())
                    
                    # Get anomaly vessels MMSIs
                    anomaly_vessels = set(anomalies_df['MMSI'].unique())
                    
                    # Split data
                    anomaly_data = all_data[all_data['MMSI'].isin(anomaly_vessels)]
                    normal_data = all_data[~all_data['MMSI'].isin(anomaly_vessels)]
                    
                    # Add normal vessels heatmap if enabled
                    if config.get('show_no_anomaly_vessels_heatmap', True) and not normal_data.empty:
                        # Vectorized approach for heatmap data
                        valid_mask = normal_data['LAT'].notna() & normal_data['LON'].notna()
                        valid_normal = normal_data[valid_mask]
                        if not valid_normal.empty:
                            normal_heatmap_data = [[lat, lon, 10] for lat, lon in 
                                                   zip(valid_normal['LAT'].values, valid_normal['LON'].values)]
                        else:
                            normal_heatmap_data = []
                        
                        if normal_heatmap_data:
                            # Green colorscale for non-anomaly vessels
                            normal_colormap = ['#edf8e9', '#c7e9c0', '#a1d99b', '#74c476', '#31a354', '#006d2c']
                            
                            # Add normal vessels heatmap layer
                            HeatMap(
                                normal_heatmap_data,
                                name='Normal Vessels Heatmap',
                                radius=15,
                                gradient=normal_colormap,
                                blur=13,
                                min_opacity=0.3,
                                max_opacity=0.6,
                                overlay=True
                            ).add_to(m)
                            logger.info("Added normal vessels heatmap layer to map")
                    
                    # Add anomaly vessels heatmap if enabled
                    if config.get('show_anomaly_heatmap', True) and not anomaly_data.empty:
                        # Vectorized approach for heatmap data
                        valid_mask = anomaly_data['LAT'].notna() & anomaly_data['LON'].notna()
                        valid_anomaly = anomaly_data[valid_mask]
                        if not valid_anomaly.empty:
                            anomaly_heatmap_data = [[lat, lon, 10] for lat, lon in 
                                                   zip(valid_anomaly['LAT'].values, valid_anomaly['LON'].values)]
                        else:
                            anomaly_heatmap_data = []
                        
                        if anomaly_heatmap_data:
                            # Red colorscale for anomalies
                            anomaly_colormap = ['#fee8c8', '#fdbb84', '#fc8d59', '#e34a33', '#b30000']
                            
                            # Add anomaly vessels heatmap layer
                            HeatMap(
                                anomaly_heatmap_data,
                                name='Anomaly Vessels Heatmap',
                                radius=15,
                                gradient=anomaly_colormap,
                                blur=13,
                                min_opacity=0.3,
                                max_opacity=0.6,
                                overlay=True
                            ).add_to(m)
                            logger.info("Added anomaly vessels heatmap layer to map")
                    
                    # Add layer control to toggle layers
                    folium.LayerControl().add_to(m)
                else:
                    logger.warning("Could not access vessel data for heatmap generation")
            except Exception as e:
                logger.error(f"Error adding heatmaps to map: {e}") '''
        
        # Create a marker cluster for better visualization
        marker_cluster = MarkerCluster().add_to(m)
        
        # Vectorized approach: prepare data first, then create markers
        # Filter valid coordinates
        valid_mask = anomalies_df['LAT'].notna() & anomalies_df['LON'].notna()
        valid_anomalies = anomalies_df[valid_mask].copy()
        
        if not valid_anomalies.empty:
            # Vectorized data extraction
            locations = list(zip(valid_anomalies['LAT'].values, valid_anomalies['LON'].values))
            mmsi_values = valid_anomalies['MMSI'].values
            vessel_names = valid_anomalies['VesselName'].values
            vessel_types = valid_anomalies['VesselType'].values
            base_datetimes = valid_anomalies['BaseDateTime'].values
            anomaly_types = valid_anomalies['AnomalyType'].values
            
            # Pre-compute which columns exist and are not NaN
            has_distance = 'Distance' in valid_anomalies.columns
            has_timediff = 'TimeDiff' in valid_anomalies.columns
            has_speed_anomaly = 'SpeedAnomaly' in valid_anomalies.columns
            has_course_anomaly = 'CourseAnomaly' in valid_anomalies.columns
            has_beacon_gap = 'BeaconGapMinutes' in valid_anomalies.columns
            
            if has_distance:
                distances = valid_anomalies['Distance'].values
            if has_timediff:
                timediffs = valid_anomalies['TimeDiff'].values
            if has_speed_anomaly:
                speed_anomalies = valid_anomalies['SpeedAnomaly'].values
                sogs = valid_anomalies['SOG'].values if 'SOG' in valid_anomalies.columns else None
            if has_course_anomaly:
                course_anomalies = valid_anomalies['CourseAnomaly'].values
                cogs = valid_anomalies['COG'].values if 'COG' in valid_anomalies.columns else None
                headings = valid_anomalies['Heading'].values if 'Heading' in valid_anomalies.columns else None
                course_heading_diffs = valid_anomalies['CourseHeadingDiff'].values if 'CourseHeadingDiff' in valid_anomalies.columns else None
            if has_beacon_gap:
                beacon_gaps = valid_anomalies['BeaconGapMinutes'].values
            
            # Create markers using vectorized data
            for i, (lat, lon) in enumerate(locations):
                # Create popup text with anomaly details
                popup_text = f"<b>MMSI:</b> {mmsi_values[i]}<br>"
                popup_text += f"<b>Vessel Name:</b> {vessel_names[i]}<br>"
                popup_text += f"<b>Vessel Type:</b> {vessel_types[i]}<br>"
                popup_text += f"<b>Date:</b> {base_datetimes[i]}<br>"
                popup_text += f"<b>Anomaly Type:</b> {anomaly_types[i]}<br>"
                
                # Add additional details based on anomaly type
                if has_distance and not pd.isna(distances[i]):
                    popup_text += f"<b>Distance (nm):</b> {distances[i]:.2f}<br>"
                if has_timediff and not pd.isna(timediffs[i]):
                    popup_text += f"<b>Time Diff (min):</b> {timediffs[i]:.1f}<br>"
                if has_speed_anomaly and speed_anomalies[i] and sogs is not None:
                    popup_text += f"<b>Speed (knots):</b> {sogs[i]:.1f}<br>"
                if has_course_anomaly and course_anomalies[i]:
                    if cogs is not None:
                        popup_text += f"<b>COG:</b> {cogs[i]:.1f} deg<br>"
                    if headings is not None:
                        popup_text += f"<b>Heading:</b> {headings[i]:.1f} deg<br>"
                    if course_heading_diffs is not None:
                        popup_text += f"<b>Difference:</b> {course_heading_diffs[i]:.1f} deg<br>"
                
                # Choose icon color based on anomaly type
                anomaly_type = anomaly_types[i]
                if anomaly_type == 'Speed':
                    icon_color = 'red'
                elif anomaly_type == 'Course':
                    icon_color = 'Yellow'
                elif anomaly_type == 'Position':
                    icon_color = 'Brown'
                elif anomaly_type == 'AIS_Beacon_On':
                    icon_color = 'orange'
                    if has_beacon_gap and not pd.isna(beacon_gaps[i]):
                        popup_text += f"<b>Gap (minutes):</b> {beacon_gaps[i]:.1f}<br>"
                elif anomaly_type == 'AIS_Beacon_Off':
                    icon_color = 'purple'
                    if has_beacon_gap and not pd.isna(beacon_gaps[i]):
                        popup_text += f"<b>Gap (minutes):</b> {beacon_gaps[i]:.1f}<br>"
                else:
                    icon_color = 'gray'
                
                # Add marker to the cluster
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_text, max_width=300),
                    icon=folium.Icon(color=icon_color, icon='info-sign')
                ).add_to(marker_cluster)
        
        # Save map to HTML file
        m.save(output_path)
        logger.info(f"Map visualization saved to {output_path}")
    
    except Exception as e:
        logger.error(f"Error creating map visualization: {e}")


def create_summary_charts(anomalies_df, output_dir, config=None):
    """
    Create summary charts for anomaly analysis.
    
    Args:
        anomalies_df (DataFrame): DataFrame containing anomalies
        output_dir (str): Directory to save the chart images
        config (dict, optional): Configuration dictionary with output controls
    """
    if anomalies_df.empty:
        logger.warning("Debug: No anomalies to visualize in charts")
        return
    
    # Default to generating all charts if no config is provided
    generate_charts = True
    generate_anomaly_type_chart = True
    generate_vessel_anomaly_chart = True
    generate_date_anomaly_chart = True
    
    # Check config settings if provided
    if config is not None:
        generate_charts = config.get('generate_charts', True)
        generate_anomaly_type_chart = config.get('generate_anomaly_type_chart', True) 
        generate_vessel_anomaly_chart = config.get('generate_vessel_anomaly_chart', True)
        generate_date_anomaly_chart = config.get('generate_date_anomaly_chart', True)
    
    # If master chart generation is turned off, return early
    if not generate_charts:
        logger.info("Chart generation disabled in configuration")
        return
    
    try:
        # Create output directory if it doesn't exist
        logger.info(f"Debug: Creating charts directory at: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Debug: Charts directory created or exists: {output_dir}")
        
        # 1. Distribution of anomaly types
        if generate_anomaly_type_chart:
            chart1_path = os.path.join(output_dir, 'anomaly_types_distribution.png')
            logger.info(f"Debug: Creating anomaly types distribution chart at: {chart1_path}")
            plt.figure(figsize=(20, 6))
            sns.countplot(x='AnomalyType', data=anomalies_df)
            plt.title('Distribution of Anomaly Types')
            plt.xlabel('Anomaly Type')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(chart1_path)
            plt.close()
            logger.info(f"Debug: Saved anomaly types chart to: {chart1_path}")
        
        # 2. MMSI counts (top 20)
        if generate_vessel_anomaly_chart:
            plt.figure(figsize=(12, 8))
            mmsi_counts = anomalies_df['MMSI'].value_counts().head(20)
            mmsi_counts.plot(kind='bar')
            plt.title('Top 20 Vessels with Most Anomalies')
            plt.xlabel('MMSI')
            plt.ylabel('Number of Anomalies')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'top_vessels_with_anomalies.png'))
            plt.close()
        
        # 3. Anomalies by date (if multiple dates are present)
        if generate_date_anomaly_chart:
            if 'Date' not in anomalies_df.columns and 'BaseDateTime' in anomalies_df.columns:
                anomalies_df['Date'] = anomalies_df['BaseDateTime'].dt.date
            
            if 'Date' in anomalies_df.columns:
                plt.figure(figsize=(10, 6))
                sns.countplot(x='Date', data=anomalies_df, order=sorted(anomalies_df['Date'].unique()))
                plt.title('Anomalies by Date')
                plt.xlabel('Date')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'anomalies_by_date.png'))
                plt.close()
        
        # 4. 3D Bar Chart: Anomalies by type (x-axis), count (y-axis), by day (z-axis)
        if generate_anomaly_type_chart and 'AnomalyType' in anomalies_df.columns and 'Date' in anomalies_df.columns:
            try:
                from mpl_toolkits.mplot3d import Axes3D
                
                # Prepare data for 3D bar chart
                if 'Date' not in anomalies_df.columns and 'BaseDateTime' in anomalies_df.columns:
                    anomalies_df['Date'] = anomalies_df['BaseDateTime'].dt.date
                
                # Get unique anomaly types and dates
                unique_types = sorted(anomalies_df['AnomalyType'].unique())
                unique_dates = sorted(anomalies_df['Date'].unique())
                
                # Create pivot table: rows = dates, columns = anomaly types
                pivot_data = anomalies_df.groupby(['Date', 'AnomalyType']).size().unstack(fill_value=0)
                
                # Ensure all types are in the pivot table
                for anomaly_type in unique_types:
                    if anomaly_type not in pivot_data.columns:
                        pivot_data[anomaly_type] = 0
                
                # Reorder columns to match unique_types
                pivot_data = pivot_data.reindex(columns=unique_types, fill_value=0)
                
                # Create 3D bar chart
                fig = plt.figure(figsize=(16, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                # Prepare data for 3D plotting
                x_pos = np.arange(len(unique_types))  # Anomaly types on x-axis
                y_pos = np.arange(len(unique_dates))   # Dates on y-axis (z-axis in 3D)
                
                # Create meshgrid for positioning
                xpos, ypos = np.meshgrid(x_pos, y_pos)
                xpos = xpos.flatten()
                ypos = ypos.flatten()
                zpos = np.zeros(len(xpos))
                
                # Get heights (counts) for each bar
                dx = dy = 0.8  # Bar width and depth
                dz = []
                colors_list = []
                
                # Color map for different anomaly types
                color_map = plt.cm.get_cmap('tab10')
                
                for i, date in enumerate(unique_dates):
                    for j, anomaly_type in enumerate(unique_types):
                        count = pivot_data.loc[date, anomaly_type] if date in pivot_data.index else 0
                        dz.append(count)
                        colors_list.append(color_map(j % 10))
                
                # Create 3D bar chart
                ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors_list, alpha=0.8, shade=True)
                
                # Set labels and title
                ax.set_xlabel('Anomaly Type', labelpad=10)
                ax.set_ylabel('Day', labelpad=10)
                ax.set_zlabel('Count', labelpad=10)
                ax.set_title('3D Bar Chart: Anomalies by Type and Day', pad=20)
                
                # Set x-axis ticks and labels
                ax.set_xticks(x_pos)
                ax.set_xticklabels(unique_types, rotation=45, ha='right')
                
                # Set y-axis ticks and labels (dates)
                ax.set_yticks(y_pos)
                ax.set_yticklabels([str(d) for d in unique_dates], rotation=0)
                
                plt.tight_layout()
                chart_3d_path = os.path.join(output_dir, 'anomalies_3d_bar_chart.png')
                plt.savefig(chart_3d_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"3D bar chart saved to {chart_3d_path}")
            except ImportError:
                logger.warning("mpl_toolkits.mplot3d not available, skipping 3D bar chart")
            except Exception as e:
                logger.error(f"Error creating 3D bar chart: {e}")
        
        # 5. Interactive Scatter Plot: Anomalies by day and "all"
        if generate_anomaly_type_chart and 'AnomalyType' in anomalies_df.columns:
            try:
                import plotly.graph_objects as go  # type: ignore
                import plotly.express as px  # type: ignore
                from plotly.subplots import make_subplots  # type: ignore
                
                # Prepare data for scatter plot
                scatter_data = anomalies_df.copy()
                
                # Add Date column if not present
                if 'Date' not in scatter_data.columns:
                    if 'BaseDateTime' in scatter_data.columns:
                        scatter_data['Date'] = scatter_data['BaseDateTime'].dt.date
                    elif 'ReportDate' in scatter_data.columns:
                        scatter_data['Date'] = pd.to_datetime(scatter_data['ReportDate']).dt.date
                    else:
                        scatter_data['Date'] = pd.Timestamp('today').date()
                
                # Create scatter plot with Plotly
                fig = go.Figure()
                
                # Get unique dates and anomaly types
                unique_dates = sorted(scatter_data['Date'].unique())
                unique_types = sorted(scatter_data['AnomalyType'].unique())
                
                # Color map for anomaly types
                color_map = px.colors.qualitative.Set3
                
                # Add trace for "All" anomalies (aggregated)
                all_counts = scatter_data.groupby('AnomalyType').size()
                fig.add_trace(go.Scatter(
                    x=all_counts.index,
                    y=all_counts.values,
                    mode='markers+lines',
                    name='All Days',
                    marker=dict(size=12, color='red', symbol='circle'),
                    line=dict(width=2, color='red')
                ))
                
                # Add trace for each day
                for i, date in enumerate(unique_dates):
                    date_data = scatter_data[scatter_data['Date'] == date]
                    date_counts = date_data.groupby('AnomalyType').size()
                    
                    # Ensure all types are represented (fill missing with 0)
                    date_counts = date_counts.reindex(unique_types, fill_value=0)
                    
                    fig.add_trace(go.Scatter(
                        x=date_counts.index,
                        y=date_counts.values,
                        mode='markers+lines',
                        name=f'Day {str(date)}',
                        marker=dict(size=10, color=color_map[i % len(color_map)], symbol='circle'),
                        line=dict(width=1.5, color=color_map[i % len(color_map)], dash='dot')
                    ))
                
                # Update layout
                fig.update_layout(
                    title='Interactive Scatter Plot: Anomalies by Type and Day',
                    xaxis_title='Anomaly Type',
                    yaxis_title='Count',
                    hovermode='closest',
                    width=1200,
                    height=700,
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02
                    )
                )
                
                # Save as HTML
                scatter_plot_path = os.path.join(output_dir, 'Anomalies_Scatterplot.html')
                fig.write_html(scatter_plot_path)
                logger.info(f"Interactive scatter plot saved to {scatter_plot_path}")
            except ImportError:
                logger.warning("plotly not available, skipping interactive scatter plot. Install with: pip install plotly")
            except Exception as e:
                logger.error(f"Error creating interactive scatter plot: {e}")
        
        logger.info(f"Summary charts saved to {output_dir}")
    
    except Exception as e:
        logger.error(f"Error creating summary charts: {e}")


def save_concatenated_dataframe(all_daily_data, config):
    """
    Save the concatenated all_daily_data dataframe to the cache directory for future use.
    If a previous concatenated file exists, it will merge with it to include all data.
    
    Args:
        all_daily_data (dict): Dictionary with date keys and DataFrame values for each day's data
        config (dict): Configuration parameters
        
    Returns:
        str: Path to the saved concatenated dataframe file
    """
    try:
        # Get the base cache directory
        base_cache_dir = get_cache_dir()
        if not base_cache_dir:
            logger.error("Cannot determine cache directory")
            return None
            
        # Create a consolidated dataframe from all daily data
        if not all_daily_data:
            logger.warning("No daily data to consolidate")
            return None
            
        consolidated_df = pd.concat(all_daily_data.values())
        logger.info(f"Created consolidated dataframe with {len(consolidated_df)} records")
        
        # Determine if we should use a date-specific subfolder
        start_date = config.get('START_DATE', '')
        end_date = config.get('END_DATE', '')
        
        if start_date and end_date:
            try:
                # Format dates for folder name
                start_fmt = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')
                end_fmt = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d')
                date_subfolder = f"{start_fmt}-{end_fmt}"
                logger.info(f"Date subfolder: {date_subfolder}")

                # Use the date subfolder
                cache_dir = os.path.join(base_cache_dir, date_subfolder)
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"Error creating date subfolder: {e}, using base cache directory")
                cache_dir = base_cache_dir
        else:
            cache_dir = base_cache_dir
        
        # Define the filename for the consolidated dataframe
        consolidated_filename = "consolidated_data.parquet"
        consolidated_path = os.path.join(cache_dir, consolidated_filename)
        
        # Check if a previous consolidated file exists and merge with it
        if os.path.exists(consolidated_path):
            try:
                logger.info(f"Previous consolidated file found, merging with new data")
                previous_df = pd.read_parquet(consolidated_path)
                
                # Merge based on unique MMSI and timestamp combinations to avoid duplicates
                # First ensure all timestamps are in datetime format
                if 'BaseDateTime' in previous_df.columns:
                    previous_df['BaseDateTime'] = pd.to_datetime(previous_df['BaseDateTime'])
                if 'BaseDateTime' in consolidated_df.columns:
                    consolidated_df['BaseDateTime'] = pd.to_datetime(consolidated_df['BaseDateTime'])
                
                # Concatenate the dataframes
                merged_df = pd.concat([previous_df, consolidated_df])
                
                # Remove duplicate rows based on MMSI and BaseDateTime
                if 'MMSI' in merged_df.columns and 'BaseDateTime' in merged_df.columns:
                    merged_df = merged_df.drop_duplicates(subset=['MMSI', 'BaseDateTime'])
                    logger.info(f"Merged with previous data, now have {len(merged_df)} records (removed duplicates)")
                    consolidated_df = merged_df
                else:
                    logger.warning("Cannot deduplicate - missing MMSI or BaseDateTime columns")
            except Exception as e:
                logger.warning(f"Error merging with previous consolidated data: {e}")
        
        # Save the consolidated dataframe
        consolidated_df.to_parquet(consolidated_path, index=False)
        logger.info(f"Saved consolidated dataframe with {len(consolidated_df)} records to {consolidated_path}")
        
        return consolidated_path
        
    except Exception as e:
        logger.error(f"Error saving consolidated dataframe: {e}")
        logger.error(traceback.format_exc())
        return None


def show_statistics_completion_notification():
    """
    Shows a notification when the statistics generation is complete.
    """
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        root.attributes('-topmost', True)
        messagebox.showinfo("Notification", "Analysis Statistics Report Complete")
        root.destroy()
    except Exception as e:
        logger.error(f"Failed to show completion notification: {str(e)}")

def run_statistics_in_background(all_daily_data, selected_dates, config, output_dir):
    """
    Run the statistics generation in a background thread.
    
    Args:
        all_daily_data (dict): Dictionary with date keys and DataFrame values for each day's data
        selected_dates (list): List of dates that were analyzed
        config (dict): Configuration parameters
        output_dir (str): Output directory for the report
    """
    global statistics_requested, statistics_completed
    
    try:
        logger.info("Starting analysis statistics generation in background thread")
        generate_analysis_statistics(all_daily_data, selected_dates, config, output_dir)
        logger.info("Analysis statistics generation completed successfully")
        statistics_completed = True
        # Show notification in main thread
        if hasattr(tk, '_default_root') and tk._default_root:
            tk._default_root.after(100, show_statistics_completion_notification)
        else:
            # Use threading for notification if no Tkinter root exists
            notification_thread = threading.Thread(target=show_statistics_completion_notification)
            notification_thread.daemon = True
            notification_thread.start()
    except Exception as e:
        logger.error(f"Error in background statistics generation: {str(e)}")
        statistics_completed = True  # Mark as completed even on error

def generate_analysis_statistics(all_daily_data, selected_dates, config, output_dir):
    """
    Generate a CSV with analysis statistics about the data processed.
    
    Args:
        all_daily_data (dict): Dictionary with date keys and DataFrame values for each day's data
        selected_dates (list): List of dates that were analyzed
        config (dict): Configuration parameters
        output_dir (str): Output directory for the report
        
    Returns:
        DataFrame: Basic statistics DataFrame
    """
    logger.info("Generating analysis statistics report")
    
    try:
        # Create the statistics dictionary
        stats = {
            'Statistic': [],
            'Value': []
        }
        
        # Number of days analyzed
        days_analyzed = len(selected_dates)
        stats['Statistic'].append("Number of Days Analyzed")
        stats['Value'].append(days_analyzed)
        
        # Total records analyzed
        total_records = 0
        for df in all_daily_data.values():
            total_records += len(df)
        stats['Statistic'].append("Number of Records Analyzed")
        stats['Value'].append(total_records)
        logger.info(f"Records analyzed: {total_records}")

        # Count unique MMSI values
        all_mmsi = set()
        for df in all_daily_data.values():
            all_mmsi.update(df['MMSI'].unique())
        unique_mmsi_count = len(all_mmsi)
        stats['Statistic'].append("Number of Unique MMSI")
        stats['Value'].append(unique_mmsi_count)
        logger.info(f"Unique MMSIs: {unique_mmsi_count}")
        
        # Create the basic statistics DataFrame
        stats_df = pd.DataFrame(stats)
        logger.info("Basic statistics DataFrame created")
        
        # Create a separate DataFrame for the null value counts by column for all days combined
        all_data = pd.concat(all_daily_data.values())
        null_counts_by_column = all_data.isnull().sum()
        null_stats = pd.DataFrame({
            'Column': null_counts_by_column.index,
            'Total Null Values': null_counts_by_column.values
        })
        logger.info("Null value counts by column created")
        
        # Calculate null values by column for each unique MMSI (total)
        mmsi_null_counts = {}
        for mmsi in all_mmsi:
            mmsi_data = all_data[all_data['MMSI'] == mmsi]
            mmsi_null_counts[mmsi] = mmsi_data.isnull().sum()
        logger.info("Null values by column for each unique MMSI (total) calculated")

        mmsi_null_df = pd.DataFrame.from_dict(mmsi_null_counts, orient='index')
        mmsi_null_df['MMSI'] = mmsi_null_df.index
        mmsi_null_df.reset_index(drop=True, inplace=True)
        
        # Calculate null values by column for each unique MMSI by day
        daily_mmsi_null_counts = {}
        for date, df in all_daily_data.items():
            date_str = date.strftime('%Y-%m-%d')
            daily_mmsi_null_counts[date_str] = {}
            for mmsi in df['MMSI'].unique():
                mmsi_data = df[df['MMSI'] == mmsi]
                daily_mmsi_null_counts[date_str][mmsi] = mmsi_data.isnull().sum()
        logger.info("Null values by column for each unique MMSI by day calculated")
        
        # Create DataFrames for daily null values by MMSI
        daily_null_dfs = {}
        for date_str, mmsi_counts in daily_mmsi_null_counts.items():
            if mmsi_counts:
                daily_df = pd.DataFrame.from_dict(mmsi_counts, orient='index')
                daily_df['MMSI'] = daily_df.index
                daily_df.reset_index(drop=True, inplace=True)
                daily_null_dfs[date_str] = daily_df
        logger.info("Daily null values by MMSI DataFrames created")
        
        # Check if Excel statistics are enabled
        generate_statistics_excel = True
        generate_statistics_csv = True
        logger.info("Checking if Excel and CSV statistics generation enabled")
        
        if isinstance(config, dict):
            generate_statistics_excel = config.get('generate_statistics_excel', True)
            generate_statistics_csv = config.get('generate_statistics_csv', True)
        logger.info(f"Excel statistics generation: {generate_statistics_excel}, CSV statistics generation: {generate_statistics_csv}")
        
        # Save all the statistics to a single Excel file with multiple sheets if enabled
        if generate_statistics_excel:
            excel_path = os.path.join(output_dir, "Analysis_Statistics.xlsx")
            try:
                # Try to create Excel file
                with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                    stats_df.to_excel(writer, sheet_name='Basic Statistics', index=False)
                    null_stats.to_excel(writer, sheet_name='Null Values by Column', index=False)
                    mmsi_null_df.to_excel(writer, sheet_name='Null Values by MMSI', index=False)
                    
                    for date_str, daily_df in daily_null_dfs.items():
                        sheet_name = f'Nulls {date_str}'
                        # Excel sheet names have a 31 character limit
                        if len(sheet_name) > 31:
                            sheet_name = sheet_name[:31]
                        daily_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Add AIS Anomalies Summary data
                    summary_path = os.path.join(output_dir, "AIS_Anomalies_Summary.csv")
                    if os.path.exists(summary_path):
                        try:
                            # Read the summary CSV file
                            summary_df = pd.read_csv(summary_path)
                            
                            # Add full summary as a worksheet
                            summary_df.to_excel(writer, sheet_name='All Anomalies', index=False)
                            logger.info(f"Added 'All Anomalies' worksheet to Excel file")
                            
                            # Group by date and create worksheets for each day
                            if 'Date' in summary_df.columns:
                                # Convert Date column to datetime if it's not already
                                summary_df['Date'] = pd.to_datetime(summary_df['Date'], errors='coerce')
                                
                                # Create a temporary date column for grouping
                                summary_df['_DateGroup'] = summary_df['Date'].dt.date
                                
                                # Group by date
                                for date, group_df in summary_df.groupby('_DateGroup'):
                                    date_str = str(date)
                                    sheet_name = f'Anomalies {date_str}'
                                    # Excel sheet names have a 31 character limit
                                    if len(sheet_name) > 31:
                                        sheet_name = sheet_name[:31]
                                    
                                    # Remove the temporary grouping column
                                    group_export = group_df.drop(columns=['_DateGroup']).copy()
                                    if 'Date' in group_export.columns:
                                        # Keep Date but ensure it's in a readable format
                                        group_export['Date'] = group_export['Date'].dt.strftime('%Y-%m-%d')
                                    
                                    group_export.to_excel(writer, sheet_name=sheet_name, index=False)
                                    logger.info(f"Added worksheet '{sheet_name}' with {len(group_export)} anomalies")
                                
                                # Remove temporary column from original dataframe
                                summary_df = summary_df.drop(columns=['_DateGroup'])
                            elif 'BaseDateTime' in summary_df.columns:
                                # Try using BaseDateTime column if Date doesn't exist
                                summary_df['BaseDateTime'] = pd.to_datetime(summary_df['BaseDateTime'], errors='coerce')
                                
                                # Group by date (extract date from BaseDateTime)
                                summary_df['_DateGroup'] = summary_df['BaseDateTime'].dt.date
                                
                                for date, group_df in summary_df.groupby('_DateGroup'):
                                    date_str = str(date)
                                    sheet_name = f'Anomalies {date_str}'
                                    # Excel sheet names have a 31 character limit
                                    if len(sheet_name) > 31:
                                        sheet_name = sheet_name[:31]
                                    
                                    # Remove the temporary grouping column
                                    group_export = group_df.drop(columns=['_DateGroup']).copy()
                                    if 'BaseDateTime' in group_export.columns:
                                        group_export['BaseDateTime'] = group_export['BaseDateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                                    
                                    group_export.to_excel(writer, sheet_name=sheet_name, index=False)
                                    logger.info(f"Added worksheet '{sheet_name}' with {len(group_export)} anomalies")
                                
                                # Remove temporary column from original dataframe
                                summary_df = summary_df.drop(columns=['_DateGroup'])
                            else:
                                logger.warning("Could not find 'Date' or 'BaseDateTime' column in summary data. Skipping daily worksheets.")
                        except Exception as e:
                            logger.warning(f"Could not add summary data to Excel file: {e}")
                    else:
                        logger.warning(f"Summary CSV file not found at {summary_path}. Skipping summary worksheets.")
                        
                logger.info(f"Excel statistics saved to {excel_path}")
            except ImportError as e:
                logger.warning(f"Attempting to install missing xlsxwriter package...")
                try:
                    # Attempt to install the missing package
                    import subprocess
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "xlsxwriter"])
                    logger.info("Successfully installed xlsxwriter package")
                    
                    # Try again to create the Excel file after installing the package
                    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                        stats_df.to_excel(writer, sheet_name='Basic Statistics', index=False)
                        null_stats.to_excel(writer, sheet_name='Null Values by Column', index=False)
                        mmsi_null_df.to_excel(writer, sheet_name='Null Values by MMSI', index=False)
                        
                        for date_str, daily_df in daily_null_dfs.items():
                            sheet_name = f'Nulls {date_str}'
                            # Excel sheet names have a 31 character limit
                            if len(sheet_name) > 31:
                                sheet_name = sheet_name[:31]
                            daily_df.to_excel(writer, sheet_name=sheet_name, index=False)
                        logger.info(f"Added 'Nulls {date_str}' worksheet to Excel file")
                        # Add AIS Anomalies Summary data
                        summary_path = os.path.join(output_dir, "AIS_Anomalies_Summary.csv")
                        if os.path.exists(summary_path):
                            try:
                                # Read the summary CSV file
                                summary_df = pd.read_csv(summary_path)
                                
                                # Add full summary as a worksheet
                                summary_df.to_excel(writer, sheet_name='All Anomalies', index=False)
                                logger.info(f"Added 'All Anomalies' worksheet to Excel file")
                                
                                # Group by date and create worksheets for each day
                                if 'Date' in summary_df.columns:
                                    # Convert Date column to datetime if it's not already
                                    summary_df['Date'] = pd.to_datetime(summary_df['Date'], errors='coerce')
                                    
                                    # Create a temporary date column for grouping
                                    summary_df['_DateGroup'] = summary_df['Date'].dt.date
                                    
                                    # Group by date
                                    for date, group_df in summary_df.groupby('_DateGroup'):
                                        date_str = str(date)
                                        sheet_name = f'Anomalies {date_str}'
                                        # Excel sheet names have a 31 character limit
                                        if len(sheet_name) > 31:
                                            sheet_name = sheet_name[:31]
                                        
                                        # Remove the temporary grouping column
                                        group_export = group_df.drop(columns=['_DateGroup']).copy()
                                        if 'Date' in group_export.columns:
                                            # Keep Date but ensure it's in a readable format
                                            group_export['Date'] = group_export['Date'].dt.strftime('%Y-%m-%d')
                                        
                                        group_export.to_excel(writer, sheet_name=sheet_name, index=False)
                                        logger.info(f"Added worksheet '{sheet_name}' with {len(group_export)} anomalies")
                                    
                                    # Remove temporary column from original dataframe
                                    summary_df = summary_df.drop(columns=['_DateGroup'])
                                elif 'BaseDateTime' in summary_df.columns:
                                    # Try using BaseDateTime column if Date doesn't exist
                                    summary_df['BaseDateTime'] = pd.to_datetime(summary_df['BaseDateTime'], errors='coerce')
                                    
                                    # Group by date (extract date from BaseDateTime)
                                    summary_df['_DateGroup'] = summary_df['BaseDateTime'].dt.date
                                    
                                    for date, group_df in summary_df.groupby('_DateGroup'):
                                        date_str = str(date)
                                        sheet_name = f'Anomalies {date_str}'
                                        # Excel sheet names have a 31 character limit
                                        if len(sheet_name) > 31:
                                            sheet_name = sheet_name[:31]
                                        
                                        # Remove the temporary grouping column
                                        group_export = group_df.drop(columns=['_DateGroup']).copy()
                                        if 'BaseDateTime' in group_export.columns:
                                            group_export['BaseDateTime'] = group_export['BaseDateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                                        
                                        group_export.to_excel(writer, sheet_name=sheet_name, index=False)
                                        logger.info(f"Added worksheet '{sheet_name}' with {len(group_export)} anomalies")
                                    
                                    # Remove temporary column from original dataframe
                                    summary_df = summary_df.drop(columns=['_DateGroup'])
                                else:
                                    logger.warning("Could not find 'Date' or 'BaseDateTime' column in summary data. Skipping daily worksheets.")
                            except Exception as e:
                                logger.warning(f"Could not add summary data to Excel file: {e}")
                        else:
                            logger.warning(f"Summary CSV file not found at {summary_path}. Skipping summary worksheets.")
                            
                    logger.info(f"Excel statistics saved to {excel_path} after installing xlsxwriter")
                except Exception as install_error:
                    logger.warning(f"Failed to install xlsxwriter package: {install_error}")
                    logger.info("Will save CSV files instead.")
            except Exception as e:
                logger.warning(f"Could not create Excel file: {e}")
                logger.info("Will save CSV files instead.")
                
                # Save multiple CSV files since we can't do Excel sheets, but only if CSV output is enabled
                logger.info("Saving CSV files...")
                if generate_statistics_csv:
                    null_stats.to_csv(os.path.join(output_dir, "Null_Values_by_Column.csv"), index=False)
                    mmsi_null_df.to_csv(os.path.join(output_dir, "Null_Values_by_MMSI.csv"), index=False)
                    
                    for date_str, daily_df in daily_null_dfs.items():
                        filename = f"Null_Values_{date_str}.csv"
                        daily_df.to_csv(os.path.join(output_dir, filename), index=False)
        
        # Also save a basic CSV version with just the main statistics if CSV output is enabled
        if generate_statistics_csv:
            csv_path = os.path.join(output_dir, "Analysis_Statistics.csv")
            stats_df.to_csv(csv_path, index=False)
            logger.info(f"CSV statistics saved to {csv_path}")
        else:
            logger.info("CSV statistics generation is disabled in configuration")
        
        # Only log CSV path if CSV statistics were generated
        if generate_statistics_csv:
            logger.info(f"Analysis statistics saved to {csv_path}")
        return stats_df
    except Exception as e:
        logger.error(f"Debug: Failed to generate analysis statistics: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of errors



def create_anomalies_heatmap(all_anomalies_df, config, output_dir):
    """
    Create a heatmap visualization of anomalies using Folium.
    
    Args:
        all_anomalies_df (DataFrame): DataFrame containing all anomalies
        config (dict): Configuration parameters
        output_dir (str): Output directory for the maps
        
    Returns:
        str: Path to the heatmap file
    """
    # Check if heatmap is enabled
    if not config.get('show_anomaly_heatmap', True):
        logger.info("Anomaly heatmap generation is disabled in configuration")
        return None
        
    logger.info("Generating anomalies heatmap")
    
    # Create maps directory if it doesn't exist
    maps_dir = os.path.join(output_dir, "Path_Maps")
    os.makedirs(maps_dir, exist_ok=True)
    
    heatmap_file = os.path.join(maps_dir, "Anomaly_Heatmap.html")
    
    # If there are no anomalies, create an empty map
    if all_anomalies_df is None or all_anomalies_df.empty:
        logger.warning("No anomalies found for heatmap generation")
        m = folium.Map(location=[0, 0], zoom_start=2)
        m.save(heatmap_file)
        return heatmap_file
        
    # Ensure we have the required columns
    if 'LAT' not in all_anomalies_df.columns or 'LON' not in all_anomalies_df.columns:
        logger.error("Missing required LAT/LON columns for heatmap generation")
        return None
        
    # Add a timestamp column if not already present
    # First check if 'TIMESTAMP' or similar date column exists
    date_columns = [col for col in all_anomalies_df.columns if 'BaseDateTime' in col.upper() or 'DATE' in col.upper()]
    
    if date_columns:
        # Use the first matching date column
        date_column = date_columns[0]
        # Ensure it's in datetime format
        if all_anomalies_df[date_column].dtype != 'datetime64[ns]':
            all_anomalies_df[date_column] = pd.to_datetime(all_anomalies_df[date_column])
        # Extract date part only (no time)
        all_anomalies_df['AnalysisDate'] = all_anomalies_df[date_column].dt.date
    else:
        # If no date column exists, create a placeholder
        logger.warning("No date/time column found in anomalies data. Using a placeholder.")
        all_anomalies_df['AnalysisDate'] = pd.to_datetime('today').date()
    
    # Get unique dates for day-by-day analysis
    unique_dates = sorted(all_anomalies_df['AnalysisDate'].unique())
    
    # Prepare data for heatmap - All data
    heat_data_all = all_anomalies_df[['LAT', 'LON']].values.tolist()
    
    # Filter data by anomaly type if the column exists
    if 'AnomalyType' in all_anomalies_df.columns:
        heat_data_AIS_on = all_anomalies_df[all_anomalies_df['AnomalyType'] == 'AIS_Beacon_On'][['LAT', 'LON']].values.tolist()
        heat_data_AIS_off = all_anomalies_df[all_anomalies_df['AnomalyType'] == 'AIS_Beacon_Off'][['LAT', 'LON']].values.tolist()
        heat_data_course = all_anomalies_df[all_anomalies_df['AnomalyType'] == 'Course'][['LAT', 'LON']].values.tolist()
        heat_data_speed = all_anomalies_df[all_anomalies_df['AnomalyType'] == 'Speed'][['LAT', 'LON']].values.tolist()
        heat_data_loitering = all_anomalies_df[all_anomalies_df['AnomalyType'] == 'Loitering'][['LAT', 'LON']].values.tolist()
        heat_data_rendezvous = all_anomalies_df[all_anomalies_df['AnomalyType'] == 'Rendezvous'][['LAT', 'LON']].values.tolist()
        heat_data_spoofing = all_anomalies_df[all_anomalies_df['AnomalyType'] == 'Identity_Spoofing'][['LAT', 'LON']].values.tolist()
        heat_data_zone = all_anomalies_df[all_anomalies_df['AnomalyType'] == 'Zone_Violation'][['LAT', 'LON']].values.tolist()
    else:
        heat_data_AIS_on = []
        heat_data_AIS_off = []
        heat_data_course = []
        heat_data_speed = []
        heat_data_loitering = []
        heat_data_rendezvous = []
        heat_data_spoofing = []
        heat_data_zone = []
    
    # Create a Base Folium Map
    map_center = [all_anomalies_df['LAT'].mean(), all_anomalies_df['LON'].mean()]
    m = folium.Map(location=map_center, zoom_start=6)
    
    # Create feature groups for different anomaly types
    # Main group: "All Anomalies (All Days)" - default view
    fg_all_anomalies = folium.FeatureGroup(name="All Anomalies (All Days)", show=True)
    
    # Sub-groups under "All Anomalies (All Days)" - All Data views
    fg_ais_on_all = folium.FeatureGroup(name="  |-- All Data - AIS Beacon On", show=False)
    fg_ais_off_all = folium.FeatureGroup(name="  |-- All Data - AIS Beacon Off", show=False)
    fg_course_all = folium.FeatureGroup(name="  |-- All Data - Course Anomalies", show=False)
    fg_speed_all = folium.FeatureGroup(name="  |-- All Data - Speed Anomalies", show=False)
    fg_loitering_all = folium.FeatureGroup(name="  |-- All Data - Loitering", show=False)
    fg_rendezvous_all = folium.FeatureGroup(name="  |-- All Data - Rendezvous", show=False)
    fg_spoofing_all = folium.FeatureGroup(name="  |-- All Data - Identity Spoofing", show=False)
    fg_zone_all = folium.FeatureGroup(name="  |-- All Data - Zone Violations", show=False)
    
    # Add heatmaps to their respective feature groups
    HeatMap(heat_data_all).add_to(fg_all_anomalies)
    if heat_data_AIS_on:
        HeatMap(heat_data_AIS_on).add_to(fg_ais_on_all)
    if heat_data_AIS_off:
        HeatMap(heat_data_AIS_off).add_to(fg_ais_off_all)
    if heat_data_course:
        HeatMap(heat_data_course).add_to(fg_course_all)
    if heat_data_speed:
        HeatMap(heat_data_speed).add_to(fg_speed_all)
    if heat_data_loitering:
        HeatMap(heat_data_loitering).add_to(fg_loitering_all)
    if heat_data_rendezvous:
        HeatMap(heat_data_rendezvous).add_to(fg_rendezvous_all)
    if heat_data_spoofing:
        HeatMap(heat_data_spoofing).add_to(fg_spoofing_all)
    if heat_data_zone:
        HeatMap(heat_data_zone).add_to(fg_zone_all)
    
    # Add overall feature groups to map
    fg_all_anomalies.add_to(m)
    fg_ais_on_all.add_to(m)
    fg_ais_off_all.add_to(m)
    fg_course_all.add_to(m)
    fg_speed_all.add_to(m)
    fg_loitering_all.add_to(m)
    fg_rendezvous_all.add_to(m)
    fg_spoofing_all.add_to(m)
    fg_zone_all.add_to(m)
    
    # Create day-specific feature groups
    day_groups = {}
    
    for date in unique_dates:
        date_str = date.strftime('%Y-%m-%d')
        
        # Filter data for this specific date
        date_df = all_anomalies_df[all_anomalies_df['AnalysisDate'] == date]
        
        # Create feature groups for this date
        day_groups[date_str] = {
            'all': folium.FeatureGroup(name=f"Day {date_str} - All Anomalies", show=False)
        }
        
        # Only add type-specific groups if AnomalyType column exists
        if 'AnomalyType' in all_anomalies_df.columns:
            day_groups[date_str].update({
                'ais_on': folium.FeatureGroup(name=f"  |-- Day {date_str} - AIS Beacon On", show=False),
                'ais_off': folium.FeatureGroup(name=f"  |-- Day {date_str} - AIS Beacon Off", show=False),
                'course': folium.FeatureGroup(name=f"  |-- Day {date_str} - Course Anomalies", show=False),
                'speed': folium.FeatureGroup(name=f"  |-- Day {date_str} - Speed Anomalies", show=False),
                'loitering': folium.FeatureGroup(name=f"  |-- Day {date_str} - Loitering", show=False),
                'rendezvous': folium.FeatureGroup(name=f"  |-- Day {date_str} - Rendezvous", show=False),
                'spoofing': folium.FeatureGroup(name=f"  |-- Day {date_str} - Identity Spoofing", show=False),
                'zone': folium.FeatureGroup(name=f"  |-- Day {date_str} - Zone Violations", show=False)
            })
        
        # Create heatmap data for this date
        day_heat_all = date_df[['LAT', 'LON']].values.tolist()
        
        # Add heatmap to date-specific feature group
        if day_heat_all:
            HeatMap(day_heat_all).add_to(day_groups[date_str]['all'])
        
        # Only add type-specific heatmaps if AnomalyType column exists
        if 'AnomalyType' in all_anomalies_df.columns:
            day_heat_ais_on = date_df[date_df['AnomalyType'] == 'AIS_Beacon_On'][['LAT', 'LON']].values.tolist()
            day_heat_ais_off = date_df[date_df['AnomalyType'] == 'AIS_Beacon_Off'][['LAT', 'LON']].values.tolist()
            day_heat_course = date_df[date_df['AnomalyType'] == 'Course'][['LAT', 'LON']].values.tolist()
            day_heat_speed = date_df[date_df['AnomalyType'] == 'Speed'][['LAT', 'LON']].values.tolist()
            day_heat_loitering = date_df[date_df['AnomalyType'] == 'Loitering'][['LAT', 'LON']].values.tolist()
            day_heat_rendezvous = date_df[date_df['AnomalyType'] == 'Rendezvous'][['LAT', 'LON']].values.tolist()
            day_heat_spoofing = date_df[date_df['AnomalyType'] == 'Identity_Spoofing'][['LAT', 'LON']].values.tolist()
            day_heat_zone = date_df[date_df['AnomalyType'] == 'Zone_Violation'][['LAT', 'LON']].values.tolist()
            
            if day_heat_ais_on:
                HeatMap(day_heat_ais_on).add_to(day_groups[date_str]['ais_on'])
            if day_heat_ais_off:
                HeatMap(day_heat_ais_off).add_to(day_groups[date_str]['ais_off'])
            if day_heat_course:
                HeatMap(day_heat_course).add_to(day_groups[date_str]['course'])
            if day_heat_speed:
                HeatMap(day_heat_speed).add_to(day_groups[date_str]['speed'])
            if day_heat_loitering:
                HeatMap(day_heat_loitering).add_to(day_groups[date_str]['loitering'])
            if day_heat_rendezvous:
                HeatMap(day_heat_rendezvous).add_to(day_groups[date_str]['rendezvous'])
            if day_heat_spoofing:
                HeatMap(day_heat_spoofing).add_to(day_groups[date_str]['spoofing'])
            if day_heat_zone:
                HeatMap(day_heat_zone).add_to(day_groups[date_str]['zone'])
        
        # Add all feature groups to map
        for group in day_groups[date_str].values():
            group.add_to(m)
    
    # Add Layer Control for toggling heatmaps
    folium.LayerControl(collapsed=False, exclusive_groups=True).add_to(m)
    
    # Save the map
    m.save(heatmap_file)
    logger.info(f"Anomaly heatmap saved to {heatmap_file}")
    
    return heatmap_file

def create_vessel_path_maps(all_daily_data, selected_dates, config, output_dir):
    """
    Create maps showing the paths of each vessel across all days and by day.
    
    Args:
        all_daily_data (dict): Dictionary with date keys and DataFrame values for each day's data
        selected_dates (list): List of dates that were analyzed
        config (dict): Configuration parameters
        output_dir (str): Output directory for the maps
        
    Returns:
        str: Path to the maps directory
    """
    global map_manager
    # Check if vessel path maps are enabled
    if isinstance(config, dict) and not config.get('generate_vessel_path_maps', True):
        logger.info("Vessel path map generation is disabled in configuration")
        return None
        
    logger.info("Generating vessel path maps")
    
    # Create a folder for the maps
    maps_dir = os.path.join(output_dir, "Path_Maps")
    os.makedirs(maps_dir, exist_ok=True)
    
    # First, create the total paths map with all vessels
    all_data = pd.concat(all_daily_data.values())
    
    # Get the unique vessels (MMSI values)
    unique_mmsi = all_data['MMSI'].unique()
    
    # Create the total paths map
    m = folium.Map(location=[all_data['LAT'].mean(), all_data['LON'].mean()], zoom_start=4)
    
    # Check if grid lines should be shown based on config
    if config.get('show_lat_long_grid', True):
        # Use global boundaries if available, otherwise calculate from current data
        if map_manager.is_valid():
            # Use pre-calculated global boundaries
            min_lat, max_lat, min_lon, max_lon = map_manager.get_boundaries()
            logger.info("Using global boundaries for vessel path map")
        else:
            # Calculate from current data as fallback
            min_lat = all_data['LAT'].min() - 5
            max_lat = all_data['LAT'].max() + 5
            min_lon = all_data['LON'].min() - 5
            max_lon = all_data['LON'].max() + 5
            logger.info("Using local boundaries for vessel path map (global not available)")
        
        # Add latitude and longitude grid lines to the total paths map
        add_lat_lon_grid_lines(m, 
                              lat_start=min_lat, 
                              lat_end=max_lat, 
                              lon_start=min_lon, 
                              lon_end=max_lon, 
                              lat_step=10, 
                              lon_step=10, 
                              label_step=10)
        logger.info("Added latitude/longitude grid lines to overall vessel path map")
    else:
        logger.info("Latitude/longitude grid lines disabled by configuration")
    
    # Use a color map to assign different colors to each vessel
    
    # Create a colormap with distinct colors for each vessel (up to 20 vessels)
    colormap = cm.get_cmap('tab20', min(20, len(unique_mmsi)))
    
    # Add paths for each vessel
    for i, mmsi in enumerate(unique_mmsi):
        vessel_data = all_data[all_data['MMSI'] == mmsi].sort_values('BaseDateTime')
        
        if len(vessel_data) > 1:  # Need at least 2 points to make a line
            # Get a color for this vessel
            if i < 20:  # Use the colormap for the first 20 vessels
                color = mcolors.rgb2hex(colormap(i)[:3])
            else:  # For additional vessels, use a default color
                color = 'gray'
                
            # Create a popup with vessel information
            vessel_name = vessel_data['VesselName'].iloc[0]
            vessel_type = vessel_data['VesselType'].iloc[0]
            popup_text = f"MMSI: {mmsi}<br>Name: {vessel_name}<br>Type: {vessel_type}"
            
            # Add the vessel's path as a line (vectorized)
            valid_mask = vessel_data['LAT'].notna() & vessel_data['LON'].notna()
            valid_data = vessel_data[valid_mask]
            if not valid_data.empty:
                points = list(zip(valid_data['LAT'].values, valid_data['LON'].values))
            else:
                points = []
            
            if points:
                # Add the path as a line
                folium.PolyLine(
                    points,
                    color=color,
                    weight=3,
                    opacity=0.7,
                    popup=popup_text
                ).add_to(m)
                
                # Add markers for the start and end points
                folium.Marker(
                    points[0],
                    popup=f"Start: {vessel_name} ({mmsi})",
                    icon=folium.Icon(color='green', icon='play', prefix='fa')
                ).add_to(m)
                
                folium.Marker(
                    points[-1],
                    popup=f"End: {vessel_name} ({mmsi})",
                    icon=folium.Icon(color='red', icon='stop', prefix='fa')
                ).add_to(m)
                
                # Add a circle marker for every AIS position report
                for i, (lat, lon) in enumerate(points):
                    # Skip first and last points as they already have markers
                    if i == 0 or i == len(points) - 1:
                        continue
                        
                    # Get the timestamp for this point
                    row = vessel_data.iloc[i]
                    timestamp = row['BaseDateTime'].strftime('%Y-%m-%d %H:%M:%S') if 'BaseDateTime' in vessel_data.columns else 'Unknown'
                    
                    # Create popup with detailed information
                    point_popup = f"{vessel_name} ({mmsi})<br>Time: {timestamp}<br>"
                    if 'SOG' in vessel_data.columns:
                        point_popup += f"Speed: {row['SOG']} knots<br>"
                    if 'COG' in vessel_data.columns:
                        point_popup += f"Course: {row['COG']} deg<br>"
                    
                    # Add the circle marker
                    folium.CircleMarker(
                        location=(lat, lon),
                        radius=4,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        popup=point_popup,
                        tooltip=f"{vessel_name}: {timestamp}"
                    ).add_to(m)
    
    # Save the total paths map
    total_map_path = os.path.join(maps_dir, "Total_Paths.html")
    m.save(total_map_path)
    
    # Create daily path maps
    for date, df in all_daily_data.items():
        date_str = date.strftime('%Y-%m-%d')
        
        # Create a map for this day
        daily_m = folium.Map(location=[df['LAT'].mean(), df['LON'].mean()], zoom_start=4)
        
        # Check if grid lines should be shown based on config
        if config.get('show_lat_long_grid', True):
            # Use global boundaries if available, otherwise calculate from current data
            if map_manager.is_valid():
                # Use pre-calculated global boundaries
                min_lat, max_lat, min_lon, max_lon = map_manager.get_boundaries()
                logger.info(f"Using global boundaries for daily vessel path map ({date_str})")
            else:
                # Calculate from current data as fallback
                min_lat = df['LAT'].min() - 5
                max_lat = df['LAT'].max() + 5
                min_lon = df['LON'].min() - 5
                max_lon = df['LON'].max() + 5
                logger.info(f"Using local boundaries for daily vessel path map ({date_str})")
            
            # Add latitude and longitude grid lines to the daily map
            add_lat_lon_grid_lines(daily_m, 
                                  lat_start=min_lat, 
                                  lat_end=max_lat, 
                                  lon_start=min_lon, 
                                  lon_end=max_lon, 
                                  lat_step=10, 
                                  lon_step=10, 
                                  label_step=10)
            logger.debug(f"Added latitude/longitude grid lines to vessel path map for {date_str}")
        
        # Get the unique vessels for this day
        daily_mmsi = df['MMSI'].unique()
        
        # Add paths for each vessel on this day
        for i, mmsi in enumerate(daily_mmsi):
            vessel_data = df[df['MMSI'] == mmsi].sort_values('BaseDateTime')
            
            if len(vessel_data) > 1:  # Need at least 2 points to make a line
                # Get a color for this vessel
                if i < 20:
                    color = mcolors.rgb2hex(colormap(i)[:3])
                else:
                    color = 'gray'
                    
                # Create a popup with vessel information
                vessel_name = vessel_data['VesselName'].iloc[0]
                vessel_type = vessel_data['VesselType'].iloc[0]
                popup_text = f"MMSI: {mmsi}<br>Name: {vessel_name}<br>Type: {vessel_type}"
                
                # Add the vessel's path as a line (vectorized)
                valid_mask = vessel_data['LAT'].notna() & vessel_data['LON'].notna()
                valid_data = vessel_data[valid_mask]
                if not valid_data.empty:
                    points = list(zip(valid_data['LAT'].values, valid_data['LON'].values))
                else:
                    points = []
                
                if points:
                    # Add the path as a line
                    folium.PolyLine(
                        points,
                        color=color,
                        weight=3,
                        opacity=0.7,
                        popup=popup_text
                    ).add_to(daily_m)
                    
                    # Add markers for the start and end points
                    folium.Marker(
                        points[0],
                        popup=f"Start: {vessel_name} ({mmsi})",
                        icon=folium.Icon(color='green', icon='play', prefix='fa')
                    ).add_to(daily_m)
                    
                    folium.Marker(
                        points[-1],
                        popup=f"End: {vessel_name} ({mmsi})",
                        icon=folium.Icon(color='red', icon='stop', prefix='fa')
                    ).add_to(daily_m)
                    
                    # Add a circle marker for every AIS position report
                    for i, (lat, lon) in enumerate(points):
                        # Skip first and last points as they already have markers
                        if i == 0 or i == len(points) - 1:
                            continue
                            
                        # Get the timestamp for this point
                        row = vessel_data.iloc[i]
                        timestamp = row['BaseDateTime'].strftime('%Y-%m-%d %H:%M:%S') if 'BaseDateTime' in vessel_data.columns else 'Unknown'
                        
                        # Create popup with detailed information
                        point_popup = f"{vessel_name} ({mmsi})<br>Time: {timestamp}<br>"
                        if 'SOG' in vessel_data.columns:
                            point_popup += f"Speed: {row['SOG']} knots<br>"
                        if 'COG' in vessel_data.columns:
                            point_popup += f"Course: {row['COG']} deg<br>"
                        
                        # Add the circle marker
                        folium.CircleMarker(
                            location=(lat, lon),
                            radius=4,
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.7,
                            popup=point_popup,
                            tooltip=f"{vessel_name}: {timestamp}"
                        ).add_to(daily_m)
        
        # Save the daily map
        daily_map_path = os.path.join(maps_dir, f"Path_Map_{date_str}.html")
        daily_m.save(daily_map_path)
    
    logger.info(f"Vessel path maps saved to {maps_dir}")
    return maps_dir


def _process_anomaly_detection(file_paths, dates_in_order, config, use_dask=True):
    """
    Internal function that handles the actual anomaly detection process.
    
    Args:
        file_paths (list): List of file paths to process
        dates_in_order (list): List of dates corresponding to file_paths
        config (dict): Configuration dictionary
        use_dask (bool): Whether to use Dask for processing
        
    Returns:
        DataFrame: Detected anomalies
    """
    
    # Store each day's data for later statistics and path mapping
    all_daily_data = {}
    
    if not file_paths or len(file_paths) <= 1:
        logger.error("Not enough valid daily files found for comparison. Need at least 2 days.")
        return pd.DataFrame()
    
    # Process files day by day for comparisons
    df_previous_day = None
    processed_first_day = False
    all_anomalies = []
    
    for i in range(len(file_paths)):
        current_file_path = file_paths[i]
        current_date = dates_in_order[i]
        logger.info(f"Processing data for: {current_date.strftime('%Y-%m-%d')} ({current_file_path})")
        
        df_current_day = load_and_preprocess_day(current_file_path, config, use_dask)
        
        # Check if DataFrame is None or empty
        if df_current_day is None or df_current_day.empty:
            if df_current_day is None:
                logger.warning(f"Skipping {current_file_path} due to loading errors (returned None).")
            else:
                logger.warning(f"Skipping {current_file_path} - DataFrame is empty after filtering.")
            
            # Reset previous day if current fails
            df_previous_day = None
            continue
            
        # Store the daily data for later analysis
        all_daily_data[current_date] = df_current_day
        
        if not processed_first_day:
            df_previous_day = df_current_day
            processed_first_day = True
            logger.info(f"Loaded initial day: {current_date.strftime('%Y-%m-%d')}. No comparisons possible yet.")
            continue  # Skip to the next day for comparisons
        
        # --- ANOMALY DETECTION ---
        anomalies = []
        report_date = current_date.strftime('%Y-%m-%d')
        
        # Group data by MMSI for analysis (move this up from below)
        prev_grouped = df_previous_day.groupby('MMSI')
        current_grouped = df_current_day.groupby('MMSI')
        
        # 1. AIS Beacon on/off anomalies (sudden appearance/disappearance)
        logger.info("Detecting AIS beacon on/off anomalies...")
        beacon_anomalies = []
        
        # Set threshold for beacon anomalies (in hours, convert to minutes)
        beacon_time_threshold = config.get('BEACON_TIME_THRESHOLD_HOURS', 6) * 60  # Convert hours to minutes
        
        # Find vessels that appeared in current day but not in previous day (beacon on)
        if config.get('ais_beacon_on', True):  # Check if this anomaly type is enabled
            # For beacon on, we need to check if this vessel has been absent for at least 6 hours
            # This requires looking at all previous days, not just the last one
            # For now, we'll implement a basic version that just checks between consecutive days
            beacon_on_mmsi = set(df_current_day['MMSI'].unique()) - set(df_previous_day['MMSI'].unique())
            logger.info(f"Found {len(beacon_on_mmsi)} potential vessels with AIS beacon on")
            
            # Vessels that meet the 6-hour threshold
            confirmed_beacon_on = []
            
            for mmsi in beacon_on_mmsi:
                # Get the vessel's first appearance in the current day
                vessel_curr = current_grouped.get_group(mmsi).copy()
                vessel_curr = vessel_curr.sort_values('BaseDateTime')
                
                if len(vessel_curr) > 0:
                    first_appearance = vessel_curr.iloc[0]['BaseDateTime']
                    
                    # Check if the first appearance is at least 6 hours after the start of the current day
                    # This is a simplification - ideally we'd check against the last known position
                    day_start = pd.Timestamp(current_date).replace(hour=0, minute=0, second=0)
                    time_since_day_start = (first_appearance - day_start).total_seconds() / 60  # in minutes
                    
                    # If the vessel appears more than 6 hours after the day start, or
                    # if it's the first record of the day, count it as a beacon on
                    if time_since_day_start >= beacon_time_threshold:
                        first_pos = vessel_curr.iloc[0]
                        anomaly_record = first_pos.copy()
                        anomaly_record['AnomalyType'] = 'AIS_Beacon_On'
                        anomaly_record['SpeedAnomaly'] = False
                        anomaly_record['PositionAnomaly'] = True
                        anomaly_record['CourseAnomaly'] = False
                        anomaly_record['BeaconAnomaly'] = True
                        anomaly_record['BeaconGapMinutes'] = time_since_day_start
                        anomaly_record['Date'] = current_date
                        anomaly_record['ReportDate'] = report_date
                        
                        beacon_anomalies.append(anomaly_record)
                        confirmed_beacon_on.append(mmsi)
            
            logger.info(f"Confirmed {len(confirmed_beacon_on)} vessels with AIS beacon on (gap >= {beacon_time_threshold/60:.1f} hours)")
        
        # Find vessels that disappeared in current day but were in previous day (beacon off)
        if config.get('ais_beacon_off', True):  # Check if this anomaly type is enabled
            beacon_off_mmsi = set(df_previous_day['MMSI'].unique()) - set(df_current_day['MMSI'].unique())
            logger.info(f"Found {len(beacon_off_mmsi)} potential vessels with AIS beacon off")
            
            # Vessels that meet the 6-hour threshold
            confirmed_beacon_off = []
            
            for mmsi in beacon_off_mmsi:
                # Get the vessel's last appearance in the previous day
                vessel_prev = prev_grouped.get_group(mmsi).copy()
                vessel_prev = vessel_prev.sort_values('BaseDateTime')
                
                if len(vessel_prev) > 0:
                    last_appearance = vessel_prev.iloc[-1]['BaseDateTime']
                    
                    # Check if the last appearance is at least 6 hours before the end of the previous day
                    day_end = pd.Timestamp(dates_in_order[i-1]).replace(hour=23, minute=59, second=59)
                    time_to_day_end = (day_end - last_appearance).total_seconds() / 60  # in minutes
                    
                    # If the vessel disappears more than 6 hours before the day end, count it as a beacon off
                    if time_to_day_end >= beacon_time_threshold:
                        last_pos = vessel_prev.iloc[-1]
                        anomaly_record = last_pos.copy()
                        anomaly_record['AnomalyType'] = 'AIS_Beacon_Off'
                        anomaly_record['SpeedAnomaly'] = False
                        anomaly_record['PositionAnomaly'] = True
                        anomaly_record['CourseAnomaly'] = False
                        anomaly_record['BeaconAnomaly'] = True
                        anomaly_record['BeaconGapMinutes'] = time_to_day_end
                        anomaly_record['Date'] = current_date
                        anomaly_record['ReportDate'] = report_date
                        
                        beacon_anomalies.append(anomaly_record)
                        confirmed_beacon_off.append(mmsi)
            
            logger.info(f"Confirmed {len(confirmed_beacon_off)} vessels with AIS beacon off (gap >= {beacon_time_threshold/60:.1f} hours)")
        
        if beacon_anomalies:
            # Convert Series objects to dictionaries for consistent DataFrame creation
            beacon_anomalies_dicts = [record.to_dict() if isinstance(record, pd.Series) else record for record in beacon_anomalies]
            beacon_anomalies_df = pd.DataFrame(beacon_anomalies_dicts)
            anomalies.extend(beacon_anomalies_dicts)
            logger.info(f"Found {len(beacon_anomalies)} AIS beacon anomalies.")
        
        # 2. Position jumps (Speed anomalies)
        speed_anomalies = []
        
        # Check if speed anomalies are enabled
        if config.get('excessive_travel_distance_fast', True):  # Check if this anomaly type is enabled
            logger.info("Detecting speed anomalies (position jumps)...")
            
            # Look for common vessels between days
            common_mmsi = set(df_previous_day['MMSI'].unique()) & set(df_current_day['MMSI'].unique())
            
            for mmsi in common_mmsi:
                # Get the vessel data for previous and current day
                vessel_prev = prev_grouped.get_group(mmsi).copy()
                vessel_curr = current_grouped.get_group(mmsi).copy()
                
                # Get the last position from previous day and first position from current day
                vessel_prev = vessel_prev.sort_values('BaseDateTime')
                vessel_curr = vessel_curr.sort_values('BaseDateTime')
                
                if len(vessel_prev) == 0 or len(vessel_curr) == 0:
                    continue
                    
                last_pos_prev = vessel_prev.iloc[-1]
                first_pos_curr = vessel_curr.iloc[0]
                
                # Calculate time difference in minutes
                time_diff = (first_pos_curr['BaseDateTime'] - last_pos_prev['BaseDateTime']).total_seconds() / 60
                
                # Skip if positions are too far apart in time (e.g., data gaps)
                time_threshold = config.get('TIME_DIFF_THRESHOLD_MIN', 240)  # Default 4 hours
                if time_diff > time_threshold:
                    continue
                    
                # Calculate distance between points (using vectorized function)
                # Create a DataFrame for vectorized haversine calculation
                distance_df = pd.DataFrame({
                    'LAT1': [last_pos_prev['LAT']],
                    'LON1': [last_pos_prev['LON']],
                    'LAT2': [first_pos_curr['LAT']],
                    'LON2': [first_pos_curr['LON']]
                })
                
                # Use vectorized haversine function (pass USE_GPU config setting)
                dist_result = haversine_vectorized(distance_df, use_gpu=config.get('USE_GPU', GPU_AVAILABLE))
                if dist_result.empty or pd.isna(dist_result.iloc[0]):
                    continue
                dist_nm = dist_result.iloc[0]
                    
                # Calculate implied speed
                implied_speed = dist_nm / (time_diff / 60)  # Convert minutes to hours for knots
                
                # Check if the implied speed exceeds our threshold
                if implied_speed > config.get('SPEED_THRESHOLD', 102):
                    # This is an anomaly - position jump detected
                    anomaly_record = first_pos_curr.copy()
                    anomaly_record['AnomalyType'] = 'Speed'
                    anomaly_record['SpeedAnomaly'] = True
                    anomaly_record['PositionAnomaly'] = False
                    anomaly_record['CourseAnomaly'] = False
                    anomaly_record['Distance'] = dist_nm
                    anomaly_record['TimeDiff'] = time_diff
                    anomaly_record['ImpliedSpeed'] = implied_speed
                    anomaly_record['Date'] = current_date
                    anomaly_record['ReportDate'] = report_date
                    
                    speed_anomalies.append(anomaly_record)
            
            if speed_anomalies:
                # Convert Series objects to dictionaries for consistent DataFrame creation
                speed_anomalies_dicts = [record.to_dict() if isinstance(record, pd.Series) else record for record in speed_anomalies]
                speed_anomalies_df = pd.DataFrame(speed_anomalies_dicts)
                anomalies.extend(speed_anomalies_dicts)
                logger.info(f"Found {len(speed_anomalies)} speed anomalies.")
        
        # 2. Course vs. Heading anomalies
        course_anomalies = []
        
        # Check if course anomalies are enabled
        if config.get('cog-heading_inconsistency', True):  # Check if this anomaly type is enabled
            logger.info("Detecting course vs. heading anomalies...")
            
            for _, vessel_data in current_grouped:
                # Filter rows with sufficient speed and valid COG and Heading
                min_speed = config.get('MIN_SPEED_FOR_COG_CHECK', 10)
                valid_rows = vessel_data[
                    (vessel_data['SOG'] >= min_speed) &
                    vessel_data['COG'].notna() &
                    vessel_data['Heading'].notna()
                ]
                
                if len(valid_rows) == 0:
                    continue
                    
                # Calculate the difference between COG and Heading
                valid_rows['CourseHeadingDiff'] = valid_rows.apply(
                    lambda row: normalize_angle_difference(row['COG'] - row['Heading']),
                    axis=1
                )
                
                # Identify potential anomalies where the difference exceeds the threshold
                max_diff = config.get('COG_HEADING_MAX_DIFF', 45)
                anomalous_rows = valid_rows[abs(valid_rows['CourseHeadingDiff']) > max_diff]
                
                if not anomalous_rows.empty:
                    # Mark these as course anomalies (vectorized)
                    # Create a copy of the anomalous rows and add the required columns
                    anomaly_records = anomalous_rows.copy()
                    anomaly_records['AnomalyType'] = 'Course'
                    anomaly_records['SpeedAnomaly'] = False
                    anomaly_records['PositionAnomaly'] = False
                    anomaly_records['CourseAnomaly'] = True
                    anomaly_records['Date'] = current_date
                    anomaly_records['ReportDate'] = report_date
                    
                    # Convert to list of dictionaries for compatibility with existing code
                    course_anomalies.extend(anomaly_records.to_dict('records'))
            
            if course_anomalies:
                course_anomalies_df = pd.DataFrame(course_anomalies)
                anomalies.extend(course_anomalies)
                logger.info(f"Found {len(course_anomalies)} course anomalies.")
        
        # 3. Loitering detection
        if config.get('loitering', True):  # Check if this anomaly type is enabled
            logger.info("Detecting loitering vessels...")
            loitering_anomalies = []
            
            # Get thresholds from config
            loitering_radius_nm = config.get('LOITERING_RADIUS_NM', 5.0)  # Default 5 nautical miles
            loitering_duration_hours = config.get('LOITERING_DURATION_HOURS', 24.0)  # Default 24 hours
            
            for mmsi, vessel_data in current_grouped:
                if len(vessel_data) < 10:  # Need at least 10 records
                    continue
                
                # Sort by time
                vessel_data = vessel_data.sort_values('BaseDateTime').copy()
                
                # Calculate time span in hours
                time_span = (vessel_data['BaseDateTime'].max() - vessel_data['BaseDateTime'].min()).total_seconds() / 3600
                
                if time_span < loitering_duration_hours:
                    continue
                
                # Calculate center point
                center_lat = vessel_data['LAT'].mean()
                center_lon = vessel_data['LON'].mean()
                
                # Calculate maximum distance from center for all positions
                max_dist = 0
                for _, row in vessel_data.iterrows():
                    if pd.notna(row['LAT']) and pd.notna(row['LON']):
                        # Use haversine_vectorized for distance calculation
                        distance_df = pd.DataFrame({
                            'LAT1': [center_lat],
                            'LON1': [center_lon],
                            'LAT2': [row['LAT']],
                            'LON2': [row['LON']]
                        })
                        dist_result = haversine_vectorized(distance_df, use_gpu=config.get('USE_GPU', GPU_AVAILABLE))
                        if not dist_result.empty and pd.notna(dist_result.iloc[0]):
                            dist_nm = dist_result.iloc[0]
                            max_dist = max(max_dist, dist_nm)
                
                # If all positions are within the radius, it's loitering
                if max_dist < loitering_radius_nm:
                    # Use the first position as the anomaly record
                    anomaly_record = vessel_data.iloc[0].copy()
                    anomaly_record['AnomalyType'] = 'Loitering'
                    anomaly_record['SpeedAnomaly'] = False
                    anomaly_record['PositionAnomaly'] = True
                    anomaly_record['CourseAnomaly'] = False
                    anomaly_record['LoiteringRadiusNM'] = max_dist
                    anomaly_record['LoiteringDurationHours'] = time_span
                    anomaly_record['LoiteringRecordCount'] = len(vessel_data)
                    anomaly_record['Date'] = current_date
                    anomaly_record['ReportDate'] = report_date
                    
                    loitering_anomalies.append(anomaly_record)
            
            if loitering_anomalies:
                loitering_anomalies_dicts = [record.to_dict() if isinstance(record, pd.Series) else record for record in loitering_anomalies]
                anomalies.extend(loitering_anomalies_dicts)
                logger.info(f"Found {len(loitering_anomalies)} loitering anomalies.")
        
        # 4. Rendezvous detection
        if config.get('rendezvous', True):  # Check if this anomaly type is enabled
            logger.info("Detecting vessel rendezvous...")
            rendezvous_anomalies = []
            
            # Get thresholds from config
            rendezvous_proximity_nm = config.get('RENDEZVOUS_PROXIMITY_NM', 0.5)  # Default 0.5 nautical miles
            rendezvous_duration_minutes = config.get('RENDEZVOUS_DURATION_MINUTES', 30)  # Default 30 minutes
            
            # Group data by time windows (e.g., 1-hour windows)
            df_current_day_sorted = df_current_day.sort_values('BaseDateTime').copy()
            
            # Create time windows (1-hour bins) - convert datetime to numeric for binning
            if len(df_current_day_sorted) > 0:
                # Use hour of day as the time window
                df_current_day_sorted['TimeWindow'] = df_current_day_sorted['BaseDateTime'].dt.hour
            else:
                df_current_day_sorted['TimeWindow'] = []
            
            # Process each time window
            for window_id, window_group in df_current_day_sorted.groupby('TimeWindow'):
                if len(window_group) < 2:
                    continue
                
                # Group by MMSI within this time window
                window_vessels = window_group.groupby('MMSI')
                if len(window_vessels) < 2:
                    continue
                
                # Calculate average position for each vessel in this window
                vessel_positions = {}
                for mmsi, vessel_group in window_vessels:
                    if len(vessel_group) >= 3:  # Need at least 3 records
                        avg_lat = vessel_group['LAT'].mean()
                        avg_lon = vessel_group['LON'].mean()
                        if pd.notna(avg_lat) and pd.notna(avg_lon):
                            vessel_positions[mmsi] = (avg_lat, avg_lon, len(vessel_group))
                
                # Check all pairs of vessels in this window
                vessel_list = list(vessel_positions.keys())
                for i in range(len(vessel_list)):
                    for j in range(i + 1, len(vessel_list)):
                        mmsi1, mmsi2 = vessel_list[i], vessel_list[j]
                        lat1, lon1, count1 = vessel_positions[mmsi1]
                        lat2, lon2, count2 = vessel_positions[mmsi2]
                        
                        # Calculate distance between vessels
                        distance_df = pd.DataFrame({
                            'LAT1': [lat1],
                            'LON1': [lon1],
                            'LAT2': [lat2],
                            'LON2': [lon2]
                        })
                        dist_result = haversine_vectorized(distance_df, use_gpu=config.get('USE_GPU', GPU_AVAILABLE))
                        if dist_result.empty or pd.isna(dist_result.iloc[0]):
                            continue
                        distance_nm = dist_result.iloc[0]
                        
                        # Check if vessels are close enough
                        if distance_nm < rendezvous_proximity_nm:
                            # Get the first record from vessel 1 as the anomaly record
                            vessel1_records = window_group[window_group['MMSI'] == mmsi1]
                            if len(vessel1_records) > 0:
                                anomaly_record = vessel1_records.iloc[0].copy()
                                anomaly_record['AnomalyType'] = 'Rendezvous'
                                anomaly_record['SpeedAnomaly'] = False
                                anomaly_record['PositionAnomaly'] = True
                                anomaly_record['CourseAnomaly'] = False
                                anomaly_record['RendezvousMMSI2'] = mmsi2
                                anomaly_record['RendezvousDistanceNM'] = distance_nm
                                anomaly_record['RendezvousLat'] = (lat1 + lat2) / 2
                                anomaly_record['RendezvousLon'] = (lon1 + lon2) / 2
                                anomaly_record['Date'] = current_date
                                anomaly_record['ReportDate'] = report_date
                                
                                rendezvous_anomalies.append(anomaly_record)
            
            if rendezvous_anomalies:
                rendezvous_anomalies_dicts = [record.to_dict() if isinstance(record, pd.Series) else record for record in rendezvous_anomalies]
                anomalies.extend(rendezvous_anomalies_dicts)
                logger.info(f"Found {len(rendezvous_anomalies)} rendezvous anomalies.")
        
        # 5. Identity Spoofing detection
        if config.get('identity_spoofing', True):  # Check if this anomaly type is enabled
            logger.info("Detecting identity spoofing...")
            spoofing_anomalies = []
            
            # Check for multiple vessel names for same MMSI
            if 'VesselName' in df_current_day.columns:
                for mmsi, vessel_data in current_grouped:
                    unique_names = vessel_data['VesselName'].dropna().unique()
                    if len(unique_names) > 1:
                        # Multiple names for same MMSI - potential spoofing
                        anomaly_record = vessel_data.iloc[0].copy()
                        anomaly_record['AnomalyType'] = 'Identity_Spoofing'
                        anomaly_record['SpeedAnomaly'] = False
                        anomaly_record['PositionAnomaly'] = False
                        anomaly_record['CourseAnomaly'] = False
                        anomaly_record['SpoofingIssue'] = 'multiple_vessel_names'
                        anomaly_record['NameCount'] = len(unique_names)
                        anomaly_record['VesselNames'] = ', '.join(unique_names[:5].tolist())  # First 5 names
                        anomaly_record['Date'] = current_date
                        anomaly_record['ReportDate'] = report_date
                        
                        spoofing_anomalies.append(anomaly_record)
            
            # Check for impossible speeds (already detected in speed anomalies, but flag as spoofing too)
            # This is handled by the speed anomaly detection above, so we'll skip duplicate detection here
            
            if spoofing_anomalies:
                spoofing_anomalies_dicts = [record.to_dict() if isinstance(record, pd.Series) else record for record in spoofing_anomalies]
                anomalies.extend(spoofing_anomalies_dicts)
                logger.info(f"Found {len(spoofing_anomalies)} identity spoofing anomalies.")
        
        # 6. Zone Violations detection
        if config.get('zone_violations', True):  # Check if this anomaly type is enabled
            logger.info("Detecting zone violations...")
            zone_violation_anomalies = []
            
            # Get restricted zones from config (default zones if not specified)
            restricted_zones = config.get('RESTRICTED_ZONES', None)
            if restricted_zones is None:
                # Default restricted zones
                restricted_zones = [
                    {'name': 'Strait of Hormuz', 'lat_min': 25.0, 'lat_max': 27.0, 'lon_min': 55.0, 'lon_max': 57.5},
                    {'name': 'South China Sea', 'lat_min': 5.0, 'lat_max': 25.0, 'lon_min': 105.0, 'lon_max': 120.0},
                ]
            
            # Check each zone using geometry helper
            try:
                from zone_geometry import point_in_zone
            except ImportError:
                # Fallback to rectangle check if geometry helper not available
                logger.warning("zone_geometry module not found, using rectangle-only zone detection")
                point_in_zone = None
            
            for zone in restricted_zones:
                zone_name = zone.get('name', 'Unknown Zone')
                
                # Use geometry helper if available, otherwise fallback to rectangle
                if point_in_zone:
                    # Filter using geometry helper
                    def check_point(row):
                        return point_in_zone(row['LAT'], row['LON'], zone)
                    
                    in_zone = df_current_day[df_current_day.apply(check_point, axis=1)]
                else:
                    # Fallback to rectangle check
                    lat_min = zone.get('lat_min', -90)
                    lat_max = zone.get('lat_max', 90)
                    lon_min = zone.get('lon_min', -180)
                    lon_max = zone.get('lon_max', 180)
                    
                    in_zone = df_current_day[
                        (df_current_day['LAT'] >= lat_min) & 
                        (df_current_day['LAT'] <= lat_max) & 
                        (df_current_day['LON'] >= lon_min) & 
                        (df_current_day['LON'] <= lon_max)
                    ]
                
                if len(in_zone) > 0:
                    # Get unique vessels in this zone
                    unique_vessels = in_zone['MMSI'].unique()
                    
                    # Create an anomaly record for each unique vessel
                    for mmsi in unique_vessels:
                        vessel_in_zone = in_zone[in_zone['MMSI'] == mmsi]
                        if len(vessel_in_zone) > 0:
                            anomaly_record = vessel_in_zone.iloc[0].copy()
                            anomaly_record['AnomalyType'] = 'Zone_Violation'
                            anomaly_record['SpeedAnomaly'] = False
                            anomaly_record['PositionAnomaly'] = True
                            anomaly_record['CourseAnomaly'] = False
                            anomaly_record['ZoneName'] = zone_name
                            anomaly_record['ZoneLatMin'] = lat_min
                            anomaly_record['ZoneLatMax'] = lat_max
                            anomaly_record['ZoneLonMin'] = lon_min
                            anomaly_record['ZoneLonMax'] = lon_max
                            anomaly_record['Date'] = current_date
                            anomaly_record['ReportDate'] = report_date
                            
                            zone_violation_anomalies.append(anomaly_record)
            
            if zone_violation_anomalies:
                zone_violation_anomalies_dicts = [record.to_dict() if isinstance(record, pd.Series) else record for record in zone_violation_anomalies]
                anomalies.extend(zone_violation_anomalies_dicts)
                logger.info(f"Found {len(zone_violation_anomalies)} zone violation anomalies.")
        
        # Update previous day reference for next iteration
        df_previous_day = df_current_day
        
        # Add this day's anomalies to the overall list
        all_anomalies.extend(anomalies)
        logger.info(f"Total anomalies detected for {report_date}: {len(anomalies)}")
    
    # Process all anomalies
    if all_anomalies:
        # Create DataFrame from all detected anomalies
        all_anomalies_df = pd.DataFrame(all_anomalies)
        
        # Filter anomalies by selected anomaly types
        if 'AnomalyType' in all_anomalies_df.columns:
            # Map AnomalyType values to config keys
            anomaly_type_mapping = {
                'AIS_Beacon_Off': 'ais_beacon_off',
                'AIS_Beacon_On': 'ais_beacon_on',
                'Speed': 'excessive_travel_distance_fast',  # Speed anomalies are fast travel
                'Course': 'cog-heading_inconsistency',
                'Loitering': 'loitering',
                'Rendezvous': 'rendezvous',
                'Identity_Spoofing': 'identity_spoofing',
                'Zone_Violation': 'zone_violations'
            }
            
            # Build list of enabled anomaly types
            enabled_anomaly_types = []
            for anomaly_type, config_key in anomaly_type_mapping.items():
                # Check if this anomaly type is enabled in config
                if config.get(config_key, True):  # Default to True if not specified
                    enabled_anomaly_types.append(anomaly_type)
            
            # Filter to only include enabled anomaly types
            if enabled_anomaly_types:
                original_count = len(all_anomalies_df)
                all_anomalies_df = all_anomalies_df[all_anomalies_df['AnomalyType'].isin(enabled_anomaly_types)]
                filtered_count = len(all_anomalies_df)
                logger.info(f"Filtered anomalies by type: {filtered_count} of {original_count} anomalies retained (enabled types: {', '.join(enabled_anomaly_types)})")
            else:
                logger.warning("No anomaly types are enabled. All anomalies will be filtered out.")
                all_anomalies_df = pd.DataFrame()  # Return empty DataFrame
        
        # Apply filters based on ANALYSIS_FILTERS settings
        logger.info("Applying analysis filters to detected anomalies")
        all_anomalies_df = filter_anomalies_by_settings(all_anomalies_df, config)
        
        # Create output directory if it doesn't exist
        # First, check if a normalized OUTPUT_DIRECTORY or output_directory key exists
        output_dir_key = None
        for key in ['OUTPUT_DIRECTORY', 'output_directory']:
            if key in config:
                output_dir_key = key
                break
                
        # If not found, try case-insensitive search
        if output_dir_key is None:
            output_dir_key = get_config_key_case_insensitive(config, 'output_directory')
        
        if output_dir_key is None:
            logger.error("No output_directory key found in configuration, using default")
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
            os.makedirs(output_dir, exist_ok=True)
            config['OUTPUT_DIRECTORY'] = output_dir  # Use uppercase for consistency
        else:
            output_dir = config[output_dir_key]
            # Store the path with a consistent key name
            config['OUTPUT_DIRECTORY'] = output_dir
            
        logger.info(f"Debug: Attempting to use output directory: {output_dir}")
        
        # Fix Windows path handling issues
        if '\\' in output_dir or (len(output_dir) > 1 and output_dir[1] == ':'):
            logger.info(f"Debug: Windows path detected, ensuring proper format: {output_dir}")
            # Special case for C:AIS_Data format (without backslash after C:)
            if output_dir.startswith('C:') and not output_dir.startswith('C:\\'):
                logger.info(f"Debug: Detected C: path without proper separator: {output_dir}")
                # Convert C:AIS_Data to C:\AIS_Data
                output_dir = output_dir.replace('C:', 'C:\\')
                logger.info(f"Debug: Fixed C: path: {output_dir}")
                
            # Normalize path for Windows
            output_dir = os.path.normpath(output_dir)
            
            # Ensure the drive letter exists and is accessible
            if len(output_dir) > 1 and output_dir[1] == ':':
                drive_letter = output_dir[0]
                logger.info(f"Debug: Detected drive letter: {drive_letter}")
                
                # Check if this is a valid drive
                if platform.system() == "Windows":
                    try:
                        # Try to import win32api
                        try:
                            import win32api
                        except ImportError:
                            logger.info("Debug: win32api not found. Attempting to install pywin32...")
                            try:
                                # Attempt to install the pywin32 package
                                import subprocess
                                subprocess.check_call([sys.executable, "-m", "pip", "install", "pywin32"])
                                logger.info("Debug: Successfully installed pywin32 package")
                                import win32api
                            except Exception as install_err:
                                logger.warning(f"Debug: Failed to install pywin32 package: {install_err}")
                                # Assume drive exists since we can't verify
                                logger.info("Debug: Proceeding with the provided drive path")
                                # Don't return here - continue with the rest of the function
                                pass
                        
                        # Check available drives
                        drives = win32api.GetLogicalDriveStrings().split('\000')[:-1]
                        logger.info(f"Debug: Available drives: {drives}")
                        
                        # Clean up drive list for comparison (handle different formats)
                        clean_drives = [d.replace('\\', '').upper() for d in drives]
                        clean_drives += [d.strip('\\').upper() for d in drives]
                        
                        # Check if our drive is in the list
                        if f"{drive_letter}:".upper() not in clean_drives and f"{drive_letter}:\\".upper() not in clean_drives:
                            logger.warning(f"Debug: Drive {drive_letter}: may not exist on this system")
                            
                            # Skip fallback if output directory appears to exist
                            if os.path.exists(output_dir):
                                logger.info(f"Debug: Output directory exists despite drive detection issue. Proceeding with: {output_dir}")
                            else:
                                # Use a default path as fallback
                                fallback_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
                                logger.info(f"Debug: Drive not found, using fallback path: {fallback_dir}")
                                os.makedirs(fallback_dir, exist_ok=True)
                                # Use the standardized OUTPUT_DIRECTORY key
                                config['OUTPUT_DIRECTORY'] = fallback_dir
                                output_dir = fallback_dir
                    except Exception as e:
                        logger.warning(f"Debug: Error checking drive availability: {str(e)}")
                        # Proceed with the provided path as a fallback
                        logger.info(f"Debug: Proceeding with the provided output path despite error: {output_dir}")
                        os.makedirs(output_dir, exist_ok=True)
            
            # Ensure the directory ends with backslash for Windows paths if it doesn't already
            if not output_dir.endswith('\\') and '\\' in output_dir:
                output_dir += '\\'
        
        # Make directory with detailed debug messages
        try:
            logger.info(f"Debug: Creating directory if it doesn't exist: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Debug: Directory exists or was created: {output_dir}")
            
            # Test if directory is writable
            test_file = os.path.join(output_dir, '.write_test')
            logger.info(f"Debug: Testing write permissions with file: {test_file}")
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                logger.info(f"Debug: Write permission verified for directory: {output_dir}")
            except Exception as e:
                logger.warning(f"Debug: Output directory {output_dir} exists but may not be writable: {str(e)}")
                # If file operations fail, try to diagnose the issue
                logger.warning(f"Debug: File path details - absolute: {os.path.abspath(output_dir)}, exists: {os.path.exists(output_dir)}, is_dir: {os.path.isdir(output_dir) if os.path.exists(output_dir) else 'N/A'}")
        except Exception as e:
            logger.error(f"Debug: Failed to create output directory {output_dir}: {str(e)}")
            # Try to use a fallback directory
            fallback_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
            logger.info(f"Debug: Using fallback output directory: {fallback_dir}")
            os.makedirs(fallback_dir, exist_ok=True)
            # Use the standardized OUTPUT_DIRECTORY key
            config['OUTPUT_DIRECTORY'] = fallback_dir
            output_dir = fallback_dir
        
        # Always save anomalies to CSV
        try:
            # Use the standardized OUTPUT_DIRECTORY key
            summary_path = os.path.join(config['OUTPUT_DIRECTORY'], "AIS_Anomalies_Summary.csv")
            logger.info(f"Debug: Attempting to save CSV to: {summary_path}")
            all_anomalies_df.to_csv(summary_path, index=False)
            logger.info(f"Debug: Successfully saved CSV to: {summary_path}")
            
            # Calculate global boundaries for all maps
            calculate_global_boundaries(all_anomalies_df)
        except Exception as e:
            logger.error(f"Debug: Failed to save CSV file: {str(e)}")
        
        # Create Charts directory if it doesn't exist
        charts_dir = os.path.join(config['OUTPUT_DIRECTORY'], "Charts")
        try:
            logger.info(f"Debug: Creating charts directory if it doesn't exist: {charts_dir}")
            os.makedirs(charts_dir, exist_ok=True)
            logger.info(f"Debug: Charts directory exists or was created: {charts_dir}")
            
            # Create summary charts
            logger.info(f"Debug: Generating summary charts in: {charts_dir}")
            logger.info(f"Debug: Number of anomalies for charts: {len(all_anomalies_df)}")
            logger.info(f"Debug: Chart generation enabled: {config.get('generate_charts', True)}")
            create_summary_charts(all_anomalies_df, charts_dir, config)
            logger.info(f"Debug: Successfully generated charts in: {charts_dir}")
        except Exception as e:
            logger.error(f"Debug: Failed to create charts: {str(e)}")
            logger.error(f"Debug: Chart generation error details: {traceback.format_exc()}")
            # Try to continue with other reports
        
        # Create overall map
        try:
            # Use the standardized OUTPUT_DIRECTORY key
            overall_map_path = os.path.join(config['OUTPUT_DIRECTORY'], "All Anomalies Map.html")
            logger.info(f"Debug: Creating overall map at: {overall_map_path}")
            create_map_visualization(all_anomalies_df, overall_map_path, config)
            logger.info(f"Debug: Successfully created overall map at: {overall_map_path}")
        except Exception as e:
            logger.error(f"Debug: Failed to create overall map: {str(e)}")
        
        # Generate analysis statistics and save to CSV/Excel if requested
        # Check if stats generation is requested
        global statistics_thread, statistics_requested, statistics_completed
        statistics_completed = False
        
        if isinstance(config, dict):
            generate_statistics_excel = config.get('generate_statistics_excel', True)
            generate_statistics_csv = config.get('generate_statistics_csv', True)
            statistics_requested = generate_statistics_excel or generate_statistics_csv
        else:
            statistics_requested = True  # Default to true if config is not a dict
        
        # If statistics generation is requested, start it in a background thread
        if statistics_requested:
            try:
                # Use the standardized OUTPUT_DIRECTORY key
                logger.info(f"Debug: Starting analysis statistics generation in background: {config['OUTPUT_DIRECTORY']}")
                # Create and start the background thread
                statistics_thread = threading.Thread(
                    target=run_statistics_in_background,
                    args=(all_daily_data, dates_in_order, config, config['OUTPUT_DIRECTORY'])
                )
                statistics_thread.daemon = True  # Thread will exit when main program exits
                statistics_thread.start()
            except Exception as e:
                logger.error(f"Debug: Failed to start background statistics generation: {str(e)}")
                statistics_completed = True  # Mark as completed on error
        else:
            logger.info("Debug: Analysis statistics generation not requested")
            statistics_completed = True  # Mark as completed if not requested
        
        # Create vessel path maps
        try:
            # Use the standardized OUTPUT_DIRECTORY key
            logger.info(f"Debug: Creating vessel path maps in: {config['OUTPUT_DIRECTORY']}")
            create_vessel_path_maps(all_daily_data, dates_in_order, config, config['OUTPUT_DIRECTORY'])
            logger.info(f"Debug: Successfully created vessel path maps")
        except Exception as e:
            logger.error(f"Debug: Failed to create vessel path maps: {str(e)}")
            
        # Generate heatmaps
        try:
            if config.get('show_anomaly_heatmap', True):
                logger.info(f"Debug: Generating anomaly heatmaps")
                create_anomalies_heatmap(all_anomalies_df, config, config['OUTPUT_DIRECTORY'])
                logger.info(f"Debug: Successfully created anomaly heatmap")
            else:
                logger.info("Anomaly heatmap generation is disabled in configuration")
        except Exception as e:
            logger.error(f"Failed to generate anomaly heatmap: {str(e)}")
            
        # Final verification of output directory contents
        try:
            # Use the standardized OUTPUT_DIRECTORY key
            logger.info(f"Debug: Verifying output directory contents at: {config['OUTPUT_DIRECTORY']}")
            files_created = os.listdir(config['OUTPUT_DIRECTORY'])
            logger.info(f"Debug: Files in output directory: {files_created}")
        except Exception as e:
            logger.error(f"Debug: Failed to list output directory contents: {str(e)}")
        
        # Save the consolidated dataframe for future use
        try:
            logger.info("Saving consolidated dataframe for future analysis...")
            consolidated_path = save_concatenated_dataframe(all_daily_data, config)
            if consolidated_path:
                logger.info(f"Consolidated dataframe saved to: {consolidated_path}")
            else:
                logger.warning("Failed to save consolidated dataframe")
        except Exception as e:
            logger.error(f"Error while saving consolidated dataframe: {e}")
            
        logger.info(f"AIS Fraud Detection Complete. Found {len(all_anomalies)} anomalies across {len(dates_in_order)} days.")
        return all_anomalies_df
    else:
        logger.info(f"AIS Fraud Detection Complete. No anomalies detected.")
        return pd.DataFrame()


def get_config_key_case_insensitive(config, key):
    """
    Helper function to get a configuration key in a case-insensitive manner.
    
    Args:
        config (dict): Configuration dictionary
        key (str): Key to look for (case insensitive)
        
    Returns:
        str: The actual key in the config dict that matches case-insensitively, or None if not found
    """
    for config_key in config.keys():
        if config_key.lower() == key.lower():
            return config_key
    return None


def filter_anomalies_by_settings(anomalies_df, config):
    """
    Filter anomalies based on configuration settings in the ANALYSIS_FILTERS section.
    
    Args:
        anomalies_df (DataFrame): DataFrame containing detected anomalies
        config (dict): Configuration dictionary with filter settings
    
    Returns:
        DataFrame: Filtered anomalies DataFrame
    """
    if anomalies_df.empty:
        return anomalies_df
        
    # Create a copy of the original DataFrame to avoid modifying it
    filtered_df = anomalies_df.copy()
    orig_count = len(filtered_df)
    logger.info(f"Applying analysis filters to {orig_count} anomalies")
    
    # Note: Anomaly type filters have been removed
    logger.info(f"Anomaly type filtering: No filter applied, all anomalies retained")
    
    # 1. Geographic filtering
    if ('min_latitude' in config and 'max_latitude' in config and 
            'min_longitude' in config and 'max_longitude' in config):
        # Apply geographic bounds
        geo_filtered = filtered_df[
            (filtered_df['LAT'] >= config['min_latitude']) & 
            (filtered_df['LAT'] <= config['max_latitude']) & 
            (filtered_df['LON'] >= config['min_longitude']) & 
            (filtered_df['LON'] <= config['max_longitude'])
        ]
        logger.info(f"Geographic filtering: {len(geo_filtered)} of {orig_count} anomalies within bounds")
        filtered_df = geo_filtered
    
    # 2. Time-based filtering (if BaseDateTime exists)
    if 'BaseDateTime' in filtered_df.columns and 'time_start_hour' in config and 'time_end_hour' in config:
        # Extract hour from BaseDateTime
        filtered_df['Hour'] = filtered_df['BaseDateTime'].dt.hour
        
        # Apply time filter
        if config['time_start_hour'] < config['time_end_hour']:  # Normal time range (e.g., 8-17)
            time_filtered = filtered_df[
                (filtered_df['Hour'] >= config['time_start_hour']) & 
                (filtered_df['Hour'] < config['time_end_hour'])
            ]
        else:  # Overnight range (e.g., 22-4)
            time_filtered = filtered_df[
                (filtered_df['Hour'] >= config['time_start_hour']) | 
                (filtered_df['Hour'] < config['time_end_hour'])
            ]
            
        logger.info(f"Time filtering: {len(time_filtered)} of {len(filtered_df)} anomalies within time range {config['time_start_hour']}-{config['time_end_hour']}")
        filtered_df = time_filtered.drop(columns=['Hour'])  # Remove the temporary hour column
    
    # 3. MMSI filtering
    if 'filter_mmsi_list' in config and config['filter_mmsi_list']:
        try:
            # Convert string MMSIs to integers if they're not already
            mmsi_list = config['filter_mmsi_list']
            if isinstance(mmsi_list, str):
                mmsi_list = [int(mmsi.strip()) for mmsi in mmsi_list.split(',') if mmsi.strip()]
                
            mmsi_filtered = filtered_df[filtered_df['MMSI'].isin(mmsi_list)]
            logger.info(f"MMSI filtering: {len(mmsi_filtered)} of {len(filtered_df)} anomalies from specified MMSI list")
            filtered_df = mmsi_filtered
        except Exception as e:
            logger.warning(f"Error in MMSI filtering: {e}. Using all vessels.")
            # Continue with unfiltered data if there's an error
    
    # 4. Apply anomaly count limits per vessel
    if 'max_anomalies_per_vessel' in config and config['max_anomalies_per_vessel'] > 0:
        # Group by MMSI and sort by confidence or another relevant factor
        # For simplicity, we're just taking the first n anomalies per vessel
        mmsi_groups = filtered_df.groupby('MMSI')
        limited_dfs = []
        
        for mmsi, group in mmsi_groups:
            # Sort by significance if we have a confidence metric, otherwise use default order
            # For this example, we're assuming higher confidence is better
            if 'Confidence' in group.columns:
                sorted_group = group.sort_values('Confidence', ascending=False)
            else:
                sorted_group = group
                
            # Take only the top N anomalies per vessel
            limited_dfs.append(sorted_group.head(config['max_anomalies_per_vessel']))
        
        # Combine all the limited groups back together
        if limited_dfs:
            count_limited = pd.concat(limited_dfs)
            logger.info(f"Anomaly count limiting: {len(count_limited)} of {len(filtered_df)} anomalies after applying max {config['max_anomalies_per_vessel']} per vessel")
            filtered_df = count_limited
    
    logger.info(f"Analysis filtering complete: {len(filtered_df)} anomalies remain from original {orig_count}")
    return filtered_df


def detect_shipping_anomalies_by_date_range(start_date, end_date, config_input='config.ini', use_dask=True):
    """
    Main function to orchestrate the loading, processing, and anomaly detection using date range.
    
    Args:
        start_date (date or str): Start date for analysis
        end_date (date or str): End date for analysis
        config_input (str or dict): Path to configuration file or configuration dictionary
        use_dask (bool): Whether to use Dask for large data processing
        
    Returns:
        DataFrame: Detected anomalies
    """
    # Parse date strings if provided
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    # Load configuration
    if isinstance(config_input, dict):
        config = config_input
    else:
        config = load_config(config_input)
    
    # Store the start and end dates in the config dictionary
    # Convert dates to string format for consistent handling
    config['START_DATE'] = start_date.strftime('%Y-%m-%d')
    config['END_DATE'] = end_date.strftime('%Y-%m-%d')
    logger.info(f"Set date range in config: {config['START_DATE']} to {config['END_DATE']}")
    
    # Get the data directory (local or S3)
    data_dir_key = get_config_key_case_insensitive(config, 'data_directory')
    
    if data_dir_key is None:
        logger.error("No data_directory key found in configuration")
        return pd.DataFrame()
    
    if config.get('USE_S3', False):
        data_dir = config.get('S3_DATA_URI', '')
        
        # Validate S3 data URI
        if not data_dir or not is_s3_uri(data_dir):
            logger.error(f"Invalid S3 data URI: {data_dir}")
            # Fall back to local data directory
            if 'DATA_DIRECTORY' in config:
                logger.warning("Falling back to local data directory")
                data_dir = config['DATA_DIRECTORY']
                config['USE_S3'] = False
            else:
                logger.error("No valid data directory found in config")
                return pd.DataFrame()
        else:
            # Test if AWS credentials work properly
            logger.info(f"Using S3 data from: {data_dir}")
            if not test_aws_credentials(config):
                logger.error("AWS credentials failed validation")
                # Offer to try local mode instead
                if 'DATA_DIRECTORY' in config and config['DATA_DIRECTORY']:
                    logger.warning("Falling back to local data directory due to S3 credential failure")
                    data_dir = config['DATA_DIRECTORY']
                    config['USE_S3'] = False
                else:
                    logger.error("No valid local data directory found as fallback")
                    return pd.DataFrame()
    else:
        data_dir = config[data_dir_key]
    
    # Find files for the date range
    file_paths, dates_in_order = get_files_for_date_range(data_dir, start_date, end_date, config)
    
    if not file_paths:
        logger.error("No valid files found for the specified date range.")
        return pd.DataFrame()
    
    return _process_anomaly_detection(file_paths, dates_in_order, config, use_dask)


def test_aws_credentials(config):
    """
    Test if the AWS credentials in config are valid by making a simple API call
    
    Args:
        config (dict): Configuration dictionary with AWS credentials
        
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    try:
        if not config or 'AWS' not in config:
            logger.error("No AWS configuration found in config")
            return False
        
        # Get credentials
        access_key = config['AWS'].get('s3_access_key', '').strip()
        secret_key = config['AWS'].get('s3_secret_key', '').strip()
        session_token = config['AWS'].get('s3_session_token', '').strip()
        region = config['AWS'].get('s3_region', '')
        
        if not access_key or not secret_key:
            logger.error("AWS access key or secret key is missing")
            return False
            
        # Try to create a session and make a simple API call
        try:
            if session_token:
                session = boto3.Session(
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    aws_session_token=session_token,
                    region_name=region if region else None
                )
            else:
                session = boto3.Session(
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    region_name=region if region else None
                )
                
            # Test if credentials work by listing S3 buckets
            s3 = session.client('s3')
            s3.list_buckets()
            return True
        except Exception as e:
            error_str = str(e)
            logger.error(f"AWS credential test failed: {e}")
            
            # Check if this is an ExpiredToken error
            if 'ExpiredToken' in error_str or 'expired' in error_str.lower():
                # Check if running from GUI
                is_gui_mode = (not sys.stdin.isatty() if hasattr(sys.stdin, 'isatty') else False) or os.environ.get('SFD_GUI_MODE', '').lower() == 'true'
                
                if is_gui_mode:
                    # Show popup message
                    try:
                        root = tk.Tk()
                        root.withdraw()  # Hide the root window
                        # Bring window to front and ensure it's visible
                        root.lift()
                        root.attributes('-topmost', True)
                        root.update()
                        
                        result = messagebox.showwarning(
                            "AWS Tokens Failed",
                            "AWS Tokens Failed\n\nPlease input updated AWS token and retry analysis.",
                            parent=root
                        )
                        root.attributes('-topmost', False)
                        root.destroy()
                        # Exit with code 101 to signal GUI to close progress window
                        sys.exit(101)
                    except Exception as popup_error:
                        logger.error(f"Error showing AWS token popup: {popup_error}")
                        # Still exit even if popup fails
                        sys.exit(101)
                else:
                    # Command line mode - just log and exit
                    logger.error("AWS session token has expired. Please update your AWS credentials and retry.")
                    sys.exit(101)
            
            return False
    except Exception as e:
        logger.error(f"Error testing AWS credentials: {e}")
        return False

def check_aws_configuration(config):
    """
    Validate AWS configuration and log details about the S3 setup
    
    Args:
        config (dict): Configuration dictionary
    """
    if not config.get('USE_S3', False):
        logger.info("S3 access is not enabled in config.ini")
        return
    
    # Test if credentials actually work
    credentials_valid = test_aws_credentials(config)
    if credentials_valid:
        logger.info("AWS credentials verified successfully!")
    else:
        logger.warning("AWS credentials verification FAILED - S3 access may not work correctly")

    
    logger.info("AWS S3 configuration:")
    s3_data_uri = config.get('S3_DATA_URI', '')
    
    if not s3_data_uri:
        logger.warning("USE_S3 is enabled but S3_DATA_URI is not set in config.ini")
    else:
        logger.info(f"  S3 Data URI: {s3_data_uri}")
    
    # Using keys authentication method
    logger.info("  Authentication method: keys")
    
    # Log credential presence (not the actual values)
    aws_config = config.get('AWS', {})
    access_key = aws_config.get('s3_access_key')
    secret_key = aws_config.get('s3_secret_key')
    session_token = aws_config.get('s3_session_token')
    region = aws_config.get('s3_region', 'us-east-1')
    
    has_access_key = bool(access_key)
    has_secret_key = bool(secret_key)
    has_token = bool(session_token)
    
    # Set AWS credentials as environment variables if found in config
    if has_access_key and has_secret_key:
        logger.info("  AWS access key and secret key are set")
        os.environ['AWS_ACCESS_KEY_ID'] = access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key
        
        if has_token:
            logger.info("  AWS session token is set")
            os.environ['AWS_SESSION_TOKEN'] = session_token
        else:
            logger.info("  AWS session token is not set (not required for permanent credentials)")
            
        # Set region if available
        if region:
            os.environ['AWS_DEFAULT_REGION'] = region
            logger.info(f"  AWS region set to: {region}")
    else:
        logger.warning("  AWS access key and/or secret key not set in config.ini")
    
    # Check for environment variables
    aws_env_vars = [
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'AWS_SESSION_TOKEN',
        'AWS_PROFILE',
        'AWS_DEFAULT_REGION'
    ]
    
    env_vars_present = [var for var in aws_env_vars if var in os.environ]
    if env_vars_present:
        logger.info(f"  AWS environment variables found: {', '.join(env_vars_present)}")


def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(description='AIS Shipping Fraud Detection System')
    parser.add_argument('--start-date', type=str, required=False, help='Start date in format YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, required=False, help='End date in format YYYY-MM-DD')
    parser.add_argument('--config', type=str, default='config.ini', help='Path to configuration file')
    parser.add_argument('--data-source', type=str, choices=['noaa', 'local', 's3'], help='Source of AIS data')
    parser.add_argument('--noaa-year', type=str, help='DEPRECATED: Year for NOAA data - now automatically extracted from start-date')
    parser.add_argument('--no-dask', action='store_true', help='Disable Dask processing for large files')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU processing even if available')
    parser.add_argument('--force-gpu', action='store_true', help='Try to use GPU even if not detected (may cause errors)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--show-warnings', action='store_true', help='Show Python warnings (overrides suppress_warnings in config.ini)')
    parser.add_argument('--ship-types', type=str, help='Comma-separated list of ship types to analyze (e.g., "70,80,81")')
    parser.add_argument('--output-directory', type=str, help='Directory to save analysis output files')
    parser.add_argument('--data-directory', type=str, help='Directory containing input data files')
    parser.add_argument('--disable-cache', action='store_true', help='Disable data caching')
    
    # Analysis filter options
    parser.add_argument('--min-latitude', type=float, help='Minimum latitude for geographic filtering')
    parser.add_argument('--max-latitude', type=float, help='Maximum latitude for geographic filtering')
    parser.add_argument('--min-longitude', type=float, help='Minimum longitude for geographic filtering')
    parser.add_argument('--max-longitude', type=float, help='Maximum longitude for geographic filtering')
    parser.add_argument('--time-start-hour', type=int, help='Start hour for time filtering (0-24)')
    parser.add_argument('--time-end-hour', type=int, help='End hour for time filtering (0-24)')
    parser.add_argument('--min-confidence', type=int, help='Minimum confidence level for anomalies (0-100)')
    parser.add_argument('--max-anomalies-per-vessel', type=int, help='Maximum anomalies to report per vessel')
    parser.add_argument('--mmsi-list', type=str, help='Comma-separated list of MMSIs to filter')
    
    # AWS authentication options (keys method only)
    parser.add_argument('--access-key', type=str, help='AWS access key ID for S3 access')
    parser.add_argument('--secret-key', type=str, help='AWS secret access key for S3 access')
    parser.add_argument('--region', type=str, help='AWS region name', default='us-east-1')
    parser.add_argument('--session-token', type=str, help='AWS session token for temporary credentials')
    parser.add_argument('--bucket', type=str, help='S3 bucket name')
    parser.add_argument('--prefix', type=str, help='S3 object prefix (path)')
    
    # Advanced Analysis options
    parser.add_argument('--advanced-analysis', type=str, 
                       choices=['export-full-dataset', 'summary-report', 'vessel-statistics', 
                               'anomaly-timeline', 'temporal-patterns', 'vessel-clustering',
                               'anomaly-frequency', 'full-spectrum-map', 'vessel-map'],
                       help='Run advanced analysis feature')
    parser.add_argument('--vessel-mmsi', type=int, 
                       help='MMSI number for vessel-specific analysis (required for vessel-map)')
    parser.add_argument('--map-type', type=str, choices=['path', 'anomaly', 'heatmap'],
                       help='Map type for vessel-map (default: path)')
    parser.add_argument('--no-show-pins', dest='show_pins', action='store_false', default=True,
                       help='Hide pins on full spectrum map (default: show pins)')
    parser.add_argument('--no-show-heatmap', dest='show_heatmap', action='store_false', default=True,
                       help='Hide heatmap on full spectrum map (default: show heatmap)')
    parser.add_argument('--extended-start-date', type=str,
                       help='Start date for extended time analysis (YYYY-MM-DD)')
    parser.add_argument('--extended-end-date', type=str,
                       help='End date for extended time analysis (YYYY-MM-DD)')
    parser.add_argument('--n-clusters', type=int, default=5,
                       help='Number of clusters for vessel behavior clustering (default: 5)')
    
    # Catch any parser errors
    try:
        args = parser.parse_args()
    except Exception as e:
        print(f"ERROR: Failed to parse arguments: {e}")
        print(f"Arguments received: {sys.argv[1:]}")
        return 1
    
    # Set logging level if debug mode is enabled
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Always print startup information to help with debugging
    logger.info("======================================")
    logger.info("SFD.py starting with parameters:")
    logger.info(f"Start date: {args.start_date}")
    logger.info(f"End date: {args.end_date}")
    logger.info(f"Data source: {args.data_source}")
    logger.info(f"Cache: {'Disabled' if args.disable_cache else 'Enabled'}")
    if args.data_source == 'noaa':
        if args.noaa_year:
            logger.info(f"NOAA year: {args.noaa_year} (from parameter)")
        elif args.start_date:
            # Extract year from start date for logging
            try:
                from datetime import datetime
                year = datetime.strptime(args.start_date, '%Y-%m-%d').year
                logger.info(f"NOAA year: {year} (from start date)")
            except ValueError:
                logger.info("NOAA year: Not available (could not parse start date)")
    logger.info(f"Data directory: {args.data_directory}")
    logger.info(f"Output directory: {args.output_directory}")
    logger.info(f"Debug mode: {args.debug}")
    logger.info("======================================")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Check AWS configuration for S3 access
        check_aws_configuration(config)
        
        # Apply warning suppression based on config and command-line args
        # Command-line --show-warnings overrides config setting
        if args.show_warnings:
            suppress_warnings(False)  # Show warnings
            logger.info("Showing warnings due to --show-warnings command-line option")
        else:
            suppress_warnings(config.get('suppress_warnings', True))
        
        # Get start and end dates from config file if not provided in command line
        if args.start_date is None:
            # Check for case-insensitive match for start_date and end_date
            for key in config.keys():
                if key.lower() == 'start_date':
                    args.start_date = config[key]
                    logger.info(f"Using start date from config: {args.start_date}")
                    break
        if args.end_date is None:
            for key in config.keys():
                if key.lower() == 'end_date':
                    args.end_date = config[key]
                    logger.info(f"Using end date from config: {args.end_date}")
                    break
        
        # Override configuration with command-line arguments
        if args.ship_types:
            try:
                config['SELECTED_SHIP_TYPES'] = [int(t.strip()) for t in args.ship_types.split(',') if t.strip()]
            except ValueError:
                logger.error("Invalid ship types specified. Using default types.")
        
        # Handle output directory if specified, with path normalization
        if args.output_directory:
            output_dir = args.output_directory
            
            # Special case for C:path format (without backslash after C:)
            if output_dir.startswith('C:') and not output_dir.startswith('C:\\'):
                output_dir = output_dir.replace('C:', 'C:\\')
                logger.info(f"Fixed C: path format to: {output_dir}")
            
            # Normalize the path
            output_dir = os.path.normpath(output_dir)
            config['OUTPUT_DIRECTORY'] = output_dir
            logger.info(f"Output directory set to: {config['OUTPUT_DIRECTORY']}")
            
            # Verify the directory exists or can be created
            try:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Verified output directory exists or was created: {output_dir}")
            except Exception as e:
                logger.error(f"Failed to create output directory: {str(e)}")
                logger.warning(f"Will attempt to use it anyway when needed")

        # Store data source in config
        if args.data_source:
            config['DATA_SOURCE'] = args.data_source.lower()
            logger.info(f"Data source set to: {config['DATA_SOURCE']}")
            
            # Handle NOAA year - extract from start date if not provided
            if args.data_source.lower() == 'noaa':
                if args.noaa_year:
                    # Use provided year if specified (backward compatibility)
                    config['NOAA_YEAR'] = args.noaa_year
                    logger.info(f"NOAA year set to: {config['NOAA_YEAR']} (from command line)")
                elif args.start_date:
                    # Extract year from start_date
                    try:
                        from datetime import datetime
                        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
                        noaa_year = str(start_date.year)
                        config['NOAA_YEAR'] = noaa_year
                        logger.info(f"NOAA year set to: {config['NOAA_YEAR']} (extracted from start date)")
                    except ValueError as e:
                        logger.warning(f"Failed to extract NOAA year from start date: {e}")
                else:
                    logger.warning("No NOAA year specified and couldn't extract from start date")
        
        # Handle data directory if specified, with path normalization
        if args.data_directory:
            data_dir = args.data_directory
            
            # Special case for C:path format (without backslash after C:)
            if data_dir.startswith('C:') and not data_dir.startswith('C:\\'):
                data_dir = data_dir.replace('C:', 'C:\\')
                logger.info(f"Fixed C: path format to: {data_dir}")
            
            # Normalize the path
            data_dir = os.path.normpath(data_dir)
            config['DATA_DIRECTORY'] = data_dir
            logger.info(f"Data directory set to: {config['DATA_DIRECTORY']}")
            
            # Verify the directory exists
            if not os.path.exists(data_dir):
                logger.error(f"Data directory does not exist: {data_dir}")
                logger.warning("Continuing anyway, but analysis may fail without valid data directory")
        # Print a debug message with all configuration parameters to help troubleshoot
        if args.debug:
            logger.debug("Configuration parameters:")
            for key, value in config.items():
                if isinstance(value, dict):
                    logger.debug(f"{key}:")
                    for subkey, subvalue in value.items():
                        logger.debug(f"  {subkey}: {subvalue}")
                else:
                    logger.debug(f"{key}: {value}")

        # Handle GPU options
        if args.no_gpu:
            config['USE_GPU'] = False
            logger.info("GPU processing disabled via command line")
        elif args.force_gpu:
            config['USE_GPU'] = True
            logger.info("GPU processing forced via command line (may cause errors if GPU libraries not available)")
            
        # Handle caching options
        if args.disable_cache:
            config['DISABLE_CACHE'] = True
            logger.info("Data caching disabled via command line")
        else:
            config['DISABLE_CACHE'] = False
            
        # No more filter toggle processing
        
        # Process analysis filter parameters
        # Geographic boundaries
        if args.min_latitude is not None:
            config['min_latitude'] = args.min_latitude
            logger.info(f"Minimum latitude set to: {args.min_latitude}")
        if args.max_latitude is not None:
            config['max_latitude'] = args.max_latitude
            logger.info(f"Maximum latitude set to: {args.max_latitude}")
        if args.min_longitude is not None:
            config['min_longitude'] = args.min_longitude
            logger.info(f"Minimum longitude set to: {args.min_longitude}")
        if args.max_longitude is not None:
            config['max_longitude'] = args.max_longitude
            logger.info(f"Maximum longitude set to: {args.max_longitude}")
        
        # Time filters
        if args.time_start_hour is not None:
            config['time_start_hour'] = args.time_start_hour
            logger.info(f"Time start hour set to: {args.time_start_hour}")
        if args.time_end_hour is not None:
            config['time_end_hour'] = args.time_end_hour
            logger.info(f"Time end hour set to: {args.time_end_hour}")
        
        # Anomaly filtering
        if args.min_confidence is not None:
            config['min_confidence'] = args.min_confidence
            logger.info(f"Minimum confidence level set to: {args.min_confidence}")
        if args.max_anomalies_per_vessel is not None:
            config['max_anomalies_per_vessel'] = args.max_anomalies_per_vessel
            logger.info(f"Maximum anomalies per vessel set to: {args.max_anomalies_per_vessel}")
        
        # MMSI filtering
        if args.mmsi_list:
            try:
                # Parse comma-separated list of MMSIs to integers
                mmsi_list = [int(mmsi.strip()) for mmsi in args.mmsi_list.split(',') if mmsi.strip()]
                config['filter_mmsi_list'] = mmsi_list
                logger.info(f"MMSI filter list set to: {mmsi_list}")
            except Exception as e:
                logger.warning(f"Error parsing MMSI list: {e}. Format should be comma-separated integers.")
        
        # Set AWS credentials if provided via command line
        if args.access_key and args.secret_key:
            logger.info(f"Using provided AWS access keys")
            os.environ['AWS_ACCESS_KEY_ID'] = args.access_key
            os.environ['AWS_SECRET_ACCESS_KEY'] = args.secret_key
            if args.region:
                os.environ['AWS_DEFAULT_REGION'] = args.region
                
        # Add session token if provided
        if args.session_token:
            logger.info("Using provided AWS session token")
            os.environ['AWS_SESSION_TOKEN'] = args.session_token
                
        # Using only key-based authentication
        
        # Set S3 URI if bucket was specified
        if args.bucket:
            s3_uri = f"s3://{args.bucket}/{args.prefix.lstrip('/') if args.prefix else ''}"
            logger.info(f"Using S3 URI: {s3_uri}")
            config['USE_S3'] = True  # Enable S3 access
            config['S3_DATA_URI'] = s3_uri
            config['DATA_DIRECTORY'] = s3_uri
        
        # Handle advanced analysis if requested (can run without main analysis)
        if args.advanced_analysis:
            if not ADVANCED_ANALYSIS_AVAILABLE:
                logger.error("Advanced analysis module is not available")
                print("ERROR: Advanced analysis module is not available.")
                print("Please ensure advanced_analysis.py is in the same directory as SFD.py")
                return 1
            
            logger.info(f"Running advanced analysis: {args.advanced_analysis}")
            
            try:
                output_dir = config.get('OUTPUT_DIRECTORY', 'output')
                analysis = AdvancedAnalysis(None, output_dir, args.config)
                
                result = None
                
                if args.advanced_analysis == 'export-full-dataset':
                    result = analysis.export_full_dataset()
                    print(f"Full dataset exported to: {result}")
                
                elif args.advanced_analysis == 'summary-report':
                    result = analysis.generate_summary_report()
                    print(f"Summary report generated: {result}")
                
                elif args.advanced_analysis == 'vessel-statistics':
                    result = analysis.export_vessel_statistics()
                    print(f"Vessel statistics exported to: {result}")
                
                elif args.advanced_analysis == 'anomaly-timeline':
                    result = analysis.generate_anomaly_timeline()
                    print(f"Anomaly timeline generated: {result}")
                
                elif args.advanced_analysis == 'temporal-patterns':
                    result = analysis.temporal_pattern_analysis()
                    print(f"Temporal pattern analysis completed: {result}")
                
                elif args.advanced_analysis == 'vessel-clustering':
                    result = analysis.vessel_behavior_clustering(n_clusters=args.n_clusters)
                    print(f"Vessel clustering completed: {result}")
                
                elif args.advanced_analysis == 'anomaly-frequency':
                    result = analysis.anomaly_frequency_analysis()
                    print(f"Anomaly frequency analysis completed: {result}")
                
                elif args.advanced_analysis == 'full-spectrum-map':
                    result = analysis.create_full_spectrum_map(
                        show_pins=args.show_pins,
                        show_heatmap=args.show_heatmap
                    )
                    print(f"Full spectrum map created: {result}")
                
                elif args.advanced_analysis == 'vessel-map':
                    if not args.vessel_mmsi:
                        logger.error("--vessel-mmsi is required for vessel-map")
                        print("ERROR: --vessel-mmsi is required for vessel-map")
                        return 1
                    
                    map_type = args.map_type or 'path'
                    result = analysis.create_vessel_map(args.vessel_mmsi, map_type)
                    print(f"Vessel map created: {result}")
                
                if result:
                    logger.info(f"Advanced analysis completed successfully: {result}")
                    # Open the result file if it's an HTML file
                    if result.endswith('.html'):
                        try:
                            if platform.system() == "Windows":
                                os.startfile(result)
                            elif platform.system() == "Darwin":  # macOS
                                subprocess.call(["open", result])
                            else:  # Linux
                                subprocess.call(["xdg-open", result])
                        except Exception as e:
                            logger.warning(f"Could not open result file: {e}")
                    return 0
                else:
                    logger.error("Advanced analysis failed to produce output")
                    return 1
                    
            except Exception as e:
                logger.error(f"Error running advanced analysis: {e}")
                logger.error(traceback.format_exc())
                print(f"ERROR: Advanced analysis failed: {e}")
                return 1
        
        # Run anomaly detection (only if dates provided)
        if args.start_date is None or args.end_date is None:
            logger.error("Start date and end date are required. Please provide them as command-line arguments or in the config file.")
            return 1
            
        logger.info(f"Running fraud detection for date range {args.start_date} to {args.end_date}")
        detect_shipping_anomalies_by_date_range(
            args.start_date, 
            args.end_date, 
            config,
            not args.no_dask
        )
        
        # Open the output directory after completion
        output_dir = config.get('OUTPUT_DIRECTORY', 'output')
        try:
            if platform.system() == "Windows":
                os.startfile(output_dir)
            elif platform.system() == "Darwin":  # macOS
                subprocess.call(["open", output_dir])
            else:  # Linux
                subprocess.call(["xdg-open", output_dir])
        except Exception as e:
            logger.error(f"Failed to open output directory: {e}")
        
        # Check if running from GUI or command line
        # Detect GUI mode by checking if stdin is not a TTY (piped/redirected) or if environment variable is set
        is_gui_mode = (not sys.stdin.isatty() if hasattr(sys.stdin, 'isatty') else False) or os.environ.get('SFD_GUI_MODE', '').lower() == 'true'
        
        # Only show dialog if running from GUI
        # Otherwise, just log completion and exit cleanly
        if is_gui_mode:
            # Create a simple tkinter root window (hidden)
            root = None
            try:
                root = tk.Tk()
                root.withdraw()  # Hide the root window
                # Bring window to front and ensure it's visible
                root.lift()
                root.attributes('-topmost', True)
                root.update()
                
                # Show dialog asking if user wants to conduct additional analysis
                # Determine the status of the statistics report
                stats_status = ""
                if not statistics_requested:
                    stats_status = "was not requested."
                elif statistics_completed:
                    stats_status = "is complete."
                else:
                    stats_status = "is not yet complete, but running in the background."
                
                result = messagebox.askquestion(
                    "Initial Analysis Phase complete.", 
                    f"Analysis Statistics Report {stats_status}\n\nWould you like to conduct additional analysis on this dataset?",
                    type="yesnocancel",
                    parent=root
                )
                root.attributes('-topmost', False)
                
                if result == 'yes':
                    # User wants to conduct additional analysis on current dataset
                    # Return with special code 101 to signal SFD_GUI.py to open additional analysis window
                    return 101
                elif result == 'no':
                    # User wants to continue with a new analysis
                    # Return 0 which keeps SFD_GUI.py running for a new analysis
                    return 0
                else:  # 'cancel' or window closed
                    # User doesn't want to continue
                    # Exit with code 100 to signal SFD_GUI.py to close as well
                    sys.exit(100)
            except Exception as e:
                # If there's an error with the messagebox, log and exit
                logger.error(f"Error showing completion dialog: {e}")
                return 0
            finally:
                # Always destroy tkinter window to ensure CMD window closes properly
                if root is not None:
                    try:
                        root.destroy()
                    except:
                        pass
        else:
            # Running from command line - just log completion and exit cleanly
            logger.info("Analysis completed successfully. Exiting...")
            return 0
            
        # This code is unreachable due to the sys.exit() call above
        # return 0 is left here as a comment for clarity
    
    except Exception as e:
        logger.exception(f"Error running AIS Fraud Detection: {e}")
        return 1


if __name__ == "__main__":
    # Debug: Print command-line arguments
    print("DEBUG: SFD.py starting with arguments:")
    print(f"DEBUG: {sys.argv}")
    print("DEBUG: Current working directory:", os.getcwd())
    print("DEBUG: Python executable:", sys.executable)
    
    # Check dependencies first
    if check_dependencies():
        sys.exit(main())
    else:
        print("\nPlease install the missing dependencies before running this script.")
        sys.exit(1)

