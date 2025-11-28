#!/usr/bin/env python3
"""
AIS Shipping Fraud Detection System GUI
Integrated version for SFD.py

[VERSION}
Team = Dreadnaught
Alex Giacomello, Christopher Matherne, Rupert Rigg, Zachary Zhao
version = 2.1 Beta
"""

import os
import sys
import configparser
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from datetime import datetime, timedelta
import subprocess
import platform
import logging
import time
import re
import glob
import traceback
import threading
import importlib
import importlib.metadata  # Modern replacement for pkg_resources
import tempfile
import shutil
import traceback

# Import utilities
from utils import check_dependencies

# Advanced Analysis import
# Note: logger is not yet defined here, so we use print for import errors
try:
    from advanced_analysis import AdvancedAnalysisGUI
    ADVANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    ADVANCED_ANALYSIS_AVAILABLE = False
    # Logger not yet initialized, use print instead
    print("Warning: Advanced analysis module not available. Install required dependencies.")
import requests
import zipfile
from urllib.parse import urljoin
import concurrent.futures

class DataManager:
    """Handles data download, processing, and management for various data sources"""
    
    def __init__(self, logger, progress_callback=None):
        """Initialize the data manager
        
        Args:
            logger: Logger instance for recording messages
            progress_callback: Optional function to report progress
        """
        self.logger = logger
        self.progress_callback = progress_callback
        self.base_temp_dir = None
        self.parquet_dir = None
        self._validate_dependencies()
        
    def _validate_dependencies(self):
        """Check if required dependencies are available"""
        self.pandas_available = False
        try:
            import pandas as pd
            self.pandas_available = True
        except ImportError:
            self.log("Warning: pandas not installed. CSV to parquet conversion will not work.")
    
    def log(self, message):
        """Log a message and optionally send to progress callback"""
        # Ensure we have a string
        if not isinstance(message, str):
            message = str(message)
            
        # Log to logger if available
        if self.logger:
            self.logger.info(message)
            
        # Send to progress callback if available
        if self.progress_callback:
            try:
                self.progress_callback(message)
            except Exception as e:
                # If callback fails, at least try to print
                print(f"Progress callback failed: {str(e)}")
                print(f"Original message: {message}")
            
    def _verify_url(self, url):
        """Verify that a URL is accessible"""
        try:
            response = requests.head(url, timeout=5)
            return response.status_code < 400
        except requests.exceptions.RequestException:
            return False
        
    def _download_file(self, url, target_path):
        """Download a file from a URL
        
        Args:
            url: URL to download
            target_path: Path to save the file
            
        Returns:
            bool: True if download was successful, False otherwise
        """
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(target_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True
            return False
        except Exception:
            return False
    
    def _get_cache_dir(self):
        """Get the cache directory for storing processed parquet files"""
        from utils import get_cache_dir
        return get_cache_dir()

    # def _get_cache_dir(self):
    #     """Get the cache directory for storing processed parquet files"""
    #     home_dir = os.path.expanduser("~")
    #     cache_dir = os.path.join(home_dir, ".ais_data_cache")
    #     if not os.path.exists(cache_dir):
    #         try:
    #             os.makedirs(cache_dir, exist_ok=True)
    #         except Exception:
    #             pass
    #     return cache_dir
    
    def _get_temp_dir(self):
        """Get the AISDataTemp directory for temporary files during analysis.
        Creates it if it doesn't exist, reuses it if it does.
        
        Returns:
            str: Path to AISDataTemp directory
        """
        # Use current working directory or script directory as base
        base_dir = os.getcwd()
        temp_dir = os.path.join(base_dir, "AISDataTemp")
        
        # Create directory if it doesn't exist
        if not os.path.exists(temp_dir):
            try:
                os.makedirs(temp_dir, exist_ok=True)
                self.log(f"Created AISDataTemp directory: {temp_dir}")
            except Exception as e:
                self.log(f"Warning: Could not create AISDataTemp directory: {e}")
                # Fallback to system temp if current directory fails
                temp_dir = os.path.join(tempfile.gettempdir(), "AISDataTemp")
                os.makedirs(temp_dir, exist_ok=True)
        else:
            self.log(f"Using existing AISDataTemp directory: {temp_dir}")
        
        return temp_dir
    
    def _check_cached_parquet(self, date):
        """Check if a parquet file for the given date already exists in cache
        
        Args:
            date: datetime object for the date to check
            
        Returns:
            str or None: Path to cached parquet file if exists, None otherwise
        """
        cache_dir = self._get_cache_dir()
        parquet_filename = f"ais-{date.year}-{date.month:02d}-{date.day:02d}.parquet"
        cached_path = os.path.join(cache_dir, parquet_filename)
        
        if os.path.exists(cached_path) and os.path.getsize(cached_path) > 0:
            return cached_path
        return None
    
    def _process_zip_file(self, zip_path, output_dir):
        """Process a ZIP file containing CSV data
        
        Args:
            zip_path: Path to the ZIP file
            output_dir: Directory to save processed parquet files
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        if not self.pandas_available:
            self.log("Cannot process CSV: pandas not available")
            return False
            
        try:
            import pandas as pd
            # Extract the zip file to a temporary directory within AISDataTemp
            temp_base = self._get_temp_dir()
            extract_dir = os.path.join(temp_base, "extract")
            os.makedirs(extract_dir, exist_ok=True)
            self.log(f"EXTRACTING: {os.path.basename(zip_path)}...")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find all CSV files
            csv_files = [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) 
                        if f.lower().endswith('.csv')]
            
            if not csv_files:
                self.log(f"No CSV files found in {os.path.basename(zip_path)}")
                shutil.rmtree(extract_dir)
                return False
            
            # Process each CSV file
            success = True
            for csv_file in csv_files:
                csv_filename = os.path.basename(csv_file)
                
                # Extract date from filename like AIS_YYYY_MM_DD.csv to create compliant format
                # First try to extract from the format AIS_YYYY_MM_DD.csv
                date_match = re.search(r'AIS_(\d{4})_(\d{2})_(\d{2})\.csv', csv_filename, re.IGNORECASE)
                
                if date_match:
                    year, month, day = date_match.groups()
                    # Format to ais-YYYY-MM-DD.parquet (preferred SFD.py format)
                    parquet_filename = f"ais-{year}-{month}-{day}.parquet"
                else:
                    # Fallback to original name if pattern doesn't match
                    parquet_filename = csv_filename.replace('.csv', '.parquet')
                    self.log(f"Warning: Could not extract date from {csv_filename}, using default name")
                    
                parquet_path = os.path.join(output_dir, parquet_filename)
                
                self.log(f"CONVERTING: {csv_filename} to parquet format...")
                try:
                    # Show progress on large files
                    filesize = os.path.getsize(csv_file) / (1024 * 1024)  # Size in MB
                    self.log(f"   - File size: {filesize:.1f} MB")
                    
                    df = pd.read_csv(csv_file)
                    self.log(f"   - Loaded {len(df)} records from CSV")
                    df.to_parquet(parquet_path, index=False)
                    self.log(f"   - Conversion complete: {os.path.basename(parquet_path)}")
                except Exception as e:
                    self.log(f"Error converting {csv_filename}: {str(e)}")
                    success = False
            
            # Clean up extraction directory (but keep AISDataTemp base directory)
            try:
                shutil.rmtree(extract_dir, ignore_errors=True)
            except Exception as e:
                self.log(f"Warning: Could not clean up extract directory: {e}")
            
            # Remove the zip file
            try:
                os.remove(zip_path)
            except Exception as e:
                self.log(f"Warning: Could not remove zip file: {e}")
            
            return success
        except Exception as e:
            self.log(f"Error processing zip file: {str(e)}")
            return False
    
    def download_noaa_data(self, start_date, end_date):
        """Download NOAA AIS data for the specified date range
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            tuple: (success, data_directory)
        """
        # Extract year from start date - no need for separate year parameter
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            year = start_dt.year
        except ValueError as e:
            self.log(f"Invalid date format: {e}")
            return False, None
        
        # Use single AISDataTemp directory for all temporary files
        self.base_temp_dir = self._get_temp_dir()
        download_dir = os.path.join(self.base_temp_dir, "downloads")
        self.parquet_dir = os.path.join(self.base_temp_dir, "parquet")
        
        os.makedirs(download_dir, exist_ok=True)
        os.makedirs(self.parquet_dir, exist_ok=True)
        
        self.log(f"Using temporary directory:\n - Base: {self.base_temp_dir}\n - Parquet: {self.parquet_dir}")
        self.log(f"STARTING DOWNLOAD PROCESS: NOAA AIS data from {start_date} to {end_date}")
        self.log("Download phase will be followed by extraction and conversion.")
        
        # Construct and validate base URL
        base_url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/"
        if not self._verify_url(base_url):
            self.log(f"NOAA URL not accessible: {base_url}")
            return False, None
        
        # Generate list of dates to download
        dates = []
        current_date = start_dt
        while current_date <= end_dt:
            dates.append(current_date)
            current_date += timedelta(days=1)
        
        total_files = len(dates)
        self.log(f"Will download {total_files} daily files")
        
        # Check for cached files first
        cache_dir = self._get_cache_dir()
        cached_files = []
        files_to_download = []
        
        for date in dates:
            cached_path = self._check_cached_parquet(date)
            if cached_path:
                cached_files.append((date, cached_path))
            else:
                files_to_download.append(date)
        
        # Calculate total files (cached + to download) for progress tracking
        total_files = len(cached_files) + len(files_to_download)
        if total_files > 0:
            self.log(f"Total files to process: {total_files} ({len(cached_files)} cached, {len(files_to_download)} to download)")
        
        # Report cached files found and copy them to output directory
        cached_success_count = 0
        if cached_files:
            self.log(f"Found {len(cached_files)} cached file(s), skipping download for those dates")
            for idx, (date, cached_path) in enumerate(cached_files):
                file_num = idx + 1
                self.log(f"   Using cached: ais-{date.year}-{date.month:02d}-{date.day:02d}.parquet ({file_num}/{total_files})")
                # Copy cached file to output directory
                parquet_filename = f"ais-{date.year}-{date.month:02d}-{date.day:02d}.parquet"
                target_path = os.path.join(self.parquet_dir, parquet_filename)
                try:
                    shutil.copy2(cached_path, target_path)
                    cached_success_count += 1
                except Exception as e:
                    self.log(f"Error: Could not copy cached file {parquet_filename}: {e}")
                    # Don't count failed copies as successes
        
        # Download and process files that aren't cached
        success_count = cached_success_count
        
        if files_to_download:
            self.log(f"Downloading {len(files_to_download)} file(s) that are not cached...")
        
        # Track current file number (including cached files)
        current_file_num = len(cached_files)
        
        for i, date in enumerate(files_to_download):
            # Construct filename and URL
            filename = f"AIS_{date.year}_{date.month:02d}_{date.day:02d}.zip"
            url = urljoin(base_url, filename)
            zip_path = os.path.join(download_dir, filename)
            
            # Download the file (use absolute file number including cached)
            current_file_num += 1
            self.log(f"DOWNLOADING: {filename} ({current_file_num}/{total_files})")
            if self._download_file(url, zip_path):
                self.log(f"Download complete: {filename} ({current_file_num}/{total_files})")
                if self._process_zip_file(zip_path, self.parquet_dir):
                    success_count += 1
                    self.log(f"Successfully processed {filename}")
                    
                    # Save to cache for future use
                    parquet_filename = f"ais-{date.year}-{date.month:02d}-{date.day:02d}.parquet"
                    parquet_path = os.path.join(self.parquet_dir, parquet_filename)
                    if os.path.exists(parquet_path):
                        cache_path = os.path.join(cache_dir, parquet_filename)
                        try:
                            shutil.copy2(parquet_path, cache_path)
                            self.log(f"Cached: {parquet_filename} for future use")
                        except Exception as e:
                            self.log(f"Warning: Could not cache {parquet_filename}: {e}")
                else:
                    self.log(f"Failed to process {filename}")
            else:
                self.log(f"Failed to download {filename}")
        
        if success_count == 0:
            self.log("No files were successfully downloaded and processed")
            return False, None
        
        self.log(f"Successfully processed {success_count}/{len(dates)} files")
        return True, self.parquet_dir


# def check_dependencies():
#     """
#     Check if all required dependencies are installed.
#     If not, offer to install them from requirements.txt.
    
#     Returns:
#         bool: True if all dependencies are satisfied, False otherwise
#     """
#     # Use module-level logger (defined at line 585)
#     logger.info("Checking dependencies...")
    
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
        
#         # Add required dependencies for NOAA data processing
#         required_packages = ['pandas', 'pyarrow', 'requests']
#         for pkg in required_packages:
#             if pkg not in requirements:
#                 logger.info(f"Adding {pkg} to requirements for NOAA data processing")
#                 requirements.append(pkg)
        
#         # Check installed packages
#         missing = []
#         for req in requirements:
#             try:
#                 importlib.import_module(req.lower().replace('-', '_'))
#             except ImportError:
#                 try:
#                     # Double-check with importlib.metadata as some package names don't match import names
#                     importlib.metadata.version(req)
#                 except importlib.metadata.PackageNotFoundError:
#                     missing.append(req)
        
#         if missing:
#             logger.warning(f"Missing dependencies: {', '.join(missing)}")
            
#             # Show dialog in GUI context
#             result = messagebox.askyesno(
#                 "Missing Dependencies",
#                 f"The following dependencies are missing: {', '.join(missing)}\n\nWould you like to install them now?"
#             )
            
#             if result:
#                 logger.info("Installing missing dependencies...")
#                 try:
#                     # Create a progress dialog
#                     progress_window = tk.Toplevel()
#                     progress_window.title("Installing Dependencies")
#                     progress_window.geometry("400x150")
#                     progress_window.transient()
#                     progress_window.grab_set()
                    
#                     # Add progress label
#                     label = ttk.Label(progress_window, text=f"Installing dependencies: {', '.join(missing)}")
#                     label.pack(pady=10)
                    
#                     # Add progress bar
#                     progress = ttk.Progressbar(progress_window, mode="indeterminate")
#                     progress.pack(fill="x", padx=20, pady=10)
#                     progress.start()
                    
#                     # Update function to keep GUI responsive
#                     def update_gui():
#                         progress_window.update()
#                         progress_window.after(100, update_gui)
                    
#                     progress_window.after(100, update_gui)
                    
#                     # Run pip install
#                     result = subprocess.run(
#                         [sys.executable, "-m", "pip", "install", "-r", req_file],
#                         capture_output=True,
#                         text=True
#                     )
                    
#                     # Stop progress
#                     progress.stop()
#                     progress_window.destroy()
                    
#                     if result.returncode == 0:
#                         messagebox.showinfo(
#                             "Installation Complete", 
#                             "Dependencies installed successfully!"
#                         )
#                         logger.info("Dependencies installed successfully!")
#                         return True
#                     else:
#                         logger.error(f"Failed to install dependencies: {result.stderr}")
#                         messagebox.showerror(
#                             "Installation Failed",
#                             f"Failed to install dependencies. Please install them manually:\n\n{result.stderr[:500]}..."
#                         )
#                         return False
#                 except Exception as e:
#                     logger.error(f"Error installing dependencies: {e}")
#                     messagebox.showerror(
#                         "Installation Error",
#                         f"Error installing dependencies: {str(e)}"
#                     )
#                     return False
#             else:
#                 logger.warning("User chose not to install missing dependencies")
#                 messagebox.showwarning(
#                     "Missing Dependencies",
#                     "Some features may not work without the required dependencies.\nYou can install them manually with:\n\n"
#                     f"pip install {' '.join(missing)}"
#                 )
#                 return False
        
#         logger.info("All dependencies are satisfied.")
#         return True
    
#     except Exception as e:
#         logger.error(f"Error checking dependencies: {e}")
#         return True  # Continue anyway

# Import PIL for image processing
try:
    from PIL import Image, ImageTk
    pil_available = True
except ImportError:
    pil_available = False

# Check for tkcalendar
try:
    from tkcalendar import DateEntry
    tkcalendar_available = True
except ImportError:
    tkcalendar_available = False

# Check for AWS boto3
try:
    import boto3
    import botocore.exceptions
    from urllib.parse import urlparse
    aws_available = True
except ImportError:
    aws_available = False

# Import Windows constants if on Windows
if platform.system() == "Windows":
    try:
        import win32con
    except ImportError:
        pass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ais_fraud_detection_gui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SDF_GUI")

# Define constants
BATCH_FILE_NAME = "run_sfd.bat"


class DownloadProgressWindow(tk.Toplevel):
    """Progress window specifically for download/extraction/conversion phase"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Downloading Data")
        self.geometry("800x400")
        self.resizable(False, False)
        self.transient(parent)
        self.parent = parent
        
        # Progress tracking
        self.total_files = 0
        self.current_file = 0
        self.current_phase = "Initializing"  # Download, Extract, Convert
        self.current_activity = "Preparing download..."
        
        # Main frame
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(main_frame, text="Downloading NOAA AIS Data", 
                 font=("Arial", 14, "bold")).pack(pady=(0, 10))
        
        # Progress bar frame
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(progress_frame, text="Overall Progress:", 
                 font=("Arial", 10)).pack(anchor=tk.W)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                            maximum=100, length=700, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="0%", font=("Arial", 9))
        self.progress_label.pack(anchor=tk.E)
        
        # Current activity frame
        activity_frame = ttk.Frame(main_frame)
        activity_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(activity_frame, text="Current Activity:", 
                 font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        self.activity_var = tk.StringVar(value="Initializing...")
        activity_label = ttk.Label(activity_frame, textvariable=self.activity_var, 
                                  font=("Arial", 9), wraplength=750, justify=tk.LEFT)
        activity_label.pack(anchor=tk.W, pady=5)
        
        # Status text area
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        ttk.Label(status_frame, text="Status Messages:", 
                 font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        self.status_text = scrolledtext.ScrolledText(status_frame, height=8, 
                                                    font=("Consolas", 9), wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        # Center window
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (self.winfo_width() // 2)
        y = (self.winfo_screenheight() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")
        
        # Prevent closing during download
        self.protocol("WM_DELETE_WINDOW", lambda: None)
    
    def add_message(self, message):
        """Add a message to the status text and update progress if applicable"""
        def update_gui():
            try:
                if not self.winfo_exists():
                    return
                
                # Add message to status text
                message_str = str(message) if isinstance(message, str) else str(message)
                self.status_text.insert(tk.END, message_str + "\n")
                self.status_text.see(tk.END)
                
                # Parse message to update progress
                self._parse_progress_message(message_str)
                
                self.update_idletasks()
            except Exception as e:
                print(f"Error updating download progress: {str(e)}")
        
        self.after(0, update_gui)
    
    def _parse_progress_message(self, message):
        """Parse log messages to extract progress information"""
        import re
        message_upper = message.upper()
        
        # Extract total file count - match "Will download X daily files" or "Total files to process: X"
        if "TOTAL FILES TO PROCESS" in message_upper:
            match = re.search(r'TOTAL FILES TO PROCESS:\s*(\d+)', message_upper)
            if match:
                self.total_files = int(match.group(1))
                self._update_progress()
        elif ("WILL DOWNLOAD" in message_upper or ("WILL" in message_upper and "DOWNLOAD" in message_upper)):
            if "DAILY FILES" in message_upper or "FILES" in message_upper:
                match = re.search(r'(\d+)\s+(?:DAILY\s+)?FILES?', message_upper)
                if match:
                    self.total_files = int(match.group(1))
                    self._update_progress()
        
        # Track cached files - look for "Using cached" with file numbers
        if "USING CACHED:" in message_upper or ("CACHED:" in message_upper and "(" in message):
            # Extract file number from cached message like "(1/3)"
            match = re.search(r'\((\d+)/(\d+)\)', message)
            if match:
                file_num = int(match.group(1))
                total = int(match.group(2))
                if not self.total_files:
                    self.total_files = total
                self.current_file = file_num
                self.current_phase = "Cached"
                self.current_activity = message.strip()
                self._update_progress()
        
        # Track current file and phase for downloads
        elif "DOWNLOADING:" in message_upper:
            match = re.search(r'\((\d+)/(\d+)\)', message)
            if match:
                self.current_file = int(match.group(1))
                if not self.total_files:
                    self.total_files = int(match.group(2))
                self.current_phase = "Download"
                self.current_activity = message.strip()
                self._update_progress()
        
        elif "DOWNLOAD COMPLETE:" in message_upper or "DOWNLOAD COMPLETE" in message_upper:
            # Extract file number from completion message
            match = re.search(r'\((\d+)/(\d+)\)', message)
            if match:
                self.current_file = int(match.group(1))
                if not self.total_files:
                    self.total_files = int(match.group(2))
            self.current_phase = "Download"
            self.current_activity = message.strip()
            self._update_progress()
        
        elif "EXTRACTING:" in message_upper:
            self.current_phase = "Extract"
            self.current_activity = message.strip()
            self._update_progress()
        
        elif "CONVERTING:" in message_upper:
            self.current_phase = "Convert"
            self.current_activity = message.strip()
            self._update_progress()
        
        elif "CONVERSION COMPLETE:" in message_upper or "SUCCESSFULLY PROCESSED" in message_upper:
            self.current_phase = "Convert"
            self.current_activity = message.strip()
            self._update_progress()
            # Move to next file after conversion completes
            if self.current_file < self.total_files:
                self.current_file += 1
                self._update_progress()
        
        elif "STARTING DOWNLOAD PROCESS" in message_upper:
            self.current_activity = message.strip()
            self._update_progress()
        
        elif "FOUND" in message_upper and ("CACHED FILE" in message_upper or "CACHED" in message_upper):
            # Extract number of cached files
            match = re.search(r'FOUND\s+(\d+)', message_upper)
            if match:
                cached_count = int(match.group(1))
                # Update activity but don't change phase yet
                self.current_activity = message.strip()
                self._update_progress()
        
        # Always update activity for other messages
        else:
            # Generic update for initialization and other messages
            if "STARTING" in message_upper or "PROCESS" in message_upper or "USING TEMPORARY" in message_upper:
                self.current_activity = message.strip()
                self._update_progress()
    
    def _update_progress(self):
        """Update the progress bar based on current state"""
        if self.total_files == 0:
            self.progress_var.set(0)
            self.progress_label.config(text="0% - Initializing...")
            self.activity_var.set(self.current_activity)
            return
        
        # Calculate progress
        # Each file has 3 phases: Download (0-33%), Extract (33-66%), Convert (66-100%)
        # Overall: (file-1)/total * 100 + (phase_progress)/total
        
        phase_progress = 0
        if self.current_phase == "Cached":
            phase_progress = 100  # Cached files are already complete
        elif self.current_phase == "Download":
            phase_progress = 0  # Start of download phase
        elif self.current_phase == "Extract":
            phase_progress = 33  # Start of extract phase
        elif self.current_phase == "Convert":
            phase_progress = 66  # Start of convert phase
        elif self.current_phase == "Initializing":
            phase_progress = 0
        
        # Calculate overall progress
        if self.total_files > 0:
            if self.current_file > 0:
                # Progress from completed files (files before current)
                completed_files_progress = (self.current_file - 1) / self.total_files * 100
                # Progress from current file's phase
                current_file_phase_progress = phase_progress / self.total_files
                total_progress = completed_files_progress + current_file_phase_progress
            else:
                # No file started yet, but we know total
                total_progress = 0
        else:
            total_progress = 0
        
        # Clamp to 0-100
        total_progress = max(0, min(100, total_progress))
        
        self.progress_var.set(total_progress)
        self.progress_label.config(text=f"{total_progress:.1f}%")
        
        # Update activity - show current file and phase
        if self.current_file > 0 and self.total_files > 0:
            # Extract just the filename or key info from activity
            activity_display = self.current_activity
            if len(activity_display) > 60:
                activity_display = activity_display[:57] + "..."
            activity_text = f"File {self.current_file}/{self.total_files} - {self.current_phase}: {activity_display}"
        else:
            activity_text = self.current_activity
        
        self.activity_var.set(activity_text)
    
    def set_total_files(self, count):
        """Set the total number of files to process"""
        self.total_files = count
        self._update_progress()
    
    def close(self):
        """Close the download progress window"""
        self.destroy()


class ProgressWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Analysis in Progress")
        self.geometry("1000x1200")  # Increased window size
        self.resizable(True, True)  # Allow resizing
        self.transient(parent)
        self.parent = parent
        
        # Storage for temporary NOAA data directory
        self.noaa_temp_dir = None
        
        # Load and display the SFDLoad.png image with resizing
        try:
            img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SFDLoad.png")
            if os.path.exists(img_path):
                if pil_available:  # Use PIL to resize if available
                    # Load and resize the image
                    original_img = Image.open(img_path)
                    
                    # Calculate new dimensions (75% of original size)
                    width, height = original_img.size
                    new_width = int(width * 0.75)
                    new_height = int(height * 0.75)
                    
                    # Resize the image
                    resized_img = original_img.resize((new_width, new_height), Image.LANCZOS)
                    
                    # Convert to PhotoImage
                    self.img = ImageTk.PhotoImage(resized_img)
                else:  # Fallback to regular PhotoImage if PIL not available
                    self.img = tk.PhotoImage(file=img_path)
                    
                # Create a frame to better contain the image and add a subtle border
                img_frame = ttk.Frame(self, relief="groove", borderwidth=1)
                img_frame.pack(pady=20, padx=20)  # Increased padding for larger window
                
                # Display the image with better padding
                img_label = ttk.Label(img_frame, image=self.img)
                img_label.pack(padx=10, pady=10)
            else:
                ttk.Label(self, text="Image not found: SFDLoad.png").pack(pady=5)
        except Exception as e:
            ttk.Label(self, text=f"Error loading image: {str(e)}").pack(pady=5)
        
        # Add a header with reduced padding
        ttk.Label(self, text="Analysis Status:", font=("Arial", 11, "bold")).pack(anchor='w', padx=8, pady=(5,0))
        
        # Add a scrolled text widget for progress messages with larger dimensions
        self.log_text = scrolledtext.ScrolledText(self, width=120, height=30, font=("Consolas", 10))
        self.log_text.pack(padx=8, pady=5, fill=tk.BOTH, expand=True)
        
        # Buttons frame with reduced padding
        self.buttons_frame = ttk.Frame(self)
        self.buttons_frame.pack(fill=tk.X, padx=8, pady=5)
        
        # Add a cancel button
        self.cancel_btn = ttk.Button(self.buttons_frame, text="Cancel Analysis", command=self.cancel)
        self.cancel_btn.pack(side=tk.LEFT, padx=3)
        
        # Create an "Open Results" button (hidden initially)
        self.open_results_btn = ttk.Button(self.buttons_frame, text="Open Results Folder", command=self.open_results, state="disabled")
        
        # Create a "Close" button (hidden initially)
        self.close_btn = ttk.Button(self.buttons_frame, text="Exit", command=self.destroy, state="disabled")
        
        # Create a button for additional analysis on current dataset (hidden initially)
        self.additional_analysis_btn = ttk.Button(self.buttons_frame, text="Conduct Additional Analysis", 
                                                command=self.conduct_additional_analysis, state="disabled")
        
        # Create a button for starting a new analysis (hidden initially)
        self.new_analysis_btn = ttk.Button(self.buttons_frame, text="Conduct New Analysis", 
                                       command=self.conduct_new_analysis, state="disabled")
        
        # Initialize process variable
        self.process = None
        self.canceled = False
        self.output_directory = None
    
    def analysis_complete(self, output_directory, success=True):
        """Called when analysis is complete to update the UI
        
        Args:
            output_directory: Path to the directory containing analysis results
            success: Whether the analysis completed successfully (default: True)
        """
        self.output_directory = output_directory
        self.cancel_btn.configure(state="disabled")
        
        # Log action being taken
        logger.info(f"Setting up analysis completion UI with output_directory={output_directory}, success={success}")
        
        # Update the message in the log
        self.add_message("\n==== Analysis Complete ====\n")
        self.add_message("You can now:")
        self.add_message("1. Open the results folder to view reports and visualizations")
        self.add_message("2. Conduct additional analysis on the current dataset")
        self.add_message("3. Start a new analysis with different parameters")
        self.add_message("4. Exit the application")
        
        # Add visual separator before buttons
        self.add_message("\n" + "-" * 40)
        
        # Show and enable all the buttons
        self.open_results_btn.pack(side=tk.LEFT, padx=3)
        self.open_results_btn.configure(state="normal")
        
        self.additional_analysis_btn.pack(side=tk.LEFT, padx=3)
        self.additional_analysis_btn.configure(state="normal")
        
        self.new_analysis_btn.pack(side=tk.LEFT, padx=3)
        self.new_analysis_btn.configure(state="normal")
        
        self.close_btn.pack(side=tk.RIGHT, padx=3)
        self.close_btn.configure(state="normal")
    
    def open_results(self):
        """Open the results folder"""
        if self.output_directory and os.path.exists(self.output_directory):
            try:
                if platform.system() == "Windows":
                    os.startfile(self.output_directory)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.call(["open", self.output_directory])
                else:  # Linux
                    subprocess.call(["xdg-open", self.output_directory])
            except Exception:
                messagebox.showerror("Error", f"Could not open folder: {self.output_directory}")
        else:
            messagebox.showerror("Error", "Output directory does not exist.")
            
    def conduct_additional_analysis(self):
        """Launch advanced analysis tools on the current dataset"""
        # Hide this window temporarily
        self.withdraw()
        
        try:
            # Create a new window for additional analysis options
            additional_window = tk.Toplevel(self.parent)
            additional_window.title("Advanced Analytical Tools")
            additional_window.geometry("900x700")  # Larger window for more content
            additional_window.transient(self.parent)
            
            # Set a minimum size to prevent controls from getting squished
            additional_window.minsize(800, 600)
        
            # Create a main frame with padding
            main_frame = ttk.Frame(additional_window, padding=10)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Add a header with title and description
            header_frame = ttk.Frame(main_frame)
            header_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(header_frame, text="Advanced Analytical Tools", 
                     font=("Arial", 16, "bold")).pack(side=tk.TOP, anchor=tk.W)
            ttk.Label(header_frame, 
                     text="Perform additional analysis on the previously generated dataset",
                     font=("Arial", 11)).pack(side=tk.TOP, anchor=tk.W, pady=5)
            
            # Create a notebook for better organization of many options
            notebook = ttk.Notebook(main_frame)
            notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
            # Create tabs for different categories of analysis
            tab_output = ttk.Frame(notebook, padding=10)
            tab_analysis = ttk.Frame(notebook, padding=10)
            tab_maps = ttk.Frame(notebook, padding=10)
            tab_vessel = ttk.Frame(notebook, padding=10)
            
            # Add tabs to the notebook
            notebook.add(tab_output, text="Additional Outputs")
            notebook.add(tab_analysis, text="Further Analysis")
            notebook.add(tab_maps, text="Mapping Tools")
            notebook.add(tab_vessel, text="Vessel-Specific Analysis")
        
            # =====================================================================
            # TAB 1: ADDITIONAL OUTPUTS OPTIONS
            # =====================================================================
            ttk.Label(tab_output, text="Generate Additional Outputs from Dataset", 
                     font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
                     
            # Create frames for each output option
            output_options = [
                ("Export Full Dataset to CSV", "Export the complete analysis dataset to CSV format"),
                ("Generate Summary Report", "Create a summary report with key findings and statistics"),
                ("Export Vessel Statistics", "Export vessel-specific statistics to Excel format"),
                ("Generate Anomaly Timeline", "Create a timeline visualization of anomalies")
            ]
        
            for btn_text, btn_desc in output_options:
                option_frame = ttk.Frame(tab_output)
                option_frame.pack(fill=tk.X, pady=5)
                
                ttk.Button(option_frame, text=btn_text, width=25,
                         command=lambda t=btn_text: self.launch_analysis_tool(t, additional_window)).pack(side=tk.LEFT, padx=5)
                
                ttk.Label(option_frame, text=btn_desc, wraplength=500, 
                         font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
            # =====================================================================
            # TAB 2: FURTHER ANALYSIS OPTIONS
            # =====================================================================
            ttk.Label(tab_analysis, text="Further Analysis Tools", 
                     font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
                     
            # Create frames for each analysis option
            analysis_options = [
                ("Anomaly Correlation Analysis", "Analyze correlations between different types of anomalies"),
                ("Temporal Pattern Analysis", "Analyze patterns over time, including hourly/daily distributions"),
                ("Vessel Behavior Clustering", "Apply clustering algorithms to identify similar vessel behaviors"),
                ("Anomaly Frequency Analysis", "Analyze frequency and distribution of different anomaly types")
            ]
            
            for btn_text, btn_desc in analysis_options:
                option_frame = ttk.Frame(tab_analysis)
                option_frame.pack(fill=tk.X, pady=5)
                
                ttk.Button(option_frame, text=btn_text, width=25,
                         command=lambda t=btn_text: self.launch_analysis_tool(t, additional_window)).pack(side=tk.LEFT, padx=5)
                
                ttk.Label(option_frame, text=btn_desc, wraplength=500, 
                         font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
            # =====================================================================
            # TAB 3: MAPPING TOOLS
            # =====================================================================
            ttk.Label(tab_maps, text="Advanced Mapping Tools", 
                     font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))

            # Create a section for merged anomaly map
            merged_map_frame = ttk.LabelFrame(tab_maps, text="Full Spectrum Anomaly Map", padding=10)
            merged_map_frame.pack(fill=tk.X, pady=10)
            
            ttk.Label(merged_map_frame, 
                     text="Create a comprehensive map showing all anomalies with various visualization options").pack(anchor=tk.W, pady=5)
            
            # Checkbuttons for map options
            map_options_frame = ttk.Frame(merged_map_frame)
            map_options_frame.pack(fill=tk.X, pady=5)
            
            # Variables for checkbuttons
            show_pins = tk.BooleanVar(value=True)
            show_heatmap = tk.BooleanVar(value=False)
            
            ttk.Checkbutton(map_options_frame, text="Show anomaly pins", variable=show_pins).pack(side=tk.LEFT, padx=10)
            ttk.Checkbutton(map_options_frame, text="Show heatmap overlay", variable=show_heatmap).pack(side=tk.LEFT, padx=10)
            
            ttk.Button(merged_map_frame, text="Generate Full Spectrum Anomaly Map", 
                      command=lambda: self.launch_analysis_tool("Full Spectrum Anomaly Map", additional_window)).pack(pady=5)
            
            # Vessel-specific map section
            vessel_map_frame = ttk.LabelFrame(tab_maps, text="Vessel-Specific Maps", padding=10)
            vessel_map_frame.pack(fill=tk.X, pady=10)
            
            ttk.Label(vessel_map_frame, 
                     text="Create maps focused on specific vessels by MMSI").pack(anchor=tk.W, pady=5)
            
            # MMSI entry for vessel maps
            mmsi_frame = ttk.Frame(vessel_map_frame)
            mmsi_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(mmsi_frame, text="MMSI:").pack(side=tk.LEFT, padx=(0, 5))
            mmsi_entry = ttk.Entry(mmsi_frame, width=15)
            mmsi_entry.pack(side=tk.LEFT)
            
            # Map type options for vessel
            map_type_frame = ttk.Frame(vessel_map_frame)
            map_type_frame.pack(fill=tk.X, pady=5)
            
            # Map type radio buttons
            map_type = tk.StringVar(value="path")
            ttk.Radiobutton(map_type_frame, text="Path Map", variable=map_type, value="path").pack(side=tk.LEFT, padx=10)
            ttk.Radiobutton(map_type_frame, text="Anomaly Map", variable=map_type, value="anomaly").pack(side=tk.LEFT, padx=10)
            ttk.Radiobutton(map_type_frame, text="Heatmap", variable=map_type, value="heatmap").pack(side=tk.LEFT, padx=10)
            
            ttk.Button(vessel_map_frame, text="Generate Vessel-Specific Map",
                     command=lambda: self.launch_analysis_tool(f"Vessel Map for MMSI {mmsi_entry.get()}", additional_window)).pack(pady=5)
        
            # =====================================================================
            # TAB 4: VESSEL-SPECIFIC ANALYSIS
            # =====================================================================
            ttk.Label(tab_vessel, text="Vessel-Specific Analysis Tools", 
                     font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
            
            # Extended analysis section
            extended_frame = ttk.LabelFrame(tab_vessel, text="Extended Time Range Analysis", padding=10)
            extended_frame.pack(fill=tk.X, pady=10)
            
            ttk.Label(extended_frame, 
                     text="Analyze additional days of data for a specific vessel beyond the current date range").pack(anchor=tk.W, pady=5)
            
            # MMSI and additional days entry
            extended_entry_frame = ttk.Frame(extended_frame)
            extended_entry_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(extended_entry_frame, text="MMSI:").pack(side=tk.LEFT, padx=(0, 5))
            extended_mmsi_entry = ttk.Entry(extended_entry_frame, width=15)
            extended_mmsi_entry.pack(side=tk.LEFT, padx=(0, 10))
            
            ttk.Label(extended_entry_frame, text="Additional Days:").pack(side=tk.LEFT, padx=(10, 5))
            days_entry = ttk.Entry(extended_entry_frame, width=5)
            days_entry.insert(0, "7")
            days_entry.pack(side=tk.LEFT)
            
            ttk.Button(extended_frame, text="Perform Extended Analysis",
                     command=lambda: self.launch_analysis_tool(f"Extended Analysis for MMSI {extended_mmsi_entry.get()}", additional_window)).pack(pady=5)
            
            # AI predicted path section
            ai_path_frame = ttk.LabelFrame(tab_vessel, text="AI Predicted Path (Placeholder)", padding=10)
            ai_path_frame.pack(fill=tk.X, pady=10)
            
            ttk.Label(ai_path_frame, 
                     text="Use AI to predict the next 48-hour path for a specific vessel").pack(anchor=tk.W, pady=5)
        
            # MMSI entry for AI prediction
            ai_entry_frame = ttk.Frame(ai_path_frame)
            ai_entry_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(ai_entry_frame, text="MMSI:").pack(side=tk.LEFT, padx=(0, 5))
            ai_mmsi_entry = ttk.Entry(ai_entry_frame, width=15)
            ai_mmsi_entry.pack(side=tk.LEFT)
            
            ttk.Button(ai_path_frame, text="Generate 48-hour Predicted Path",
                     command=lambda: self.launch_analysis_tool(f"AI Path Prediction for MMSI {ai_mmsi_entry.get()}", additional_window)).pack(pady=5)
            
            # =====================================================================
            # BOTTOM CONTROL BUTTONS
            # =====================================================================
            # Create bottom frame for control buttons
            bottom_frame = ttk.Frame(main_frame)
            bottom_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)
            
            # Add buttons to close or start over
            ttk.Button(bottom_frame, text="Return to Results", 
                      command=lambda: [additional_window.destroy(), self.deiconify()]).pack(side=tk.RIGHT, padx=5)
            
            ttk.Button(bottom_frame, text="End Current Analysis and Start Over", 
                      command=lambda: [additional_window.destroy(), self.conduct_new_analysis()]).pack(side=tk.RIGHT, padx=5)
            
            ttk.Button(bottom_frame, text="End Analysis and Exit", 
                      command=lambda: [additional_window.destroy(), self.destroy()]).pack(side=tk.RIGHT, padx=5)
            
            # Make sure this window is shown
            additional_window.protocol("WM_DELETE_WINDOW", lambda: [additional_window.destroy(), self.deiconify()])
            additional_window.focus_set()
            
        except Exception as e:
            logger.error(f"Error creating additional analysis window: {str(e)}")
            # Print full traceback for debugging
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Could not open additional analysis window: {str(e)}")
            return
    
    def launch_analysis_tool(self, tool_name, parent_window):
        """Launch a specific analysis tool on the current dataset"""
        # Log the selected tool
        logger.info(f"Selected analysis tool: {tool_name}")
        
        # Check if we have a valid output directory
        if not hasattr(self, 'output_directory') or not self.output_directory:
            self.output_directory = "C:/AIS_Data/Output"  # Default path if not set
        
        # Close the parent window temporarily
        parent_window.withdraw()
        
        # Show processing message
        processing_window = tk.Toplevel(self.parent)
        processing_window.title("Processing")
        processing_window.geometry("400x150")
        processing_window.transient(self.parent)
        processing_window.resizable(False, False)
        
        # Add a message
        ttk.Label(processing_window, text=f"Processing {tool_name}...", 
                 font=("Arial", 12)).pack(pady=20)
        
        # Add a progress bar
        progress = ttk.Progressbar(processing_window, mode="indeterminate")
        progress.pack(fill=tk.X, padx=20, pady=10)
        progress.start()
        
        # Center the processing window
        processing_window.update_idletasks()
        width = processing_window.winfo_width()
        height = processing_window.winfo_height()
        x = (processing_window.winfo_screenwidth() // 2) - (width // 2)
        y = (processing_window.winfo_screenheight() // 2) - (height // 2)
        processing_window.geometry(f"{width}x{height}+{x}+{y}")
        
        # Make sure this window stays on top
        processing_window.grab_set()
        processing_window.focus_set()
        
        # Handle the specific tool being launched
        if tool_name == "Export Full Dataset to CSV":
            self.parent.after(500, lambda: self._export_full_dataset(processing_window, parent_window))
        elif tool_name == "Generate Summary Report":
            self.parent.after(500, lambda: self._generate_summary_report(processing_window, parent_window))
        elif tool_name == "Export Vessel Statistics":
            self.parent.after(500, lambda: self._export_vessel_statistics(processing_window, parent_window))
        elif tool_name == "Generate Anomaly Timeline":
            self.parent.after(500, lambda: self._generate_anomaly_timeline(processing_window, parent_window))
        else:
            # Generic handling for other tools
            def finish_processing():
                progress.stop()
                processing_window.destroy()
                
                # Show result message
                messagebox.showinfo("Analysis Complete", 
                                  f"The {tool_name} has been completed successfully.\n\n"
                                  "In a production version, this would display actual results.")
                
                # Show parent window again
                parent_window.deiconify()
            
            # Simulate processing time (2 seconds)
            self.parent.after(2000, finish_processing)
    
    def conduct_new_analysis(self):
        """Close this window and reset the main UI for a new analysis"""
        # Ask for confirmation
        if messagebox.askyesno("Confirm New Analysis", 
                              "Are you sure you want to start a new analysis? "
                              "This will close the current results."):
            # Pass control back to the main application to start a new analysis
            self.parent.reset_for_new_analysis()
            self.destroy()
    
    def add_message(self, message):
        """Add a message to the log text widget (thread-safe)"""
        # Use after() to schedule GUI update from any thread
        def update_gui():
            try:
                # Ensure the widget still exists
                if not self.winfo_exists():
                    return
                    
                # Ensure message is a string
                if not isinstance(message, str):
                    message_str = str(message)
                else:
                    message_str = message
                    
                # Add the message to the text widget
                self.log_text.insert(tk.END, message_str + "\n")
                self.log_text.see(tk.END)  # Scroll to the end
                self.update_idletasks()  # Force update of the widget
            except Exception as e:
                # Log the error but don't crash
                print(f"Error updating progress window: {str(e)}")
                
        # Schedule the update on the main thread
        self.after(0, update_gui)
    
    def set_process(self, process):
        """Set the subprocess process"""
        self.process = process
    
    def cancel(self):
        """Cancel the running process"""
        if self.process:
            try:
                self.process.terminate()
                self.add_message("Analysis canceled by user.")
            except Exception as e:
                self.add_message(f"Error canceling process: {str(e)}")
                
        # Try to delete the batch file if it exists
        batch_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), BATCH_FILE_NAME)
        try:
            if os.path.exists(batch_path):
                os.remove(batch_path)
                self.add_message("Temporary batch file removed.")
        except Exception as e:
            self.add_message(f"Could not remove batch file: {e}")
                
        self.canceled = True
        self.destroy()
    
    def conduct_additional_analysis(self):
        """Launch advanced analysis tools on the current dataset"""
        # Access the main application through the parent reference
        logger.info("ProgressWindow: conduct_additional_analysis requested")
        import traceback
        
        try:
            # The parent should be set directly when creating the ProgressWindow
            if self.parent and hasattr(self.parent, 'conduct_additional_analysis'):
                # Log success
                logger.info(f"Found parent with conduct_additional_analysis method: {self.parent}")
                
                # Close the progress window
                self.destroy()
                
                # Call the main application's conduct_additional_analysis method
                self.parent.conduct_additional_analysis()
                return
                
            # If direct parent reference doesn't work, try to find it by traversing widget hierarchy
            logger.debug("Direct parent reference failed, trying to traverse widget hierarchy")
            
            # Start with this window's Tk parent
            parent_window = self.master
            while parent_window:
                # Log what we're checking
                logger.debug(f"Checking parent: {parent_window}")
                
                # Check if this is an instance of the application class with the method we need
                # Look for common attributes that might indicate this is our main application
                if (hasattr(parent_window, 'conduct_additional_analysis') and 
                    hasattr(parent_window, 'run_analysis')):
                    
                    logger.info(f"Found application via widget hierarchy: {parent_window}")
                    self.destroy()
                    parent_window.conduct_additional_analysis()
                    return
                    
                # Move up one level in the hierarchy
                if hasattr(parent_window, 'master') and parent_window.master != parent_window:
                    parent_window = parent_window.master
                else:
                    break
                    
            # If we get here, we couldn't find the application
            # Try using the SFD_GUI module's global variables
            import sys
            logger.debug("Trying to find app instance through module globals")
            
            # Look for modules that might contain our application instance
            for name, module in sys.modules.items():
                if 'SFD_GUI' in name and hasattr(module, 'app') and hasattr(module.app, 'conduct_additional_analysis'):
                    logger.info(f"Found app through module: {name}")
                    self.destroy()
                    module.app.conduct_additional_analysis()
                    return
            
            # Last resort - try to create a new interface with minimal functionality
            logger.warning("Could not find main application, creating simple analysis interface")
            self.create_simple_analysis_interface()
            
        except Exception as e:
            logger.error(f"Error in conduct_additional_analysis: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            messagebox.showerror("Error", f"Cannot conduct additional analysis. Error: {str(e)}")
    
    def create_simple_analysis_interface(self):
        """Create a simple analysis interface when main app can't be found"""
        # Hide current window
        self.withdraw()
        
        # Create a new basic window
        basic_window = tk.Toplevel(self.master)
        basic_window.title("Basic Analysis Tools")
        basic_window.geometry("700x500")
        
        # Add header
        ttk.Label(basic_window, text="Basic Analysis Tools", font=("Arial", 16, "bold")).pack(pady=20)
        ttk.Label(basic_window, text="Select an analysis option:", font=("Arial", 12)).pack(pady=10)
        
        # Add some basic options
        ttk.Button(basic_window, text="View Vessel Statistics", 
                 command=lambda: messagebox.showinfo("Info", "This would show vessel statistics.")).pack(pady=10)
                 
        ttk.Button(basic_window, text="Export Data to CSV", 
                 command=lambda: messagebox.showinfo("Info", "This would export data to CSV.")).pack(pady=10)
                 
        ttk.Button(basic_window, text="Generate Visualization", 
                 command=lambda: messagebox.showinfo("Info", "This would generate visualizations.")).pack(pady=10)
        
        # Close button returns to previous window
        ttk.Button(basic_window, text="Close", command=lambda: [basic_window.destroy(), self.deiconify()]).pack(pady=20)
        
        # Ensure this window is shown
        basic_window.transient(self.master)
        basic_window.lift()
        basic_window.focus_force()
        basic_window.grab_set()
        
    def withdraw(self):
        """Hide the window but keep it available"""
        super().withdraw()
    
    def destroy(self):
        """Clean up resources before destroying the window"""
        # Clean up temporary files if analysis was canceled
        self.cleanup_temp_files()
        super().destroy()
    
    def cleanup_temp_files(self):
        """Clean up temporary files created during analysis.
        Note: AISDataTemp directory is preserved for reuse, only cleans up subdirectories.
        """
        # Clean up extract subdirectory if it exists
        if hasattr(self, 'noaa_temp_dir') and self.noaa_temp_dir:
            # AISDataTemp is now persistent, so we only clean up specific subdirectories
            extract_dir = os.path.join(self.noaa_temp_dir, "extract")
            if os.path.exists(extract_dir):
                try:
                    self.add_message("\n===== CLEANING UP TEMPORARY FILES =====\n")
                    self.add_message(f"Cleaning up extraction directory: {extract_dir}")
                    
                    # Wait a moment in case any files are still in use
                    time.sleep(0.5)
                    
                    shutil.rmtree(extract_dir, ignore_errors=True)
                    self.add_message("\nTemporary extraction files cleaned up successfully.")
                    self.add_message(f"Note: AISDataTemp directory is preserved for reuse: {self.noaa_temp_dir}")
                except Exception as e:
                    self.add_message(f"\nWarning: Error cleaning up temporary files: {str(e)}")
                    self.add_message("This is not critical, but you may want to manually delete temporary files later.")
                    # Don't raise the exception, just log it
                    logger.error(f"Error cleaning up temporary files: {str(e)}")

class SDFGUI:
    def _get_ais_temp_dir(self):
        """Get the AISDataTemp directory path"""
        base_dir = os.getcwd()
        temp_dir = os.path.join(base_dir, "AISDataTemp")
        # Fallback to system temp if not in current directory
        if not os.path.exists(temp_dir):
            temp_dir = os.path.join(tempfile.gettempdir(), "AISDataTemp")
        return temp_dir
    
    def _delete_temp_directory(self):
        """Delete the AISDataTemp directory and all its contents"""
        temp_dir = self._get_ais_temp_dir()
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"Deleted temporary directory: {temp_dir}")
                return True
            except Exception as e:
                logger.error(f"Error deleting temporary directory: {e}")
                return False
        return True  # Directory doesn't exist, so nothing to delete

    def on_closing(self):
        """Handle window closing - ask user if they want to delete temp files"""
        temp_dir = self._get_ais_temp_dir()
        temp_exists = os.path.exists(temp_dir)
        
        if temp_exists:
            # Ask user if they want to delete temp files
            response = messagebox.askyesno(
                "Delete Temporary Files?",
                f"The temporary directory exists:\n{temp_dir}\n\n"
                "Would you like to delete it before closing?\n\n"
                "Yes - Delete temporary files and close\n"
                "No - Keep temporary files and close",
                icon='question'
            )
            
            if response:  # User clicked Yes
                try:
                    self._delete_temp_directory()
                    messagebox.showinfo("Success", "Temporary files deleted successfully.")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to delete temporary files:\n{str(e)}\n\n"
                                                "You may need to delete them manually.")
        
        # Now ask about cached data
        from SFD import get_cache_dir, clear_cache
        cache_dir = get_cache_dir()
        if cache_dir and os.path.exists(cache_dir):
            try:
                # Check if there are files in the cache directory
                files_exist = any(os.path.isfile(os.path.join(cache_dir, f)) for f in os.listdir(cache_dir))
                if files_exist:
                    # Ask user if they want to delete cached files
                    cache_response = messagebox.askyesno(
                        "Delete Cached Files?",
                        f"Cached AIS data files exist in:\n{cache_dir}\n\n"
                        "These files help speed up future analysis by avoiding downloads.\n\n"
                        "Would you like to delete the cached files?\n\n"
                        "Yes - Delete cached files\n"
                        "No - Keep cached files",
                        icon='question'
                    )
                    
                    if cache_response:  # User clicked Yes
                        try:
                            if clear_cache():
                                messagebox.showinfo("Success", "Cached files deleted successfully.")
                            else:
                                messagebox.showerror("Error", "Failed to delete some cached files.\n"
                                                        "You may need to delete them manually.")
                        except Exception as e:
                            messagebox.showerror("Error", f"Failed to delete cached files:\n{str(e)}\n\n"
                                                    "You may need to delete them manually.")
            except Exception:
                # If we can't access the cache directory or there's any other error,
                # just continue with closing the window
                pass
                
        # Close the window
        self.root.destroy()    

    # def on_closing(self):
    #     """Handle window closing - ask user if they want to delete temp files"""
    #     temp_dir = self._get_ais_temp_dir()
    #     temp_exists = os.path.exists(temp_dir)
        
    #     if temp_exists:
    #         # Ask user if they want to delete temp files
    #         response = messagebox.askyesno(
    #             "Delete Temporary Files?",
    #             f"The temporary directory exists:\n{temp_dir}\n\n"
    #             "Would you like to delete it before closing?\n\n"
    #             "Yes - Delete temporary files and close\n"
    #             "No - Keep temporary files and close",
    #             icon='question'
    #         )
            
    #         if response:  # User clicked Yes
    #             try:
    #                 self._delete_temp_directory()
    #                 messagebox.showinfo("Success", "Temporary files deleted successfully.")
    #             except Exception as e:
    #                 messagebox.showerror("Error", f"Failed to delete temporary files:\n{str(e)}\n\n"
    #                                             "You may need to delete them manually.")
        
    #     # Close the window
    #     self.root.destroy()
    
    def reset_for_new_analysis(self):
        """Reset the UI for a new analysis run"""
        # This method would be called when the user wants to start a fresh analysis
        # It should reset any necessary state while preserving settings
        
        # No need to reset date range, ship types, etc. unless specifically required
        # Just focus on the first tab to start a new analysis flow
        self.notebook.select(0)  # Select the first tab (Startup tab)
        
        # Optionally show a message
        messagebox.showinfo("New Analysis", "Ready to start a new analysis. Adjust parameters as needed and click 'Run Analysis'.")
    
    def conduct_additional_analysis(self):
        """Launch advanced analysis tools on the current dataset"""
        logger.info("Opening advanced analytical tools for additional analysis")
        
        try:
            if not ADVANCED_ANALYSIS_AVAILABLE:
                messagebox.showerror("Error", 
                    "Advanced analysis module is not available.\\n\\n"
                    "Please ensure advanced_analysis.py is in the same directory as SFD_GUI.py")
                return
            
            # Get output directory from config or use default
            output_dir = self.output_directory.get()
            if not output_dir:
                output_dir = "C:/AIS_Data/Output"
            
            # Get config path relative to script directory (same as load_config does)
            if getattr(sys, 'frozen', False):
                script_dir = os.path.dirname(sys.executable)
            else:
                script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, 'config.ini')
            
            # Launch the advanced analysis GUI
            try:
                advanced_gui = AdvancedAnalysisGUI(self.root, output_dir, config_path)
                logger.info("Advanced analysis GUI launched successfully")
            except Exception as e:
                logger.error(f"Error launching advanced analysis GUI: {e}")
                logger.error(traceback.format_exc())
                messagebox.showerror("Error", 
                    f"Could not launch advanced analysis interface:\\n{str(e)}\\n\\n"
                    "Please check that config.ini exists and contains valid settings.")
            
        except Exception as e:
            import traceback
            logger.error(f"Error creating additional analysis window: {str(e)}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Could not open additional analysis window: {str(e)}")
            return
    
    def launch_analysis_tool(self, tool_name, parent_window):
        """Launch a specific analysis tool on the current dataset"""
        # This is a placeholder function that would be implemented to handle different analysis tools
        # In a real implementation, we would access the dataframe from the initial analysis
        
        # For demonstration purposes, we'll just log the tool that was selected and show a message
        logger.info(f"Selected analysis tool: {tool_name}")
        
        # Show processing message
        processing_window = tk.Toplevel(self.root)
        processing_window.title("Processing")
        processing_window.geometry("400x150")
        processing_window.transient(self.root)
        processing_window.resizable(False, False)
        
        # Add a message
        ttk.Label(processing_window, text=f"Processing {tool_name}...", 
                 font=("Arial", 12)).pack(pady=20)
        
        # Add a progress bar
        progress = ttk.Progressbar(processing_window, mode="indeterminate")
        progress.pack(fill=tk.X, padx=20, pady=10)
        progress.start()
        
        # Simulate processing
        def finish_processing():
            progress.stop()
            processing_window.destroy()
            
            # Show result message
            messagebox.showinfo("Analysis Complete", 
                              f"The {tool_name} has been completed successfully.\n\n"
                              "In a production version, this would display actual results.")
        
        # Simulate processing time (2 seconds)
        self.root.after(2000, finish_processing)
        
        # Center the processing window
        processing_window.update_idletasks()
        width = processing_window.winfo_width()
        height = processing_window.winfo_height()
        x = (processing_window.winfo_screenwidth() // 2) - (width // 2)
        y = (processing_window.winfo_screenheight() // 2) - (height // 2)
        processing_window.geometry(f"{width}x{height}+{x}+{y}")
        
        # Make sure this window stays on top
        processing_window.grab_set()
        processing_window.focus_set()
    
    def create_collapsible_section(self, parent, title, default_expanded=True):
        """Create a collapsible section with header and content frame"""
        # Create section header
        section_header = ttk.Frame(parent)
        section_header.pack(fill=tk.X, pady=5)
        
        # Create section state variable
        expanded = tk.BooleanVar(value=default_expanded)
        
        # Create content frame
        content_frame = ttk.Frame(parent)
        
        # Define toggle function
        def toggle_section():
            if expanded.get():
                content_frame.pack(fill=tk.X, padx=10, pady=5)
                toggle_btn.configure(text=f"[EXPANDED] {title}")
            else:
                content_frame.pack_forget()
                toggle_btn.configure(text=f"[COLLAPSED] {title}")
        
        # Create toggle button
        toggle_btn = ttk.Button(
            section_header, 
            text=f"[EXPANDED] {title}" if default_expanded else f"[COLLAPSED] {title}", 
            command=lambda: [expanded.set(not expanded.get()), toggle_section()]
        )
        toggle_btn.pack(fill=tk.X, padx=10)
        
        # Initial state
        if default_expanded:
            content_frame.pack(fill=tk.X, padx=10, pady=5)
        
        return content_frame
    
    def __init__(self, root):
        self.root = root
        self.root.title("AIS Shipping Fraud Detection System")
        self.root.geometry("1680x840")
        
        # Set up window close protocol to ask about temp files
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Try to load the banner image
        try:
            # Get the directory of this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            banner_path = os.path.join(script_dir, "SFD_AI_banner.png")
            
            self.banner_img = tk.PhotoImage(file=banner_path)
            self.banner_label = tk.Label(root, image=self.banner_img)
            self.banner_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
            print(f"Loaded banner image from {banner_path}")
        except Exception as e:
            print(f"Warning: Could not load banner image: {e}")
            # Create a text banner instead
            self.banner_label = tk.Label(root, text="AIS SHIPPING FRAUD DETECTION", 
                                    font=("Arial", 16, "bold"), bg="#003366", fg="white", padx=20, pady=15)
            self.banner_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Create a style for the notebook tabs with rounded top corners and darker borders
        # We need to use ttk.Style().configure directly to style the tabs
        style = ttk.Style()
        
        # Configure the notebook
        style.configure('TNotebook', background='#f0f0f0', borderwidth=2)
        
        # Configure scrollbars to be wider
        style.configure('Vertical.TScrollbar', width=100)
        style.configure('Horizontal.TScrollbar', width=100)
        
        # Configure the tabs with darker borders and rounded corners
        style.configure('TNotebook.Tab',
                        padding=[15, 6],  # Increased padding for a better look
                        borderwidth=2,     # Thicker border
                        relief='ridge',    # Ridge relief gives a 3D effect
                        background='#e0e0e0')
        
        # Configure tab states
        style.map('TNotebook.Tab',
                  background=[('selected', '#ffffff'), ('active', '#f0f0f0')],
                  foreground=[('selected', '#000000'), ('active', '#000000')],
                  borderwidth=[('selected', 3), ('active', 2)],
                  relief=[('selected', 'ridge')])
        
        # Apply custom layout for tabs to create rounded corners effect
        # This is the best approximation in tkinter/ttk as it doesn't directly support rounded corners
        # We're using a combination of padding and relief to create a visual effect
        style.layout('TNotebook.Tab', [
            ('Notebook.tab', {
                'sticky': 'nswe',
                'children': [
                    ('Notebook.padding', {
                        'side': 'top',
                        'sticky': 'nswe',
                        'children': [
                            ('Notebook.label', {'side': 'top', 'sticky': ''})],
                    })],
            })
        ])
        
        # Initialize variables
        self.data_directory = tk.StringVar()
        self.output_directory = tk.StringVar(value="C:\\AIS_Data\\Reports")
        
        # Date range variables
        self.start_date = tk.StringVar(value="2024-10-15")  # Default to Oct 15, 2024
        self.end_date = tk.StringVar(value="2024-10-20")  # Default to Oct 20, 2024
        
        # Anomaly types and thresholds
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
        
        # Output controls variables
        self.output_controls = {
            # Controls which reports and visualizations are generated
            # Statistical analysis reports
            'generate_statistics_excel': tk.BooleanVar(value=True),
            #'generate_statistics_csv': tk.BooleanVar(value=True),
            # Visualization maps
            'generate_overall_map': tk.BooleanVar(value=True),
            'generate_vessel_path_maps': tk.BooleanVar(value=True),
            # Charts and visualizations
            'generate_charts': tk.BooleanVar(value=True),
            # Chart types (only used if generate_charts is True)
            'generate_anomaly_type_chart': tk.BooleanVar(value=True),
            'generate_vessel_anomaly_chart': tk.BooleanVar(value=True),
            'generate_date_anomaly_chart': tk.BooleanVar(value=True),
            # Only generate reports for vessels with anomalies
            'filter_to_anomaly_vessels_only': tk.BooleanVar(value=False),
            # Show latitude/longitude grid lines on maps
            'show_lat_long_grid': tk.BooleanVar(value=True),
            # Show vessel heatmaps
            'show_anomaly_heatmap': tk.BooleanVar(value=True)
        }
        
        # Processing options
        self.use_dask = tk.BooleanVar(value=True)
        self.use_gpu = tk.BooleanVar(value=True)
        
        # Analysis Filters
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
        
        # S3 variables
        self.use_s3 = tk.BooleanVar(value=False)
        self.s3_data_uri = tk.StringVar(value="s3://ai-dsde-txg-datathon2025/teams/dreadnought/parquet/")
        self.s3_auth_method = tk.StringVar(value="keys") # only keys authentication is supported
        self.s3_access_key = tk.StringVar()
        self.s3_secret_key = tk.StringVar()
        self.s3_session_token = tk.StringVar()
        self.s3_region = tk.StringVar(value="us-east-1")
        self.s3_profile_name = tk.StringVar() # Kept for backward compatibility
        self.s3_role_arn = tk.StringVar() # Kept for backward compatibility
        self.s3_bucket_name = tk.StringVar()
        self.s3_prefix = tk.StringVar()
        self.s3_local_dir = tk.StringVar()
        
        # Ship type variables - using dict with ship type number as key and BooleanVar as value
        self.ship_types = {
            # Wing in ground (WIG)
            20: {'name': 'Wing in ground (WIG), all ships of this type', 'var': tk.BooleanVar(value=True), 'category': 'WIG'},
            21: {'name': 'Wing in ground (WIG), Hazardous category A', 'var': tk.BooleanVar(value=True), 'category': 'WIG'},
            22: {'name': 'Wing in ground (WIG), Hazardous category B', 'var': tk.BooleanVar(value=True), 'category': 'WIG'},
            23: {'name': 'Wing in ground (WIG), Hazardous category C', 'var': tk.BooleanVar(value=True), 'category': 'WIG'},
            24: {'name': 'Wing in ground (WIG), Hazardous category D', 'var': tk.BooleanVar(value=True), 'category': 'WIG'},
            
            # Special craft
            30: {'name': 'Fishing', 'var': tk.BooleanVar(value=True), 'category': 'Special'},
            31: {'name': 'Towing', 'var': tk.BooleanVar(value=True), 'category': 'Special'},
            32: {'name': 'Towing: length exceeds 200m or breadth exceeds 25m', 'var': tk.BooleanVar(value=True), 'category': 'Special'},
            33: {'name': 'Dredging or underwater ops', 'var': tk.BooleanVar(value=True), 'category': 'Special'},
            34: {'name': 'Diving ops', 'var': tk.BooleanVar(value=True), 'category': 'Special'},
            35: {'name': 'Military ops', 'var': tk.BooleanVar(value=True), 'category': 'Special'},
            36: {'name': 'Sailing', 'var': tk.BooleanVar(value=True), 'category': 'Special'},
            37: {'name': 'Pleasure Craft', 'var': tk.BooleanVar(value=True), 'category': 'Special'},
            38: {'name': 'Reserved', 'var': tk.BooleanVar(value=True), 'category': 'Special'},
            39: {'name': 'Reserved', 'var': tk.BooleanVar(value=True), 'category': 'Special'},
            # High speed craft (HSC)
            40: {'name': 'High speed craft (HSC), all ships of this type', 'var': tk.BooleanVar(value=True), 'category': 'HSC'},
            41: {'name': 'High speed craft (HSC), Hazardous category A', 'var': tk.BooleanVar(value=True), 'category': 'HSC'},
            42: {'name': 'High speed craft (HSC), Hazardous category B', 'var': tk.BooleanVar(value=True), 'category': 'HSC'},
            43: {'name': 'High speed craft (HSC), Hazardous category C', 'var': tk.BooleanVar(value=True), 'category': 'HSC'},
            44: {'name': 'High speed craft (HSC), Hazardous category D', 'var': tk.BooleanVar(value=True), 'category': 'HSC'},
            45: {'name': 'High speed craft (HSC), Reserved', 'var': tk.BooleanVar(value=True), 'category': 'HSC'},
            46: {'name': 'High speed craft (HSC), Reserved', 'var': tk.BooleanVar(value=True), 'category': 'HSC'},
            47: {'name': 'High speed craft (HSC), Reserved', 'var': tk.BooleanVar(value=True), 'category': 'HSC'},
            48: {'name': 'High speed craft (HSC), Reserved', 'var': tk.BooleanVar(value=True), 'category': 'HSC'},
            49: {'name': 'High speed craft (HSC), No additional information', 'var': tk.BooleanVar(value=True), 'category': 'HSC'},
            
            # Special purpose
            50: {'name': 'Pilot Vessel', 'var': tk.BooleanVar(value=True), 'category': 'Special Purpose'},
            51: {'name': 'Search and Rescue vessel', 'var': tk.BooleanVar(value=True), 'category': 'Special Purpose'},
            52: {'name': 'Tug', 'var': tk.BooleanVar(value=True), 'category': 'Special Purpose'},
            53: {'name': 'Port Tender', 'var': tk.BooleanVar(value=True), 'category': 'Special Purpose'},
            54: {'name': 'Anti-pollution equipment', 'var': tk.BooleanVar(value=True), 'category': 'Special Purpose'},
            55: {'name': 'Law Enforcement', 'var': tk.BooleanVar(value=True), 'category': 'Special Purpose'},
            56: {'name': 'Spare - Local Vessel', 'var': tk.BooleanVar(value=True), 'category': 'Special Purpose'},
            57: {'name': 'Spare - Local Vessel', 'var': tk.BooleanVar(value=True), 'category': 'Special Purpose'},
            58: {'name': 'Medical Transport', 'var': tk.BooleanVar(value=True), 'category': 'Special Purpose'},
            59: {'name': 'Noncombatant ship (RR Resolution No. 18)', 'var': tk.BooleanVar(value=True), 'category': 'Special Purpose'},
            
            # Passenger
            60: {'name': 'Passenger, all ships of this type', 'var': tk.BooleanVar(value=True), 'category': 'Passenger'},
            61: {'name': 'Passenger, Hazardous category A', 'var': tk.BooleanVar(value=True), 'category': 'Passenger'},
            62: {'name': 'Passenger, Hazardous category B', 'var': tk.BooleanVar(value=True), 'category': 'Passenger'},
            63: {'name': 'Passenger, Hazardous category C', 'var': tk.BooleanVar(value=True), 'category': 'Passenger'},
            64: {'name': 'Passenger, Hazardous category D', 'var': tk.BooleanVar(value=True), 'category': 'Passenger'},
            69: {'name': 'Passenger, No additional information', 'var': tk.BooleanVar(value=True), 'category': 'Passenger'},
            
            # Cargo
            70: {'name': 'Cargo, all ships of this type', 'var': tk.BooleanVar(value=True), 'category': 'Cargo'},
            71: {'name': 'Cargo, Hazardous category A', 'var': tk.BooleanVar(value=True), 'category': 'Cargo'},
            72: {'name': 'Cargo, Hazardous category B', 'var': tk.BooleanVar(value=True), 'category': 'Cargo'},
            73: {'name': 'Cargo, Hazardous category C', 'var': tk.BooleanVar(value=True), 'category': 'Cargo'},
            74: {'name': 'Cargo, Hazardous category D', 'var': tk.BooleanVar(value=True), 'category': 'Cargo'},
            79: {'name': 'Cargo, No additional information', 'var': tk.BooleanVar(value=True), 'category': 'Cargo'},
            
            # Tanker
            80: {'name': 'Tanker, all ships of this type', 'var': tk.BooleanVar(value=True), 'category': 'Tanker'},
            81: {'name': 'Tanker, Hazardous category A', 'var': tk.BooleanVar(value=True), 'category': 'Tanker'},
            82: {'name': 'Tanker, Hazardous category B', 'var': tk.BooleanVar(value=True), 'category': 'Tanker'},
            83: {'name': 'Tanker, Hazardous category C', 'var': tk.BooleanVar(value=True), 'category': 'Tanker'},
            84: {'name': 'Tanker, Hazardous category D', 'var': tk.BooleanVar(value=True), 'category': 'Tanker'},
            89: {'name': 'Tanker, No additional information', 'var': tk.BooleanVar(value=True), 'category': 'Tanker'},
            
            # Other
            90: {'name': 'Other Type, all ships of this type', 'var': tk.BooleanVar(value=True), 'category': 'Other'},
            91: {'name': 'Other Type, Hazardous category A', 'var': tk.BooleanVar(value=True), 'category': 'Other'},
            92: {'name': 'Other Type, Hazardous category B', 'var': tk.BooleanVar(value=True), 'category': 'Other'},
            93: {'name': 'Other Type, Hazardous category C', 'var': tk.BooleanVar(value=True), 'category': 'Other'},
            94: {'name': 'Other Type, Hazardous category D', 'var': tk.BooleanVar(value=True), 'category': 'Other'}
        }
        
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))  # Reduced top padding to account for banner
        
        # Bind tab change event to update data
        self.notebook.bind("<<NotebookTabChanged>>", self.notebook_tab_changed)
        
        # Initialize zone violations list (before creating tab that uses it)
        self.zone_violations = []  # List of dicts: {'name': str, 'lat_min': float, 'lat_max': float, 'lon_min': float, 'lon_max': float, 'is_selected': bool}
        
        # Create tabs
        self.create_date_selection_tab()
        self.create_ship_types_tab()
        self.create_anomaly_types_tab()
        self.create_analysis_filters_tab()
        self.create_data_tab()
        self.create_zone_violations_tab()
        self.create_output_controls_tab()
        self.create_instructions_tab()
                
        # Create bottom buttons
        self.create_bottom_buttons()
        
        # Make sure all ship types are selected by default
        for ship_type in self.ship_types:
            self.ship_types[ship_type]['var'].set(True)
            
        # Load config if exists
        self.load_config()
        
    def create_date_selection_tab(self):
        """Create the Startup tab with descriptive text and date range settings"""
        date_selection_frame = ttk.Frame(self.notebook)
        self.notebook.add(date_selection_frame, text="Startup")
        
        # Add buttons frame at the top in two equal rows
        button_frame = ttk.Frame(date_selection_frame)
        button_frame.grid(row=0, column=0, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=10)
        
        # Create top row frame
        top_row_frame = ttk.Frame(button_frame)
        top_row_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # Create bottom row frame
        bottom_row_frame = ttk.Frame(button_frame)
        bottom_row_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # Top row: Run Analysis button (green) - alone on top
        run_btn = tk.Button(top_row_frame, text="Run Analysis", command=self.run_analysis, bg="green", fg="white", font=("Arial", 10, "bold"))
        run_btn.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
        
        # Bottom row: Save Configuration, Check for GPU Acceleration, Exit (red)
        save_btn = ttk.Button(bottom_row_frame, text="Save Configuration", command=self.save_config)
        save_btn.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
        
        ttk.Button(bottom_row_frame, text="Check for GPU Acceleration", command=self.check_gpu_acceleration).pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
        
        # Exit button (red)
        exit_btn = tk.Button(bottom_row_frame, text="Exit", command=self.on_closing, bg="red", fg="white", font=("Arial", 10, "bold"))
        exit_btn.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
        
        # Add descriptive text below buttons
        desc_frame = ttk.Frame(date_selection_frame)
        desc_frame.grid(row=1, column=0, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=5)
        
        desc_text = """This system analyzes Automatic Identification System (AIS) data to detect potentially fraudulent shipping activities. It does so by identifing anomalies such as vessels turning off their AIS beacons (sudden disappearances), sudden reappearances, unusual travel distances, and large inconsistencies between reported Course Over Ground (COG) and Heading.  Users have the option of examining data over specific date ranges, or examining data leading up to the current date.  
NOTE: In this test implementation only data from OCT 2024 to MAR 2025 is available for use.

Select the time period on this tab and press the Run Analysis button to execute your inquery.  You can select the specific types of ships and anomalies you want to search for in those tabs.  The Data and Results folders can be set on the Data Tab."""
        
        desc_label = ttk.Label(desc_frame, text=desc_text, wraplength=750, justify="left")
        desc_label.pack(fill=tk.X, padx=10, pady=10)
        
        # Time Range Selection Frame
        time_frame = ttk.LabelFrame(date_selection_frame, text="Time Range Selection")
        time_frame.grid(row=2, column=0, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=10)
        
        # Date range label
        ttk.Label(time_frame, text="Date Range Selection", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5, columnspan=3)
        
        date_range_frame = ttk.Frame(time_frame)
        date_range_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Check if tkcalendar is available
        if tkcalendar_available:
            ttk.Label(date_range_frame, text="Start Date:").pack(side=tk.LEFT, padx=2)
            self.start_date_picker = DateEntry(date_range_frame, width=12, background='darkblue',
                                             foreground='white', borderwidth=2, date_pattern='y-mm-dd',
                                             state='readonly')
            self.start_date_picker.pack(side=tk.LEFT, padx=2)
            
            ttk.Label(date_range_frame, text="End Date:").pack(side=tk.LEFT, padx=2)
            self.end_date_picker = DateEntry(date_range_frame, width=12, background='darkblue',
                                           foreground='white', borderwidth=2, date_pattern='y-mm-dd',
                                           state='readonly')
            self.end_date_picker.pack(side=tk.LEFT, padx=2)
            
            # Set default dates to Oct 15-20, 2024
            self.start_date_picker.set_date(datetime(2024, 10, 15).date())
            self.end_date_picker.set_date(datetime(2024, 10, 20).date())
        else:
            # If tkcalendar is not available, use basic entry fields with validation
            self.date_warning = ttk.Label(date_range_frame, 
                                        text="Warning: tkcalendar not installed. Using basic date fields.",
                                        foreground='red')
            self.date_warning.pack(side=tk.TOP, pady=2)
            
            date_fields_frame = ttk.Frame(date_range_frame)
            date_fields_frame.pack(side=tk.TOP)
            
            ttk.Label(date_fields_frame, text="Start Date (YYYY-MM-DD):").pack(side=tk.LEFT, padx=2)
            self.start_date_entry = ttk.Entry(date_fields_frame, textvariable=self.start_date, width=12)
            self.start_date_entry.pack(side=tk.LEFT, padx=2)
            
            ttk.Label(date_fields_frame, text="End Date (YYYY-MM-DD):").pack(side=tk.LEFT, padx=2)
            self.end_date_entry = ttk.Entry(date_fields_frame, textvariable=self.end_date, width=12)
            self.end_date_entry.pack(side=tk.LEFT, padx=2)
            
            # Recommendation to install tkcalendar
            ttk.Label(time_frame, text="For better date selection, install tkcalendar: pip install tkcalendar",
                     foreground='blue').grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
    
    def create_ship_types_tab(self):
        """Create the Ship Types tab with checkboxes for each ship type"""
        ship_types_frame = ttk.Frame(self.notebook)
        self.notebook.add(ship_types_frame, text="Ship Types")
        
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
        categories_added = []
        for category in sorted(categories.keys()):
            if category not in categories_added:
                select_method = getattr(self, f"select_all_{category.lower().replace(' ', '_')}", None)
                deselect_method = getattr(self, f"deselect_all_{category.lower().replace(' ', '_')}", None)
                
                if select_method is None:
                    # Create method dynamically if it doesn't exist
                    setattr(self, f"select_all_{category.lower().replace(' ', '_')}", lambda c=category: self.select_category(c))
                    select_method = getattr(self, f"select_all_{category.lower().replace(' ', '_')}")
                    
                if deselect_method is None:
                    # Create method dynamically if it doesn't exist
                    setattr(self, f"deselect_all_{category.lower().replace(' ', '_')}", lambda c=category: self.deselect_category(c))
                    deselect_method = getattr(self, f"deselect_all_{category.lower().replace(' ', '_')}")
                
                category_buttons.append((f"Select All {category}", select_method))
                category_buttons.append((f"Deselect All {category}", deselect_method))
                categories_added.append(category)
        
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
        ttk.Label(scrollable_frame, text="Select ship types to include in analysis:", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky=tk.W, padx=5, pady=10, columnspan=3)
            
        # Create frames for each category
        row = 1
        col = 0
        max_columns = 3  # Maximum columns in grid
        
        for category, ship_types_list in sorted(categories.items()):
            # Create a frame for this category
            category_frame = ttk.LabelFrame(scrollable_frame, text=f"{category} Ships")
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
    
    def select_all_cargo(self):
        """Select all cargo ship types"""
        for ship_type in range(70, 80):
            self.ship_types[ship_type]['var'].set(True)
    
    def deselect_all_cargo(self):
        """Deselect all cargo ship types"""
        for ship_type in range(70, 80):
            self.ship_types[ship_type]['var'].set(False)
    
    def select_all_tanker(self):
        """Select all tanker ship types"""
        for ship_type in range(80, 90):
            self.ship_types[ship_type]['var'].set(True)
    
    def deselect_all_tanker(self):
        """Deselect all tanker ship types"""
        for ship_type in range(80, 90):
            self.ship_types[ship_type]['var'].set(False)
    
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
    
    def create_anomaly_types_tab(self):
        """Create the Anomaly Types tab with checkboxes for each type and thresholds"""
        anomaly_frame = ttk.Frame(self.notebook)
        self.notebook.add(anomaly_frame, text="Anomaly Types")
        
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
    
    def select_all_anomalies(self):
        """Select all anomaly types"""
        for var in self.anomaly_types.values():
            var.set(True)

    def deselect_all_anomalies(self):
        """Deselect all anomaly types"""
        for var in self.anomaly_types.values():
            var.set(False)
            
    def create_analysis_filters_tab(self):
        """Create the Analysis Filters tab with settings for filtering analysis"""
        filters_frame = ttk.Frame(self.notebook)
        self.notebook.add(filters_frame, text="Analysis Filters")
        
        # Create a canvas with scrollbar for the filters tab
        canvas = tk.Canvas(filters_frame, width=750)
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
        
        # Use Defaults button at the top
        defaults_frame = ttk.Frame(scrollable_frame)
        defaults_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(defaults_frame, text="Use Defaults", 
                   command=self.reset_analysis_filters_to_defaults).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Geographic Boundaries Section
        geo_frame = ttk.LabelFrame(scrollable_frame, text="Geographic Boundaries")
        geo_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Latitude controls
        lat_frame = ttk.Frame(geo_frame)
        lat_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(lat_frame, text="Latitude Range:").grid(row=0, column=0, sticky=tk.W, padx=5)
        
        min_lat_frame = ttk.Frame(lat_frame)
        min_lat_frame.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(min_lat_frame, text="Min:").pack(side=tk.LEFT)
        ttk.Entry(min_lat_frame, textvariable=self.analysis_filters['min_latitude'], width=8).pack(side=tk.LEFT)
        
        max_lat_frame = ttk.Frame(lat_frame)
        max_lat_frame.grid(row=0, column=2, sticky=tk.W, padx=5)
        ttk.Label(max_lat_frame, text="Max:").pack(side=tk.LEFT)
        ttk.Entry(max_lat_frame, textvariable=self.analysis_filters['max_latitude'], width=8).pack(side=tk.LEFT)
        
        # Longitude controls
        lon_frame = ttk.Frame(geo_frame)
        lon_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(lon_frame, text="Longitude Range:").grid(row=0, column=0, sticky=tk.W, padx=5)
        
        min_lon_frame = ttk.Frame(lon_frame)
        min_lon_frame.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(min_lon_frame, text="Min:").pack(side=tk.LEFT)
        ttk.Entry(min_lon_frame, textvariable=self.analysis_filters['min_longitude'], width=8).pack(side=tk.LEFT)
        
        max_lon_frame = ttk.Frame(lon_frame)
        max_lon_frame.grid(row=0, column=2, sticky=tk.W, padx=5)
        ttk.Label(max_lon_frame, text="Max:").pack(side=tk.LEFT)
        ttk.Entry(max_lon_frame, textvariable=self.analysis_filters['max_longitude'], width=8).pack(side=tk.LEFT)
        
        # Draw Box button
        draw_box_frame = ttk.Frame(geo_frame)
        draw_box_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(draw_box_frame, text="Draw Box on Map", 
                   command=self.draw_geographic_box).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Time Filters Section
        time_frame = ttk.LabelFrame(scrollable_frame, text="Time Filters")
        time_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Hour range controls
        hour_frame = ttk.Frame(time_frame)
        hour_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(hour_frame, text="Hour of Day Range (0-24):").grid(row=0, column=0, sticky=tk.W, padx=5)
        
        start_hour_frame = ttk.Frame(hour_frame)
        start_hour_frame.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(start_hour_frame, text="Start:").pack(side=tk.LEFT)
        ttk.Entry(start_hour_frame, textvariable=self.analysis_filters['time_start_hour'], width=5).pack(side=tk.LEFT)
        
        end_hour_frame = ttk.Frame(hour_frame)
        end_hour_frame.grid(row=0, column=2, sticky=tk.W, padx=5)
        ttk.Label(end_hour_frame, text="End:").pack(side=tk.LEFT)
        ttk.Entry(end_hour_frame, textvariable=self.analysis_filters['time_end_hour'], width=5).pack(side=tk.LEFT)
        
        # Anomaly Filtering Section
        anomaly_frame = ttk.LabelFrame(scrollable_frame, text="Anomaly Filtering")
        anomaly_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Confidence threshold
        conf_frame = ttk.Frame(anomaly_frame)
        conf_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(conf_frame, text="Minimum Confidence Level (0-100):").pack(side=tk.LEFT, padx=5)
        ttk.Entry(conf_frame, textvariable=self.analysis_filters['min_confidence'], width=5).pack(side=tk.LEFT, padx=5)
        
        # Max anomalies per vessel
        max_anom_frame = ttk.Frame(anomaly_frame)
        max_anom_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(max_anom_frame, text="Maximum Anomalies Per Vessel:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(max_anom_frame, textvariable=self.analysis_filters['max_anomalies_per_vessel'], width=5).pack(side=tk.LEFT, padx=5)
        
        # MMSI Filter Section
        mmsi_frame = ttk.LabelFrame(scrollable_frame, text="MMSI Filtering")
        mmsi_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # MMSI list input
        mmsi_input_frame = ttk.Frame(mmsi_frame)
        mmsi_input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(mmsi_input_frame, text="Filter by MMSI (comma-separated, leave empty for all):").pack(anchor=tk.W, padx=5)
        ttk.Entry(mmsi_input_frame, textvariable=self.analysis_filters['filter_mmsi_list'], width=50).pack(fill=tk.X, padx=5, pady=5)
        
        # Note: Anomaly Filter Toggles Section has been removed
        
        # Help text
        help_frame = ttk.Frame(scrollable_frame)
        help_frame.pack(fill=tk.X, padx=10, pady=10)
        
        help_text = """Analysis Filters allow you to narrow down the scope of the analysis by geographical area, 
        time of day, and specific vessels. You can also set the minimum confidence threshold for anomaly detection 
        and limit the number of anomalies reported per vessel. All anomaly types are now automatically enabled."""
        
        ttk.Label(help_frame, text=help_text, wraplength=700, justify="left").pack(anchor=tk.W, padx=5, pady=5)
            
    def get_year_from_date_range(self):
        """Extract year or years from start_date and end_date in Startup tab"""
        try:
            # Get years from both start and end dates
            start_date = self.start_date.get()
            end_date = self.end_date.get()
            
            if start_date and end_date:
                start_year = datetime.strptime(start_date, '%Y-%m-%d').year
                end_year = datetime.strptime(end_date, '%Y-%m-%d').year
                
                # If dates span different years
                if start_year != end_year:
                    # Return a range of years (will be used in display)
                    years_range = f"{start_year}-{end_year}"
                    return years_range
                else:
                    # If same year, just return that year
                    return str(start_year)
            elif start_date:
                return str(datetime.strptime(start_date, '%Y-%m-%d').year)
            elif end_date:
                return str(datetime.strptime(end_date, '%Y-%m-%d').year)
        except (ValueError, AttributeError):
            pass
            
        # Default to current year if both fail
        return str(datetime.now().year)
    
    def create_data_tab(self):
        """Create the Data tab with additional settings and collapsible sections"""
        advanced_frame = ttk.Frame(self.notebook)
        self.notebook.add(advanced_frame, text="Data")
        
        # Create a canvas with scrollbar for the data tab
        canvas = tk.Canvas(advanced_frame, width=750)
        scrollbar = ttk.Scrollbar(advanced_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # File Locations Section
        file_frame = self.create_collapsible_section(scrollable_frame, "File Locations", True)
        
        # Data Directory
        ttk.Label(file_frame, text="Data Directory:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(file_frame, textvariable=self.data_directory, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_data_directory).grid(row=0, column=2, padx=5, pady=5)
        
        # Output Directory
        ttk.Label(file_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(file_frame, textvariable=self.output_directory, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_output_directory).grid(row=1, column=2, padx=5, pady=5)
        
        # Data Source Selection Section
        data_source_frame = self.create_collapsible_section(scrollable_frame, "Data Source Selection", True)
        
        # Radio buttons for data source selection
        self.data_source = tk.StringVar(value="noaa")
        
        # Auto-populate year from Startup tab date range
        self.noaa_year = tk.StringVar(value=self.get_year_from_date_range())
        
        # NOAA Data option
        noaa_frame = ttk.Frame(data_source_frame)
        noaa_frame.pack(fill=tk.X, padx=5, pady=5)
        
        noaa_radio = ttk.Radiobutton(noaa_frame, text="Use NOAA AIS Data", variable=self.data_source, 
                       value="noaa", command=self.toggle_data_source)
        noaa_radio.pack(side=tk.LEFT, padx=5)
        
        # We still auto-detect the year but don't display it in the UI
        # The self.noaa_year variable is still populated and used for processing
        ttk.Label(noaa_frame, text="(Using date range from Startup tab)").pack(side=tk.LEFT, padx=5)
        
        # Local Data option
        ttk.Radiobutton(data_source_frame, text="Use Local Data Folder", variable=self.data_source, 
                       value="local", command=self.toggle_data_source).pack(anchor=tk.W, padx=15, pady=5)
            
        # AWS S3 option
        ttk.Radiobutton(data_source_frame, text="Use AWS S3 Data Bucket", variable=self.data_source, 
                       value="s3", command=self.toggle_data_source,
                       state="normal" if aws_available else "disabled").pack(anchor=tk.W, padx=15, pady=5)
        
        # S3 Settings Section
        s3_frame = self.create_collapsible_section(scrollable_frame, "Amazon S3 Settings", True)
        
        # Check if boto3 is available
        if not aws_available:
            ttk.Label(s3_frame, text="AWS boto3 library not installed. Install with: pip install boto3", 
                     foreground="red").pack(anchor=tk.W, padx=15, pady=5)
            self.s3_frame_row = 1
        else:
            self.s3_frame_row = 0
        
        # S3 Bucket Configuration
        bucket_frame = ttk.Frame(s3_frame)
        bucket_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create rows for each input
        uri_row = ttk.Frame(bucket_frame)
        uri_row.pack(fill=tk.X, pady=5)
        
        bucket_row = ttk.Frame(bucket_frame)
        bucket_row.pack(fill=tk.X, pady=5)
        
        prefix_row = ttk.Frame(bucket_frame)
        prefix_row.pack(fill=tk.X, pady=5)
        
        # S3 URI (derived from bucket and prefix)
        ttk.Label(uri_row, text="S3 URI:", width=12).pack(side=tk.LEFT, padx=5)
        self.s3_uri_entry = ttk.Entry(uri_row, textvariable=self.s3_data_uri, width=60)
        self.s3_uri_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Bucket name
        ttk.Label(bucket_row, text="Bucket Name:", width=12).pack(side=tk.LEFT, padx=5)
        self.bucket_name_entry = ttk.Entry(bucket_row, textvariable=self.s3_bucket_name, width=40)
        self.bucket_name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.s3_bucket_name.set("ai-dsde-txg-datathon2025")
        
        # Prefix
        ttk.Label(prefix_row, text="Prefix:", width=12).pack(side=tk.LEFT, padx=5)
        self.prefix_entry = ttk.Entry(prefix_row, textvariable=self.s3_prefix, width=40)
        self.prefix_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.s3_prefix.set("teams/dreadnought/parquet/")
        
        # Auto-update S3 URI when bucket or prefix changes
        def update_s3_uri(*args):
            bucket = self.s3_bucket_name.get()
            prefix = self.s3_prefix.get()
            if bucket:
                self.s3_data_uri.set(f"s3://{bucket}/{prefix}")
        
        # Register trace callbacks
        self.s3_bucket_name.trace_add("write", update_s3_uri)
        self.s3_prefix.trace_add("write", update_s3_uri)
        
        # Add callback for credential changes (auto-saving removed)
        def save_aws_changes(*args):
            # This function previously auto-saved the config on each change
            # Now we just update the UI as needed but don't save to config.ini
            pass
        
        # Register callbacks for credential changes to save config
        self.s3_access_key.trace_add("write", save_aws_changes)
        self.s3_secret_key.trace_add("write", save_aws_changes)
        self.s3_session_token.trace_add("write", save_aws_changes)
        self.s3_region.trace_add("write", save_aws_changes)
        self.s3_auth_method.trace_add("write", save_aws_changes)
        
        # Initial update
        update_s3_uri()
        
        # AWS Authentication using Access Keys
        ttk.Label(s3_frame, text="AWS Authentication Details (Access Keys):").pack(anchor=tk.W, padx=5, pady=5)
        
        auth_keys_frame = ttk.Frame(s3_frame)
        auth_keys_frame.pack(fill=tk.X, padx=15, pady=5)
        
        # Create rows for each authentication input
        access_key_row = ttk.Frame(auth_keys_frame)
        access_key_row.pack(fill=tk.X, pady=3)
        
        secret_key_row = ttk.Frame(auth_keys_frame)
        secret_key_row.pack(fill=tk.X, pady=3)
        
        token_row = ttk.Frame(auth_keys_frame)
        token_row.pack(fill=tk.X, pady=3)
        
        # Access Key
        ttk.Label(access_key_row, text="Access Key:", width=12).pack(side=tk.LEFT, padx=5)
        self.access_key_entry = ttk.Entry(access_key_row, textvariable=self.s3_access_key, width=40)
        self.access_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Secret Key
        ttk.Label(secret_key_row, text="Secret Key:", width=12).pack(side=tk.LEFT, padx=5)
        self.secret_key_entry = ttk.Entry(secret_key_row, textvariable=self.s3_secret_key, width=40, show="*")
        self.secret_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Session Token - Expanded width to handle very long tokens
        ttk.Label(token_row, text="Session Token:", width=12).pack(side=tk.LEFT, padx=5)
        self.token_entry = ttk.Entry(token_row, textvariable=self.s3_session_token, width=60)
        self.token_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Advanced Authentication Options
        # Toggle for advanced auth options (region)
        self.show_advanced_auth = tk.BooleanVar(value=False)
        ttk.Checkbutton(s3_frame, text="Show advanced authentication options", 
                      variable=self.show_advanced_auth, command=self.toggle_advanced_auth).pack(
            anchor=tk.W, padx=5, pady=(15,5))
            
        # Advanced Authentication frame
        self.advanced_auth_frame = ttk.Frame(s3_frame)
        # Don't pack it initially - will be managed by toggle_advanced_auth
        
        # Create frames for each row in advanced auth frame
        # Keys are the only auth method, so we don't need auth method selection UI
        
        region_row = ttk.Frame(self.advanced_auth_frame)
        region_row.pack(fill=tk.X, pady=3)
        
        # Keys are the only supported authentication method
        
        # Region settings
        ttk.Label(region_row, text="AWS Region:", width=12).pack(side=tk.LEFT, padx=5)
        self.region_entry = ttk.Entry(region_row, textvariable=self.s3_region, width=20)
        self.region_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Test Connection Button
        button_frame = ttk.Frame(s3_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.test_s3_button = ttk.Button(button_frame, text="Test S3 Connection", 
                                      command=self.test_s3_connection,
                                      state="normal" if aws_available else "disabled")
        self.test_s3_button.pack(side=tk.RIGHT, padx=15)
        
        # Add Processing Options Section
        processing_frame = self.create_collapsible_section(scrollable_frame, "Processing Options", False)
        
        # Dask option
        dask_option = ttk.Frame(processing_frame)
        dask_option.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(dask_option, text="Use Dask for distributed processing", 
                        variable=self.use_dask).pack(anchor=tk.W, padx=15)
        
        # GPU option
        self.gpu_option_frame = ttk.Frame(processing_frame)
        self.gpu_option_frame.pack(fill=tk.X, pady=5)
        
        self.use_gpu = tk.BooleanVar(value=True)  # Default to True, will be disabled if not available
        ttk.Checkbutton(self.gpu_option_frame, text="Use GPU acceleration if available", 
                        variable=self.use_gpu, command=self.check_gpu_support).pack(anchor=tk.W, padx=15)
        
        # Add an install button for GPU support
        self.gpu_status_label = ttk.Label(self.gpu_option_frame, text="GPU status: Checking...")
        self.gpu_status_label.pack(anchor=tk.W, padx=15, pady=2)
        
        self.install_gpu_button = ttk.Button(self.gpu_option_frame, text="Install GPU Support", command=self.install_gpu_packages)
        # Initially hide the button until we check GPU status
        # self.install_gpu_button.pack(anchor=tk.W, padx=15, pady=2)
        
        # Add note label placeholder
        self.gpu_note_label = ttk.Label(self.gpu_option_frame, 
                                     text="Note: GPU acceleration is optional. The application will run without it.")
        
        # Check GPU status on startup
        self.root.after(1000, self.check_gpu_support)
        
        # Initialize the auth method settings
        self.toggle_auth_method()
        
        # Initialize S3 settings
        self.toggle_s3_settings()
    
    def create_zone_violations_tab(self):
        """Create the Zone Violations tab for managing restricted zones"""
        zone_frame = ttk.Frame(self.notebook)
        self.notebook.add(zone_frame, text="Zone Violations")
        
        # Top section: Buttons
        button_frame = ttk.Frame(zone_frame)
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=10)
        
        ttk.Button(button_frame, text="Add Zone", command=self.add_zone).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Middle section: Zone list with checkboxes (similar to anomaly types)
        list_frame = ttk.LabelFrame(zone_frame, text="Restricted Zones")
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
        self.zone_checkboxes = {}  # Maps zone name to {'selected': BooleanVar, 'frame': Frame}
        self.zone_frames = {}  # Maps zone name to frame widget
        
        # Zone list container
        self.zone_list_container = scrollable_frame
        
        # Add "Select All" buttons at the top
        select_button_frame = ttk.Frame(scrollable_frame)
        select_button_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        ttk.Button(select_button_frame, text="Select All", command=self.select_all_zones).pack(side=tk.LEFT, padx=5)
        ttk.Button(select_button_frame, text="Deselect All", command=self.deselect_all_zones).pack(side=tk.LEFT, padx=5)
        
        # Refresh zone list display
        self._refresh_zone_list()
        
        # Description
        desc_label = ttk.Label(zone_frame, 
                              text="Manage restricted zones for Zone Violation detection. Check 'Selected' to include in analysis.",
                              font=("Arial", 9))
        desc_label.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
    
    def add_zone(self):
        """Add a new zone violation area"""
        self._show_zone_dialog()
    
    def draw_geographic_box(self):
        """Draw a geographic box on a map and update lat/long bounds"""
        self._draw_box_for_bounds(
            self.root,
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
        m.save(temp_map_file)
        
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
        instructions = tk.Text(coord_dialog, height=6, wrap=tk.WORD, font=("Arial", 9))
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
                        lat_min_var.set(coords[0])
                        lat_max_var.set(coords[1])
                        lon_min_var.set(coords[2])
                        lon_max_var.set(coords[3])
                        messagebox.showinfo("Success", "Rectangle coordinates parsed and filled in!")
                        coord_dialog.destroy()
                    else:
                        messagebox.showerror("Error", "Please enter 4 comma-separated values: lat_min,lat_max,lon_min,lon_max\nOr paste JSON format from the map")
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid format: {e}\n\nPlease enter:\n- JSON format from map, or\n- 4 comma-separated values: lat_min,lat_max,lon_min,lon_max")
        
        ttk.Button(paste_frame, text="Parse & Fill", command=parse_pasted_coords).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(coord_dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(button_frame, text="Close", command=coord_dialog.destroy).pack(side=tk.LEFT, padx=5)
    
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
    
    def draw_zone(self):
        """Draw a zone on a map and extract coordinates (legacy method - now calls add_zone)"""
        # This method is kept for backward compatibility but now just opens the add zone dialog
        self.add_zone()
    
    def edit_zone(self):
        """Edit the selected zone (first selected zone if multiple)"""
        # Find first zone with selected checkbox checked
        selected_zone = None
        for zone in self.zone_violations:
            if zone.get('is_selected', True):
                selected_zone = zone
                break
        
        if selected_zone is None:
            messagebox.showwarning("No Selection", "Please select a zone to edit (check the 'Selected' checkbox).")
            return
        
        # Show edit dialog - pass the entire zone as zone_data for proper loading
        self._show_zone_dialog(
            zone_index=self.zone_violations.index(selected_zone),
            zone_name=selected_zone['name'],
            lat_min=selected_zone.get('lat_min', 0.0),
            lat_max=selected_zone.get('lat_max', 0.0),
            lon_min=selected_zone.get('lon_min', 0.0),
            lon_max=selected_zone.get('lon_max', 0.0),
            is_selected=selected_zone.get('is_selected', True),
            zone_data=selected_zone  # Pass full zone data for proper loading of all geometry types
        )
    
    def delete_zone(self):
        """Delete the selected zone (first selected zone if multiple)"""
        # Find first zone with selected checkbox checked
        selected_zone = None
        for zone in self.zone_violations:
            if zone.get('is_selected', True):
                selected_zone = zone
                break
        
        if selected_zone is None:
            messagebox.showwarning("No Selection", "Please select a zone to delete (check the 'Selected' checkbox).")
            return
        
        zone_name = selected_zone['name']
        
        # Confirm deletion
        if not messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete zone '{zone_name}'?"):
            return
        
        # Remove from list
        self.zone_violations = [z for z in self.zone_violations if z['name'] != zone_name]
        
        # Refresh display
        self._refresh_zone_list()
        # Save only zones to config.ini to avoid errors with uninitialized variables
        self._save_zones_only()
    
    def _show_zone_dialog(self, zone_index=None, zone_name="", lat_min=0.0, lat_max=0.0, 
                          lon_min=0.0, lon_max=0.0, is_selected=True, zone_data=None):
        """Show dialog for adding/editing a zone - supports all geometry types"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Zone" if zone_index is None else "Edit Zone")
        dialog.geometry("500x600")
        dialog.transient(self.root)
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
        
        # Zone name
        ttk.Label(dialog, text="Zone Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        name_var = tk.StringVar(value=zone_name)
        ttk.Entry(dialog, textvariable=name_var, width=30).grid(row=0, column=1, padx=5, pady=5)
        
        # Geometry type selection
        ttk.Label(dialog, text="Geometry Type:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        geometry_type_var = tk.StringVar(value=current_geometry_type)
        geometry_frame = ttk.Frame(dialog)
        geometry_frame.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Radiobutton(geometry_frame, text="Rectangle", variable=geometry_type_var, value="rectangle").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(geometry_frame, text="Circle", variable=geometry_type_var, value="circle").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(geometry_frame, text="Polygon", variable=geometry_type_var, value="polygon").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(geometry_frame, text="Polyline", variable=geometry_type_var, value="polyline").pack(side=tk.LEFT, padx=5)
        
        # Container for coordinate fields (will be updated based on geometry type)
        coord_frame = ttk.LabelFrame(dialog, text="Coordinates")
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
        ttk.Checkbutton(dialog, text="Selected (use in analysis)", variable=is_selected_var).grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        
        # Draw Zone button
        def draw_zone_from_dialog():
            """Draw zone on map and populate coordinate fields"""
            # Call the drawing method with parent dialog reference
            # Use a wrapper to wait for coordinate dialog to close and then update
            def draw_and_update():
                self._draw_zone_for_dialog(dialog, None, None, None, None, parent_dialog_ref=dialog)
                # Wait a bit for the coordinate dialog to process and close, then update
                dialog.after(500, lambda: self._update_dialog_from_drawn_coords(dialog, geometry_type_var, lat_min_var, lat_max_var, 
                                                                                  lon_min_var, lon_max_var, center_lat_var, center_lon_var,
                                                                                  radius_var, coords_text_widget, tolerance_var, update_coord_fields))
            draw_and_update()
        
        draw_button_frame = ttk.Frame(dialog)
        draw_button_frame.grid(row=4, column=0, columnspan=2, pady=5)
        ttk.Button(draw_button_frame, text="Draw Zone on Map", command=draw_zone_from_dialog).pack(side=tk.LEFT, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=7, column=0, columnspan=2, pady=10)
        
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
            # Save only zones to config.ini to avoid errors with uninitialized variables
            self._save_zones_only()
            dialog.destroy()
        
        def delete_zone_from_dialog():
            """Delete the zone from within the dialog"""
            if zone_index is None:
                messagebox.showwarning("Cannot Delete", "Cannot delete a zone that hasn't been saved yet.")
                return
            
            zone_name_to_delete = name_var.get().strip() or zone_name
            if not messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete zone '{zone_name_to_delete}'?"):
                return
            
            # Remove from list
            if zone_index < len(self.zone_violations):
                self.zone_violations.pop(zone_index)
            
            # Refresh display
            self._refresh_zone_list()
            # Save only zones to config.ini
            self._save_zones_only()
            dialog.destroy()
        
        ttk.Button(button_frame, text="Save", command=save_zone).pack(side=tk.LEFT, padx=5)
        if zone_index is not None:
            # Only show Delete button when editing an existing zone
            ttk.Button(button_frame, text="Delete", command=delete_zone_from_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
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
        self._save_zones_only()
    
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
                    is_selected=zone.get('is_selected', True)
                )
                break
    
    def select_all_zones(self):
        """Select all zones"""
        for zone in self.zone_violations:
            zone['is_selected'] = True
            if zone['name'] in self.zone_checkboxes:
                self.zone_checkboxes[zone['name']]['selected'].set(True)
        self._save_zones_only()
    
    def deselect_all_zones(self):
        """Deselect all zones"""
        for zone in self.zone_violations:
            zone['is_selected'] = False
            if zone['name'] in self.zone_checkboxes:
                self.zone_checkboxes[zone['name']]['selected'].set(False)
        self._save_zones_only()
    
    
    def create_output_controls_tab(self):
        """Create the Output Controls tab with checkboxes for each output"""
        output_controls_frame = ttk.Frame(self.notebook)
        self.notebook.add(output_controls_frame, text="Output Controls")
        
        # Output Controls section
        controls_frame = ttk.LabelFrame(output_controls_frame, text="Select Report Outputs")
        controls_frame.grid(row=0, column=0, sticky=tk.N+tk.W, padx=10, pady=10)
        
        # Group the controls by type for better organization
        groups = {
            "Reports": ["generate_statistics_excel", "generate_statistics_csv"],
            "Maps": ["generate_overall_map", "generate_vessel_path_maps", "show_lat_long_grid", "show_anomaly_heatmap"],
            "Charts": ["generate_charts", "generate_anomaly_type_chart", "generate_vessel_anomaly_chart", "generate_date_anomaly_chart"],
            "Filtering": ["filter_to_anomaly_vessels_only"]
        }
        
        row = 0
        # Add section headers and checkboxes for each group
        for group_name, control_keys in groups.items():
            ttk.Label(controls_frame, text=f"{group_name}:", font=("Arial", 10, "bold")).grid(
                row=row, column=0, sticky=tk.W, padx=5, pady=(10, 5))
            row += 1
            
            for key in control_keys:
                if key in self.output_controls:
                    # Create a more readable label from the key
                    label = " ".join(word.capitalize() for word in key.replace("generate_", "").split("_"))
                    ttk.Checkbutton(controls_frame, text=label, variable=self.output_controls[key]).grid(
                        row=row, column=0, sticky=tk.W, padx=20, pady=3)
                    row += 1
        
        # Add "Select All" buttons
        button_frame = ttk.Frame(controls_frame)
        button_frame.grid(row=row, column=0, sticky=tk.W, padx=5, pady=10, columnspan=2)
        ttk.Button(button_frame, text="Select All", command=self.select_all_output_controls).grid(
            row=0, column=0, sticky=tk.W, padx=5)
        ttk.Button(button_frame, text="Deselect All", command=self.deselect_all_output_controls).grid(
            row=0, column=1, sticky=tk.W, padx=5)
        

    def select_all_output_controls(self):
        """Select all output controls"""
        for var in self.output_controls.values():
            var.set(True)

    def deselect_all_output_controls(self):
        """Deselect all output controls"""
        for var in self.output_controls.values():
            var.set(False)

    def create_instructions_tab(self):
        """Create the Instructions tab displaying README.md content"""
        instructions_frame = ttk.Frame(self.notebook)
        self.notebook.add(instructions_frame, text="Instructions")
        
        # Create a frame with scrollbar
        main_frame = ttk.Frame(instructions_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(main_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create text widget with scrollbar
        text_widget = tk.Text(main_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set,
                             font=("Consolas", 10), bg="white", fg="black",
                             padx=10, pady=10)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbar
        scrollbar.config(command=text_widget.yview)
        
        # Read README.md file
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            readme_path = os.path.join(script_dir, "README.md")
            
            if os.path.exists(readme_path):
                with open(readme_path, 'r', encoding='utf-8') as f:
                    readme_content = f.read()
                
                # Insert content into text widget
                text_widget.insert(tk.END, readme_content)
                
                # Make text widget read-only
                text_widget.config(state=tk.DISABLED)
            else:
                # If README.md doesn't exist, show a message
                error_msg = f"README.md file not found at:\n{readme_path}\n\nPlease ensure README.md is in the same directory as SFD_GUI.py"
                text_widget.insert(tk.END, error_msg)
                text_widget.config(state=tk.DISABLED)
                logger.warning(f"README.md not found at {readme_path}")
        except Exception as e:
            error_msg = f"Error reading README.md file:\n{str(e)}\n\nCheck the log file for more details."
            text_widget.insert(tk.END, error_msg)
            text_widget.config(state=tk.DISABLED)
            logger.error(f"Error reading README.md: {str(e)}")
            logger.error(traceback.format_exc())

    def browse_data_directory(self):
        """Open file dialog to select data directory"""
        directory = filedialog.askdirectory()
        if directory:
            self.data_directory.set(directory)

    def browse_output_directory(self):
        """Open file dialog to select output directory"""
        directory = filedialog.askdirectory()
        if directory:
            self.output_directory.set(directory)
    
    def reset_analysis_filters_to_defaults(self):
        """Reset all analysis filter values to their defaults"""
        if hasattr(self, 'analysis_filters'):
            # Reset to default values
            self.analysis_filters['min_latitude'].set(-90.0)
            self.analysis_filters['max_latitude'].set(90.0)
            self.analysis_filters['min_longitude'].set(-180.0)
            self.analysis_filters['max_longitude'].set(180.0)
            self.analysis_filters['time_start_hour'].set(0)
            self.analysis_filters['time_end_hour'].set(24)
            self.analysis_filters['min_confidence'].set(75)
            self.analysis_filters['max_anomalies_per_vessel'].set(10)
            self.analysis_filters['filter_mmsi_list'].set('')
            messagebox.showinfo("Defaults Applied", "All analysis filter values have been reset to their defaults.")
    
    def check_gpu_support(self):
        """Check if GPU support packages are available"""
        gpu_available = False
        missing_packages = []
        
        # Check for each GPU package
        for package in ["cudf", "cupy", "cuml"]:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_packages.append(package)
        
        # Update the use_gpu checkbox state based on what's installed
        # This doesn't change the user's preference, just informs them of capability
        
        if not missing_packages:
            self.gpu_status_label.config(text="GPU status: Support available - acceleration enabled")
            if hasattr(self, 'install_gpu_button') and self.install_gpu_button.winfo_ismapped():
                self.install_gpu_button.pack_forget()
            gpu_available = True
        else:
            # Check if conda is available
            conda_available = self.is_conda_available()
            if conda_available:
                self.gpu_status_label.config(text=f"GPU status: Optional acceleration not available (Conda available)")
            else:
                self.gpu_status_label.config(text=f"GPU status: Optional acceleration not available (using pip)")
                
            # Show the install button if packages are missing
            if hasattr(self, 'install_gpu_button') and not self.install_gpu_button.winfo_ismapped():
                self.install_gpu_button.pack(anchor=tk.W, padx=15, pady=2)
                
            # Add note about GPU being optional
            if not hasattr(self, 'gpu_note_label') or not self.gpu_note_label.winfo_ismapped():
                self.gpu_note_label = ttk.Label(self.gpu_option_frame, 
                                              text="Note: GPU acceleration is optional. The application will run without it.")
                self.gpu_note_label.pack(anchor=tk.W, padx=15, pady=(0, 5))
        
        return gpu_available
    
    def detect_gpu_hardware(self):
        """
        Detect GPU hardware (AMD, Intel, NVIDIA) using system commands.
        Returns a list of detected GPUs with their types.
        """
        detected_gpus = []
        is_windows = platform.system() == 'Windows'
        
        try:
            if is_windows:
                # Windows: Use wmic to get video controller information
                result = subprocess.run(
                    ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines[1:]:  # Skip header
                        line = line.strip()
                        if line and line.lower() != 'name':
                            gpu_name = line.strip()
                            gpu_type = None
                            gpu_name_lower = gpu_name.lower()
                            
                            # Detect GPU type
                            if 'nvidia' in gpu_name_lower or 'geforce' in gpu_name_lower or 'quadro' in gpu_name_lower or 'tesla' in gpu_name_lower:
                                gpu_type = 'NVIDIA'
                            elif 'amd' in gpu_name_lower or 'radeon' in gpu_name_lower:
                                gpu_type = 'AMD'
                            elif 'intel' in gpu_name_lower or 'iris' in gpu_name_lower or 'uhd' in gpu_name_lower or 'hd graphics' in gpu_name_lower:
                                gpu_type = 'Intel'
                            
                            if gpu_type:
                                detected_gpus.append({'name': gpu_name, 'type': gpu_type})
            else:
                # Linux: Use lspci to get GPU information
                result = subprocess.run(
                    ['lspci'],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if 'vga' in line.lower() or 'display' in line.lower() or '3d' in line.lower():
                            gpu_type = None
                            line_lower = line.lower()
                            
                            # Detect GPU type
                            if 'nvidia' in line_lower:
                                gpu_type = 'NVIDIA'
                            elif 'amd' in line_lower or 'radeon' in line_lower or 'ati' in line_lower:
                                gpu_type = 'AMD'
                            elif 'intel' in line_lower:
                                gpu_type = 'Intel'
                            
                            if gpu_type:
                                detected_gpus.append({'name': line.strip(), 'type': gpu_type})
        except Exception as e:
            logger.warning(f"Error detecting GPU hardware: {e}")
        
        return detected_gpus
    
    def test_cupy_functionality(self):
        """
        Test if cupy or HIP is available and working by performing a simple computation.
        Returns (available, working, error_message, backend_type)
        """
        # First try cupy
        cupy_available = False
        cupy_working = False
        cupy_error = None
        
        try:
            import cupy as cp  # type: ignore
            cupy_available = True
        except ImportError:
            cupy_available = False
        
        if cupy_available:
            try:
                # Test if GPU is available
                if not hasattr(cp, 'cuda') or not cp.cuda.is_available():
                    cupy_error = "cupy is installed but GPU is not available"
                else:
                    # Perform a simple computation test
                    test_array = cp.array([1.0, 2.0, 3.0, 4.0, 5.0])
                    result = cp.sum(test_array * 2)
                    result_cpu = float(result)
                    
                    # Verify the result
                    expected = 30.0  # (1+2+3+4+5) * 2
                    if abs(result_cpu - expected) < 0.001:
                        cupy_working = True
                        return (True, True, "cupy is working correctly", "cupy")
                    else:
                        cupy_error = f"cupy computation test failed (got {result_cpu}, expected {expected})"
            except Exception as e:
                cupy_error = f"cupy test failed: {str(e)}"
        
        # If cupy is not available or not working, try PyHIP
        if not cupy_working:
            try:
                # Prioritize pyhip (standard PyHIP package name)
                try:
                    import pyhip as hip  # type: ignore
                    # Check if HIP is available
                    if hasattr(hip, 'is_available') and hip.is_available():
                        return (True, True, "PyHIP is available and working", "HIP")
                    elif hasattr(hip, 'getDeviceCount'):
                        # Alternative check: try to get device count
                        device_count = hip.getDeviceCount()
                        if device_count > 0:
                            return (True, True, f"PyHIP is available and working ({device_count} device(s))", "HIP")
                        else:
                            hip_error = "PyHIP is installed but no HIP devices available"
                    else:
                        # If no availability check, assume it's available if imported
                        return (True, True, "PyHIP is available and working", "HIP")
                except ImportError:
                    # Try alternative HIP import name
                    try:
                        import hip  # type: ignore
                        if hasattr(hip, 'is_available') and hip.is_available():
                            return (True, True, "HIP is available and working", "HIP")
                        elif hasattr(hip, 'getDeviceCount'):
                            device_count = hip.getDeviceCount()
                            if device_count > 0:
                                return (True, True, f"HIP is available and working ({device_count} device(s))", "HIP")
                            else:
                                hip_error = "HIP is installed but no devices available"
                        else:
                            return (True, True, "HIP is available and working", "HIP")
                    except ImportError:
                        hip_error = "PyHIP libraries not found"
            except Exception as e:
                hip_error = f"PyHIP test failed: {str(e)}"
        
        # Return cupy status if available but not working
        if cupy_available:
            return (True, False, cupy_error or "cupy is installed but not working", "cupy")
        
        # Neither cupy nor HIP available
        return (False, False, "Neither cupy nor HIP is installed. Install cupy, cupy-rocm, or PyHIP for GPU acceleration.", None)
    
    def check_gpu_acceleration(self):
        """
        Comprehensive GPU detection and testing function.
        Detects AMD, Intel, and NVIDIA GPUs, tests cupy availability and functionality,
        and updates config.ini if GPU acceleration is available and working.
        """
        # Show progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("GPU Detection")
        progress_window.geometry("500x300")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        status_label = ttk.Label(progress_window, text="Checking for GPU acceleration...")
        status_label.pack(pady=10)
        
        log_text = scrolledtext.ScrolledText(progress_window, height=10, wrap=tk.WORD)
        log_text.pack(fill="both", expand=True, padx=20, pady=10)
        
        def update_log(message):
            log_text.insert(tk.END, message + "\n")
            log_text.see(tk.END)
            log_text.update()
        
        def run_detection():
            try:
                update_log("=" * 60)
                update_log("GPU Acceleration Detection")
                update_log("=" * 60)
                update_log("")
                
                # Step 1: Detect GPU hardware
                update_log("Step 1: Detecting GPU hardware...")
                detected_gpus = self.detect_gpu_hardware()
                
                if not detected_gpus:
                    update_log("  No GPU hardware detected.")
                    update_log("")
                    update_log("Result: GPU acceleration is NOT available")
                    self.root.after(100, lambda: self._complete_gpu_check(progress_window, False, "No GPU hardware detected"))
                    return
                
                update_log(f"  Found {len(detected_gpus)} GPU(s):")
                for i, gpu in enumerate(detected_gpus, 1):
                    update_log(f"    {i}. {gpu['name']} ({gpu['type']})")
                update_log("")
                
                # Step 2: Check if cupy or HIP is available
                update_log("Step 2: Checking if cupy or HIP is available...")
                gpu_lib_available, gpu_lib_working, error_msg, backend_type = self.test_cupy_functionality()
                
                if not gpu_lib_available:
                    update_log(f"  GPU acceleration library not found: {error_msg}")
                    update_log("")
                    update_log("Result: GPU acceleration is NOT available")
                    update_log("  Install options:")
                    update_log("    - For NVIDIA: pip install cupy")
                    update_log("    - For AMD: pip install cupy-rocm")
                    update_log("    - For AMD HIP: pip install pyhip")
                    self.root.after(100, lambda: self._complete_gpu_check(progress_window, False, error_msg))
                    return
                
                if backend_type == "cupy":
                    update_log("  cupy is installed")
                elif backend_type == "HIP":
                    update_log("  HIP (PyHIP) is installed")
                else:
                    update_log("  GPU acceleration library is installed")
                
                if not gpu_lib_working:
                    update_log(f"  GPU library is installed but not working: {error_msg}")
                    update_log("")
                    update_log("Result: GPU acceleration is NOT available")
                    self.root.after(100, lambda: self._complete_gpu_check(progress_window, False, error_msg))
                    return
                
                if backend_type == "cupy":
                    update_log("  cupy is working correctly!")
                elif backend_type == "HIP":
                    update_log("  HIP is working correctly!")
                else:
                    update_log("  GPU acceleration library is working correctly!")
                update_log("")
                
                # Step 3: Get GPU details if available
                try:
                    import cupy as cp  # type: ignore
                    if hasattr(cp, 'cuda') and cp.cuda.is_available():
                        device_count = cp.cuda.runtime.getDeviceCount()
                        update_log(f"  GPU device count: {device_count}")
                        for i in range(device_count):
                            device_props = cp.cuda.runtime.getDeviceProperties(i)
                            device_name = device_props['name'].decode() if isinstance(device_props['name'], bytes) else device_props['name']
                            update_log(f"    Device {i}: {device_name}")
                except Exception as e:
                    update_log(f"  Could not get detailed GPU info: {e}")
                
                update_log("")
                update_log("=" * 60)
                update_log("Result: GPU Acceleration IS AVAILABLE")
                update_log("=" * 60)
                
                # Update config.ini
                self._update_gpu_config(True)
                update_log("")
                update_log("Configuration updated: USE_GPU = True in config.ini")
                
                self.root.after(100, lambda: self._complete_gpu_check(progress_window, True, "GPU acceleration is available and working"))
                
            except Exception as e:
                error_msg = f"Error during GPU detection: {str(e)}"
                update_log("")
                update_log(f"ERROR: {error_msg}")
                import traceback
                update_log(traceback.format_exc())
                self.root.after(100, lambda: self._complete_gpu_check(progress_window, False, error_msg))
        
        # Run detection in a separate thread
        threading.Thread(target=run_detection, daemon=True).start()
    
    def _update_gpu_config(self, enable_gpu):
        """
        Update the GPU setting in config.ini
        """
        try:
            # Get the script directory for the config file path
            if getattr(sys, 'frozen', False):
                script_dir = os.path.dirname(sys.executable)
            else:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                
            config_path = os.path.join(script_dir, 'config.ini')
            
            config = configparser.ConfigParser()
            
            # Read existing config to preserve other sections
            if os.path.exists(config_path):
                config.read(config_path)
            
            # Ensure Processing section exists
            if 'Processing' not in config:
                config['Processing'] = {}
            
            # Update GPU setting (use uppercase to match save_config format)
            config['Processing']['USE_GPU'] = str(enable_gpu)
            
            # Also update the use_gpu variable in the GUI
            self.use_gpu.set(enable_gpu)
            
            # Write to file
            with open(config_path, 'w') as configfile:
                config.write(configfile)
                
        except Exception as e:
            logger.error(f"Error updating GPU config: {e}")
    
    def _complete_gpu_check(self, progress_window, success, message):
        """
        Complete the GPU check process and show results
        """
        # Close the progress window
        progress_window.destroy()
        
        # Show completion message
        if success:
            messagebox.showinfo("GPU Acceleration Available", 
                f"GPU Acceleration is available and working!\n\n{message}\n\n"
                "The GPU setting in config.ini has been set to True.")
        else:
            messagebox.showwarning("GPU Acceleration Not Available", 
                f"GPU Acceleration is not available.\n\n{message}\n\n"
                "The application will use CPU-based processing.")
        
    def is_conda_available(self):
        """Check if conda is available in PATH"""
        try:
            # Check if conda is in PATH
            result = subprocess.run(['conda', '--version'], 
                                  capture_output=True, text=True, check=False)
            return result.returncode == 0
        except:
            return False
    
    def install_gpu_packages(self):
        """Install GPU support packages"""
        try:
            # Create progress window
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Installing GPU Support")
            progress_window.geometry("400x200")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            # Add status label
            status_label = ttk.Label(progress_window, text="Installing GPU support packages...")
            status_label.pack(pady=10)
            
            # Add progress bar
            progress = ttk.Progressbar(progress_window, mode="indeterminate")
            progress.pack(fill="x", padx=20, pady=10)
            progress.start()
            
            # Add log text area
            log_text = scrolledtext.ScrolledText(progress_window, height=5)
            log_text.pack(fill="both", expand=True, padx=20, pady=10)
            
            def update_log(message):
                log_text.insert(tk.END, message + "\n")
                log_text.see(tk.END)
                log_text.update()
            
            # Define installation function to run in thread
            def run_installation():
                installation_success = False
                conda_available = self.is_conda_available()
                
                # Try conda installation first if available
                if conda_available:
                    update_log("Conda detected - attempting to install RAPIDS using conda...")
                    try:
                        # Try to install RAPIDS using conda
                        cmd = ["conda", "install", "-y", "-c", "rapidsai", "-c", "conda-forge", "-c", "nvidia", 
                               "cudf", "cuml", "cupy"]
                        process = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if process.returncode == 0:
                            installation_success = True
                            update_log("Successfully installed RAPIDS using conda!")
                        else:
                            update_log(f"Failed to install with conda: {process.stderr}")
                            update_log("Falling back to pip installation...")
                    except Exception as e:
                        update_log(f"Error during conda installation: {str(e)}")
                        update_log("Falling back to pip installation...")
                else:
                    update_log("Conda not found - using pip for installation")
                
                # Fall back to pip with lower version requirements if conda failed or is not available
                if not installation_success:
                    update_log("Attempting to install RAPIDS using pip with lower version requirements...")
                    
                    # Packages to install with lower version requirements
                    gpu_packages = ["cudf<1.0", "cupy<8.0", "cuml<1.0"]
                    pip_success = True
                    
                    for package in gpu_packages:
                        update_log(f"Installing {package}...")
                        try:
                            # Use pip to install the package
                            process = subprocess.run(
                                [sys.executable, "-m", "pip", "install", package],
                                capture_output=True,
                                text=True
                            )
                            
                            if process.returncode != 0:
                                pip_success = False
                                update_log(f"Failed to install {package}: {process.stderr}")
                            else:
                                update_log(f"{package} installed successfully.")
                        except Exception as e:
                            pip_success = False
                            update_log(f"Error installing {package}: {str(e)}")
                    
                    installation_success = pip_success
                
                # Complete installation
                if installation_success:
                    update_log("\nAll GPU packages installed successfully!")
                else:
                    update_log("\nSome packages could not be installed. Check output for details.")
                
                # Update UI back on the main thread
                self.root.after(100, lambda: self._complete_installation(progress_window, installation_success))
            
            # Start installation in a separate thread
            threading.Thread(target=run_installation, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to install GPU packages: {str(e)}")
    
    def _complete_installation(self, progress_window, success):
        """Complete the GPU package installation process"""
        # Close the progress window
        progress_window.destroy()
        
        # Show completion message
        if success:
            messagebox.showinfo("Success", "GPU support packages installed successfully! GPU acceleration is now enabled.")
        else:
            messagebox.showwarning("Warning", 
                "Some GPU packages could not be installed. The application will still run without GPU acceleration.\n\n"
                "You can continue using the application with CPU-only processing.")
        
        # Check GPU support status again
        self.check_gpu_support()
    
    def toggle_data_source(self):
        """Toggle between NOAA data, local data folder, and AWS S3 data bucket"""
        if not aws_available and self.data_source.get() == "s3":
            messagebox.showerror("Error", "AWS boto3 library not installed. Install with: pip install boto3")
            self.data_source.set("noaa")  # Default to NOAA if S3 not available
            return
        
        data_source = self.data_source.get()
        
        # Disable all S3 settings first
        if hasattr(self, 'bucket_name_entry'):
            self.bucket_name_entry.configure(state="disabled")
            self.prefix_entry.configure(state="disabled")
            self.access_key_entry.configure(state="disabled")
            self.secret_key_entry.configure(state="disabled")
            self.token_entry.configure(state="disabled")
            self.test_s3_button.configure(state="disabled")
        self.use_s3.set(False)
        
        if data_source == "s3":
            # Using AWS S3
            if hasattr(self, 'bucket_name_entry'):
                self.bucket_name_entry.configure(state="normal")
                self.prefix_entry.configure(state="normal")
                self.access_key_entry.configure(state="normal")
                self.secret_key_entry.configure(state="normal")
                self.token_entry.configure(state="normal")
                self.test_s3_button.configure(state="normal")
            self.use_s3.set(True)
            
    def get_noaa_data_url(self, year=None):
        """Generate URL for NOAA AIS data based on year"""
        if year is None:
            year = self.noaa_year.get()
        
        # Handle year ranges (e.g. "2023-2024")
        if '-' in str(year):
            # For a range, use the first year (DataManager.download_noaa_data will handle multiple years)
            # This is just for getting the base URL
            try:
                start_year = year.split('-')[0].strip()
                base_url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{start_year}/"
                return base_url
            except (ValueError, IndexError):
                messagebox.showerror("Error", f"Invalid year range format: {year}. Expected format like '2023-2024'.")
                return None
        else:
            # Handle single year
            try:
                year_int = int(year)
                base_url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year_int}/"
                return base_url
            except ValueError:
                messagebox.showerror("Error", f"Invalid year: {year}. Please enter a valid year.")
                return None
    
    def toggle_s3_settings(self):
        """Enable/disable S3 settings based on use_s3 checkbox"""
        if not aws_available:
            # Disable all S3 settings if boto3 is not available
            if hasattr(self, 's3_uri_entry'):
                self.s3_uri_entry.configure(state="disabled")
            if hasattr(self, 'test_s3_button'):
                self.test_s3_button.configure(state="disabled")
            # Keys authentication is the only method supported
            return
            
        if self.use_s3.get():
            # Enable S3 settings
            if hasattr(self, 's3_uri_entry'):
                self.s3_uri_entry.configure(state="normal")
            if hasattr(self, 'test_s3_button'):
                self.test_s3_button.configure(state="normal")
            # Keys authentication is the only method supported
            # Update authentication settings based on selected method
            self.toggle_auth_method()
            # Toggle advanced authentication if needed
            self.toggle_advanced_auth()
            
            # Set data source to S3 if available
            if hasattr(self, 'data_source'):
                self.data_source.set("s3")
                
            # Configuration will be saved when 'Save Configuration' or 'Run Analysis' is clicked
                
        else:
            # Disable S3 settings
            if hasattr(self, 's3_uri_entry'):
                self.s3_uri_entry.configure(state="disabled")
            if hasattr(self, 'test_s3_button'):
                self.test_s3_button.configure(state="disabled")
            # Disable all authentication settings
            if hasattr(self, 'access_key_entry'):
                self.access_key_entry.configure(state="disabled")
                self.secret_key_entry.configure(state="disabled")
            if hasattr(self, 'token_entry'):
                self.token_entry.configure(state="disabled")
            if hasattr(self, 'bucket_name_entry'):
                self.bucket_name_entry.configure(state="disabled")
                self.prefix_entry.configure(state="disabled")
            if hasattr(self, 'region_entry'):
                self.region_entry.configure(state="disabled")
            # Only keys authentication is supported
                
            # Set data source to local
            if hasattr(self, 'data_source'):
                self.data_source.set("local")
                
            # Configuration will be saved when 'Save Configuration' or 'Run Analysis' is clicked
                
    def toggle_advanced_auth(self):
        """Show or hide advanced authentication options"""
        if not hasattr(self, 'show_advanced_auth') or not hasattr(self, 'advanced_auth_frame'):
            return
            
        if self.show_advanced_auth.get():
            # Show advanced authentication frame using pack instead of grid
            self.advanced_auth_frame.pack(fill=tk.X, padx=5, pady=5, after=self.show_advanced_auth.master)
            self.toggle_auth_method()
        else:
            # Hide advanced authentication frame
            self.advanced_auth_frame.pack_forget()
    
    def toggle_auth_method(self, *args):
        """Enable/disable authentication settings based on selected auth method - only keys supported"""
        if not aws_available or not self.use_s3.get() or not hasattr(self, 'show_advanced_auth'):
            return
            
        # If advanced auth is not shown, return
        if not self.show_advanced_auth.get():
            return
            
        # Only keys auth method is supported now
        # Just enable the region entry
        if hasattr(self, 'region_entry'):
            self.region_entry.configure(state="normal")
            
    def validate_session_token(self, session_token):
        """Validate the format of a session token"""
        # Basic checks for AWS session token format
        # This doesn't verify the token is valid with AWS, just that it looks like a valid format
        if not session_token:
            return True  # Empty token is technically valid (means don't use a token)
            
        # Most AWS session tokens follow this pattern
        if len(session_token) < 100:
            return False  # Too short for a session token
        
        # Check for invalid characters
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n")
        return all(c in valid_chars for c in session_token)
    
    def test_s3_connection(self):
        """Test the connection to the S3 bucket"""
        if not aws_available:
            messagebox.showerror("Error", "AWS boto3 library not installed. Install with: pip install boto3")
            return
        
        s3_uri = self.s3_data_uri.get()
        if not s3_uri.startswith('s3://'):
            messagebox.showerror("Error", "Invalid S3 URI format. It should start with 's3://'") 
            return
        
        try:
            # Parse the S3 URI to extract bucket name and prefix
            parts = s3_uri.replace("s3://", "").split("/")
            bucket_name = parts[0]
            prefix = "/".join(parts[1:]) if len(parts) > 1 else ""
            
            # Only keys authentication is supported
            access_key = self.s3_access_key.get()
            secret_key = self.s3_secret_key.get()
            session_token = self.s3_session_token.get()
            region = self.s3_region.get()
            s3_client = None
            
            if not access_key or not secret_key:
                messagebox.showerror("Error", "Access key and secret key cannot be empty")
                return
                
            # Validate session token if provided
            if session_token and not self.validate_session_token(session_token):
                messagebox.showerror("Error", "Session token appears to be invalid. Please check the format.")
                return
                
            if session_token:
                messagebox.showinfo("Auth Method", f"Using AWS access keys with session token in region {region}")
                try:
                    s3_client = boto3.client('s3', 
                                           aws_access_key_id=access_key, 
                                           aws_secret_access_key=secret_key, 
                                           aws_session_token=session_token,
                                           region_name=region)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to create S3 client with session token: {str(e)}\n\nYour session token may be invalid or expired.")
                    return
            else:
                messagebox.showinfo("Auth Method", f"Using AWS access keys in region {region}")
                try:
                    s3_client = boto3.client('s3', 
                                           aws_access_key_id=access_key, 
                                           aws_secret_access_key=secret_key, 
                                           region_name=region)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to create S3 client: {str(e)}")
                    return
            
            # Check if we got a valid S3 client
            if s3_client is None:
                messagebox.showerror("Error", "Failed to create S3 client. Check your authentication details.")
                return
                
            # Test the connection by listing objects
            try:
                response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=1)
                if prefix:
                    messagebox.showinfo("Success", f"Successfully connected to S3 bucket '{bucket_name}' with prefix '{prefix}'")
                else:
                    messagebox.showinfo("Success", f"Successfully connected to S3 bucket '{bucket_name}'")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to list objects in bucket: {str(e)}")
                return
            
        except botocore.exceptions.NoCredentialsError:
            messagebox.showerror("Error", "AWS credentials not found. Please configure your AWS credentials.")
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error while testing S3 connection: {str(e)}")
            logger.error(traceback.format_exc())
    
    def notebook_tab_changed(self, event=None):
        """Handle tab changes in the notebook"""
        try:
            # Get the current tab index
            current_tab = self.notebook.select()
            if current_tab:
                tab_idx = self.notebook.index(current_tab)
                
                # If switching to the Data tab, update the NOAA year
                if tab_idx == 4:  # Data tab is at index 4 (0-indexed)
                    # Update NOAA year from date range in Startup tab
                    self.noaa_year.set(self.get_year_from_date_range())
        except Exception as e:
            logger.error(f"Error in notebook_tab_changed: {e}")
    
    def update_date_values(self):
        """Update date string variables from the date picker widgets"""
        if tkcalendar_available and hasattr(self, 'start_date_picker') and hasattr(self, 'end_date_picker'):
            # Using DateEntry widgets
            self.start_date.set(self.start_date_picker.get_date().strftime('%Y-%m-%d'))
            self.end_date.set(self.end_date_picker.get_date().strftime('%Y-%m-%d'))
        elif hasattr(self, 'start_date_entry') and hasattr(self, 'end_date_entry'):
            # Using basic Entry widgets
            self.start_date.set(self.start_date_entry.get())
            self.end_date.set(self.end_date_entry.get())
            
        # Also update the NOAA year when dates change
        self.noaa_year.set(self.get_year_from_date_range())
                    
    def create_bottom_buttons(self):
        """Create buttons only on the Startup tab below the time range selection"""
        # We'll create these buttons directly on the Startup tab instead of at the bottom of the window
        pass
    
    def load_config(self):
        """Load configuration from config.ini if it exists"""
        paths_synced = False  # Track if we synced [Paths] with [DEFAULT]
        try:
            # Get the script directory for the config file path
            if getattr(sys, 'frozen', False):
                script_dir = os.path.dirname(sys.executable)
            else:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                
            config_path = os.path.join(script_dir, 'config.ini')
            
            if not os.path.exists(config_path):
                return
                
            config = configparser.ConfigParser()
            config.read(config_path)
            
            # Load general settings
            if 'DEFAULT' in config:
                if 'DATA_DIRECTORY' in config['DEFAULT']:
                    self.data_directory.set(config['DEFAULT']['DATA_DIRECTORY'])
                if 'OUTPUT_DIRECTORY' in config['DEFAULT']:
                    self.output_directory.set(config['DEFAULT']['OUTPUT_DIRECTORY'])
                    # Sync [Paths] section with [DEFAULT] output_directory
                    if 'Paths' not in config:
                        config['Paths'] = {}
                        paths_synced = True
                    if config['Paths'].get('OUTPUT_DIRECTORY') != config['DEFAULT']['OUTPUT_DIRECTORY']:
                        config['Paths']['OUTPUT_DIRECTORY'] = config['DEFAULT']['OUTPUT_DIRECTORY']
                        paths_synced = True
                    if 'DATA_DIRECTORY' in config['DEFAULT']:
                        if config['Paths'].get('DATA_DIRECTORY') != config['DEFAULT']['DATA_DIRECTORY']:
                            config['Paths']['DATA_DIRECTORY'] = config['DEFAULT']['DATA_DIRECTORY']
                            paths_synced = True
                
                # Load date range settings
                if 'START_DATE' in config['DEFAULT'] and config['DEFAULT']['START_DATE']:
                    self.start_date.set(config['DEFAULT']['START_DATE'])
                    try:
                        if tkcalendar_available and hasattr(self, 'start_date_picker'):
                            self.start_date_picker.set_date(datetime.strptime(config['DEFAULT']['START_DATE'], '%Y-%m-%d').date())
                    except (ValueError, AttributeError):
                        pass  # If date is invalid, keep the default
                if 'END_DATE' in config['DEFAULT'] and config['DEFAULT']['END_DATE']:
                    self.end_date.set(config['DEFAULT']['END_DATE'])
                    try:
                        if tkcalendar_available and hasattr(self, 'end_date_picker'):
                            self.end_date_picker.set_date(datetime.strptime(config['DEFAULT']['END_DATE'], '%Y-%m-%d').date())
                    except (ValueError, AttributeError):
                        pass  # If date is invalid, keep the default
            
            # Load ship filters
            if 'SHIP_FILTERS' in config:
                if 'SELECTED_SHIP_TYPES' in config['SHIP_FILTERS']:
                    selected_types = config['SHIP_FILTERS']['SELECTED_SHIP_TYPES'].split(',')
                    # Reset all to false first
                    for ship_type in self.ship_types:
                        self.ship_types[ship_type]['var'].set(False)
                    # Then set selected ones to true
                    for type_str in selected_types:
                        try:
                            ship_type = int(type_str.strip())
                            if ship_type in self.ship_types:
                                self.ship_types[ship_type]['var'].set(True)
                        except (ValueError, KeyError):
                            pass
            
            # Load anomaly thresholds
            if 'ANOMALY_THRESHOLDS' in config:
                if 'MIN_TRAVEL_NM' in config['ANOMALY_THRESHOLDS']:
                    self.min_travel_nm.set(float(config['ANOMALY_THRESHOLDS']['MIN_TRAVEL_NM']))
                if 'MAX_TRAVEL_NM' in config['ANOMALY_THRESHOLDS']:
                    self.max_travel_nm.set(float(config['ANOMALY_THRESHOLDS']['MAX_TRAVEL_NM']))
                if 'COG_HEADING_MAX_DIFF' in config['ANOMALY_THRESHOLDS']:
                    self.cog_heading_max_diff.set(float(config['ANOMALY_THRESHOLDS']['COG_HEADING_MAX_DIFF']))
                if 'MIN_SPEED_FOR_COG_CHECK' in config['ANOMALY_THRESHOLDS']:
                    self.min_speed_for_cog_check.set(float(config['ANOMALY_THRESHOLDS']['MIN_SPEED_FOR_COG_CHECK']))
            
            # Load anomaly types to detect
            if 'ANOMALY_TYPES' in config:
                for anomaly_type in self.anomaly_types:
                    # Map GUI names to config keys
                    if anomaly_type == "Course over Ground-Heading Inconsistency":
                        key = "cog-heading_inconsistency"
                    elif anomaly_type == "Excessive Travel Distance (Fast)":
                        key = "excessive_travel_distance_fast"
                    elif anomaly_type == "Excessive Travel Distance (Slow)":
                        key = "excessive_travel_distance_slow"
                    elif anomaly_type == "AIS Beacon Off":
                        key = "ais_beacon_off"
                    elif anomaly_type == "AIS Beacon On":
                        key = "ais_beacon_on"
                    elif anomaly_type == "Loitering":
                        key = "loitering"
                    elif anomaly_type == "Rendezvous - **This dramatically increases processing time":
                        key = "rendezvous"
                    elif anomaly_type == "Identity Spoofing":
                        key = "identity_spoofing"
                    elif anomaly_type == "Zone Violations":
                        key = "zone_violations"
                    else:
                        key = anomaly_type.replace(' ', '_').replace('(', '').replace(')', '').lower()
                    if key in config['ANOMALY_TYPES']:
                        self.anomaly_types[anomaly_type].set(config['ANOMALY_TYPES'].getboolean(key))
                        
            # Load output controls settings
            if 'OUTPUT_CONTROLS' in config:
                for control_key in self.output_controls:
                    if control_key in config['OUTPUT_CONTROLS']:
                        self.output_controls[control_key].set(config['OUTPUT_CONTROLS'].getboolean(control_key))
                        
            # Load analysis filters
            if 'ANALYSIS_FILTERS' in config:
                # Only try to load existing filter variables
                # Skip the removed anomaly filter toggles that might still be in the config file
                for filter_key, var in self.analysis_filters.items():
                    if filter_key in config['ANALYSIS_FILTERS']:
                        # Handle different variable types appropriately
                        if isinstance(var, tk.DoubleVar):
                            try:
                                var.set(float(config['ANALYSIS_FILTERS'][filter_key]))
                            except ValueError:
                                pass  # Use default if value cannot be converted
                        elif isinstance(var, tk.IntVar):
                            try:
                                var.set(int(config['ANALYSIS_FILTERS'][filter_key]))
                            except ValueError:
                                pass  # Use default if value cannot be converted
                        else:  # StringVar or other
                            var.set(config['ANALYSIS_FILTERS'][filter_key])
            
            # Load zone violations
            if 'ZONE_VIOLATIONS' in config:
                self.zone_violations = []
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
                for i in sorted(zone_indices):
                    zone_key = f'zone_{i}'
                    try:
                        zone = {
                            'name': config['ZONE_VIOLATIONS'][f'{zone_key}_name'],
                            'lat_min': float(config['ZONE_VIOLATIONS'].get(f'{zone_key}_lat_min', '0')),
                            'lat_max': float(config['ZONE_VIOLATIONS'].get(f'{zone_key}_lat_max', '0')),
                            'lon_min': float(config['ZONE_VIOLATIONS'].get(f'{zone_key}_lon_min', '0')),
                            'lon_max': float(config['ZONE_VIOLATIONS'].get(f'{zone_key}_lon_max', '0')),
                            'is_selected': config['ZONE_VIOLATIONS'].getboolean(f'{zone_key}_is_selected', True)
                        }
                        self.zone_violations.append(zone)
                    except (ValueError, KeyError, configparser.NoOptionError):
                        continue
                
                # Refresh zone list display
                if hasattr(self, 'zone_list_container'):
                    self._refresh_zone_list()
            else:
                # Initialize with default zones if none exist
                if not self.zone_violations:
                    self.zone_violations = [
                        {'name': 'Strait of Hormuz', 'lat_min': 25.0, 'lat_max': 27.0, 'lon_min': 55.0, 'lon_max': 57.5, 'is_selected': True},
                        {'name': 'South China Sea', 'lat_min': 5.0, 'lat_max': 25.0, 'lon_min': 105.0, 'lon_max': 120.0, 'is_selected': True}
                    ]
            
            # Refresh zone list display if container exists
            if hasattr(self, 'zone_list_container'):
                self._refresh_zone_list()
            
            # Load Data Source settings
            if 'DATA_SOURCE' in config:
                if 'source' in config['DATA_SOURCE']:
                    self.data_source.set(config['DATA_SOURCE']['source'])
                if 'noaa_year' in config['DATA_SOURCE']:
                    self.noaa_year.set(config['DATA_SOURCE']['noaa_year'])
                    
            # Load AWS settings
            if 'AWS' in config:
                # Use getboolean for boolean values
                if 'use_s3' in config['AWS']:
                    try:
                        self.use_s3.set(config['AWS'].getboolean('use_s3'))
                    except ValueError:
                        pass
                        
                # String values
                if 's3_data_uri' in config['AWS']:
                    self.s3_data_uri.set(config['AWS']['s3_data_uri'])
                if 's3_auth_method' in config['AWS']:
                    self.s3_auth_method.set(config['AWS']['s3_auth_method'])
                if 's3_profile_name' in config['AWS']:
                    self.s3_profile_name.set(config['AWS']['s3_profile_name'])
                if 's3_access_key' in config['AWS']:
                    self.s3_access_key.set(config['AWS']['s3_access_key'])
                if 's3_secret_key' in config['AWS']:
                    self.s3_secret_key.set(config['AWS']['s3_secret_key'])
                if 's3_session_token' in config['AWS']:
                    self.s3_session_token.set(config['AWS']['s3_session_token'])
                if 's3_region' in config['AWS']:
                    self.s3_region.set(config['AWS']['s3_region'])
                if 's3_role_arn' in config['AWS']:
                    self.s3_role_arn.set(config['AWS']['s3_role_arn'])
                if 's3_bucket_name' in config['AWS']:
                    self.s3_bucket_name.set(config['AWS']['s3_bucket_name'])
                if 's3_prefix' in config['AWS']:
                    self.s3_prefix.set(config['AWS']['s3_prefix'])
                if 's3_local_dir' in config['AWS']:
                    self.s3_local_dir.set(config['AWS']['s3_local_dir'])
            
            # Write the config back to file if we synced [Paths] section with [DEFAULT]
            # This ensures [Paths] output_directory always matches [DEFAULT] output_directory
            if paths_synced:
                with open(config_path, 'w') as configfile:
                    config.write(configfile)
        
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    def save_config(self, show_message=True):
        """Save configuration to config.ini
        
        Args:
            show_message (bool): Whether to show a message box when configuration is saved. 
                                 Default is True. Set to False when saving automatically before analysis.
        """
        try:
            # Get the script directory for the config file path
            if getattr(sys, 'frozen', False):
                script_dir = os.path.dirname(sys.executable)
            else:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                
            config_path = os.path.join(script_dir, 'config.ini')
            
            config = configparser.ConfigParser()
            
            # Read existing config to preserve other sections
            if os.path.exists(config_path):
                config.read(config_path)
            
            if 'DEFAULT' not in config:
                config['DEFAULT'] = {}
            
            # Update with current values
            # Safely get directory values, handling cases where they might not be initialized
            try:
                if hasattr(self, 'data_directory'):
                    data_dir = self.data_directory.get()
                else:
                    data_dir = ''
            except (AttributeError, tk.TclError, RuntimeError):
                data_dir = ''
            
            try:
                if hasattr(self, 'output_directory'):
                    output_dir = self.output_directory.get()
                else:
                    output_dir = 'C:/Users/chris/OneDrive/Desktop/MVP/Output'
            except (AttributeError, tk.TclError, RuntimeError):
                output_dir = 'C:/Users/chris/OneDrive/Desktop/MVP/Output'
            
            config['DEFAULT']['DATA_DIRECTORY'] = data_dir
            config['DEFAULT']['OUTPUT_DIRECTORY'] = output_dir
            
            # Update [Paths] section to match [DEFAULT] output_directory
            if 'Paths' not in config:
                config['Paths'] = {}
            config['Paths']['OUTPUT_DIRECTORY'] = output_dir
            config['Paths']['DATA_DIRECTORY'] = data_dir
            
            # Get date values based on available widgets
            try:
                if tkcalendar_available and hasattr(self, 'start_date_picker'):
                    self.start_date.set(self.start_date_picker.get_date().strftime('%Y-%m-%d'))
                    self.end_date.set(self.end_date_picker.get_date().strftime('%Y-%m-%d'))
                
                start_date_val = self.start_date.get() if hasattr(self, 'start_date') else '2024-10-15'
                end_date_val = self.end_date.get() if hasattr(self, 'end_date') else '2024-10-20'
            except (AttributeError, tk.TclError):
                start_date_val = '2024-10-15'
                end_date_val = '2024-10-20'
            
            config['DEFAULT']['START_DATE'] = start_date_val
            config['DEFAULT']['END_DATE'] = end_date_val
            
            # Ship filters - get selected ship types
            try:
                if hasattr(self, 'ship_types'):
                    selected_types = [str(ship_type) for ship_type in self.ship_types if self.ship_types[ship_type]['var'].get()]
                else:
                    selected_types = []
            except (AttributeError, tk.TclError, KeyError):
                selected_types = []
            
            if 'SHIP_FILTERS' not in config:
                config['SHIP_FILTERS'] = {}
            config['SHIP_FILTERS']['SELECTED_SHIP_TYPES'] = ','.join(selected_types)
            
            # Anomaly thresholds
            if 'ANOMALY_THRESHOLDS' not in config:
                config['ANOMALY_THRESHOLDS'] = {}
            
            try:
                config['ANOMALY_THRESHOLDS']['MIN_TRAVEL_NM'] = str(self.min_travel_nm.get()) if hasattr(self, 'min_travel_nm') else '200.0'
                config['ANOMALY_THRESHOLDS']['MAX_TRAVEL_NM'] = str(self.max_travel_nm.get()) if hasattr(self, 'max_travel_nm') else '550.0'
                config['ANOMALY_THRESHOLDS']['COG_HEADING_MAX_DIFF'] = str(self.cog_heading_max_diff.get()) if hasattr(self, 'cog_heading_max_diff') else '45.0'
                config['ANOMALY_THRESHOLDS']['MIN_SPEED_FOR_COG_CHECK'] = str(self.min_speed_for_cog_check.get()) if hasattr(self, 'min_speed_for_cog_check') else '10.0'
            except (AttributeError, tk.TclError):
                config['ANOMALY_THRESHOLDS']['MIN_TRAVEL_NM'] = '200.0'
                config['ANOMALY_THRESHOLDS']['MAX_TRAVEL_NM'] = '550.0'
                config['ANOMALY_THRESHOLDS']['COG_HEADING_MAX_DIFF'] = '45.0'
                config['ANOMALY_THRESHOLDS']['MIN_SPEED_FOR_COG_CHECK'] = '10.0'
            
            # Anomaly types to detect
            if 'ANOMALY_TYPES' not in config:
                config['ANOMALY_TYPES'] = {}
            
            try:
                if hasattr(self, 'anomaly_types'):
                    for anomaly_type, var in self.anomaly_types.items():
                        # Map GUI names to config keys
                        if anomaly_type == "Course over Ground-Heading Inconsistency":
                            key = "cog-heading_inconsistency"
                        elif anomaly_type == "Excessive Travel Distance (Fast)":
                            key = "excessive_travel_distance_fast"
                        elif anomaly_type == "Excessive Travel Distance (Slow)":
                            key = "excessive_travel_distance_slow"
                        elif anomaly_type == "AIS Beacon Off":
                            key = "ais_beacon_off"
                        elif anomaly_type == "AIS Beacon On":
                            key = "ais_beacon_on"
                        elif anomaly_type == "Loitering":
                            key = "loitering"
                        elif anomaly_type == "Rendezvous - **This dramatically increases processing time":
                            key = "rendezvous"
                        elif anomaly_type == "Rendezvous":
                            key = "rendezvous"
                        elif anomaly_type == "Identity Spoofing":
                            key = "identity_spoofing"
                        elif anomaly_type == "Zone Violations":
                            key = "zone_violations"
                        else:
                            key = anomaly_type.replace(' ', '_').replace('(', '').replace(')', '').lower()
                        try:
                            config['ANOMALY_TYPES'][key] = str(var.get())
                        except (AttributeError, tk.TclError):
                            config['ANOMALY_TYPES'][key] = 'True'
            except (AttributeError, KeyError):
                pass
            
            # Output Controls settings
            if 'OUTPUT_CONTROLS' not in config:
                config['OUTPUT_CONTROLS'] = {}
            
            try:
                if hasattr(self, 'output_controls'):
                    for control_key, var in self.output_controls.items():
                        try:
                            config['OUTPUT_CONTROLS'][control_key] = str(var.get())
                        except (AttributeError, tk.TclError):
                            config['OUTPUT_CONTROLS'][control_key] = 'True'
            except (AttributeError, KeyError):
                pass
                
            # Analysis Filters settings
            if 'ANALYSIS_FILTERS' not in config:
                config['ANALYSIS_FILTERS'] = {}
            
            # Save only the existing analysis filter variables
            try:
                if hasattr(self, 'analysis_filters'):
                    for filter_key, var in self.analysis_filters.items():
                        try:
                            config['ANALYSIS_FILTERS'][filter_key] = str(var.get())
                        except (AttributeError, tk.TclError):
                            pass
            except (AttributeError, KeyError):
                pass
                
            # Processing Options
            if 'Processing' not in config:
                config['Processing'] = {}
                
            # Save processing options
            try:
                config['Processing']['USE_GPU'] = str(self.use_gpu.get()) if hasattr(self, 'use_gpu') else 'True'
                config['Processing']['USE_DASK'] = str(self.use_dask.get()) if hasattr(self, 'use_dask') else 'True'
            except (AttributeError, tk.TclError):
                config['Processing']['USE_GPU'] = 'True'
                config['Processing']['USE_DASK'] = 'True'
                
            # Data Source settings
            if 'DATA_SOURCE' not in config:
                config['DATA_SOURCE'] = {}
                
            # Save data source selection
            try:
                config['DATA_SOURCE']['source'] = self.data_source.get() if hasattr(self, 'data_source') else 'noaa'
                config['DATA_SOURCE']['noaa_year'] = self.noaa_year.get() if hasattr(self, 'noaa_year') else '2024'
            except (AttributeError, tk.TclError):
                config['DATA_SOURCE']['source'] = 'noaa'
                config['DATA_SOURCE']['noaa_year'] = '2024'
                
            # AWS S3 settings
            if 'AWS' not in config:
                config['AWS'] = {}
            
            # Save AWS credentials and settings
            try:
                config['AWS']['use_s3'] = str(self.use_s3.get()) if hasattr(self, 'use_s3') else 'False'
                config['AWS']['s3_data_uri'] = self.s3_data_uri.get() if hasattr(self, 's3_data_uri') else ''
                # Keys authentication is the only supported method
                config['AWS']['s3_auth_method'] = "keys"
                config['AWS']['s3_access_key'] = self.s3_access_key.get() if hasattr(self, 's3_access_key') else ''
                config['AWS']['s3_secret_key'] = self.s3_secret_key.get() if hasattr(self, 's3_secret_key') else ''
                
                # Handle session token - very long tokens may cause issues with the config file
                session_token = self.s3_session_token.get() if hasattr(self, 's3_session_token') else ''
                if len(session_token) > 1000:
                    logger.warning(f"Session token is very long ({len(session_token)} chars). Storing in config file.")
                config['AWS']['s3_session_token'] = session_token
                config['AWS']['s3_region'] = self.s3_region.get() if hasattr(self, 's3_region') else 'us-east-1'
                config['AWS']['s3_bucket_name'] = self.s3_bucket_name.get() if hasattr(self, 's3_bucket_name') else ''
                config['AWS']['s3_prefix'] = self.s3_prefix.get() if hasattr(self, 's3_prefix') else ''
                config['AWS']['s3_local_dir'] = self.s3_local_dir.get() if hasattr(self, 's3_local_dir') else ''
            except (AttributeError, tk.TclError) as e:
                # If AWS variables aren't initialized, use defaults
                config['AWS']['use_s3'] = 'False'
                config['AWS']['s3_data_uri'] = ''
                config['AWS']['s3_auth_method'] = "keys"
                config['AWS']['s3_access_key'] = ''
                config['AWS']['s3_secret_key'] = ''
                config['AWS']['s3_session_token'] = ''
                config['AWS']['s3_region'] = 'us-east-1'
                config['AWS']['s3_bucket_name'] = ''
                config['AWS']['s3_prefix'] = ''
                config['AWS']['s3_local_dir'] = ''
            
            # Zone Violations
            if 'ZONE_VIOLATIONS' not in config:
                config['ZONE_VIOLATIONS'] = {}
            
            # Get existing zone indices from config
            existing_zone_indices = set()
            for key in list(config['ZONE_VIOLATIONS'].keys()):
                # Only process keys that match the zone pattern (zone_N_*)
                if key.startswith('zone_') and '_name' in key:
                    try:
                        # Extract zone index from key like "zone_0_name"
                        zone_index = int(key.split('_')[1])
                        existing_zone_indices.add(zone_index)
                    except (ValueError, IndexError):
                        pass  # Skip invalid keys
            
            # Save/update zones
            current_zone_indices = set()
            for i, zone in enumerate(self.zone_violations):
                zone_key = f'zone_{i}'
                current_zone_indices.add(i)
                
                # Update or create zone keys
                config['ZONE_VIOLATIONS'][f'{zone_key}_name'] = zone['name']
                config['ZONE_VIOLATIONS'][f'{zone_key}_lat_min'] = str(zone['lat_min'])
                config['ZONE_VIOLATIONS'][f'{zone_key}_lat_max'] = str(zone['lat_max'])
                config['ZONE_VIOLATIONS'][f'{zone_key}_lon_min'] = str(zone['lon_min'])
                config['ZONE_VIOLATIONS'][f'{zone_key}_lon_max'] = str(zone['lon_max'])
                config['ZONE_VIOLATIONS'][f'{zone_key}_is_selected'] = str(zone.get('is_selected', True))
            
            # Only delete keys for zones that no longer exist (removed zones)
            zones_to_remove = existing_zone_indices - current_zone_indices
            for zone_index in zones_to_remove:
                zone_key = f'zone_{zone_index}'
                # Delete all keys for this zone
                keys_to_delete = [
                    f'{zone_key}_name', f'{zone_key}_geometry_type', f'{zone_key}_is_selected',
                    f'{zone_key}_lat_min', f'{zone_key}_lat_max', f'{zone_key}_lon_min', f'{zone_key}_lon_max',
                    f'{zone_key}_center_lat', f'{zone_key}_center_lon', f'{zone_key}_radius_meters',
                    f'{zone_key}_coordinates', f'{zone_key}_tolerance_meters'
                ]
                for key in keys_to_delete:
                    try:
                        del config['ZONE_VIOLATIONS'][key]
                    except KeyError:
                        pass  # Key doesn't exist, skip it

            # Write to file
            try:
                # Ensure the directory exists
                config_dir = os.path.dirname(config_path)
                if config_dir and not os.path.exists(config_dir):
                    os.makedirs(config_dir, exist_ok=True)
                
                # Write config with error handling
                with open(config_path, 'w', encoding='utf-8') as configfile:
                    config.write(configfile)
            except IOError as io_err:
                error_msg = f"IO Error writing config file: {str(io_err)}"
                try:
                    logger = logging.getLogger("SDF_GUI")
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                except:
                    print(error_msg)
                    print(traceback.format_exc())
                messagebox.showerror("Error", f"Failed to save configuration: {error_msg}")
                return
            except Exception as write_err:
                error_msg = f"Error writing config file: {str(write_err)}"
                try:
                    logger = logging.getLogger("SDF_GUI")
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                except:
                    print(error_msg)
                    print(traceback.format_exc())
                messagebox.showerror("Error", f"Failed to save configuration: {error_msg}")
                return
                
            # Only show message if explicitly requested (e.g., from Save Configuration button)
            if show_message:
                messagebox.showinfo("Configuration Saved", f"Configuration has been saved to {config_path}")
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            full_error = f"{error_type}: {error_msg}"
            try:
                logger = logging.getLogger("SDF_GUI")
                logger.error(f"Error saving configuration: {full_error}")
                logger.error(traceback.format_exc())
            except:
                # If logger isn't available, just print to console
                print(f"Error saving configuration: {full_error}")
                print(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to save configuration: {full_error}")
    
    def _save_zones_only(self):
        """Save zone violations and essential config values to config.ini"""
        try:
            # Get the script directory for the config file path
            if getattr(sys, 'frozen', False):
                script_dir = os.path.dirname(sys.executable)
            else:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                
            config_path = os.path.join(script_dir, 'config.ini')
            
            config = configparser.ConfigParser()
            
            # Read existing config to preserve other sections
            if os.path.exists(config_path):
                config.read(config_path)
            
            # Ensure DEFAULT section exists
            if 'DEFAULT' not in config:
                config['DEFAULT'] = {}
            
            # Save data_directory and output_directory if available
            try:
                if hasattr(self, 'data_directory'):
                    data_dir = self.data_directory.get()
                    config['DEFAULT']['DATA_DIRECTORY'] = data_dir
                    if 'Paths' not in config:
                        config['Paths'] = {}
                    config['Paths']['DATA_DIRECTORY'] = data_dir
            except (AttributeError, tk.TclError, RuntimeError):
                pass  # Keep existing value if not available
            
            try:
                if hasattr(self, 'output_directory'):
                    output_dir = self.output_directory.get()
                    config['DEFAULT']['OUTPUT_DIRECTORY'] = output_dir
                    if 'Paths' not in config:
                        config['Paths'] = {}
                    config['Paths']['OUTPUT_DIRECTORY'] = output_dir
            except (AttributeError, tk.TclError, RuntimeError):
                pass  # Keep existing value if not available
            
            # Zone Violations
            if 'ZONE_VIOLATIONS' not in config:
                config['ZONE_VIOLATIONS'] = {}
            
            # Get existing zone indices from config
            existing_zone_indices = set()
            for key in list(config['ZONE_VIOLATIONS'].keys()):
                # Only process keys that match the zone pattern (zone_N_*)
                if key.startswith('zone_') and '_name' in key:
                    try:
                        # Extract zone index from key like "zone_0_name"
                        zone_index = int(key.split('_')[1])
                        existing_zone_indices.add(zone_index)
                    except (ValueError, IndexError):
                        pass  # Skip invalid keys
            
            # Save/update zones
            current_zone_indices = set()
            for i, zone in enumerate(self.zone_violations):
                zone_key = f'zone_{i}'
                current_zone_indices.add(i)
                
                import json
                geometry_type = zone.get('geometry_type', 'rectangle')
                
                # Update or create zone keys
                config['ZONE_VIOLATIONS'][f'{zone_key}_name'] = str(zone.get('name', ''))
                config['ZONE_VIOLATIONS'][f'{zone_key}_geometry_type'] = geometry_type
                config['ZONE_VIOLATIONS'][f'{zone_key}_is_selected'] = str(zone.get('is_selected', True))
                
                # Save coordinates based on geometry type
                if geometry_type == 'rectangle':
                    # Legacy format for backward compatibility
                    config['ZONE_VIOLATIONS'][f'{zone_key}_lat_min'] = str(zone.get('lat_min', 0.0))
                    config['ZONE_VIOLATIONS'][f'{zone_key}_lat_max'] = str(zone.get('lat_max', 0.0))
                    config['ZONE_VIOLATIONS'][f'{zone_key}_lon_min'] = str(zone.get('lon_min', 0.0))
                    config['ZONE_VIOLATIONS'][f'{zone_key}_lon_max'] = str(zone.get('lon_max', 0.0))
                elif geometry_type == 'circle':
                    config['ZONE_VIOLATIONS'][f'{zone_key}_center_lat'] = str(zone.get('center_lat', 0.0))
                    config['ZONE_VIOLATIONS'][f'{zone_key}_center_lon'] = str(zone.get('center_lon', 0.0))
                    config['ZONE_VIOLATIONS'][f'{zone_key}_radius_meters'] = str(zone.get('radius_meters', 0))
                else:  # polygon or polyline
                    # Store coordinates as JSON string
                    coords = zone.get('coordinates', [])
                    config['ZONE_VIOLATIONS'][f'{zone_key}_coordinates'] = json.dumps(coords)
                    if geometry_type == 'polyline':
                        config['ZONE_VIOLATIONS'][f'{zone_key}_tolerance_meters'] = str(zone.get('tolerance_meters', 100))
            
            # Only delete keys for zones that no longer exist (removed zones)
            zones_to_remove = existing_zone_indices - current_zone_indices
            for zone_index in zones_to_remove:
                zone_key = f'zone_{zone_index}'
                # Delete all keys for this zone
                keys_to_delete = [
                    f'{zone_key}_name', f'{zone_key}_geometry_type', f'{zone_key}_is_selected',
                    f'{zone_key}_lat_min', f'{zone_key}_lat_max', f'{zone_key}_lon_min', f'{zone_key}_lon_max',
                    f'{zone_key}_center_lat', f'{zone_key}_center_lon', f'{zone_key}_radius_meters',
                    f'{zone_key}_coordinates', f'{zone_key}_tolerance_meters'
                ]
                for key in keys_to_delete:
                    try:
                        del config['ZONE_VIOLATIONS'][key]
                    except KeyError:
                        pass  # Key doesn't exist, skip it
            
            # Write to file
            try:
                # Ensure the directory exists
                config_dir = os.path.dirname(config_path)
                if config_dir and not os.path.exists(config_dir):
                    os.makedirs(config_dir, exist_ok=True)
                
                # Write config with error handling
                with open(config_path, 'w', encoding='utf-8') as configfile:
                    config.write(configfile)
            except Exception as write_err:
                error_msg = f"Error writing config file: {str(write_err)}"
                try:
                    logger = logging.getLogger("SDF_GUI")
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                except:
                    print(error_msg)
                    print(traceback.format_exc())
                messagebox.showerror("Error", f"Failed to save zones: {error_msg}")
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            full_error = f"{error_type}: {error_msg}"
            try:
                logger = logging.getLogger("SDF_GUI")
                logger.error(f"Error saving zones: {full_error}")
                logger.error(traceback.format_exc())
            except:
                print(f"Error saving zones: {full_error}")
                print(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to save zones: {full_error}")
    
    def run_analysis(self):
        """Run the shipping fraud detection analysis using SFD.py"""
        # Validate data and output directories based on selected data source
        if self.data_source.get() == "noaa":
            # For NOAA data, validate year or year range
            noaa_year = self.noaa_year.get()
            
            # Check if it's a year range (e.g., "2023-2024")
            if '-' in noaa_year:
                try:
                    start_year, end_year = map(int, noaa_year.split('-'))
                    # Validate both years
                    for year in [start_year, end_year]:
                        if year < 2015 or year > 2024:  # Assume valid year range
                            messagebox.showwarning("Warning", f"Year {year} may not have data available. The system will still attempt to access it.")
                except ValueError:
                    messagebox.showerror("Error", f"Invalid year range: {noaa_year}. Please check your date selection.")
                    return
            else:
                # Single year validation
                try:
                    year = int(noaa_year)
                    if year < 2015 or year > 2024:  # Assume valid year range
                        messagebox.showwarning("Warning", f"Year {year} may not have data available. The system will still attempt to access it.")
                except ValueError:
                    messagebox.showerror("Error", f"Invalid year: {noaa_year}. Please check your date selection.")
                    return
                
        elif self.data_source.get() == "local" and not self.data_directory.get():
            messagebox.showerror("Error", "Please specify a data directory in the Data tab")
            return
            
        if not self.output_directory.get():
            messagebox.showerror("Error", "Please specify an output directory in the Data tab")
            return
            
        # Validate S3 settings if using S3
        if self.data_source.get() == "s3":
            if not self.s3_bucket_name.get():
                messagebox.showerror("Error", "Please specify an S3 bucket name")
                return
                
            # Check for required access keys
            if not (self.s3_access_key.get() and self.s3_secret_key.get()):
                messagebox.showerror("Error", "Both AWS Access Key and Secret Key are required")
                return
        
        # Get the script directory for the config file path
        if getattr(sys, 'frozen', False):
            script_dir = os.path.dirname(sys.executable)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
        # Get absolute paths with debug output
        # Look for both possible script filenames (SFD.py and SDF.py)
        script_path = None
        for script_name in ["SFD.py", "SDF.py"]:
            potential_path = os.path.abspath(os.path.join(script_dir, script_name))
            if os.path.exists(potential_path):
                script_path = potential_path
                logger.info(f"Found script file: {script_name}")
                break
                
        if script_path is None:
            # If no script found, default to SFD.py and let the error handling deal with it
            script_path = os.path.abspath(os.path.join(script_dir, "SFD.py"))
            
        config_path = os.path.abspath(os.path.join(script_dir, "config.ini"))
        
        # Log the paths for debugging
        script_filename = os.path.basename(script_path)
        logger.info(f"Script path: {script_path}")
        logger.info(f"Config path: {config_path}")
        
        if not os.path.exists(script_path):
            messagebox.showerror("Error", f"{script_filename} not found.")
            return
        
        # Get date values based on available widgets
        if tkcalendar_available and hasattr(self, 'start_date_picker'):
            start_date_str = self.start_date_picker.get_date().strftime('%Y-%m-%d')
            end_date_str = self.end_date_picker.get_date().strftime('%Y-%m-%d')
            # Update internal variables to match calendar widget values
            self.start_date.set(start_date_str)
            self.end_date.set(end_date_str)
        else:
            start_date_str = self.start_date.get()
            end_date_str = self.end_date.get()
            
        # Save all configuration settings to config.ini first (without showing message)
        self.save_config(show_message=False)
        
        # Build command to run SDF.py with appropriate command-line arguments
        cmd = [sys.executable, script_path]
        
        # Get date values from the UI
        start_date = self.start_date.get()
        end_date = self.end_date.get()
        
        # Add date range arguments
        cmd.extend(["--start-date", start_date, "--end-date", end_date])
        
        # Add data source type and directory path
        cmd.extend(["--data-source", self.data_source.get()])
        
        # Only add data-directory if it's already set (for local/S3)
        # For NOAA, it will be set after downloading data
        if self.data_directory.get() and self.data_source.get() != "noaa":
            cmd.extend(["--data-directory", self.data_directory.get()])
        
        # Add output directory
        cmd.extend(["--output-directory", self.output_directory.get()])
        
        # Add debug flag to get more information
        cmd.append("--debug")
        
        # Debug: Log the command that will be run
        debug_msg = f"Command: {' '.join(cmd)}\nWorking directory: {script_dir}"
        logger.info(debug_msg)
            
        # Note: All other settings will be read from the config.ini file by SDF.py
        
        # Create and show progress window immediately
        progress_window = ProgressWindow(self.root)
        
        # Set the SDFGUI instance as the parent for the progress window
        # This will help it find the conduct_additional_analysis method
        progress_window.parent = self
        progress_window.deiconify()  # Ensure window is visible
        progress_window.lift()  # Bring window to front
        progress_window.focus_force()  # Force focus
        progress_window.update()  # Force update to ensure window is displayed
        
        # Add initial messages
        if tkcalendar_available and hasattr(self, 'start_date_picker'):
            start_date = self.start_date_picker.get_date().strftime('%Y-%m-%d')
            end_date = self.end_date_picker.get_date().strftime('%Y-%m-%d')
            message = f"Starting analysis for date range {start_date} to {end_date}."
        else:
            message = f"Starting analysis for date range {self.start_date.get()} to {self.end_date.get()}."
        
        # Add data source info to progress window
        data_source = self.data_source.get()
        if data_source == "noaa":
            noaa_year = self.noaa_year.get()
            noaa_url = self.get_noaa_data_url(noaa_year)
            progress_window.add_message(f"Using NOAA AIS data for year {noaa_year}")
            progress_window.add_message(f"NOAA data URL: {noaa_url}")
            
            # Get start and end dates
            start_date = self.start_date.get()
            end_date = self.end_date.get()
            
            # Create and show download progress window
            download_progress = DownloadProgressWindow(self.root)
            download_progress.update()
            
            # Add a section separator for better visual organization in the log
            progress_window.add_message("\n===== DOWNLOADING NOAA AIS DATA =====\n")
            progress_window.add_message(f"Downloading AIS data for date range: {start_date} to {end_date}")
            progress_window.add_message("This step may take several minutes depending on the amount of data.")
            progress_window.add_message("All terminal output will be displayed below:")
            progress_window.add_message("-------------------------------------------")
            
            # Use the new DataManager to handle downloads
            # Run download in a separate thread to avoid blocking GUI
            download_complete = threading.Event()
            download_result = {"success": False, "parquet_dir": None, "error": None, "data_manager": None}
            
            def download_thread():
                """Run download in background thread"""
                try:
                    # Create a callback that updates both windows
                    def progress_callback(message):
                        progress_window.add_message(message)
                        download_progress.add_message(message)
                    
                    # Create a data manager instance with the combined callback
                    data_manager = DataManager(logger, progress_callback)
                    download_result["data_manager"] = data_manager
                    
                    # Download NOAA data - this extracts the year from the start date internally
                    success, parquet_dir = data_manager.download_noaa_data(start_date, end_date)
                    
                    download_result["success"] = success
                    download_result["parquet_dir"] = parquet_dir
                    
                except Exception as e:
                    download_result["error"] = str(e)
                finally:
                    download_complete.set()
            
            # Start download in background thread
            download_thread_obj = threading.Thread(target=download_thread, daemon=True)
            download_thread_obj.start()
            
            # Process GUI events while waiting for download to complete
            while not download_complete.is_set():
                self.root.update_idletasks()
                self.root.update()
                download_complete.wait(timeout=0.1)  # Check every 100ms
            
            # Wait for thread to finish
            download_thread_obj.join(timeout=1)
            
            # Check results
            if download_result["error"]:
                error_msg = f"Error downloading NOAA data: {download_result['error']}"
                logger.error(error_msg)
                progress_window.add_message(f"\nERROR: {error_msg}")
                download_progress.close()
                return  # Exit without starting analysis
            
            if not download_result["success"] or not download_result["parquet_dir"]:
                progress_window.add_message("\nERROR: Failed to process NOAA AIS data. Analysis cannot continue.")
                download_progress.close()
                return  # Exit without starting analysis
            
            # Close download progress window when complete
            download_progress.close()
            
            progress_window.add_message("\nNOAA AIS data download and processing complete!")
            progress_window.add_message(f"Files are ready at: {download_result['parquet_dir']}")
            progress_window.add_message("\n===== STARTING ANALYSIS =====\n")
            
            # Update data directory to point to the temporary parquet directory
            self.data_directory.set(download_result["parquet_dir"])
            
            # Add the data directory to the command
            cmd.extend(["--data-directory", download_result["parquet_dir"]])
            
            # Mark for preservation after process completes
            if download_result["data_manager"]:
                progress_window.noaa_temp_dir = download_result["data_manager"].base_temp_dir
            
            logger.info(f"Using NOAA data source: {noaa_url}")
        elif data_source == "local":
            progress_window.add_message(f"Using local data directory: {self.data_directory.get()}")
            logger.info(f"Using local data directory: {self.data_directory.get()}")
        elif data_source == "s3":
            progress_window.add_message(f"Using AWS S3 data: {self.s3_data_uri.get()}")
            logger.info(f"Using S3 data source: {self.s3_data_uri.get()}")
        
        progress_window.add_message(message)
        progress_window.add_message("This may take some time depending on the amount of data.")
        progress_window.add_message("-------------------------------------------")
        progress_window.add_message(f"Command: {' '.join(cmd)}")
        progress_window.add_message("-------------------------------------------")
        
        # Function to read output from process and update the window
        def read_output(process):
            # Show initial status
            progress_window.add_message(f"{script_filename} process started. Capturing output...")
            progress_window.add_message("You can monitor the progress in this window.")
            progress_window.add_message("-------------------------------------------")
            progress_window.add_message("STARTING ANALYSIS")
            progress_window.add_message("All terminal output from the process will be shown below:")
            progress_window.add_message("============================================")
            logger.info("Starting output capture for subprocess")
            
            # Read both stdout and stderr from the process
            try:
                import select
                import queue
                
                # Create queues for stdout and stderr
                stdout_queue = queue.Queue()
                stderr_queue = queue.Queue()
                
                # Function to read from stdout
                def read_stdout():
                    # Only log to debug, don't show in progress window
                    logger.info("Started stdout reading thread")
                    try:
                        while True:
                            if progress_window.canceled:
                                logger.info("Stdout reader detected cancellation")
                                break
                                
                            try:
                                line = process.stdout.readline()
                                if not line:
                                    if process.poll() is not None:
                                        logger.info("Stdout reader detected process exit")
                                        break
                                    time.sleep(0.1)  # Prevent CPU spinning
                                    continue
                                    
                                # Decode bytes to string if needed
                                if isinstance(line, bytes):
                                    line_str = line.decode('utf-8', errors='replace').strip()
                                else:
                                    line_str = line.strip()
                                    
                                if line_str:
                                    logger.debug(f"STDOUT: {line_str}")
                                    # Check if this is a significant message we want to highlight
                                    if "download" in line_str.lower() or "convert" in line_str.lower() or "extract" in line_str.lower():
                                        # Add a special prefix for download/conversion messages
                                        stdout_queue.put(f"[PROCESSING] {line_str}")
                                    elif any(pattern in line_str for pattern in ["Processing", "Starting", "Completed", "Loading", "Analyzing"]):
                                        # Highlight other important process messages
                                        stdout_queue.put(f">> {line_str}")
                                    else:
                                        stdout_queue.put(line_str)
                                    
                            except Exception as decode_error:
                                error_msg = f"Error decoding stdout line: {str(decode_error)}"
                                logger.error(error_msg)
                                stdout_queue.put(error_msg)
                                
                    except Exception as e:
                        error_msg = f"Critical error in stdout reader: {str(e)}"
                        logger.error(error_msg)
                        stdout_queue.put(error_msg)
                        
                    logger.info("Stdout reader thread exiting")
                
                # Function to read from stderr
                def read_stderr():
                    # Only log to debug, don't show in progress window
                    logger.info("Started stderr reading thread")
                    try:
                        while True:
                            if progress_window.canceled:
                                logger.info("Stderr reader detected cancellation")
                                break
                                
                            try:
                                line = process.stderr.readline()
                                if not line:
                                    if process.poll() is not None:
                                        logger.info("Stderr reader detected process exit")
                                        break
                                    time.sleep(0.1)  # Prevent CPU spinning
                                    continue
                                    
                                # Decode bytes to string if needed
                                if isinstance(line, bytes):
                                    line_str = line.decode('utf-8', errors='replace').strip()
                                else:
                                    line_str = line.strip()
                                    
                                if line_str:
                                    logger.debug(f"STDERR: {line_str}")
                                    
                                    # Python's logging module writes to stderr by default
                                    # Format log messages to be more readable
                                    if " - ERROR - " in line_str or " - CRITICAL - " in line_str:
                                        # Make errors stand out
                                        stderr_queue.put(f"ERROR: {line_str}")
                                    elif " - WARNING - " in line_str:
                                        stderr_queue.put(f"WARNING: {line_str}")
                                    elif "download" in line_str.lower() or "convert" in line_str.lower() or "extract" in line_str.lower():
                                        # Add a special prefix for download/conversion messages
                                        stderr_queue.put(f"[PROCESSING] {line_str}")
                                    elif any(pattern in line_str for pattern in ["Processing", "Starting", "Completed", "Loading", "Analyzing"]):
                                        # Highlight other processing messages
                                        stderr_queue.put(f">> {line_str}")
                                    else:
                                        # Regular log messages - no prefix needed
                                        stderr_queue.put(line_str)
                            except Exception as decode_error:
                                error_msg = f"Error decoding stderr line: {str(decode_error)}"
                                logger.error(error_msg)
                                stderr_queue.put(error_msg)
                                
                    except Exception as e:
                        error_msg = f"Critical error in stderr reader: {str(e)}"
                        logger.error(error_msg)
                        stderr_queue.put(error_msg)
                        
                    logger.info("Stderr reader thread exiting")
                
                # Start threads to read stdout and stderr
                stdout_thread = threading.Thread(target=read_stdout, daemon=True)
                stderr_thread = threading.Thread(target=read_stderr, daemon=True)
                stdout_thread.start()
                stderr_thread.start()
                
                # Add marker to indicate processing has started
                logger.info("Starting to process output queues")
                
                # Track if we've processed any output
                output_processed = False
                
                # Read from queues and display messages
                while True:
                    if progress_window.canceled:
                        logger.info("Main queue processor detected cancellation")
                        break
                    
                    # Check if process is done
                    poll_result = process.poll()
                    if poll_result is not None:
                        logger.info(f"Process exited with code {poll_result}, reading remaining output")
                        progress_window.add_message(f"Process exited with code {poll_result}")
                        
                        # Process is done, read any remaining output
                        try:
                            messages_processed = 0
                            while not stdout_queue.empty():
                                line = stdout_queue.get_nowait()
                                progress_window.add_message(line)
                                messages_processed += 1
                            if messages_processed > 0:
                                logger.info(f"Processed {messages_processed} remaining stdout messages")
                        except Exception as e:
                            logger.error(f"Error reading remaining stdout: {str(e)}")
                            
                        try:
                            messages_processed = 0
                            while not stderr_queue.empty():
                                line = stderr_queue.get_nowait()
                                progress_window.add_message(line)
                                messages_processed += 1
                            if messages_processed > 0:
                                logger.info(f"Processed {messages_processed} remaining stderr messages")
                        except Exception as e:
                            logger.error(f"Error reading remaining stderr: {str(e)}")
                            
                        break
                    
                    # Track if we processed any messages in this cycle
                    cycle_processed = 0
                    
                    # Read from stdout queue
                    try:
                        messages_processed = 0
                        while not stdout_queue.empty():
                            line = stdout_queue.get_nowait()
                            progress_window.add_message(line)
                            messages_processed += 1
                            cycle_processed += 1
                        if messages_processed > 0:
                            output_processed = True
                    except Exception as e:
                        logger.error(f"Error processing stdout queue: {str(e)}")
                    
                    # Read from stderr queue
                    try:
                        messages_processed = 0
                        while not stderr_queue.empty():
                            line = stderr_queue.get_nowait()
                            progress_window.add_message(line)
                            messages_processed += 1
                            cycle_processed += 1
                        if messages_processed > 0:
                            output_processed = True
                    except Exception as e:
                        logger.error(f"Error processing stderr queue: {str(e)}")
                    
                    # If we haven't processed any output yet, log it but don't show in UI
                    if not output_processed and cycle_processed > 0:
                        logger.info("First output processed from subprocess")
                    
                    # Small sleep to avoid busy waiting
                    time.sleep(0.1)
                
                # Wait for threads to finish
                stdout_thread.join(timeout=1)
                stderr_thread.join(timeout=1)
                
            except Exception as e:
                # Fallback to simple reading if threading approach fails
                error_msg = f"Threading output capture failed: {str(e)}. Using fallback method."
                logger.error(error_msg)
                progress_window.add_message(error_msg)
                
                try:
                    # Different handling for Windows vs Unix systems
                    if platform.system() == "Windows":
                        progress_window.add_message("Using Windows-specific fallback reader")
                        logger.info("Using Windows-specific fallback reader")
                        
                        # Windows doesn't support fcntl, but we can use a different approach
                        import msvcrt
                        import io
                        
                        # Try to set binary mode for Windows
                        try:
                            msvcrt.setmode(process.stdout.fileno(), os.O_BINARY)
                            msvcrt.setmode(process.stderr.fileno(), os.O_BINARY)
                            progress_window.add_message("Set binary mode for Windows pipes")
                        except (AttributeError, ImportError, io.UnsupportedOperation) as e:
                            logger.warning(f"Could not set binary mode: {str(e)}")
                            progress_window.add_message("WARNING: Binary mode not available, output encoding may be affected")
                    else:
                        # Unix-like system (Linux/Mac)
                        import io
                        import fcntl
                        import os
                        import select
                        
                        # Try to set non-blocking mode
                        try:
                            # Make stdout non-blocking
                            fl = fcntl.fcntl(process.stdout, fcntl.F_GETFL)
                            fcntl.fcntl(process.stdout, fcntl.F_SETFL, fl | os.O_NONBLOCK)
                            
                            # Make stderr non-blocking
                            fl = fcntl.fcntl(process.stderr, fcntl.F_GETFL)
                            fcntl.fcntl(process.stderr, fcntl.F_SETFL, fl | os.O_NONBLOCK)
                            
                            progress_window.add_message("Configured non-blocking mode for fallback reader")
                            logger.info("Configured non-blocking mode for fallback reader")
                        except (AttributeError, ImportError, io.UnsupportedOperation) as e:
                            progress_window.add_message(f"WARNING: Non-blocking mode not available: {str(e)}")
                            logger.warning(f"Non-blocking mode not available in fallback reader: {str(e)}")
                
                    # Fallback read loop
                    while True:
                        if progress_window.canceled:
                            logger.info("Fallback reader detected cancellation")
                            break
                            
                        if process.poll() is not None:
                            logger.info(f"Fallback reader detected process exit with code {process.poll()}")
                            progress_window.add_message(f"Process exited with code {process.poll()}")
                            break
                        
                        # Try to read from stdout
                        try:
                            line = process.stdout.readline()
                            if line:
                                if isinstance(line, bytes):
                                    line_str = line.decode('utf-8', errors='replace').strip()
                                else:
                                    line_str = line.strip()
                                if line_str:
                                    logger.debug(f"FALLBACK STDOUT: {line_str}")
                                    progress_window.add_message(line_str)
                        except Exception as stdout_error:
                            pass  # Ignore errors in non-blocking mode
                        
                        # Try to read from stderr
                        try:
                            line = process.stderr.readline()
                            if line:
                                if isinstance(line, bytes):
                                    line_str = line.decode('utf-8', errors='replace').strip()
                                else:
                                    line_str = line.strip()
                                if line_str:
                                    logger.debug(f"FALLBACK STDERR: {line_str}")
                                    progress_window.add_message(line_str)
                        except Exception as stderr_error:
                            pass  # Ignore errors in non-blocking mode
                        
                        # Small sleep to avoid busy waiting
                        time.sleep(0.1)
                
                except Exception as fallback_error:
                    error_msg = f"Error in fallback reading: {str(fallback_error)}"
                    logger.error(error_msg)
                    progress_window.add_message(error_msg)
            
            # Process has finished - handle completion regardless of which method was used
            if not progress_window.canceled:
                try:
                    return_code = process.poll()
                    if return_code is None:
                        return_code = process.wait(timeout=5)
                except Exception:
                    return_code = process.poll() or -1
                    
                if return_code == 0:
                    # Add section separator and success message
                    progress_window.add_message("\n===== ANALYSIS COMPLETED SUCCESSFULLY =====\n")
                    progress_window.add_message("The AIS data analysis has completed successfully!")
                    logger.info("Analysis completed with return code 0 - showing completion options")
                    progress_window.add_message(f"\nResults have been saved to: {self.output_directory.get()}")
                    progress_window.add_message("\nYou can review the results and choose further actions from the options below.")
                    
                    # Show completion message and update the progress window
                    def show_completion():
                        if not progress_window.canceled:
                            # No need to show another message box, since SDF.py shows one
                            # with an option to continue or exit
                            
                            # Update the progress window with completion options
                            progress_window.analysis_complete(self.output_directory.get(), success=True)
                            
                            # Remove focus from progress window - letting the SDF.py dialog
                            # be visible to the user
                            progress_window.withdraw()
                    
                    # Schedule the completion message to be shown after a delay
                    progress_window.after(500, show_completion)
                elif return_code == 100:
                    # Special exit code 100 means user chose not to continue
                    # Close the entire application
                    progress_window.add_message("\n===== USER REQUESTED EXIT =====\n")
                    progress_window.add_message("User chose not to continue. Closing application...")
                    # Create message box to make sure the user understands the application is closing
                    messagebox.showinfo("Closing Application", "The analysis is complete and you've chosen not to continue.\n\nThe application will now close.")
                    # Force destroy both windows to ensure complete exit
                    progress_window.destroy()
                    self.root.destroy()
                    # Additional exit call to guarantee termination
                    sys.exit(0)
                elif return_code == 101:
                    # Special exit code 101 means the user wants to conduct additional analysis
                    # Update the progress window with completion options
                    progress_window.analysis_complete(self.output_directory.get(), success=True)
                    
                    # Launch the additional analysis window after a short delay
                    def launch_additional_analysis():
                        if not progress_window.canceled:
                            # Call the conduct_additional_analysis method through the button command
                            progress_window.conduct_additional_analysis()
                    
                    # Schedule the additional analysis window to be shown after a delay
                    progress_window.after(500, launch_additional_analysis)
                else:
                    # Add section separator and error message
                    progress_window.add_message("\n===== ANALYSIS FAILED =====\n")
                    progress_window.add_message(f"Analysis failed with return code {return_code}")
                    
                    # If there's an error, show the error output
                    try:
                        error_lines = process.stderr.readlines()
                        if error_lines:
                            progress_window.add_message("Error details:")
                            for error_line in error_lines:
                                try:
                                    # Decode bytes to string if needed
                                    if isinstance(error_line, bytes):
                                        line_str = error_line.decode('utf-8', errors='replace').strip()
                                    else:
                                        line_str = error_line.strip()
                                    if line_str:
                                        progress_window.add_message(f"  {line_str}")
                                except Exception:
                                    pass
                    except Exception as e:
                        progress_window.add_message(f"Error reading error details: {str(e)}")
                    
                    # Add suggestions for troubleshooting
                    progress_window.add_message("\nSuggestions for troubleshooting:")
                    progress_window.add_message("1. Check if the data files exist and are accessible")
                    progress_window.add_message("2. Verify that the date range contains valid data")
                    progress_window.add_message("3. Ensure you have sufficient disk space")
                    
                    # Show error message and provide options
                    def show_error():
                        if not progress_window.canceled:
                            messagebox.showerror("Analysis Failed", f"The analysis failed with return code {return_code}.\n\nCheck the log for details and try again.")
                            # Still show completion options, but indicate failure
                            progress_window.analysis_complete(self.output_directory.get(), success=False)
                    
                    # Schedule the error message to be shown after a delay
                    progress_window.after(2000, show_error)
        
        try:
            # Check if the script file actually exists
            if not os.path.exists(script_path):
                error_msg = f"Error: {script_filename} not found at {script_path}"
                logger.error(error_msg)
                messagebox.showerror("Error", error_msg)
                return
                
            # Check permissions
            try:
                with open(script_path, 'r') as f:
                    pass  # Just testing if we can read the file
            except Exception as e:
                error_msg = f"Cannot read {script_filename}: {str(e)}"
                logger.error(error_msg)
                messagebox.showerror("Error", error_msg)
                return
            
            # Start the process - use subprocess.Popen to capture output for all platforms
            try:
                logger.info(f"Attempting to start process with cmd: {cmd}")
                # Use CREATE_NO_WINDOW on Windows to capture output properly
                # This allows us to capture stdout/stderr and display in the progress window
                # If user wants to see console window, they can run SFD.py directly
                creation_flags = 0
                if platform.system() == "Windows":
                    # Use CREATE_NO_WINDOW to capture output (no separate console window)
                    # Output will be displayed in the progress window instead
                    try:
                        creation_flags = subprocess.CREATE_NO_WINDOW
                    except AttributeError:
                        # CREATE_NO_WINDOW might not be available in older Python versions
                        # Fall back to no flags (output will still be captured)
                        creation_flags = 0
                
                # Set environment variable to indicate GUI mode
                env = os.environ.copy()
                env['SFD_GUI_MODE'] = 'true'
                env['PYTHONIOENCODING'] = 'utf-8'  # Force UTF-8 encoding for Python I/O
                
                # Debug: Print detailed information about the process we're about to start
                logger.info("=========================")
                logger.info("DEBUG: Starting subprocess")
                logger.info(f"DEBUG: Command: {' '.join(cmd)}")
                logger.info(f"DEBUG: Working directory: {script_dir}")
                logger.info(f"DEBUG: Env variables: SFD_GUI_MODE={env.get('SFD_GUI_MODE')}")
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=False,  # Keep as bytes for consistent handling
                    cwd=script_dir,  # Ensure correct working directory
                    creationflags=creation_flags,
                    bufsize=0,  # Unbuffered for real-time output (required for binary mode)
                    env=env  # Pass environment with GUI mode flag
                )
                
                logger.info(f"DEBUG: Process started with PID: {process.pid}")
                logger.info("=========================")
                logger.info(f"Process started with PID: {process.pid}")
                
                # Set the process on the progress window
                progress_window.set_process(process)
                
                # Start a thread to read the output
                threading.Thread(target=read_output, args=(process,), daemon=True).start()
            except Exception as e:
                error_msg = f"Failed to start {script_filename} process: {str(e)}"
                logger.error(error_msg)
                messagebox.showerror("Process Error", error_msg)
                return
                
        except Exception as e:
            error_msg = f"Failed to run analysis: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", error_msg)
            
            # Try an alternative method as a fallback
            if messagebox.askyesno("Attempt Alternative Launch", "Would you like to try an alternative method to launch SFD.py?"):
                try:
                    logger.info("Attempting alternative launch method")
                    # Construct a command string for direct execution with NO command line arguments
                    # We use a variable since this command is shown to the user in the dialog
                    alt_cmd = f"cd \"{script_dir}\" && \"{sys.executable}\" \"{script_path}\""
                    
                    # Log the alternative command
                    logger.info(f"Alternative command: {alt_cmd}")
                    messagebox.showinfo("Alternative Method", f"Trying to launch with command:\n{alt_cmd}")
                    
                    # Create a batch file to execute the command - use a different name than main batch file
                    temp_batch_name = f"run_sfd_alt_{int(time.time())}.bat"
                    batch_path = os.path.join(script_dir, temp_batch_name)
                    with open(batch_path, "w") as batch_file:
                        batch_file.write(f"@echo off\n")
                        batch_file.write(f"echo Running {script_filename}...\n")
                        batch_file.write(f"{alt_cmd}\n")
                        batch_file.write(f"pause\n")
                    
                    # Execute the batch file
                    os.startfile(batch_path)
                    messagebox.showinfo("Batch File Created", f"A batch file has been created at {batch_path}. Please check if {script_filename} is running.")
                    
                except Exception as alt_e:
                    logger.error(f"Alternative launch method failed: {str(alt_e)}")
                    messagebox.showerror("Error", f"Alternative launch method failed: {str(alt_e)}")
                    messagebox.showinfo("Manual Instructions", f"To run {script_filename} manually:\n1. Open a command prompt\n2. Navigate to: {script_dir}\n3. Run: {sys.executable} {script_path}")
                    return

def main():
    root = tk.Tk()
    app = SDFGUI(root)
    return root

if __name__ == "__main__":
    # Set up basic root window first for messagebox to work
    root = tk.Tk()
    root.withdraw()  # Hide window initially
    
    # Check dependencies first
    if check_dependencies():
        # Destroy initial hidden window
        root.destroy()
        # Start the application normally
        root = main()
        root.mainloop()
    else:
        # If dependencies are missing and user didn't install them, exit
        messagebox.showerror(
            "Dependencies Missing",
            "Cannot start application due to missing dependencies."
        )
        root.destroy()