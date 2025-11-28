#!/usr/bin/env python3
"""
Shared Utilities Module for SFD Project

[VERSION}
Team = Dreadnaught
Alex Giacomello, Christopher Matherne, Rupert Rigg, Zachary Zhao
version = 2.1 Beta

This module contains common utility functions used across the SFD project components.
It centralizes duplicated code to improve maintainability and consistency.
"""

import os
import sys
import logging
import configparser
import platform
import subprocess
import importlib
from datetime import datetime, timedelta

# Set up module-level logger
logger = logging.getLogger(__name__)

def get_cache_dir():
    """Get the directory to use for caching data."""
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".ais_data_cache")
    
    if not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir)
            logger.info(f"Created cache directory at {cache_dir}")
        except Exception as e:
            logger.error(f"Failed to create cache directory: {e}")
            return None
    
    return cache_dir

def clear_cache():
    """Clear the data cache directory."""
    cache_dir = get_cache_dir()
    if not cache_dir or not os.path.exists(cache_dir):
        logger.info("No cache directory found")
        return True
    
    try:
        for filename in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        logger.info(f"Cache cleared: {cache_dir}")
        return True
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return False

def check_dependencies(requirements_file='requirements.txt', silent=False, offer_install=True):
    """Check if all required dependencies are installed."""
    logger.info("Checking dependencies...")
    
    # Log Python version information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    
    # Find the requirements.txt file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    req_file = os.path.join(script_dir, requirements_file)
    
    if not os.path.exists(req_file):
        req_file = os.path.join(os.getcwd(), requirements_file)
        if not os.path.exists(req_file):
            logger.warning(f"{requirements_file} not found")
            return True
    
    try:
        # Parse requirements file
        requirements = []
        with open(req_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-e'):
                    # Parse requirement line
                    condition = None
                    if ';' in line:
                        req, condition = line.split(';', 1)
                        req = req.strip()
                        condition = condition.strip()
                        
                        # Check platform-specific requirements
                        if 'platform_system==' in condition:
                            platform_name = condition.split('==')[1].strip().strip("'").strip('"')
                            if platform.system() != platform_name:
                                continue
                        
                        # Check Python version requirements
                        if 'python_version' in condition:
                            # Extract version requirement (e.g., "< '3.8'" or "python_version < '3.8'")
                            # Simple parsing for common patterns
                            current_major = sys.version_info.major
                            current_minor = sys.version_info.minor
                            
                            # Look for version pattern like "< '3.8'" or "> '3.8'"
                            should_include = True
                            if "< '3.8'" in condition or '< "3.8"' in condition:
                                # Only include if Python < 3.8
                                should_include = (current_major, current_minor) < (3, 8)
                            elif "> '3.8'" in condition or '> "3.8"' in condition:
                                # Only include if Python > 3.8
                                should_include = (current_major, current_minor) > (3, 8)
                            elif "<='3.8'" in condition or '<="3.8"' in condition:
                                should_include = (current_major, current_minor) <= (3, 8)
                            elif ">='3.8'" in condition or '>="3.8"' in condition:
                                should_include = (current_major, current_minor) >= (3, 8)
                            
                            if not should_include:
                                continue  # Skip this requirement
                    else:
                        req = line
                    
                    # Strip version specifiers
                    if '>=' in req:
                        req = req.split('>=')[0]
                    elif '==' in req:
                        req = req.split('==')[0]
                    elif '<=' in req:
                        req = req.split('<=')[0]
                    
                    req = req.strip()
                    if req:
                        requirements.append(req)
        
        # Check installed packages
        missing = []
        import_module = importlib.import_module  # Store reference to the function
        
        # Special handling for packages where package name != import name
        package_import_map = {
            'concurrent-futures': 'concurrent.futures',  # Package name vs import name
            'pywin32': 'win32api',  # pywin32 package imports as win32api
        }
        
        for req in requirements:
            # Check if this is a special case
            if req in package_import_map:
                import_name = package_import_map[req]
                try:
                    import_module(import_name)
                    continue  # Successfully imported, skip to next requirement
                except ImportError:
                    # Check if package is installed via metadata
                    try:
                        import importlib.metadata as imp_metadata
                        try:
                            imp_metadata.version(req)
                            continue  # Package installed, skip to next
                        except imp_metadata.PackageNotFoundError:
                            pass
                    except ImportError:
                        try:
                            import pkg_resources
                            try:
                                pkg_resources.get_distribution(req)
                                continue  # Package installed, skip to next
                            except pkg_resources.DistributionNotFound:
                                pass
                        except ImportError:
                            pass
                    # If we get here, package is missing
                    missing.append(req)
                    continue
            
            # Standard package checking
            try:
                import_module(req.lower().replace('-', '_'))
            except ImportError:
                try:
                    # Check with metadata if available
                    try:
                        import importlib.metadata as imp_metadata
                        try:
                            imp_metadata.version(req)
                        except imp_metadata.PackageNotFoundError:
                            missing.append(req)
                    except ImportError:
                        # Fallback for older Python versions
                        try:
                            import pkg_resources
                            try:
                                pkg_resources.get_distribution(req)
                            except pkg_resources.DistributionNotFound:
                                missing.append(req)
                        except ImportError:
                            missing.append(req)
                except Exception:
                    missing.append(req)
        
        if missing:
            logger.warning(f"Missing dependencies: {', '.join(missing)}")
            
            # Handle installation if needed
            if offer_install and not silent:
                resp = input("Would you like to install missing dependencies now? (y/n): ").strip().lower()
                
                if resp in ('y', 'yes'):
                    logger.info("Installing missing dependencies...")
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
                        logger.info("Dependencies installed successfully!")
                        return True
                    except subprocess.CalledProcessError:
                        logger.error("Failed to install dependencies.")
                        return False
                else:
                    logger.warning("Missing dependencies will not be installed.")
                    return False
            else:
                return False
        
        logger.info("All dependencies are satisfied.")
        return True
    
    except Exception as e:
        logger.error(f"Error checking dependencies: {e}")
        return True  # Continue anyway

def format_file_size(size_bytes):
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def get_memory_usage():
    """Get current memory usage information."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            'rss_mb': mem_info.rss / (1024 * 1024),
            'vms_mb': mem_info.vms / (1024 * 1024),
            'percent': process.memory_percent()
        }
    except ImportError:
        return {'rss_mb': 'N/A', 'vms_mb': 'N/A', 'percent': 'N/A'}
    except Exception:
        return {'rss_mb': 'N/A', 'vms_mb': 'N/A', 'percent': 'N/A'}

def log_memory_usage(operation_name=""):
    """Log current memory usage."""
    mem_usage = get_memory_usage()
    logger.info(f"Memory usage {operation_name}: RSS={mem_usage.get('rss_mb', 'N/A')} MB, "
               f"Percent={mem_usage.get('percent', 'N/A')}%")

def suppress_warnings(enabled=True):
    """Suppress Python warnings if enabled."""
    import warnings
    if enabled:
        warnings.filterwarnings("ignore")
        logger.info("Warnings have been suppressed")
    else:
        warnings.resetwarnings()
        logger.info("Warnings are enabled")

def validate_config(config_path='config.ini'):
    """Validate that config.ini exists and contains required sections."""
    # Resolve config path
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
        if not os.path.exists(config_path):
            config_path = os.path.join(os.getcwd(), 'config.ini')
    
    if not os.path.exists(config_path):
        return False, f"Config file not found: {config_path}"
    
    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        
        if 'DEFAULT' not in config:
            return False, "Missing required DEFAULT section in config"
        
        return True, None
        
    except Exception as e:
        return False, f"Error reading config file: {str(e)}"

def generate_cache_key(file_path, config):
    """Generate a unique cache key for a file based on path and config."""
    import hashlib
    
    file_name = os.path.basename(file_path)
    ship_types_str = ",".join(map(str, config.get('SELECTED_SHIP_TYPES', []))) if 'SELECTED_SHIP_TYPES' in config else ""
    key_string = f"{file_name}_{ship_types_str}"
    
    # Include file modification time if file exists locally
    if os.path.exists(file_path) and not file_path.startswith('s3://'):
        try:
            mtime = os.path.getmtime(file_path)
            key_string += f"_{mtime}"
        except Exception:
            pass
    
    # Generate hash of the key string
    return hashlib.md5(key_string.encode()).hexdigest()
