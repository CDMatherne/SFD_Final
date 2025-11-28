#!/usr/bin/env python3
"""
Import Manager Module for SFD Project

[VERSION}
Team = Dreadnaught
Alex Giacomello, Christopher Matherne, Rupert Rigg, Zachary Zhao
version = 2.1 Beta

This module provides centralized import management for the SFD project,
handling optional dependencies and providing appropriate fallbacks.
"""

import os
import sys
import logging
import importlib
import platform
import subprocess

# Configure module logger
logger = logging.getLogger(__name__)

# Dictionary to track module availability
AVAILABLE_MODULES = {}

def check_module(module_name, package_name=None):
    """
    Check if a module is available and mark it in the AVAILABLE_MODULES dictionary.
    
    Args:
        module_name (str): Name of the module to import
        package_name (str, optional): Name of the package if different from module_name
    
    Returns:
        bool: True if the module is available, False otherwise
    """
    if module_name in AVAILABLE_MODULES:
        return AVAILABLE_MODULES[module_name]
        
    if not package_name:
        package_name = module_name
        
    try:
        importlib.import_module(module_name)
        AVAILABLE_MODULES[module_name] = True
        return True
    except ImportError:
        AVAILABLE_MODULES[module_name] = False
        logger.info(f"Optional module {module_name} is not available")
        return False

def import_optional(module_name, package_name=None, as_name=None):
    """
    Import an optional module if available.
    
    Args:
        module_name (str): Name of the module to import
        package_name (str, optional): Name of the package if different from module_name
        as_name (str, optional): Name to import the module as
    
    Returns:
        module or None: The imported module or None if not available
    """
    if not package_name:
        package_name = module_name
        
    try:
        module = importlib.import_module(module_name)
        if as_name:
            sys.modules[as_name] = module
        AVAILABLE_MODULES[module_name] = True
        return module
    except ImportError:
        AVAILABLE_MODULES[module_name] = False
        logger.info(f"Optional module {module_name} is not available")
        return None

def install_package(package_name, upgrade=False, user=False):
    """
    Attempt to install a package using pip.
    
    Args:
        package_name (str): Name of the package to install
        upgrade (bool, optional): Whether to upgrade the package if already installed
        user (bool, optional): Whether to install in user space
    
    Returns:
        bool: True if installation succeeded, False otherwise
    """
    try:
        cmd = [sys.executable, "-m", "pip", "install"]
        
        if upgrade:
            cmd.append("--upgrade")
            
        if user:
            cmd.append("--user")
            
        cmd.append(package_name)
        
        logger.info(f"Installing package: {package_name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Successfully installed {package_name}")
            return True
        else:
            logger.error(f"Failed to install {package_name}: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error installing {package_name}: {e}")
        return False

# Initialize common optional modules
pandas = import_optional("pandas")
numpy = import_optional("numpy")
matplotlib_plt = import_optional("matplotlib.pyplot", as_name="plt")
plotly_go = import_optional("plotly.graph_objects", as_name="go")
folium = import_optional("folium")
dash = import_optional("dash")
sklearn = import_optional("sklearn")

# GPU acceleration libraries
cupy = import_optional("cupy")
torch = import_optional("torch")
tensorflow = import_optional("tensorflow")
# AMD GPU support (ROCm/HIP)
# AMD support rrquires the HIP-SDK to be installed https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html
# Note: cupy-rocm is installed as 'cupy' but uses ROCm/HIP backend
# Install cupy-rocm for AMD GPUs (replaces cupy for AMD systems)
# cupy-rocm>=12.0.0  # AMD ROCm version of CuPy (install via: pip install cupy-rocm)
# Alternatively, use PyHIP for direct HIP access:
# pyhip>=0.1.0  # PyHIP for direct AMD HIP runtime access

# Platform-specific modules
if platform.system() == "Windows":
    pywin32 = import_optional("win32api", package_name="pywin32")
    
# Define module groups for easier checking
DATA_PROCESSING_AVAILABLE = all(m in AVAILABLE_MODULES and AVAILABLE_MODULES[m] 
                              for m in ["pandas", "numpy"])
VISUALIZATION_AVAILABLE = any(m in AVAILABLE_MODULES and AVAILABLE_MODULES[m] 
                            for m in ["matplotlib.pyplot", "plotly.graph_objects", "folium"])
ML_AVAILABLE = "sklearn" in AVAILABLE_MODULES and AVAILABLE_MODULES["sklearn"]
GPU_AVAILABLE = any(m in AVAILABLE_MODULES and AVAILABLE_MODULES[m] 
                   for m in ["cupy", "torch", "tensorflow"])

def get_module_status():
    """
    Get the status of all checked modules.
    
    Returns:
        dict: Dictionary with module names as keys and availability as values
    """
    return AVAILABLE_MODULES.copy()

def print_module_status():
    """Print the status of all checked modules."""
    print("\n=== Module Availability ===")
    for module, available in sorted(AVAILABLE_MODULES.items()):
        status = "Available" if available else "Not Available"
        print(f"{module:<20}: {status}")
    print("\n=== Feature Availability ===")
    print(f"Data Processing: {'Available' if DATA_PROCESSING_AVAILABLE else 'Not Available'}")
    print(f"Visualization: {'Available' if VISUALIZATION_AVAILABLE else 'Not Available'}")
    print(f"Machine Learning: {'Available' if ML_AVAILABLE else 'Not Available'}")
    print(f"GPU Acceleration: {'Available' if GPU_AVAILABLE else 'Not Available'}")
    print("")
