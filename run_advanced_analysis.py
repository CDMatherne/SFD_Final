#!/usr/bin/env python3
"""
Run Advanced Analysis Standalone Script

[VERSION}
Team = Dreadnaught
Alex Giacomello, Christopher Matherne, Rupert Rigg, Zachary Zhao
version = 2.1 Beta

This script directly launches the Advanced Analysis GUI without going through the main SFD_GUI.
It uses data from the specified output directory and the .ais.cache_data directories.
"""

import os
import sys
import tkinter as tk
import logging
import traceback
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_analysis_standalone.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AdvancedAnalysisRunner")

def get_latest_output_dir():
    """Find the most recently modified output directory."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dirs = []
    
    # Also check C:/AIS_Data/Output which is where SFD.py typically puts its files
    # Look for directories that might contain output
    possible_dirs = [
        "C:/AIS_Data/Output",  # Add the typical SFD output path
        os.path.join(base_dir, "output"),
        os.path.join(base_dir, "results"),
        base_dir  # Check base dir itself for any outputs
    ]
    
    for search_dir in possible_dirs:
        if os.path.exists(search_dir):
            # Add the search dir itself as a possible output directory
            output_dirs.append(search_dir)
            
            # Find all directories in the search path
            for item in os.listdir(search_dir):
                item_path = os.path.join(search_dir, item)
                if os.path.isdir(item_path):
                    # Check if this directory contains AIS analysis outputs
                    if (os.path.exists(os.path.join(item_path, "AIS_Anomalies_Summary.csv")) or 
                        os.path.exists(os.path.join(item_path, "vessel_details.csv")) or
                        os.path.exists(os.path.join(item_path, "All Anomalies Map.html"))):
                        
                        output_dirs.append(item_path)
    
    # Include base_dir/output if it exists
    default_output = os.path.join(base_dir, "output")
    if os.path.exists(default_output) and default_output not in output_dirs:
        output_dirs.append(default_output)
    
    # Sort by modification time (newest first)
    output_dirs.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)
    
    # Return the most recently modified directory or the default output path
    if output_dirs:
        logger.info(f"Found most recent output directory: {output_dirs[0]}")
        return output_dirs[0]
    else:
        logger.warning(f"No output directories found, using default: {default_output}")
        os.makedirs(default_output, exist_ok=True)
        return default_output

def get_config_path():
    """Get the config.ini path, preferring one in the script directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.ini")
    
    if os.path.exists(config_path):
        return config_path
    else:
        # Search for config.ini in common locations
        search_paths = [
            os.path.join(script_dir, "conf", "config.ini"),
            os.path.join(os.path.expanduser("~"), ".config", "sfd", "config.ini"),
            "/etc/sfd/config.ini"
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        # If no config found, create a basic one
        logger.warning("No config.ini found, creating a default one")
        with open(config_path, 'w') as f:
            f.write("[DEFAULT]\n")
            f.write("data_source = local\n")
            f.write("[OUTPUT_CONTROLS]\n")
            f.write("generate_statistics_excel = False\n")
            f.write("generate_statistics_csv = False\n")
            
        return config_path

def run_advanced_analysis():
    try:
        # Import advanced_analysis here to avoid circular imports
        from advanced_analysis import AdvancedAnalysisGUI
        logger.info("Imported AdvancedAnalysisGUI successfully")
        # Get the most recent output directory
        output_dir = get_latest_output_dir()
        logger.info(f"Found output directory: {output_dir}")
        # Get config path
        config_path = get_config_path()
        logger.info(f"Found config path: {config_path}")
        # Create tkinter root
        root = tk.Tk()
        root.title("SFD Advanced Analysis")
        root.geometry("200x200")
        logger.info("Created tkinter root successfully")
        # Create and run the advanced analysis GUI
        # Make sure output_dir exists and create it if needed
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
            
        # Log the output directory we're using
        logger.info(f"Using output directory for Advanced Analysis: {output_dir}")
        
        # Initialize the Advanced Analysis GUI with the output directory
        advanced_gui = AdvancedAnalysisGUI(root, output_dir, config_path)
        root.mainloop()
        logger.info("Advanced analysis GUI closed successfully")
    except Exception as e:
        logger.error(f"Error running advanced analysis: {e}")
        logger.error(traceback.format_exc())
        
        # Show error in GUI if possible
        try:
            import tkinter.messagebox as messagebox
            messagebox.showerror("Error", f"Failed to start Advanced Analysis:\n{str(e)}\n\nSee log for details.")
        except:
            print(f"ERROR: {str(e)}")
            print("See log for details.")

if __name__ == "__main__":
    logger.info("Starting Advanced Analysis standalone")
    run_advanced_analysis()
