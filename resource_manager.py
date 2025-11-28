#!/usr/bin/env python3
"""
Resource Manager Module for SFD Project

[VERSION}
Team = Dreadnaught
Alex Giacomello, Christopher Matherne, Rupert Rigg, Zachary Zhao
version = 2.1 Beta

This module provides resource management capabilities for the SFD project,
including memory monitoring, file handles, and other system resources.
"""

import os
import sys
import logging
import gc
import time
import threading
import weakref
from datetime import datetime
import traceback

# Configure module logger
logger = logging.getLogger(__name__)

class ResourceManager:
    """
    Resource management class for monitoring and controlling system resources.
    
    This class provides methods for monitoring memory usage, managing file handles,
    and controlling other system resources used by the application.
    """
    
    def __init__(self, memory_threshold=80, check_interval=60):
        """
        Initialize the resource manager.
        
        Args:
            memory_threshold (int, optional): Memory usage percentage threshold for warnings
            check_interval (int, optional): Interval in seconds between resource checks
        """
        self.memory_threshold = memory_threshold
        self.check_interval = check_interval
        
        self.open_files = weakref.WeakValueDictionary()  # Track open file handles
        self.open_files_lock = threading.RLock()
        
        self.running = False
        self.monitor_thread = None
        
        # Determine if psutil is available for better resource monitoring
        try:
            import psutil
            self.psutil = psutil
            logger.info("Using psutil for enhanced resource monitoring")
        except ImportError:
            self.psutil = None
            logger.info("psutil not available, using basic resource monitoring")
            
    def start_monitoring(self):
        """Start the resource monitoring thread."""
        if self.running:
            logger.info("Resource monitoring already running")
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_resources,
            name="ResourceMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
        
    def stop_monitoring(self):
        """Stop the resource monitoring thread."""
        if not self.running:
            return
            
        self.running = False
        if self.monitor_thread:
            try:
                self.monitor_thread.join(timeout=5)
            except Exception:
                pass
            self.monitor_thread = None
            
        logger.info("Resource monitoring stopped")
        
    def _monitor_resources(self):
        """Monitor system resources periodically."""
        while self.running:
            try:
                mem_info = self.get_memory_usage()
                if mem_info.get('percent', 0) > self.memory_threshold:
                    logger.warning(f"High memory usage: {mem_info.get('percent')}%")
                
                # Count open file handles
                with self.open_files_lock:
                    open_file_count = len(self.open_files)
                    if open_file_count > 100:  # Arbitrary threshold
                        logger.warning(f"High number of open files: {open_file_count}")
                
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
                
            # Sleep until next check
            for _ in range(self.check_interval * 10):  # Check for stop signal every 1/10 sec
                if not self.running:
                    break
                time.sleep(0.1)
                
    def get_memory_usage(self):
        """
        Get current memory usage information.
        
        Returns:
            dict: Memory usage statistics
        """
        if self.psutil:
            try:
                process = self.psutil.Process(os.getpid())
                mem_info = process.memory_info()
                return {
                    'rss': mem_info.rss,
                    'rss_mb': mem_info.rss / (1024 * 1024),
                    'vms': mem_info.vms,
                    'vms_mb': mem_info.vms / (1024 * 1024),
                    'percent': process.memory_percent()
                }
            except Exception as e:
                logger.error(f"Error getting memory usage with psutil: {e}")
        
        # Fallback to basic info
        import platform
        if platform.system() == "Windows":
            # Only basic info available without psutil on Windows
            return {'rss_mb': 'N/A', 'vms_mb': 'N/A', 'percent': 'N/A'}
        else:
            try:
                # Try to get memory info from /proc/self/status on Unix-like systems
                mem_info = {'rss_mb': 'N/A', 'vms_mb': 'N/A', 'percent': 'N/A'}
                
                try:
                    with open('/proc/self/status', 'r') as f:
                        for line in f:
                            if 'VmRSS:' in line:
                                mem_info['rss_mb'] = float(line.split()[1]) / 1024
                            elif 'VmSize:' in line:
                                mem_info['vms_mb'] = float(line.split()[1]) / 1024
                except Exception:
                    pass
                    
                return mem_info
            except Exception:
                return {'rss_mb': 'N/A', 'vms_mb': 'N/A', 'percent': 'N/A'}
                
    def register_file(self, file_obj, file_path):
        """
        Register an open file with the resource manager for tracking.
        
        Args:
            file_obj: The file object
            file_path (str): Path to the file
            
        Returns:
            file_obj: The original file object
        """
        with self.open_files_lock:
            self.open_files[id(file_obj)] = file_obj
            
        return file_obj
        
    def safe_open(self, file_path, mode='r', **kwargs):
        """
        Safely open a file with tracking.
        
        Args:
            file_path (str): Path to the file
            mode (str): File open mode
            **kwargs: Additional arguments to pass to open()
            
        Returns:
            file: The opened file object
        """
        try:
            f = open(file_path, mode, **kwargs)
            return self.register_file(f, file_path)
        except Exception as e:
            logger.error(f"Error opening file {file_path}: {e}")
            raise
            
    def force_gc(self):
        """Force garbage collection to free memory."""
        gc.collect()
        logger.info("Forced garbage collection completed")
        
    def clear_memory(self):
        """
        Attempt to free as much memory as possible.
        
        Returns:
            dict: Memory usage before and after cleaning
        """
        # Get memory usage before cleaning
        before = self.get_memory_usage()
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
            
        # Get memory usage after cleaning
        after = self.get_memory_usage()
        
        logger.info(f"Memory cleanup: Before {before.get('rss_mb')} MB, After {after.get('rss_mb')} MB")
        
        return {'before': before, 'after': after}


# Create a global resource manager instance for convenience
resource_manager = ResourceManager()

def get_memory_usage():
    """
    Get current memory usage information using the global resource manager.
    
    Returns:
        dict: Memory usage statistics
    """
    return resource_manager.get_memory_usage()

def safe_open(file_path, mode='r', **kwargs):
    """
    Safely open a file with tracking using the global resource manager.
    
    Args:
        file_path (str): Path to the file
        mode (str): File open mode
        **kwargs: Additional arguments to pass to open()
        
    Returns:
        file: The opened file object
    """
    return resource_manager.safe_open(file_path, mode, **kwargs)

def start_monitoring():
    """Start resource monitoring with the global resource manager."""
    resource_manager.start_monitoring()

def stop_monitoring():
    """Stop resource monitoring with the global resource manager."""
    resource_manager.stop_monitoring()

def force_gc():
    """Force garbage collection with the global resource manager."""
    resource_manager.force_gc()

def clear_memory():
    """Attempt to free memory with the global resource manager."""
    return resource_manager.clear_memory()
