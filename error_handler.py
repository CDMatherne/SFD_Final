#!/usr/bin/env python3
"""
Error Handler Module for SFD Project

[VERSION}
Team = Dreadnaught
Alex Giacomello, Christopher Matherne, Rupert Rigg, Zachary Zhao
version = 2.1 Beta

This module provides standardized error handling capabilities
for the Shipping Fraud Detection (SFD) project.
"""

import os
import sys
import logging
import traceback
from datetime import datetime
import platform
import json

# Configure module logger
logger = logging.getLogger(__name__)

class ErrorHandler:
    """
    Central error handler class for standardizing error handling across the SFD application.
    
    This class provides methods for handling various types of errors in a consistent way
    throughout the application, including logging, user-friendly messages, and recovery options.
    """
    
    def __init__(self, log_dir=None, config=None):
        """
        Initialize the error handler.
        
        Args:
            log_dir (str, optional): Directory to store error logs. Defaults to a 'logs' directory
                                    in the same directory as the script.
            config (dict, optional): Configuration parameters for error handling.
        """
        self.config = config or {}
        
        # Set up error log directory
        if log_dir:
            self.log_dir = log_dir
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.log_dir = os.path.join(script_dir, "logs")
            
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up error log file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.error_log_file = os.path.join(self.log_dir, f"error_log_{timestamp}.txt")
        
    def handle_exception(self, exception, context=None, fatal=False):
        """
        Handle an exception in a standardized way.
        
        Args:
            exception (Exception): The exception to handle
            context (str, optional): Context information about where the exception occurred
            fatal (bool, optional): Whether this is a fatal error that should terminate the program
            
        Returns:
            dict: Information about the handled exception
        """
        # Get exception details
        exc_type = type(exception).__name__
        exc_message = str(exception)
        exc_traceback = traceback.format_exc()
        
        # Log the exception
        if context:
            logger.error(f"Exception in {context}: {exc_type}: {exc_message}")
        else:
            logger.error(f"Exception: {exc_type}: {exc_message}")
        
        logger.debug(exc_traceback)
        
        # Save to error log file
        self._save_error_log(exc_type, exc_message, exc_traceback, context)
        
        # Create error info dict
        error_info = {
            'type': exc_type,
            'message': exc_message,
            'traceback': exc_traceback,
            'context': context,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'fatal': fatal,
            'system_info': {
                'platform': platform.system(),
                'platform_version': platform.version(),
                'python_version': sys.version
            }
        }
        
        # Terminate program if fatal
        if fatal:
            logger.critical(f"Fatal error encountered. Application will exit. See {self.error_log_file} for details.")
            if self.config.get('show_fatal_dialog', True):
                self._show_fatal_error_dialog(error_info)
            sys.exit(1)
            
        return error_info
    
    def handle_data_error(self, message, data_path=None, recoverable=True):
        """
        Handle an error related to data processing.
        
        Args:
            message (str): Error message
            data_path (str, optional): Path to the data file causing the error
            recoverable (bool, optional): Whether this error is recoverable
            
        Returns:
            dict: Information about the handled error
        """
        # Log the data error
        if data_path:
            logger.error(f"Data error with {data_path}: {message}")
        else:
            logger.error(f"Data error: {message}")
            
        # Create error info dict
        error_info = {
            'type': 'DataError',
            'message': message,
            'data_path': data_path,
            'recoverable': recoverable,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Save to error log file
        self._save_error_log('DataError', message, None, data_path)
        
        return error_info
    
    def handle_validation_error(self, field_name, expected_type, actual_value):
        """
        Handle a validation error for input data.
        
        Args:
            field_name (str): Name of the field with invalid data
            expected_type (str): Expected type or format
            actual_value: The invalid value
            
        Returns:
            dict: Information about the validation error
        """
        message = f"Validation error for {field_name}: Expected {expected_type}, got {actual_value} ({type(actual_value).__name__})"
        logger.warning(message)
        
        # Create error info dict
        error_info = {
            'type': 'ValidationError',
            'field': field_name,
            'expected_type': expected_type,
            'actual_value': str(actual_value),
            'actual_type': type(actual_value).__name__,
            'message': message,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return error_info
        
    def _save_error_log(self, error_type, message, traceback_str=None, context=None):
        """
        Save error information to log file.
        
        Args:
            error_type (str): Type of error
            message (str): Error message
            traceback_str (str, optional): Traceback string
            context (str, optional): Context information
        """
        try:
            with open(self.error_log_file, 'a') as f:
                f.write(f"==== ERROR REPORT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ====\n")
                f.write(f"Type: {error_type}\n")
                f.write(f"Message: {message}\n")
                
                if context:
                    f.write(f"Context: {context}\n")
                    
                if traceback_str:
                    f.write("Traceback:\n")
                    f.write(traceback_str)
                    
                f.write("\nSystem Information:\n")
                f.write(f"Platform: {platform.system()} {platform.version()}\n")
                f.write(f"Python: {sys.version}\n")
                f.write(f"Working Directory: {os.getcwd()}\n\n")
                
        except Exception as e:
            logger.error(f"Failed to write to error log file: {e}")
            
    def _show_fatal_error_dialog(self, error_info):
        """
        Show a dialog for fatal errors when running in GUI mode.
        
        Args:
            error_info (dict): Information about the error
        """
        try:
            import tkinter as tk
            from tkinter import messagebox
            
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            
            message = (
                f"A fatal error has occurred:\n\n"
                f"{error_info['type']}: {error_info['message']}\n\n"
                f"The application will now exit.\n\n"
                f"Error details have been saved to:\n{self.error_log_file}"
            )
            
            messagebox.showerror("Fatal Error", message)
            root.destroy()
            
        except Exception:
            # If we can't show a GUI dialog, just log the failure
            logger.error("Failed to display error dialog")


# Create a global error handler instance for convenience
default_error_handler = ErrorHandler()

def handle_exception(exception, context=None, fatal=False):
    """
    Convenience function to handle an exception using the default error handler.
    
    Args:
        exception (Exception): The exception to handle
        context (str, optional): Context information about where the exception occurred
        fatal (bool, optional): Whether this is a fatal error that should terminate the program
        
    Returns:
        dict: Information about the handled exception
    """
    return default_error_handler.handle_exception(exception, context, fatal)

def safe_execute(func, args=None, kwargs=None, context=None, default_return=None):
    """
    Safely execute a function with exception handling.
    
    Args:
        func (callable): The function to execute
        args (tuple, optional): Positional arguments to pass to the function
        kwargs (dict, optional): Keyword arguments to pass to the function
        context (str, optional): Context information for error reporting
        default_return: Value to return if the function raises an exception
        
    Returns:
        The return value of the function, or default_return if an exception occurs
    """
    args = args or ()
    kwargs = kwargs or {}
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        handle_exception(e, context)
        return default_return
