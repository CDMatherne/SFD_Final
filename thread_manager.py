#!/usr/bin/env python3
"""
Thread Manager Module for SFD Project

[VERSION}
Team = Dreadnaught
Alex Giacomello, Christopher Matherne, Rupert Rigg, Zachary Zhao
version = 2.1 Beta

This module provides a thread management system for the SFD project,
ensuring proper thread creation, monitoring, and cleanup.
"""

import os
import sys
import time
import logging
import threading
import queue
import concurrent.futures
from datetime import datetime
import traceback

# Configure module logger
logger = logging.getLogger(__name__)

class ThreadManager:
    """
    Thread management class for controlling concurrent operations.
    
    This class provides methods for creating, monitoring, and cleaning up threads,
    as well as a thread pool for efficient task execution.
    """
    
    def __init__(self, max_workers=None, thread_name_prefix="SFD-Worker"):
        """
        Initialize the thread manager.
        
        Args:
            max_workers (int, optional): Maximum number of worker threads. 
                                        If None, defaults to number of CPUs.
            thread_name_prefix (str, optional): Prefix for thread names
        """
        self.thread_name_prefix = thread_name_prefix
        self.active_threads = {}  # Dictionary to track active threads
        self.thread_results = {}  # Dictionary to store thread results
        self.thread_errors = {}   # Dictionary to store thread errors
        self.thread_lock = threading.RLock()  # Lock for thread dictionary operations
        
        # Create thread pool
        if max_workers is None:
            import multiprocessing
            max_workers = multiprocessing.cpu_count()
            
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix
        )
        
        # Create task queue for tasks that need to be run in sequence
        self.task_queue = queue.Queue()
        self._start_queue_worker()
        
        logger.info(f"Thread manager initialized with {max_workers} workers")
        
    def _start_queue_worker(self):
        """Start the queue worker thread to process tasks in sequence."""
        def worker():
            while True:
                try:
                    # Get the next task from the queue
                    task = self.task_queue.get()
                    
                    # Check for termination sentinel
                    if task is None:
                        self.task_queue.task_done()
                        break
                        
                    # Unpack task
                    task_id, func, args, kwargs, callback = task
                    
                    # Execute the task
                    try:
                        result = func(*args, **kwargs)
                        self.thread_results[task_id] = result
                        if callback:
                            callback(result)
                    except Exception as e:
                        logger.error(f"Error in queued task {task_id}: {e}")
                        self.thread_errors[task_id] = e
                    finally:
                        self.task_queue.task_done()
                        
                except Exception as e:
                    logger.error(f"Error in queue worker: {e}")
                    time.sleep(1)  # Avoid tight loop in case of persistent errors
                    
        # Start the queue worker thread
        thread = threading.Thread(target=worker, name=f"{self.thread_name_prefix}-QueueWorker", daemon=True)
        thread.start()
        
        # Track the queue worker thread
        with self.thread_lock:
            self.active_threads['queue_worker'] = thread
            
    def submit_task(self, func, args=None, kwargs=None, callback=None):
        """
        Submit a task to the thread pool for execution.
        
        Args:
            func (callable): The function to execute
            args (tuple, optional): Positional arguments to pass to the function
            kwargs (dict, optional): Keyword arguments to pass to the function
            callback (callable, optional): Function to call with the result
            
        Returns:
            concurrent.futures.Future: Future object representing the task
        """
        args = args or ()
        kwargs = kwargs or {}
        
        # Create a wrapper function to track results and errors
        def task_wrapper():
            task_id = threading.get_ident()
            try:
                # Track the thread
                with self.thread_lock:
                    self.active_threads[task_id] = threading.current_thread()
                
                # Execute the task
                result = func(*args, **kwargs)
                
                # Store the result
                self.thread_results[task_id] = result
                
                # Call the callback if provided
                if callback:
                    callback(result)
                    
                return result
                
            except Exception as e:
                logger.error(f"Error in task {threading.current_thread().name}: {e}")
                self.thread_errors[task_id] = e
                raise
            finally:
                # Remove the thread from active threads
                with self.thread_lock:
                    if task_id in self.active_threads:
                        del self.active_threads[task_id]
        
        # Submit the task to the thread pool
        return self.executor.submit(task_wrapper)
        
    def queue_task(self, func, args=None, kwargs=None, callback=None):
        """
        Queue a task to be executed in sequence.
        
        Args:
            func (callable): The function to execute
            args (tuple, optional): Positional arguments to pass to the function
            kwargs (dict, optional): Keyword arguments to pass to the function
            callback (callable, optional): Function to call with the result
            
        Returns:
            str: Task ID
        """
        args = args or ()
        kwargs = kwargs or {}
        
        # Generate a task ID
        task_id = f"task_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        # Add the task to the queue
        self.task_queue.put((task_id, func, args, kwargs, callback))
        
        return task_id
        
    def wait_for_tasks(self, timeout=None):
        """
        Wait for all tasks in the thread pool to complete.
        
        Args:
            timeout (float, optional): Maximum time to wait in seconds
            
        Returns:
            bool: True if all tasks completed, False if timed out
        """
        return self.executor.shutdown(wait=True, timeout=timeout)
        
    def wait_for_queue(self, timeout=None):
        """
        Wait for all queued tasks to complete.
        
        Args:
            timeout (float, optional): Maximum time to wait in seconds
            
        Returns:
            bool: True if all tasks completed, False if timed out
        """
        try:
            self.task_queue.join()
            return True
        except Exception as e:
            logger.error(f"Error waiting for queue: {e}")
            return False
            
    def get_active_threads(self):
        """
        Get information about active threads.
        
        Returns:
            dict: Dictionary with thread IDs as keys and thread info as values
        """
        with self.thread_lock:
            thread_info = {}
            for thread_id, thread in self.active_threads.items():
                thread_info[thread_id] = {
                    'name': thread.name,
                    'alive': thread.is_alive(),
                    'daemon': thread.daemon
                }
            return thread_info
            
    def shutdown(self, wait=True, timeout=None):
        """
        Shutdown the thread manager, stopping all threads.
        
        Args:
            wait (bool, optional): Whether to wait for threads to complete
            timeout (float, optional): Maximum time to wait in seconds
            
        Returns:
            bool: True if shutdown completed successfully
        """
        try:
            # Signal the queue worker to stop
            self.task_queue.put(None)
            
            # Wait for queue to empty if requested
            if wait:
                try:
                    self.task_queue.join()
                except Exception:
                    pass
                    
            # Shutdown the thread pool
            self.executor.shutdown(wait=wait, timeout=timeout)
            
            # Clear thread tracking
            with self.thread_lock:
                self.active_threads.clear()
                
            logger.info("Thread manager shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Error during thread manager shutdown: {e}")
            return False


# Create a global thread manager instance for convenience
thread_manager = ThreadManager()

def submit_task(func, args=None, kwargs=None, callback=None):
    """
    Submit a task to the global thread manager for execution.
    
    Args:
        func (callable): The function to execute
        args (tuple, optional): Positional arguments to pass to the function
        kwargs (dict, optional): Keyword arguments to pass to the function
        callback (callable, optional): Function to call with the result
        
    Returns:
        concurrent.futures.Future: Future object representing the task
    """
    return thread_manager.submit_task(func, args, kwargs, callback)

def queue_task(func, args=None, kwargs=None, callback=None):
    """
    Queue a task to be executed in sequence by the global thread manager.
    
    Args:
        func (callable): The function to execute
        args (tuple, optional): Positional arguments to pass to the function
        kwargs (dict, optional): Keyword arguments to pass to the function
        callback (callable, optional): Function to call with the result
        
    Returns:
        str: Task ID
    """
    return thread_manager.queue_task(func, args, kwargs, callback)

def shutdown_threads(wait=True, timeout=None):
    """
    Shutdown the global thread manager, stopping all threads.
    
    Args:
        wait (bool, optional): Whether to wait for threads to complete
        timeout (float, optional): Maximum time to wait in seconds
        
    Returns:
        bool: True if shutdown completed successfully
    """
    return thread_manager.shutdown(wait, timeout)
