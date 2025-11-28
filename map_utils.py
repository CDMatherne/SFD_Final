#!/usr/bin/env python3
"""
Map Utilities Module for SFD Project

[VERSION}
Team = Dreadnaught
Alex Giacomello, Christopher Matherne, Rupert Rigg, Zachary Zhao
version = 2.1 Beta

This module contains map-related utilities and classes for the SFD project,
including coordinate management and map display functions.
"""

import logging
import math
import folium
from folium.plugins import MarkerCluster, HeatMap
from branca.element import Element

# Set up module-level logger
logger = logging.getLogger(__name__)

class MapCoordinateManager:
    """
    Manages map coordinates and boundaries to ensure consistent map displays
    across different visualization functions.
    """
    
    def __init__(self):
        """Initialize with default (empty) boundaries."""
        self.min_lat = None
        self.max_lat = None
        self.min_lon = None
        self.max_lon = None
        self.is_initialized = False
        logger.info("MapCoordinateManager initialized")
    
    def calculate_boundaries(self, df):
        """
        Calculate map boundaries from a DataFrame containing LAT and LON columns.
        
        Args:
            df (DataFrame): DataFrame containing LAT and LON columns
            
        Returns:
            bool: True if boundaries were successfully calculated
        """
        if df is None or df.empty or 'LAT' not in df.columns or 'LON' not in df.columns:
            logger.warning("Cannot calculate boundaries from empty or invalid dataframe")
            return False
        
        # Add padding to ensure maps have some margin
        padding = 5
        
        # Calculate boundaries from the dataframe
        self.min_lat = df['LAT'].min() - padding
        self.max_lat = df['LAT'].max() + padding
        self.min_lon = df['LON'].min() - padding
        self.max_lon = df['LON'].max() + padding
        self.is_initialized = True
        
        logger.info(f"Map boundaries calculated: LAT [{self.min_lat} to {self.max_lat}], LON [{self.min_lon} to {self.max_lon}]")
        return True
    
    def get_boundaries(self):
        """
        Get the current map boundaries.
        
        Returns:
            tuple: (min_lat, max_lat, min_lon, max_lon) or None if not initialized
        """
        if not self.is_initialized:
            return None
        return (self.min_lat, self.max_lat, self.min_lon, self.max_lon)
    
    def get_center(self):
        """
        Get the center point of the current boundaries.
        
        Returns:
            tuple: (center_lat, center_lon) or None if not initialized
        """
        if not self.is_initialized:
            return None
        
        center_lat = (self.min_lat + self.max_lat) / 2
        center_lon = (self.min_lon + self.max_lon) / 2
        return (center_lat, center_lon)
    
    def is_valid(self):
        """Check if the current boundaries are valid."""
        return (self.is_initialized and 
                self.min_lat is not None and 
                self.max_lat is not None and 
                self.min_lon is not None and 
                self.max_lon is not None)


def add_lat_lon_grid_lines(m_or_fg, lat_start=-90, lat_end=90, lon_start=-180, lon_end=180, lat_step=10, lon_step=10, label_step=10):
    """
    Add latitude and longitude grid lines to a Folium map or FeatureGroup.
    
    Args:
        m_or_fg (folium.Map or folium.FeatureGroup): Folium map or feature group to add grid lines to
        lat_start (float): Starting latitude
        lat_end (float): Ending latitude
        lon_start (float): Starting longitude
        lon_end (float): Ending longitude
        lat_step (float): Step size for latitude lines
        lon_step (float): Step size for longitude lines
        label_step (int): Add labels every N degrees
        
    Returns:
        folium.Map: The map with grid lines added
    """
    # For backward compatibility, use m internally
    m = m_or_fg
    try:
        logger.info(f"Adding grid lines from lat {lat_start} to {lat_end}, lon {lon_start} to {lon_end}")
        
        # Set reasonable limits for grid boundaries to avoid poles and edge cases
        lat_range = [max(lat_start, -85), min(lat_end, 85)]  # Avoid poles
        lon_range = [max(lon_start, -175), min(lon_end, 175)]  # Limit range
        
        # Add custom CSS to style the grid lines
        css = '''
            .lat-line {
                stroke: #ff6b6b;
                stroke-width: 1.5;
                stroke-dasharray: 5, 5;
                opacity: 0.6;
            }
            .lon-line {
                stroke: #4ecdc4;
                stroke-width: 1.5;
                stroke-dasharray: 5, 5;
                opacity: 0.6;
            }
            .coordinate-label {
                background: rgba(255, 255, 255, 0.7);
                border-radius: 3px;
                padding: 2px;
                font-size: 10px;
                font-weight: bold;
                box-shadow: 1px 1px 2px rgba(0,0,0,0.3);
                color: #333;
            }
        '''
        # Only add CSS if this is a map, not a feature group
        if hasattr(m, 'get_root'):
            element = Element(f"<style>{css}</style>")
            m.get_root().html.add_child(element)
        
        # Add latitude lines with error handling
        for lat in range(math.ceil(lat_range[0]), math.floor(lat_range[1])+1, lat_step):
            try:
                # Add grid line
                points = [[lat, lon_range[0]], [lat, lon_range[1]]]
                folium.PolyLine(
                    points, 
                    tooltip=f"Latitude: {lat} degrees", 
                    color='#ff6b6b', 
                    weight=1.5, 
                    opacity=0.6, 
                    dash_array='5,5',
                    className='lat-line'
                ).add_to(m)
                
                # Add label every label_step degrees
                if lat % label_step == 0:
                    # Both left and right side labels
                    html_left = f'<div class="coordinate-label">{lat} deg N</div>'
                    folium.Marker(
                        [lat, lon_range[0]],
                        icon=folium.DivIcon(html=html_left)
                    ).add_to(m)
                    
                    # Right side label
                    html_right = f'<div class="coordinate-label">{lat} deg N</div>'
                    folium.Marker(
                        [lat, lon_range[1]],
                        icon=folium.DivIcon(html=html_right)
                    ).add_to(m)
            except Exception as e:
                logger.warning(f"Error adding latitude line at {lat}: {e}")
                continue  # Skip this line and continue
        
        # Add longitude lines with error handling
        for lon in range(math.ceil(lon_range[0]), math.floor(lon_range[1])+1, lon_step):
            try:
                # Add grid line
                points = [[lat_range[0], lon], [lat_range[1], lon]]
                folium.PolyLine(
                    points, 
                    tooltip=f"Longitude: {lon} degrees", 
                    color='#4ecdc4', 
                    weight=1.5, 
                    opacity=0.6, 
                    dash_array='5,5',
                    className='lon-line'
                ).add_to(m)
                
                # Add label every label_step degrees
                if lon % label_step == 0:
                    # Both bottom and top labels
                    html_bottom = f'<div class="coordinate-label">{lon} deg E</div>'
                    folium.Marker(
                        [lat_range[0], lon],
                        icon=folium.DivIcon(html=html_bottom)
                    ).add_to(m)
                    
                    # Top label
                    html_top = f'<div class="coordinate-label">{lon} deg E</div>'
                    folium.Marker(
                        [lat_range[1], lon],
                        icon=folium.DivIcon(html=html_top)
                    ).add_to(m)
            except Exception as e:
                logger.warning(f"Error adding longitude line at {lon}: {e}")
                continue  # Skip this line and continue
        
        logger.info("Successfully added grid lines to map")
        return m
    
    except Exception as e:
        # If anything goes wrong, log the error and return the map without grid lines
        logger.error(f"Failed to add grid lines to map: {e}")
        return m
