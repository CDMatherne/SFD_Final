"""
Zone Geometry Utilities
Handles different geometry types for zones: polyline, polygon, rectangle, circle
"""

import math
from typing import List, Tuple, Dict, Any


def point_in_rectangle(lat: float, lon: float, lat_min: float, lat_max: float, 
                      lon_min: float, lon_max: float) -> bool:
    """Check if a point is inside a rectangle."""
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def point_in_circle(lat: float, lon: float, center_lat: float, center_lon: float, 
                   radius_meters: float) -> bool:
    """Check if a point is inside a circle (using Haversine formula)."""
    # Haversine formula to calculate distance
    R = 6371000  # Earth radius in meters
    
    lat1_rad = math.radians(lat)
    lat2_rad = math.radians(center_lat)
    delta_lat = math.radians(center_lat - lat)
    delta_lon = math.radians(center_lon - lon)
    
    a = (math.sin(delta_lat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    return distance <= radius_meters


def point_in_polygon(lat: float, lon: float, polygon_coords: List[Tuple[float, float]]) -> bool:
    """Check if a point is inside a polygon using ray casting algorithm."""
    if len(polygon_coords) < 3:
        return False
    
    n = len(polygon_coords)
    inside = False
    
    p1_lat, p1_lon = polygon_coords[0]
    for i in range(1, n + 1):
        p2_lat, p2_lon = polygon_coords[i % n]
        if lat > min(p1_lat, p2_lat):
            if lat <= max(p1_lat, p2_lat):
                if lon <= max(p1_lon, p2_lon):
                    if p1_lat != p2_lat:
                        xinters = (lat - p1_lat) * (p2_lon - p1_lon) / (p2_lat - p1_lat) + p1_lon
                    if p1_lon == p2_lon or lon <= xinters:
                        inside = not inside
        p1_lat, p1_lon = p2_lat, p2_lon
    
    return inside


def point_on_polyline(lat: float, lon: float, polyline_coords: List[Tuple[float, float]], 
                     tolerance_meters: float = 100) -> bool:
    """Check if a point is near a polyline (within tolerance distance)."""
    if len(polyline_coords) < 2:
        return False
    
    R = 6371000  # Earth radius in meters
    
    # Check distance to each segment
    for i in range(len(polyline_coords) - 1):
        p1_lat, p1_lon = polyline_coords[i]
        p2_lat, p2_lon = polyline_coords[i + 1]
        
        # Calculate distance from point to line segment
        distance = point_to_line_segment_distance(lat, lon, p1_lat, p1_lon, p2_lat, p2_lon, R)
        
        if distance <= tolerance_meters:
            return True
    
    return False


def point_to_line_segment_distance(lat: float, lon: float, 
                                   p1_lat: float, p1_lon: float,
                                   p2_lat: float, p2_lon: float,
                                   R: float = 6371000) -> float:
    """Calculate distance from a point to a line segment on Earth's surface."""
    # Convert to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    p1_lat_rad = math.radians(p1_lat)
    p1_lon_rad = math.radians(p1_lon)
    p2_lat_rad = math.radians(p2_lat)
    p2_lon_rad = math.radians(p2_lon)
    
    # Calculate distance from point to first endpoint
    d1 = haversine_distance(lat_rad, lon_rad, p1_lat_rad, p1_lon_rad, R)
    
    # Calculate distance from point to second endpoint
    d2 = haversine_distance(lat_rad, lon_rad, p2_lat_rad, p2_lon_rad, R)
    
    # Calculate distance between endpoints
    segment_length = haversine_distance(p1_lat_rad, p1_lon_rad, p2_lat_rad, p2_lon_rad, R)
    
    if segment_length == 0:
        return d1
    
    # Calculate the projection parameter
    # Using dot product approximation for small distances
    # For more accuracy, we'd need to use spherical geometry
    # This is a simplified version that works well for typical zone sizes
    
    # Calculate bearing from p1 to p2
    dlon = p2_lon_rad - p1_lon_rad
    y = math.sin(dlon) * math.cos(p2_lat_rad)
    x = (math.cos(p1_lat_rad) * math.sin(p2_lat_rad) - 
         math.sin(p1_lat_rad) * math.cos(p2_lat_rad) * math.cos(dlon))
    bearing = math.atan2(y, x)
    
    # Calculate bearing from p1 to point
    dlon_point = lon_rad - p1_lon_rad
    y_point = math.sin(dlon_point) * math.cos(lat_rad)
    x_point = (math.cos(p1_lat_rad) * math.sin(lat_rad) - 
               math.sin(p1_lat_rad) * math.cos(lat_rad) * math.cos(dlon_point))
    bearing_point = math.atan2(y_point, x_point)
    
    # Calculate cross-track distance
    cross_track_distance = math.asin(math.sin(d1 / R) * 
                                    math.sin(bearing_point - bearing)) * R
    
    # Check if projection is within segment
    along_track_distance = math.acos(math.cos(d1 / R) / math.cos(abs(cross_track_distance) / R)) * R
    
    if 0 <= along_track_distance <= segment_length:
        return abs(cross_track_distance)
    else:
        # Point is outside segment, return distance to nearest endpoint
        return min(d1, d2)


def haversine_distance(lat1_rad: float, lon1_rad: float, 
                      lat2_rad: float, lon2_rad: float, 
                      R: float = 6371000) -> float:
    """Calculate distance between two points using Haversine formula."""
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def point_in_zone(lat: float, lon: float, zone: Dict[str, Any]) -> bool:
    """Check if a point is inside/on a zone based on its geometry type."""
    geometry_type = zone.get('geometry_type', 'rectangle')
    
    if geometry_type == 'rectangle':
        return point_in_rectangle(
            lat, lon,
            zone['lat_min'], zone['lat_max'],
            zone['lon_min'], zone['lon_max']
        )
    
    elif geometry_type == 'circle':
        return point_in_circle(
            lat, lon,
            zone['center_lat'], zone['center_lon'],
            zone['radius_meters']
        )
    
    elif geometry_type == 'polygon':
        coords = zone.get('coordinates', [])
        if not coords:
            return False
        # Convert to list of tuples if needed
        if isinstance(coords[0], dict):
            coords = [(c['lat'], c['lon']) for c in coords]
        elif isinstance(coords[0], list):
            coords = [(c[0], c[1]) if len(c) >= 2 else (c[0], c[1]) for c in coords]
        return point_in_polygon(lat, lon, coords)
    
    elif geometry_type == 'polyline':
        coords = zone.get('coordinates', [])
        if not coords:
            return False
        # Convert to list of tuples if needed
        if isinstance(coords[0], dict):
            coords = [(c['lat'], c['lon']) for c in coords]
        elif isinstance(coords[0], list):
            coords = [(c[0], c[1]) if len(c) >= 2 else (c[0], c[1]) for c in coords]
        tolerance = zone.get('tolerance_meters', 100)
        return point_on_polyline(lat, lon, coords, tolerance)
    
    else:
        # Fallback to rectangle for unknown types
        return point_in_rectangle(
            lat, lon,
            zone.get('lat_min', -90), zone.get('lat_max', 90),
            zone.get('lon_min', -180), zone.get('lon_max', 180)
        )


def get_zone_bounds(zone: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Get bounding box (lat_min, lat_max, lon_min, lon_max) for any zone type."""
    geometry_type = zone.get('geometry_type', 'rectangle')
    
    if geometry_type == 'rectangle':
        return (zone['lat_min'], zone['lat_max'], zone['lon_min'], zone['lon_max'])
    
    elif geometry_type == 'circle':
        center_lat = zone['center_lat']
        center_lon = zone['center_lon']
        radius_meters = zone['radius_meters']
        
        # Approximate bounding box (not exact for large circles near poles)
        # 1 degree latitude ≈ 111 km
        # 1 degree longitude ≈ 111 km * cos(latitude)
        lat_degrees = radius_meters / 111000.0
        lon_degrees = radius_meters / (111000.0 * abs(math.cos(math.radians(center_lat))))
        
        return (
            center_lat - lat_degrees,
            center_lat + lat_degrees,
            center_lon - lon_degrees,
            center_lon + lon_degrees
        )
    
    elif geometry_type in ['polygon', 'polyline']:
        coords = zone.get('coordinates', [])
        if not coords:
            return (-90, 90, -180, 180)
        
        # Convert to list of tuples if needed
        if isinstance(coords[0], dict):
            lats = [c['lat'] for c in coords]
            lons = [c['lon'] for c in coords]
        elif isinstance(coords[0], list):
            lats = [c[0] for c in coords]
            lons = [c[1] for c in coords if len(c) >= 2]
        else:
            return (-90, 90, -180, 180)
        
        return (min(lats), max(lats), min(lons), max(lons))
    
    else:
        # Fallback
        return (
            zone.get('lat_min', -90),
            zone.get('lat_max', 90),
            zone.get('lon_min', -180),
            zone.get('lon_max', 180)
        )

