"""

Data Sources
------------
1. National Record of the Historic Environment (NRHE) Areas
- Contains archaeological finds and sites in Scotland
- Shapefile with point data for each find
- See more at: https://www.historicenvironment.scot/archives-and-research/archaeology-and-heritage-collections/
2. Scottish Parliament Regions (1999-2011)
- Shapefile with boundaries of Scottish Parliament regions
- Used for filtering cities within Scotland
- See more at: https://www.nrscotland.gov.uk/statistics-and-data/geography/our-products/scottish-parliamentary-constituencies
3. OpenStreetMap (OSM) data
- Used for fetching Scottish cities and population data
- See more at: https://www.openstreetmap.org/

"""

import geopandas as gpd
import numpy as np
import overpy

DEFAULT_CRS = 'EPSG:27700'  # British National Grid
WGS84_CRS = 'EPSG:4326'  # WGS84 standard

def read_shapefile_and_reproject(path, target_crs=WGS84_CRS):
    """
    Read a shapefile and ensure it's in the target coordinate reference system.
    
    Args:
        path: Path to the shapefile
        target_crs: Target coordinate reference system (default: WGS84)
        
    Returns:
        GeoDataFrame in the target CRS
    """
    try:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf.set_crs(DEFAULT_CRS, inplace=True)
        if gdf.crs.to_string() != target_crs:
            gdf = gdf.to_crs(target_crs)
        return gdf
    except Exception as e:
        print(f"Error reading shapefile {path}: {e}")
        raise
    

def fetch_scottish_cities(min_population=1000, scotland_boundary=None):
    """
    Fetch Scottish cities from OpenStreetMap with population filtering.
    
    Args:
        min_population: Minimum population to include
        scotland_boundary: Optional Scotland boundary for filtering
        
    Returns:
        Dictionary mapping city names to (lon, lat) coordinates
    """
    api = overpy.Overpass()
    bbox = SCOTLAND_BBOX
    
    # Query for cities with population data
    query = f"""
    [out:json];
    (
      node["place"~"city|town|village|hamlet"]["population"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    );
    out center;
    """
    
    try:
        result = api.query(query)
        cities = {}
        
        # Process results
        for node in result.nodes:
            name = node.tags.get('name')
            population = node.tags.get('population')
            if name and population and int(population) >= min_population:
                cities[name] = (float(node.lon), float(node.lat))
        
        # Filter by Scotland boundary if provided
        if scotland_boundary is not None:
            scotland_union = scotland_boundary.union_all()
            cities_to_delete = []
            
            for city, coord in tqdm(cities.items(), desc='Filtering cities', ncols=150):
                if not contains(scotland_union, coord[0], coord[1]):
                    cities_to_delete.append(city)
                    
            for city in cities_to_delete:
                del cities[city]
        
        return cities
    
    except Exception as e:
        print(f"Error fetching cities: {e}")
        return {}
    

def prepare_implot(ax, spines_off=False):
    """Configure axis for map plotting."""
    ax.set_xticks([])
    ax.set_yticks([])
    if spines_off:
        for spine in ax.spines.values():
            spine.set_visible(False)
            
    
def ordinal(n): 
    """Convert integer to ordinal string (1st, 2nd, 3rd, etc.)."""
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f'{n}{suffix}'


def euclidean_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return ((x1-x2)**2 + (y1-y2)**2)**0.5