"""
Scottish Archaeological Finds Analysis and Visualization Tool

This script analyzes and visualizes archaeological finds in Scotland using
both heatmaps and scatter plots, with focus on different historical eras.

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

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.spatial import KDTree
from shapely.geometry import Point
from shapely.vectorized import contains
from tqdm import tqdm

from utils import *

# Constants
SCOTLAND_BBOX = (-8.8, -0.4, 54.5, 61)  # bounding box for Scotland (south, west, north, east)
DISTANCE_THRESHOLD_CITY = 10000  # meters
DISTANCE_THRESHOLD_REGION = 20000  # meters

# Historical eras with date ranges
ERAS_DATES = {
    'PREHISTORIC': '10000 BC - 4000 BC',
    'NEOLITHIC': '4000 BC - 2000 BC',
    'BRONZE AGE': '2000 BC - 800 BC',
    'IRON AGE': '800 BC - 43 AD',
    'ROMAN': '43 AD - 410 AD',
    'EARLY MEDIEVAL': '410 AD - 1066 AD'
}

# Key locations
EDINBURGH_COORDS = (-3.188267, 55.953251)
TOWNS = {
    'Traprian\nLaw': (-2.6667, 55.9667),
    'Dumfries': (-3.60667, 55.07083),
    'Perthshire': (-3.72971, 56.704361),
    'Orkney': (-3.225, 58.9847),
    'Aberdeenshire': (-2.7194, 57.1621)
}

CATEGORIES_FUNCTIONS_MAP = {
    'FORT': 'PROTECTIVE',
    'PALISADED ENCLOSURE': 'PROTECTIVE',
    'PALISADED SETTLEMENT': 'PROTECTIVE',
    'EARTHWORK': 'PROTECTIVE',
    
    'HUT CIRCLE': 'DOMESTIC',
    'ROUNDHOUSE': 'DOMESTIC',
    'ENCLOSED SETTLEMENT': 'DOMESTIC',
    'UNENCLOSED PLATFORM SETTLEMENT': 'DOMESTIC',
    'PLATFORM SETTLEMENT': 'DOMESTIC',
    'SOUTERRAIN': 'DOMESTIC',
    'CRANNOG': 'DOMESTIC',
            
    'LONG CIST': 'RITUAL',
    'STANDING STONE': 'RITUAL',
    'BURIAL CAIRN': 'RITUAL',
    'CUP MARKED STONE': 'RITUAL',
    'CAIRN': 'RITUAL',
    'CAIRNFIELD': 'RITUAL',
            
    'ENCLOSURE': 'AGRICULTURAL',
    'RING DITCH': 'AGRICULTURAL',
    'PIT ALIGNMENT': 'AGRICULTURAL',
    'CROPMARK': 'AGRICULTURAL',
    'FIELD BOUNDARY': 'AGRICULTURAL',
    'FIELD SYSTEM': 'AGRICULTURAL',
    'BANK': 'AGRICULTURAL',
    'RING DITCH HOUSE': 'AGRICULTURAL',
            
    'POST HOLE': 'OTHER',
    'UNIDENTIFIED FLINT': 'OTHER'
}


def plot_edinburgh_cities(edinburgh_coords, towns, ax, padding=0.05):
    """Plot Edinburgh and other towns with labels on the given axis."""
    # Plot Edinburgh
    text = ax.text(
        edinburgh_coords[0] - padding, 
        edinburgh_coords[1] - padding, 
        'Edinburgh', 
        fontsize=12, 
        color='black', 
        fontweight='bold', 
        ha='right', 
        va='bottom'
    )
    text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'), path_effects.Normal()])
    ax.scatter(edinburgh_coords[0] - padding, edinburgh_coords[1] - padding, s=4, color='black')
    
    # Plot other towns
    for town, coord in towns.items():
        ha = 'left' if town != 'Aberdeenshire' else 'right'
        text = ax.text(
            coord[0] + padding, 
            coord[1] - padding, 
            town, 
            fontsize=12, 
            color='black', 
            ha=ha, 
            va='bottom'
        )
        text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'), path_effects.Normal()])
        ax.scatter(coord[0] + padding, coord[1] - padding, s=4, color='black')


def table_cities_finds(gdf, eras_dates, cities):
    """
    Generate a table counting archaeological finds near cities for different eras.
    
    Args:
        gdf: GeoDataFrame with archaeological finds
        eras_dates: Dictionary mapping era names to date ranges
        cities: Dictionary mapping city names to coordinates
        
    Returns:
        DataFrame with cities as rows and eras as columns, values are find counts
    """
    gdf_proj = gdf.to_crs(DEFAULT_CRS)
    gdf_proj['centroid'] = gdf_proj.geometry.centroid
    results = {}
    
    for city, coords in tqdm(cities.items(), desc='Counting artifacts', ncols=150):
        city_point = gpd.GeoSeries([Point(coords)], crs=WGS84_CRS).to_crs(DEFAULT_CRS).iloc[0]
        
        for era in eras_dates:
            era_points = gdf_proj[gdf_proj['TYPE'].str.contains(era, case=False, na=False)].copy()
            era_points['distance'] = era_points['centroid'].distance(city_point)
            results[(city, era)] = era_points[era_points['distance'] < DISTANCE_THRESHOLD_REGION].shape[0]
    
    # Process results into a pivot table
    df_results = pd.DataFrame.from_dict(results, orient='index', columns=['Finds'])
    df_results.index = pd.MultiIndex.from_tuples(df_results.index, names=['City', 'Era'])
    df_results = df_results.reset_index()
    df_pivot = df_results.pivot(index='City', columns='Era', values='Finds').fillna(0)
    df_pivot = df_pivot.reindex(index=list(cities.keys()), columns=list(eras_dates.keys()))
    print(df_pivot)
    
    return df_pivot


def print_city_finds_summary(gdf, city_coords, city_name, eras_dates):
    """
    Print summary of archaeological finds near a city for different eras.
    
    Args:
        gdf: GeoDataFrame with archaeological finds
        city_coords: (lon, lat) coordinates of the city
        city_name: Name of the city
        eras_dates: Dictionary mapping era names to date ranges
    """
    gdf_proj = gdf.to_crs(DEFAULT_CRS)
    gdf_proj['centroid'] = gdf_proj.geometry.centroid
    city_point = gpd.GeoSeries([Point(city_coords)], crs=WGS84_CRS).to_crs(DEFAULT_CRS).iloc[0]
    
    print(f"Summary of archaeological finds near {city_name}:")
    for era, dates in eras_dates.items():
        era_points = gdf_proj[gdf_proj['TYPE'].str.contains(era, case=False, na=False)].copy()
        era_points['distance'] = era_points['centroid'].distance(city_point)
        era_points_city = era_points[era_points['distance'] < DISTANCE_THRESHOLD_CITY]
        print(f"  {era} ({dates}): {era_points_city.shape[0]} finds")
        
        collapse_keywords = ['UNENCLOSED', 'ARCHAEOLOGICAL']
        
        all_finds = []
        for index, row in era_points_city.iterrows():
            finds = row['TYPE'].split(', ')
            finds = [f for f in finds if era.lower() in f.lower()]
            finds = [finds.replace('(S)', '').split(' (')[0] for finds in finds]
            for keyword in collapse_keywords:
                finds = [find.replace(keyword, '').strip() for find in finds]
            
            label = f"\t{row['PID']}: ".ljust(10)
            print(f"{label}{f"\n{label}".join(finds)}")
            all_finds.extend(finds)
        
        # Plot a bar plot of find types
        find_counts = {find: all_finds.count(find) for find in set(all_finds)}
        find_counts_sorted = dict(sorted(find_counts.items(), key=lambda x: x[1], reverse=True))
        
        hatch_patterns = ["-", "/", "o"] * 10
        colors = ['lightgray', 'white', 'black', 'gray'] * 10
        
        functions = [CATEGORIES_FUNCTIONS_MAP.get(find, 'OTHER') for find in find_counts_sorted.keys()]
        functions_unique = sorted(list(set(CATEGORIES_FUNCTIONS_MAP.values())))
        
        # Create figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=300, gridspec_kw={'width_ratios': [9, 1]})

        # Configure left subplot
        axs[0].grid(axis='y')

        # Plot bar chart with bars colored and labeled according to function
        for i, (find, count) in enumerate(find_counts_sorted.items()):
            axs[0].bar(
                find, 
                count, 
                color=colors[functions_unique.index(functions[i])], 
                edgecolor='black', 
                hatch=hatch_patterns[functions_unique.index(functions[i])], 
                label=functions[i]
            )

        axs[0].set_xlabel("Find Type")
        axs[0].set_ylabel("Count")
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=90)

        # Ensure legend labels are unique and 'OTHER' is always last
        handles, labels = axs[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        by_label = {k: by_label[k] for k in sorted(by_label, key=lambda x: x if x != 'OTHER' else 'z', reverse=False)}
        axs[0].legend(by_label.values(), by_label.keys())

        # Plot stacked bar plot with proportions of functions (sum of counts for each function)
        function_counts = {function: sum([find_counts[find] for find in find_counts_sorted.keys() if CATEGORIES_FUNCTIONS_MAP.get(find, 'OTHER') == function]) for function in functions_unique}
        for i, function in enumerate(functions_unique):
            axs[1].bar(
                'Function\nProfile', 
                function_counts[function], 
                bottom=sum([function_counts[functions_unique[j]] for j in range(i)]), 
                color=colors[i], 
                edgecolor='black', 
                hatch=hatch_patterns[i], 
                label=function
            )
        # Remove ticks, spines
        axs[1].set_ylim(0, sum(function_counts.values()))
        # axs[1].set_xticks(['Function\nprofile'])
        axs[1].set_yticks([])
        for spine in axs[1].spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{era}_finds_{city_name}.png")
        plt.show()
        

def plot_roman_roads_and_forts(points):
    """Plot Roman roads and forts on a map of Scotland."""
    
    bg_image = plt.imread('scotland_map.png')
    extent = SCOTLAND_BBOX
    pad = 0.05
    points = read_shapefile_and_reproject('NRHE_Areas/NRHE_Areas.shp')
    
    roman_points = points[points['TYPE'].str.contains('ROMAN', case=False, na=False)].copy()
    roman_points['centroid'] = roman_points.geometry.centroid
    
    roman_road_points = roman_points[roman_points['TYPE'].str.contains('ROAD', case=False, na=False)].copy()
    roman_road_coordinates = roman_road_points['centroid'].apply(lambda x: (x.x, x.y)).tolist()
    # Connect roman roads into a network of lines
    tree = KDTree(roman_road_coordinates)

    # Find nearest neighbor paths
    lines = []
    DISTANCE_THRESHOLD_REGION = 0.2
    for point in roman_road_coordinates:
        dists, idxs = tree.query(point, k=3)
        for idx in idxs[1:]:
            # Only add line if less than threshold distance
            if euclidean_distance(*point, *roman_road_coordinates[idx]) < DISTANCE_THRESHOLD_REGION:
                lines.append([point, roman_road_coordinates[idx]])

    
    roman_fort_points = roman_points[roman_points['TYPE'].str.contains('FORT', case=False, na=False)].copy()
    roman_aqueduct_points = roman_points[roman_points['TYPE'].str.contains('AQUEDUCT', case=False, na=False)].copy()
    
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.imshow(bg_image, extent=extent, aspect='auto')
    
    for i, line in enumerate(lines):
        label = 'Roman road (NN reconstruction)' if i == 0 else None
        ax.plot(*zip(*line), color='black', linewidth=1, label=label, zorder=1)
    
    ax.scatter(roman_fort_points['centroid'].x, roman_fort_points['centroid'].y, s=10, label='Roman fort/wall', c='blue', zorder=2, marker='x')
    ax.scatter(roman_aqueduct_points['centroid'].x, roman_aqueduct_points['centroid'].y, s=10, c='orange', label='Roman aqueduct', zorder=2, marker='x')
    # ax.scatter(roman_road_points['centroid'].x, roman_road_points['centroid'].y, s=5, label='Roman Road')
    
    cramond_fort_coords = (-3.2967, 55.9774)
    ax.scatter(cramond_fort_coords[0], cramond_fort_coords[1], s=60, c='blue', marker='*', label='Cramond Fort', zorder=3)
    # ax.text(cramond_fort_coords[0] + pad, cramond_fort_coords[1] - pad, 'Cramond Fort', fontsize=14, color='orange', ha='left', va='bottom')#, fontweight='bold')
    
    scatter = ax.scatter(EDINBURGH_COORDS[0], EDINBURGH_COORDS[1], s=30, c='black', zorder=3)
    scatter.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])
    text = ax.text(EDINBURGH_COORDS[0] + pad, EDINBURGH_COORDS[1], 'Edinburgh', fontsize=12, color='black', ha='left', va='bottom', zorder=3)#, fontweight='bold')
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])
    
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title('Roman Roads and Forts in Scotland')
    
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    
    # ax.set_xlim(-5.5, -1.5)
    ax.set_ylim(55, 57)

    legend = ax.legend(unique.values(), unique.keys(), fontsize=12, frameon=True, loc='upper left')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1)
    legend.get_frame().set_boxstyle('Square')
    
    plt.tight_layout()
    plt.savefig('roman_roads_forts.png')


def scottish_cities_scatterplot(cities, bg_image, extent, labels=False):
    """Plot cities on a map of Scotland."""
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(bg_image, extent=extent, aspect='auto')
    
    coords = np.array(list(cities.values()))
    ax.scatter(coords[:, 0], coords[:, 1], s=3, c='r', linewidth=1)
    
    if labels:
        padding = 0.05
        for city, coord in cities.items():
            ax.text(coord[0] + padding, coord[1] - padding, 
                   city, fontsize=8, color='black', ha='left', va='bottom')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Scottish Cities Scatterplot')
    plt.tight_layout()
    plt.show()


def table_edinburgh_rank(gdf, eras_dates):
    """
    Calculate Edinburgh's rank among Scottish cities for archaeological finds.
    
    Returns:
        DataFrame showing Edinburgh's rank for different eras
    """
    gdf_proj = gdf.to_crs(DEFAULT_CRS)
    gdf_proj['centroid'] = gdf_proj.geometry.centroid
    
    cities_dict = fetch_scottish_cities(min_population=500)
    
    # Convert city coordinates to projected CRS
    cities_proj = {}
    for city, coords in cities_dict.items():
        point = Point(coords)
        proj_coords = gpd.GeoSeries([point], crs=WGS84_CRS).to_crs(DEFAULT_CRS).iloc[0].coords[0]
        cities_proj[city] = proj_coords
    
    # Count artifacts for each city and era
    counts = {}
    for city, coord in tqdm(cities_proj.items(), desc='Counting artifacts', ncols=150):
        city_point = Point(coord)
        era_counts = {}
        for era in eras_dates.keys():
            era_points = gdf_proj[gdf_proj['TYPE'].str.contains(era, case=False, na=False)].copy()
            era_points['distance'] = era_points['centroid'].distance(city_point)
            era_counts[era] = era_points[era_points['distance'] < DISTANCE_THRESHOLD_CITY].shape[0]
        counts[city] = era_counts
    
    df_counts = pd.DataFrame.from_dict(counts, orient='index')
    
    # Calculate rankings
    rankings = {}
    total = df_counts.shape[0]
    for era in df_counts.columns:
        era_rank = df_counts[era].rank(method='min', ascending=False).astype(int)
        rankings[era] = {city: f'{ordinal(era_rank[city])}/{total}' for city in df_counts.index}
    
    edinburgh_ranks = {era: rankings[era].get('Edinburgh', 'NA') for era in df_counts.columns}
    df_rank = pd.DataFrame(edinburgh_ranks, index=['Edinburgh'])
    
    print(df_rank)
    return df_rank


def create_era_visualizations(points, scotland_boundary, eras_dates):
    """Create heatmaps and scatter plots for different historical eras."""
    # Setup
    bg_image = plt.imread('scotland_map.png')
    extent = SCOTLAND_BBOX
    scotland_union = scotland_boundary.union_all()
    n_eras = len(eras_dates)
    
    # Create figure objects
    fig_heat, axs_heat = plt.subplots(1, n_eras, dpi=300, figsize=(3*n_eras, 5))
    fig_scatter, axs_scatter = plt.subplots(1, n_eras, dpi=300, figsize=(3*n_eras, 5))
    
    # Process each era
    for i, (era, dates) in enumerate(eras_dates.items()):
        # Filter points for this era
        era_points = points[points['TYPE'].str.contains(era, case=False, na=False)].copy()
        era_points['centroid'] = era_points.geometry.centroid
        era_points['lon'] = era_points['centroid'].x
        era_points['lat'] = era_points['centroid'].y
        
        x = era_points['lon'].values
        y = era_points['lat'].values
        
        # Handle single plot case
        ax_heat = axs_heat[i] if n_eras > 1 else axs_heat
        ax_scatter = axs_scatter[i] if n_eras > 1 else axs_scatter
        
        # Generate title with proper case formatting
        title = ' '.join(word.capitalize() for word in era.split()) + f'\n({dates})'
        
        # Create heatmap
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy, bw_method=0.1)
        grid_size_x = int((extent[1]-extent[0])*100)
        grid_size_y = int((extent[3]-extent[2])*100)
        xi, yi = np.mgrid[extent[0]:extent[1]:grid_size_x*1j, extent[2]:extent[3]:grid_size_y*1j]
        zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
        mask = ~contains(scotland_union, xi, yi)
        # zi[mask] = np.nan

        ax_heat.imshow(bg_image, extent=extent, aspect='auto')
        
        # Create alpha mask that varies with intensity
        norm = plt.Normalize(vmin=np.min(zi), vmax=np.max(zi))
        alpha_mask = norm(zi) ** 2 # Map values to alpha range 0-1
        masked_zi = np.ma.array(zi, mask=np.isnan(zi))
        ax_heat.imshow(np.flipud(masked_zi.T), extent=extent, cmap='magma', alpha=np.flipud(alpha_mask.T), aspect='auto')
        # ax_heat.set_ylim(55, 58)
        
        plot_edinburgh_cities(EDINBURGH_COORDS, TOWNS, ax_heat, padding=0.05)
        prepare_implot(ax_heat, spines_off=False)
        ax_heat.set_title(title, fontsize=14)

        # Create scatter plot
        ax_scatter.imshow(bg_image, extent=extent, aspect='auto')
        ax_scatter.scatter(x, y, s=0.5, color='orange', alpha=1)
        plot_edinburgh_cities(EDINBURGH_COORDS, TOWNS, ax_scatter, padding=0.05)
        prepare_implot(ax_scatter, spines_off=False)
        ax_scatter.set_title(title, fontsize=14)
    
    # Save figures
    fig_heat.tight_layout()
    fig_heat.savefig('kde_heatmap.png')
    
    fig_scatter.tight_layout()
    fig_scatter.savefig('scatter.png')


def main():
    """Main execution function."""
    # Load data
    points = read_shapefile_and_reproject('NRHE_Areas/NRHE_Areas.shp')
    scotland_boundary = read_shapefile_and_reproject('SP_regions_1999_2011/SP_regions_1999_2011.shp')
        
    # Generate statistics, tables, and visualizations
    # print_city_finds_summary(points, TOWNS['Dumfries'], 'Dumfries', ERAS_DATES)
    # print_city_finds_summary(points, TOWNS['Aberdeenshire'], 'Aberdeenshire', ERAS_DATES)
    # print_city_finds_summary(points, EDINBURGH_COORDS, 'Edinburgh', ERAS_DATES)
    # plot_roman_roads_and_forts(points)
    # table_edinburgh_rank(points, ERAS_DATES)
    # table_cities_finds(points, ERAS_DATES, {'Edinburgh': EDINBURGH_COORDS, **TOWNS})
        
    create_era_visualizations(points, scotland_boundary, ERAS_DATES)


if __name__ == '__main__':
    main()