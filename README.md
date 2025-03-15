# Archaeological Investigation into Din Eidyn: from Prehistory to Early Medieval

---

### Data

| Folder Name               | Contents Description                                      | Size    | Access Link |
|---------------------------|----------------------------------------------------------|---------|-------------|
| NRHE_Areas               | Contains National Record of Historic Environment (NRHE) archaeological datapoints, including heritage sites, protected areas, and historical land use. | 474 MB  | [Access](https://www.historicenvironment.scot/archives-and-research/archaeology-and-heritage-collections/) |
| SP_regions_1999_2011     | Scottish Parliament regions from 1999 to 2011, representing administrative, political, or statistical boundaries. | 3 MB    | [Access](https://www.nrscotland.gov.uk/statistics-and-data/geography/our-products/scottish-parliamentary-constituencies) |

### Visualizations (included in paper)

---

Type and Function of NHRE Prehistoric Archaeological Finds <10km from the Modern Edinburgh City Center

<img width="468" alt="image" src="https://github.com/user-attachments/assets/3c33ea5f-af54-4c47-97b5-ab8dd9ce71f6" />

---

Type and Function of NHRE Prehistoric Archaeological Finds <10km from Aberdeenshire

<img width="468" alt="image" src="https://github.com/user-attachments/assets/68dd5820-9331-4d44-abdb-cac2c313aa3e" />

---

Location of NHRE Canmore Archaeological Sites (Yellow), by Era

<img width="468" alt="image" src="https://github.com/user-attachments/assets/4a2aab18-431d-44ef-b7eb-0e5caf63d14b" />

---

Location of Roman Roads and Forts from NHRE Canmore site data. Roads reconstructed from nearest-neighbors analysis of “Road”-designated sites.

<img width="468" alt="image" src="https://github.com/user-attachments/assets/3bdd9cb7-c590-412a-8312-12f9cd780b4a" />

---

Gaussian KDE Heatmap of NHRE Archaeological Sites, by Era

<img width="468" alt="image" src="https://github.com/user-attachments/assets/6f26fa68-e436-4861-9e79-3160d41e8c13" />

<img width="468" alt="image" src="https://github.com/user-attachments/assets/42126c85-0ec2-47fc-b91b-e912487a0d90" />

---


### Function descriptions

`read_shapefile_and_reproject(path, target_crs=WGS84_CRS)`

* Description: Reads a shapefile from the provided path, checks its coordinate reference system (CRS), sets a default if none exists, and reprojects it to the specified target CRS (default is WGS84).
* Output: Returns a GeoDataFrame in the target CRS.

`prepare_implot(ax, spines_off=False)`

* Description: Configures the plotting axis for map visualizations by removing axis ticks and optionally hiding the spines.
* Output: Modifies the passed Matplotlib axis (ax) for a cleaner map display.

`plot_edinburgh_cities(edinburgh_coords, towns, ax, padding=0.05)`

* Description: Plots the location of Edinburgh and other specified towns onto the provided axis with labeled points and a slight padding for clarity.
* Output: Draws labeled scatter points on the provided axis; the function does not return any value.

`euclidean_distance(x1, y1, x2, y2)`

* Description: Calculates the Euclidean distance between two points given their x and y coordinates.
* Output: Returns the Euclidean distance as a float.

`table_cities_finds(gdf, eras_dates, cities)`

* Description: Counts the number of archaeological finds (filtered by historical era) near each specified city using a GeoDataFrame, and formats these counts into a pivot table.
* Output: Returns a pandas DataFrame with cities as rows and historical eras as columns showing find counts.

`print_city_finds_summary(gdf, city_coords, city_name, eras_dates)`

* Description: Prints a summary of archaeological finds near a specified city for each historical era. It calculates distances, prints counts, lists find details, and generates bar plots to visualize the distribution of find types and their associated functions.
* Output: Prints textual summaries to the console and displays (and saves) bar plot visualizations; does not return any value.

`plot_roman_roads_and_forts(points)`

* Description: Visualizes Roman-era archaeological features by plotting roads (using a nearest-neighbor reconstruction), forts, aqueducts, and key landmarks on a map of Scotland. It uses a background image for context and annotates the map with labels.
* Output: Creates and saves a figure showing Roman roads and forts; does not return any value.

`ordinal(n)`

* Description: Converts an integer into its ordinal string representation (e.g., 1 becomes "1st", 2 becomes "2nd").
* Output: Returns a string representing the ordinal version of the input integer.

`fetch_scottish_cities(min_population=1000, scotland_boundary=None)`

* Description: Queries OpenStreetMap for cities, towns, villages, or hamlets in Scotland that meet a minimum population threshold. Optionally filters cities to those within a given Scotland boundary.
* Output: Returns a dictionary mapping city names to their (longitude, latitude) coordinates.

`scottish_cities_scatterplot(cities, bg_image, extent, labels=False)`

* Description: Plots the locations of Scottish cities as a scatter plot on a map using a provided background image and geographical extent. Optionally, city names can be added as labels.
* Output: Displays the scatter plot visualization; does not return any value.

`table_edinburgh_rank(gdf, eras_dates)`

* Description: Calculates the ranking of Edinburgh among Scottish cities based on the number of archaeological finds for each historical era. It projects city coordinates to a common CRS and compares find counts.
* Output: Returns a pandas DataFrame showing Edinburgh's rank (formatted as an ordinal out of the total) for each era.

`create_era_visualizations(points, scotland_boundary, eras_dates)`

* Description: Creates visualizations (both heatmaps and scatter plots) for different historical eras by filtering archaeological find data. It computes density estimates (using KDE), overlays data on a background map, and annotates key locations.
* Output: Saves and displays the generated heatmap and scatter plot figures; does not return any value.

`main()`

* Description: Acts as the primary execution function. It loads the necessary data, processes the shapefiles, and calls other functions.
* Output: Executes the data analysis and visualization process; does not return any value.
