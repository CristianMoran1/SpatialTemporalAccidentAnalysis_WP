# San Antonio Crash Risk Explorer (Spatial + Temporal)

This project was co-developed under the guidance of my research mentor (co-advised) as part of an ongoing research effort on traffic safety in San Antonio. It turns San Antonio crash data into an interactive experience where you can explore spatial patterns, time trends, and high risk areas. The goal is to make crash risk easier to see and talk about, especially when it overlaps with infrastructure gaps and neighborhood level conditions.

## What it can do
* Explore crash hotspots on an interactive map
* Look at trends over time (depending on what fields exist in the dataset)
* Compare patterns across boundary layers (council districts, growth areas, other cities/towns)
* Overlay context layers like streets and speed humps
## Visuals (Interactive Dashboard)

These are screenshots of the interactive dashboard (hover tooltips, zoom/pan, and filtering via the year slider + toggles).

### 2021 Snapshot (Map + District Counts)
![2021 Map and Bar Chart](assets/map_and_bar_2021.png)

### Trends Across Time (2001–2021)
![Trends](assets/trends_2001_2021.png)

### Hotspots View
![Hotspots](assets/hotspots_map.png)

### Safety + Equity (Poverty + Micromorts)
![Poverty and Micromorts](assets/poverty_and_micromorts.png)

## What is in this repo
* `main.py` runs the workflow and builds the map output
* `Council_Districts.geojson` boundary layer
* `InclusiveGrowthAreas.geojson` boundary layer
* `Streets.geojson` street layer
* `Traffic_Speed_Humps.geojson` infrastructure layer
* `Other_Cities_Towns.geojson` additional boundary layer

## Quick start (run the dashboard locally)

This project is a Dash + Plotly web app. Once it’s running, you can explore crashes across San Antonio with a year slider, layer toggles, and linked charts.

### Setup + run
```bash
# 1) Clone + enter repo
git clone https://github.com/CristianMoran1/SpatialTemporalAccidentAnalysis_WP.git
cd SpatialTemporalAccidentAnalysis_WP

# 2) Create + activate virtual environment
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\activate

# macOS/Linux (use this instead of the Windows line above)
# source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run the app
python main.py

# 5) Open locally in your browser
example: http://127.0.0.1:8050/
