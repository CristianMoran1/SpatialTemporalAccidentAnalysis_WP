# San Antonio Crash Risk Explorer (Spatial + Temporal)

This project was co-developed under the guidance of my research mentor (co-advised) as part of an ongoing research effort on traffic safety in San Antonio. It turns San Antonio crash data into an interactive experience where you can explore spatial patterns, time trends, and high risk areas. The goal is to make crash risk easier to see and talk about, especially when it overlaps with infrastructure gaps and neighborhood level conditions.

## What it can do
* Explore crash hotspots on an interactive map
* Look at trends over time (depending on what fields exist in the dataset)
* Compare patterns across boundary layers (council districts, growth areas, other cities/towns)
* Overlay context layers like streets and speed humps

## What is in this repo
* `main.py` runs the workflow and builds the map output
* `Council_Districts.geojson` boundary layer
* `InclusiveGrowthAreas.geojson` boundary layer
* `Streets.geojson` street layer
* `Traffic_Speed_Humps.geojson` infrastructure layer
* `Other_Cities_Towns.geojson` additional boundary layer
