import pandas as pd
import os
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import json
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
import numpy as np
from dash import html
from dash import dash_table


# Define variables globally
lat_col, lon_col, district_column_name = None, None, None

# Define a dictionary to store extracted fatalities count for each year
fatalities_by_year = {}

# Variable to store the latest triggered year
latest_triggered_year = 2021

# Define a dictionary to store accumulated district accident counts across 4 years
accumulated_district_counts = {}

# Loaded GeoJSON data for council districts from file
council_districts_geojson = gpd.read_file("Council_Districts.geojson")

# Define constants
DATA_DIR = rf"C:\Users\cmora\OneDrive\Bexarcounty_Data_Extraction"
MIN_YEAR = 2000
MAX_YEAR = 2021

# Function to update accumulated district counts
def update_accumulated_district_counts(district_counts):
    global accumulated_district_counts

    # Iterate over district counts and update accumulated counts
    for district, count in district_counts.items():
        accumulated_district_counts[district] = accumulated_district_counts.get(district, 0) + count


# Function to calculate total number of accidents for each district for each year
def calculate_district_accidents_by_year():
    district_accidents_by_year = {}

    # Iterate over the years
    for year in range(2000, 2022):
        # Process data for the year
        _, _, district_counts, _ = process_data(year, cached_data)

        # Update district_accidents_by_year with district_counts
        if not district_counts.empty:
            for district, count in district_counts.items():
                district_accidents_by_year.setdefault(district, {})[year] = count

    return district_accidents_by_year

def process_data(selected_year, cached_data):
    global lat_col, lon_col, district_column_name  # Add these lines to use the global variables

    # Check if data for the selected year is already cached
    if selected_year in cached_data:
        return cached_data[selected_year]

    # Constructed the file path with 'BEXAR'
    filePath = rf"Bexarcounty_Data_Extraction\{selected_year}_bexar_county.csv"

    # Checked if the file exists before reading it
    if os.path.exists(filePath):
        # Read the CSV file for the selected year into a pandas DataFrame
        bexar_county_year_data = pd.read_csv(filePath, encoding='ISO-8859-1', low_memory=False)

        # Checked if 'STATENAME' column exists and assigned the appropriate column name
        if 'STATE' in bexar_county_year_data.columns:
            statename_column = 'STATE'
        else:
            statename_column = 'STATE'

        # Checked if 'COUNTYNAME' column exists and assigned the appropriate column name
        if 'CITY' in bexar_county_year_data.columns:
            city_column = 'CITY'
        else:
            city_column = 'CITY'

        # Defined possible column names for latitude and longitude
        lat_lon_columns = ['LATITUDE', 'LATITUD', 'LATITUDENAME', 'Latitude', 'latitude', 'LAT', 'LATNAME',
                           'LONGITUDE', 'LONGITUD', 'LONGITUDENAME', 'Longitude', 'longitude', 'longitud', 'LON', 'LONNAME']

        # Found existing latitude and longitude columns
        lat_col, lon_col = None, None
        for col in lat_lon_columns:
            if col in bexar_county_year_data.columns:
                if col.lower().startswith('lat'):
                    lat_col = col
                elif col.lower().startswith('lon'):
                    lon_col = col

        # If 'LATITUDENAME' is not found, use the first latitude column available
        if lat_col is None and 'LATITUDENAME' in bexar_county_year_data.columns:
            lat_col = 'LATITUDENAME'

        # If 'LONGITUDENAME' is not found, use the first longitude column available
        if lon_col is None and 'LONGITUDENAME' in bexar_county_year_data.columns:
            lon_col = 'LONGITUDENAME'

        # Adjusted filtering conditions for the city of San Antonio
        bexar_texas_data = bexar_county_year_data[
            ((bexar_county_year_data['CITY'] == 6090) | (
                bexar_county_year_data[city_column].astype(str).str.contains('San Antonio'))) &
            ((bexar_county_year_data['STATE'] == 48) | (bexar_county_year_data[statename_column] == '48'))
        ]

        # Checked if latitude and longitude columns exist before creating GeoDataFrame
        if lat_col is not None and lon_col is not None:
            # Created a GeoDataFrame from the accident data
            gdf_accidents = gpd.GeoDataFrame(bexar_texas_data,
                                              geometry=gpd.points_from_xy(bexar_texas_data[lon_col],
                                                                          bexar_texas_data[lat_col]),
                                              crs="EPSG:4326")
            # Extracting fatalities for San Antonio, TX
            san_antonio_fatalities = bexar_texas_data['FATALS'].sum()

            # Spatial join to assign each accident to a district
            gdf_accidents = gpd.sjoin(gdf_accidents, council_districts_geojson, how="left", op="within")

            # Identified the correct column for districts in the accident dataset
            district_column_name = 'District'

            # Counted the number of accidents in each district
            district_counts = gdf_accidents[district_column_name].value_counts().nlargest(10)

            # Reindexed to include districts with count 0
            district_counts = district_counts.reindex(council_districts_geojson['District'], fill_value=0)

            # Calculated the total number of accidents for the year
            total_accidents = len(gdf_accidents)

            # Update accumulated district counts
            update_accumulated_district_counts(district_counts)

            # Printed to console and saved to a text file
            output_text = "San Antonio's 10 Districts with Accident Counts:\n" + district_counts.reset_index().rename(
                columns={'index': 'District', 'District': 'Accident Count'}).to_string(index=False)
            output_text += f"\n\nTotal Accidents for the Year: {total_accidents}"

            # Cache the processed data for the selected year
            cached_data[selected_year] = (gdf_accidents, council_districts_geojson, district_counts, output_text)

            return gdf_accidents, council_districts_geojson, district_counts, output_text

        else:
            print("Latitude or Longitude column not found. Skipping map plotting.")

    else:
        print(f"No data found for the selected year: {selected_year}")
        return None, None, None, None


# Loaded GeoJSON data for other cities and towns from file
other_cities_towns_geojson = gpd.read_file("Other_Cities_Towns_.geojson")

# List of towns to keep
towns_to_keep = ['Leon Valley', 'Castle Hills', 'Alamo Heights', 'Olmos Park', 'Shavano Park', 'Hollywood Park', 'Hill Country Village', 'Windcrest', 'Kirby','Balcones Heights', 'Terrell Hills']

# Filter other_cities_towns_geojson to include only specified towns
filtered_other_cities_towns_geojson = other_cities_towns_geojson[other_cities_towns_geojson['Name'].isin(towns_to_keep)]

# Initialize Dash app
app = dash.Dash(__name__)

# Define your research summary
research_summary = """
This website presents a comprehensive analysis of accident data within San Antonio, Texas, for the last 20 years (between 2001 and 2021),with a specific focus on the city's council districts. Utilizing information sourced from Bexar County records, encompassing reported accidents within the city limits, the analysis aims to discern prevailing accident trends across various regions, as well as to explore the relationship between accidents and demographic factors, such as poverty. The primary objective of this research is to furnish valuable insights into the distribution of accidents throughout San Antonio, pinpointing hotspots necessitating heightened attention from municipal authorities. Through this endeavor, city officials can gain a nuanced understanding of accident patterns, in order to facilitate informed decision-making for the implementation of targeted interventions aimed at enhancing public safety within the city and surrounding areas.
"""

# Define the layout for the research summary section
research_summary_layout = html.Div(style={'backgroundColor': '#000000', 'color': '#FFFFFF', 'padding': '20px'}, children=[
    html.H2("Executive Summary", style={'textAlign': 'center'}),
    dcc.Markdown(research_summary)
])

# Define the layout for the correlation section
correlation_section_layout = html.Div(style={'padding': '20px', 'display': 'flex', 'flex-direction': 'row'}, children=[
    # Left side containing the charts
    html.Div(style={'flex': '1', 'margin-right': '0px'}, children=[
        html.H2("Patterns identified throughout the years", style={'textAlign': 'center', 'color': '#FFFFFF', 'margin-bottom': '40px'}),
        dcc.Graph(id='district-accidents-graph'),  # Placeholder for district accidents chart
    ]),
    # Right side containing the paragraph
    html.Div(style={'flex': '1'}, children=[
        html.P("This chart presents the comprehensive overview of accidents by district spanning from 2001 to 2021. Our analysis reveals that District 1 (Northwest side), District 2 (Eastside), District 3 (Southeast side), and District 5 (West inner side)  emerged as the areas with the highest incidence of accidents compared to others. This  data was obtained from the San Antonio Open Database, ensuring its accuracy and reliability, and is regularly updated to reflect the latest redistricting information. Moreover, our study incorporates fatal crash data sourced from the National Highway Traffic Safety Administration. Employing a meticulous process, we sorted latitude and longitude coordinates of accidents occurring within these districts over the past two decades. This rigorous approach enhances the precision of our findings, enabling a deeper understanding of the trends and patterns associated with road safety in these specific areas.", style={'color': '#FFFFFF', 'font-size': '20px'}),
        html.P("Coupled with the district-wise analysis, our study delves deeper into understanding the root causes and contributing factors behind these accidents. By examining various parameters such as road conditions, traffic density, and demographic characteristics, we aim to provide actionable insights for enhancing road safety measures and reducing the incidence of accidents in these high-risk areas." ,style={'color': '#FFFFFF', 'font-size': '20px'})
    ])
])



# Define the data for the new table
data_table = {
    'District #': ['District 1 (Central & North side)', 'District 2 (Central & East side)', 'District 3 (Central & Southeast side)', 'District 4 (Southwest side)', 'District 5 (Central and West side)', 'District 6 (West side)', 'District 7 (West & Northwest side)', 'District 8 (Northwest side)', 'District 9 (North side)', 'District 10 (Northeast side)', 'Total'],
    'Accidents': ['12.31%', '16.00%', '11.33%', '10.32%', '11.83%', '8.20%', '7.33%', '8.08%', '5.80%', '9.00%', '2801' ],
    'Fatalities': ['11.65%', '12.57%', '6.25%', '6.74%', '15.36%', '11.00%', '11.39%', '8.27%', '8.63%', '8.14%', '2817'],
    'Micromorts y = (k/p) * (1000000/p)': ['2325 out of 1000000', '2475 out of 1000000', '1250 out of 1000000', '1397 out of 1000000', '3065 out of 1000000', '1933 out of 1000000', '2194 out of 1000000', '1605 out of 1000000', '1682 out of 1000000', '1550 out of 1000000', '19386 out of 1000000']
}


# Create a DataFrame for the new table
df_table = pd.DataFrame(data_table)

# Update the layout for the new table
table_layout = html.Div([
    html.Div([
        html.H2("Analysis",
                style={'textAlign': 'center', 'color': '#FFFFFF', 'margin-bottom': '20px', 'margin-top': '10px'}),
        dash_table.DataTable(
            id='new-summary-table',
            columns=[{'name': col, 'id': col} for col in df_table.columns],
            data=df_table.to_dict('records'),
            style_table={'height': '800px', 'width': '700px', 'overflowY': 'auto'}
        ),
    ], style={'flex': '1', 'margin-right': '20px', 'margin-bottom': '0px', 'vertical-align': 'top'}),

    html.Div([
        html.P(
            "Upon the inspection of the data analyzed and gathered, we were able to identify that out of the total 2701 accidents that occurred over the span of two decades in all of the districts, these particular 4 districts consisted of 52% of the total accidents, which is more than half.",
            style={'color': '#FFFFFF', 'font-size': '20px', 'margin-left': '20px'}),
        html.P(
            "Furthermore, we were able to identify the total number of fatalities resulting from accidents over the two decades, amounting to 2147 deaths. Among these, Districts 1, 2, 3, and 5 accounted for a combined total of 984 fatalities, while the remaining districts collectively reported 1163 fatalities. District 1 represented 14.2% of the total fatalities, District 2 contributed to 10.35% of the total fatalities, District 3 consisted of 8.3%, and District 5 accounted for 13.51% of all fatalities across the districts. This distribution highlights significant differences in fatality rates among the districts. Comparatively, the average percentage of fatalities for the other districts, excluding Districts 1, 2, 3, and 5, was calculated to be approximately 54%. This analysis highlights substantial variations in fatality rates between the specified districts and the rest.",
            style={'color': '#FFFFFF', 'font-size': '20px', 'margin-left': '20px'}),
        html.P("Additionally, based on the tabulated data, we calculated micromorts for each district. Micromorts, derived from the formula y = (k/p) * (1000000/p), represent a measure of risk in one-in-a-million terms. The micromorts for each district are as follows: District 1 (0.0153), District 2 (0.0108), District 3 (0.009), District 5 (0.0146), and the average micromorts for the other six districts (0.0015), resulting in an overall average of 0.00872 micromorts. These micromort figures offer valuable insights into the comparative risk levels across different districts.",
            style={'color': '#FFFFFF', 'font-size': '20px', 'margin-left': '20px'}),
    ], style={'flex': '1', 'margin-bottom': '0px', 'vertical-align': 'top'})
], style={'display': 'flex', 'align-items': 'flex-start', 'justify-content': 'center'})



# Define the layout for the accident hotspots section
accident_hotspots_section_layout = html.Div(style={'padding': '20px', 'display': 'flex', 'flex-direction': 'row', 'margin-bottom': '0px', 'margin-top': '0px'}, children=[
    # Left side containing the map
    html.Div(style={'flex': '1', 'margin-right': '20px', 'margin-bottom': '0px'}, children=[
        html.H2("Accident Hotspots Map", style={'textAlign': 'center', 'color': '#FFFFFF', 'margin-bottom': '20px'}),
        dcc.Graph(id='hotspot-map', figure=go.Figure(), style={'height': '100%', 'width': '100%'}),  # Placeholder for accident hotspots map
    ]),
    # Right side containing the paragraph
    html.Div(style={'flex': '1', 'margin-bottom': '0px'}, children=[
        html.P("This map provides a visual representation of accident hotspots across districts 1, 2, 3, and 5 which had the highest amount of accidents and fatalities from 2001 to 2021. The heatmap overlay highlights areas with a higher concentration of accidents, enabling a quick identification of potential risk zones. The data for this map is sourced from the San Antonio Open Database and is continuously updated to reflect the latest accident reports.", style={'color': '#FFFFFF', 'font-size': '20px', 'justify-content': 'center'}),
        html.P("In addition to identifying accident hotspots, our analysis also includes a comprehensive examination of contributing factors such as road conditions, traffic density, and environmental variables. By integrating these insights, we aim to develop effective strategies for enhancing road safety and reducing the occurrence of accidents in high-risk areas. Some of the strategies we consider include implementing traffic calming measures such as speed bumps or roundabouts, enhancing road signage and visibility, improving infrastructure for pedestrian and cyclist safety, implementing stricter enforcement of traffic laws, and launching educational campaigns to promote safe driving behaviors. These strategies, when implemented in conjunction with data-driven insights, can significantly contribute to reducing the frequency and severity of accidents in targeted areas.", style={'color': '#FFFFFF', 'font-size': '20px', 'justify-content': 'center'}),
        html.P("Furthermore, the clustering algorithm employed in this analysis, known as DBSCAN (Density-Based Spatial Clustering of Applications with Noise), is instrumental in pinpointing concentrated areas of accidents within district 10. DBSCAN operates by categorizing points in the dataset based on their density and proximity to each other, thereby automatically identifying clusters without requiring a predetermined number of clusters. Initially, the algorithm extracts the latitude and longitude coordinates from the accident data stored in a GeoDataFrame. It then proceeds to categorize points as core, border, or noise points, based on their density and proximity. Core points, which have sufficient neighboring points within a specified distance, form the foundation of clusters, while border points are within the vicinity of core points but lack the density to be considered core points themselves. Noise points, on the other hand, are isolated outliers. As the algorithm progresses, it recursively expands clusters by adding neighboring points, eventually assigning each point a cluster label. Subsequently, the algorithm calculates the centers of these clusters, which serve as focal points for visualizing the spatial distribution of accident clusters on the map. By plotting circles around these cluster centers, the algorithm effectively highlights areas with a high density of accidents, enabling targeted interventions to enhance road safety. ", style={ 'justify-content': 'center', 'color': '#FFFFFF', 'font-size': '20px'})
    ])
])




# Define the URL of the Tableau visualization
tableau_url = "https://public.tableau.com/views/SA2020SelectCityCouncilData2023PovertyRate/StoryPoverty?:embed=y&:showVizHome=no"

# Define the layout for the poverty correlation section
poverty_correlation_section_layout = html.Div(style={'padding': '20px'}, children=[
    html.H2("Correlation between Accidents and Poverty in San Antonio", style={'textAlign': 'center', 'color': '#FFFFFF', 'margin-bottom': '40px'}),
    html.Div(style={'display': 'flex', 'flexDirection': 'row', 'gap': '80px'}, children=[
        html.Iframe(src=tableau_url, style={'width': '800px', 'height': '600px', 'border': 'none'}),
        html.Div(style={'margin-left': '60px'}, children=[
            dcc.Graph(id='micromorts-bar-chart', style={'width': '800px', 'height': '650px', 'border': 'none'})
        ])
    ])
])


# Define the layout for the conclusion section
conclusion_section_layout = html.Div(style={'padding': '20px'}, children=[
    html.H2("Conclusion", style={'textAlign': 'center', 'color': '#FFFFFF', 'margin-bottom': '40px'}),
    html.Div(style={'flex': '1'}, children=[
        html.P(
            "Through the analysis of roadway accidents in San Antonio, Texas we were able to reveal several crucial insights. Firstly, Districts 1, 2, 3, and 5 were consistently shown as districts with the highest accident rates over the past two decades. Indicating that these districts  require heightened attention for road safety measures. Moreover, a significant portion of fatalities resulting from these accidents is concentrated within these districts, highlighting the urgent need for targeted interventions to reduce fatalities in these high-risk zones. Additionally, we were also able to find the micromort for every district. A micromort, is a unit of measurement that represents a one-in-a-million chance of death. In the context of road safety analysis, the micromort value for District 1, 2 ,3 , and 5 had higher values compared to the other districts, indicating a greater risk of fatal accidents in those areas.  ",
            style={'color': '#FFFFFF', 'font-size': '20px'}),
        html.P(
            "Delving deeper into our analysis, we explored the correlation between poverty rates and the frequency of accidents in each district. Our findings revealed a compelling connection between higher poverty levels and increased accident occurrences. Specifically, Districts 1, 2, 3, and 5, which exhibited the highest accident rates over the two-decade period, also demonstrated poverty levels exceeding 20%. This correlation highlights how socio-economic factors intertwine with road safety outcomes. It indicates that areas experiencing higher poverty rates may encounter extra hurdles in ensuring safe road conditions and implementing efficient traffic management measures.",
            style={'color': '#FFFFFF', 'font-size': '20px'}),
        html.P(
            "In the future, I aim to explore further into the analyzed data, honing in on specific external factors. This could involve finding the year of vehicles involved in accidents across each district, determining whether districts with higher accident rates see more incidents due to car wear and tear which lead to malfunctions. Additionally, I'm interested in exploring whether accidents attributed to poor lighting or darkness are more prevalent in certain districts compared to others. Furthermore, my dataset also raises the possibility of exploring accidents caused by road conditions or involving uninsured individuals which is another intriguing avenue for investigation. By analyzing such data, we can gain insights into the role of a variety of factors that lead to car accidents outside of infrastructure across different districts. This deeper understanding could inform targeted interventions aimed at addressing these specific challenges and improving overall road safety.",
            style={'color': '#FFFFFF', 'font-size': '20px'})
        ])
])

# Define the layout for the works cited section
citations_section_layout = html.Div(style={'padding': '20px'}, children=[
    html.H2("Citations", style={'textAlign': 'center', 'color': '#FFFFFF', 'margin-bottom': '40px'}),
    html.Div(style={'flex': '1'}, children=[
        html.P(
            "City of San Antonio. (n.d.). City of san antonio open data. City of San Antonio Open Data. https://opendata-cosagis.opendata.arcgis.com/ ",
            style={'color': '#FFFFFF', 'font-size': '20px'}),
        html.P(
            "",
            style={'color': '#FFFFFF', 'font-size': '20px'}),
        html.P(
            "",
            style={'color': '#FFFFFF', 'font-size': '20px'}),
    ])
])


@app.callback(
    [Output('micromorts-bar-chart', 'figure')],  # Note the brackets
    [Input('year-slider', 'value')]
)
def update_micromorts_bar_chart(year):
    data_table = {
        'District #': ['District 1 (Central & North side)', 'District 2 (Central & East side)',
                       'District 3 (Central & Southeast side)', 'District 4 (Southwest side)',
                       'District 5 (Central and West side)', 'District 6 (West side)',
                       'District 7 (West & Northwest side)', 'District 8 (Northwest side)', 'District 9 (North side)',
                       'District 10 (Northeast side)'],
        'Accidents': ['12.31%', '16.00%', '11.33%', '10.32%', '11.83%', '8.20%', '7.33%', '8.08%', '5.80%', '9.00%'],
        'Fatalities': ['11.65%', '12.57%', '6.25%', '6.74%', '15.36%', '11.00%', '11.39%', '8.27%', '8.63%', '8.14%'],
        'Micromorts': ['2325', '2475', '1250', '1397',
                       '3065', '1933', '2194', '1605',
                       '1682', '1550']
    }

    district_labels = data_table['District #']

    # Extract numerical part and convert to float
    micromorts_values = [float(value.split()[0]) for value in data_table['Micromorts']]

    # Create a bar chart
    bar_chart = go.Figure(data=[go.Bar(
        x=district_labels,
        y=micromorts_values,
        text=data_table['Micromorts'],  # Keep the original text
        textposition='auto',
        marker_color='skyblue'
    )])

    bar_chart.update_layout(
        title="Micromorts by District",
        xaxis_title="District",
        yaxis_title="Micromorts",
        template='plotly_white'
    )

    return [bar_chart]  # Return the Figure object inside a list


# # Update the callback to update the district accidents chart and graph
# @app.callback(
#     [Output('accident-bar-chart', 'figure')],
#     [Input('year-slider', 'value')]
# )
# def update_accident_bar_chart(selected_years_range):
#     # Provided accident data for each district across the years
#     district_accidents_data = {
#         'District 1': [23, 19, 17, 16, 22, 27, 18, 17, 18, 16, 18, 16, 13, 8, 13, 16, 16, 12, 21, 18, 11],
#         'District 2': [31, 17, 22, 14, 28, 29, 19, 23, 19, 17, 17, 15, 12, 24, 14, 18, 33, 25, 13, 18, 11],
#         'District 3': [17, 16, 17, 17, 13, 23, 19, 16, 19, 15, 15, 15, 15, 8, 10, 11, 15, 17, 12, 16, 13],
#         'District 4': [26, 17, 15, 14, 16, 18, 8, 15, 22, 19, 7, 9, 9, 11, 9, 15, 9, 15, 8, 11, 6],
#         'District 5': [20, 19, 17, 18, 15, 19, 15, 16, 14, 10, 11, 7, 13, 16, 9, 10, 6, 19, 18, 20, 11],
#         'District 6': [18, 10, 25, 11, 10, 21, 10, 12, 13, 8, 5, 11, 8, 8, 8, 9, 9, 7, 4, 10, 4],
#         'District 7': [9, 10, 9, 16, 9, 6, 13, 5, 10, 8, 11, 8, 9, 6, 9, 8, 9, 9, 7, 10, 9],
#         'District 8': [12, 11, 14, 14, 9, 14, 13, 12, 9, 9, 11, 7, 12, 9, 9, 12, 7, 3, 14, 6, 11],
#         'District 9': [11, 6, 0, 4, 7, 7, 9, 11, 10, 12, 1, 12, 6, 6, 8, 5, 4, 14, 6, 13, 3],
#         'District 10': [12, 17, 10, 15, 12, 17, 19, 7, 16, 7, 14, 5, 18, 10, 7, 8, 12, 10, 9, 10, 7]
#     }
#
#     # Calculate the total accidents for each district across all years
#     total_accidents_by_district = {district: sum(accidents_data) for district, accidents_data in
#                                    district_accidents_data.items()}
#
#     # Create the bar chart data
#     bar_chart_data = go.Figure(data=[go.Bar(
#         x=list(total_accidents_by_district.keys()),  # District names
#         y=list(total_accidents_by_district.values()),  # Total number of accidents
#         marker_color='indianred'  # Bar color
#     )])
#
#     # Update layout for the bar chart
#     bar_chart_data.update_layout(
#         title='Total Accidents by District',
#         xaxis_title='District',
#         yaxis_title='Total Accidents',
#         bargap=0.2,  # Gap between bars
#         height=600,
#         width=800
#     )
#
#     return [bar_chart_data]


# Update the layout to include the research summary section and the correlation section
app.layout = html.Div(style={'backgroundColor': '#000000', 'color': '#333', 'padding': '20px'}, children=[
    # Add a styled header with a different font
    html.H1("Unveiling Hidden Factors: Spatial-Temporal Roadway Accident Visualization and Analysis in San Antonio", style={'font-family': 'Arial, sans-serif', 'color': '#FFFFFF', 'textAlign': 'center', 'margin-bottom': '20px', 'font-size': '30px'}),  # Adjust margin for spacing

    # Add the research summary section
    research_summary_layout,


    # Use a styled div for better visual separation
    html.Div([
        # Replace the dcc.Slider with dcc.RangeSlider
        dcc.RangeSlider(
            id='year-slider',
            min=2001,
            max=2021,
            step=1,
            marks={year: {'label': str(year), 'style': {'transform': 'rotate(-45deg)', 'whiteSpace': 'nowrap'}} for
                   year in range(2001, 2022)},
            value=[2021],
            allowCross=False,  # This prevents the range from having a cross-handle
            className='custom-slider',  # Add a custom CSS class
            tooltip={'placement': 'bottom', 'always_visible': True}  # Add a tooltip to the slider
        ),

        dcc.Checklist(
            id='toggle-accidents',
            options=[
                {'label': 'Show Accidents', 'value': 'show_accidents'},
            ],
            value=[],
            inline=True,
            style={'margin-top': '10px', 'background-color': '#28a745', 'color': 'white', 'padding': '8px', 'border-radius': '5px'}  # Adjust the style of the checkbox
        ),
        dcc.Checklist(
            id='toggle-speed-humps',
            options=[
                {'label': 'Show Speed Humps', 'value': 'show_speed_humps'},
            ],
            value=[],
            inline=True,
            style={'margin-top': '10px', 'background-color': '#007BFF', 'color': 'white', 'padding': '8px', 'border-radius': '5px'}  # Adjust the style of the checkbox
        ),
    ], style={'margin-bottom': '20px'}),  # Add some margin at the bottom for better spacing

    # Add the graph to your layout
    html.Div([
        # Map component
        dcc.Graph(
            id='accident-map',
            config={'displayModeBar': False},
            style={'width': '50%', 'height': '90vh', 'margin': 'auto', 'margin-top': '10px', 'margin-left': '10px',
                   'box-shadow': '0 4px 8px rgba(0,0,0,0.1)'},
            clickData={'points': [{'location': '...', 'text': '...'}]}
        ),
        # Graph component
        dcc.Graph(
            id='district-accidents-chart',
            config={'displayModeBar': False},
            style={'width': '50%', 'height': '90vh', 'margin': 'auto', 'margin-top': '10px', 'margin-right': '10px',
                   'box-shadow': '0 4px 8px rgba(0,0,0,0.1)'},
        )
    ], style={'display': 'flex'}),

    # Add the correlation section
    correlation_section_layout,

    # Add the table section
    table_layout,

    # Add the accident hotspots section
    accident_hotspots_section_layout,

    # Add the poverty correlation section
    poverty_correlation_section_layout,

    conclusion_section_layout,

    citations_section_layout
])


# Define a variable for the last selected year
last_selected_year = 2011

# Initialize an empty figure
empty_fig = px.choropleth_mapbox()
empty_fig.update_layout(mapbox_style="open-street-map", mapbox_zoom=9, mapbox_center={"lat": 29.4201, "lon": -98.5721})

# Define a dictionary to cache processed data
cached_data = {}

# Number of years to preload data for (adjust as needed)
preload_years_before = 10  # Adjust the number of years to preload before the selected year
preload_years_after = 10   # Adjust the number of years to preload after the selected year

# Preload data for a range of years
preloaded_years_before = range(last_selected_year - preload_years_before, last_selected_year)
preloaded_years_after = range(last_selected_year + 1, last_selected_year + preload_years_after + 1)

# Concatenate the two ranges to cover both before and after the selected year
preloaded_years = list(preloaded_years_before) + [last_selected_year] + list(preloaded_years_after)

for year in preloaded_years:
    process_data(year, cached_data)


# Calculate district accidents by year
district_accidents_by_year = calculate_district_accidents_by_year()


# Define function to perform DBSCAN clustering
def perform_dbscan_clustering(gdf_accidents, eps=0.01, min_samples=5):
    # Extract latitude and longitude coordinates from GeoDataFrame
    coordinates = np.column_stack((gdf_accidents.geometry.x, gdf_accidents.geometry.y))

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    dbscan.fit(coordinates)

    # Get labels assigned to each point (-1 represents outliers)
    cluster_labels = dbscan.labels_

    return cluster_labels


# Update the callback to update the hotspot map with clustered accidents
@app.callback(
    Output('hotspot-map', 'figure'),
    [Input('hotspot-map', 'clickData')]
)
def update_hotspot_map(click_data):
    # Define the years and districts of interest
    years_of_interest = range(2001, 2022)  # All years from 2000 to 2021
    districts_of_interest = [1, 2, 3, 5]  # Districts 1, 2, and 3

    # Create an empty list to store coordinates of accidents in districts of interest
    hotspot_latitudes = []
    hotspot_longitudes = []

    # Iterate over the years and districts of interest
    for year in years_of_interest:
        # Get the processed data for the selected year
        gdf_accidents, council_districts_geojson, district_counts, _ = cached_data.get(year, (None, None, None, None))

        # Check if data is available for the selected year
        if gdf_accidents is not None:
            # Iterate over the districts
            for district in districts_of_interest:
                # Check if the district is in the processed data
                if district in district_counts.index:
                    # Get the accidents in the district of interest for the current year
                    accidents_in_district = gdf_accidents[gdf_accidents[district_column_name] == district]

                    # Extract latitude and longitude of accidents in the district
                    district_latitudes = accidents_in_district.geometry.y.tolist()
                    district_longitudes = accidents_in_district.geometry.x.tolist()

                    # Extend the lists of coordinates with the coordinates of accidents in the district
                    hotspot_latitudes.extend(district_latitudes)
                    hotspot_longitudes.extend(district_longitudes)

    # Create map figure
    fig = go.Figure()

    # Plot the scatter map of accident hotspots
    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=hotspot_longitudes,
        lat=hotspot_latitudes,
        marker=dict(
            size=7,
            opacity=0.8,
            color='purple',
        ),
        name='Accident Hotspots',
        showlegend=True,
    ))

    # Add GeoJSON trace for districts 1, 6, and 10
    for district in districts_of_interest:
        district_geojson = council_districts_geojson[council_districts_geojson['District'] == district].to_json()
        district_data = json.loads(district_geojson)
        fig.add_trace(go.Choroplethmapbox(
            geojson=district_data,
            locations=[0],  # Dummy locations
            z=[1],  # Dummy values
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],  # Transparent color
            marker_opacity=0,
            showlegend=False,
            hoverinfo='none'  # Remove hover info for this trace
        ))

    # Perform DBSCAN clustering on accidents
    cluster_labels = perform_dbscan_clustering(gdf_accidents)

    # Get unique cluster labels (excluding outliers)
    unique_clusters = np.unique(cluster_labels[cluster_labels != -1])

    # Iterate over unique clusters and plot circles around cluster centers
    for cluster_label in unique_clusters:
        # Extract coordinates of accidents in the cluster
        cluster_indices = np.where(cluster_labels == cluster_label)[0]
        cluster_latitudes = np.array(hotspot_latitudes)[cluster_indices]
        cluster_longitudes = np.array(hotspot_longitudes)[cluster_indices]

        # Fit kernel density estimation to estimate density of accidents in the cluster with adjusted bandwidth
        kde = KernelDensity(bandwidth=0.005, kernel='gaussian')  # Adjust bandwidth
        kde.fit(np.column_stack((cluster_longitudes, cluster_latitudes)))

        # Sample from KDE to get density values at points within cluster
        lon_min, lon_max = np.min(cluster_longitudes), np.max(cluster_longitudes)
        lat_min, lat_max = np.min(cluster_latitudes), np.max(cluster_latitudes)
        lon_range = np.linspace(lon_min, lon_max, 100)
        lat_range = np.linspace(lat_min, lat_max, 100)
        lon_mesh, lat_mesh = np.meshgrid(lon_range, lat_range)
        sample_points = np.column_stack((lon_mesh.ravel(), lat_mesh.ravel()))
        density_values = np.exp(kde.score_samples(sample_points)).reshape(lon_mesh.shape)

        # Normalize density values
        density_values /= np.max(density_values)

        # Plot circles around areas with higher density of accidents
        for lon, lat, density in zip(lon_mesh.ravel(), lat_mesh.ravel(), density_values.ravel()):
            if density > 0.5:  # Adjust density threshold as needed
                fig.add_trace(go.Scattermapbox(
                    mode="markers",
                    lon=[lon],
                    lat=[lat],
                    marker=dict(
                        size=density * 10,  # Adjust size based on density
                        opacity=0.3,  # Adjust opacity of circle
                        color='blue',
                    ),
                    name='Accident Cluster',
                    showlegend=False,
                ))
    fig.update_layout(
        mapbox=dict(
            center=dict(lat=29.4241, lon=-98.4936),
            zoom=10,
            style='carto-positron'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        width=700
    )

    return fig

# Update the callback to update the district accidents chart and graph
@app.callback(
    [Output('district-accidents-chart', 'figure'),
     Output('district-accidents-graph', 'figure')],
    [Input('year-slider', 'value')]
)
def update_district_accidents_graph(selected_years_range):
    # Calculate the average of the selected range
    selected_year = int(sum(selected_years_range) / len(selected_years_range))

    # Get the processed data for the selected year
    gdf_accidents, council_districts_geojson, district_counts, _ = cached_data.get(selected_year,
                                                                                   (None, None, None, None))

    if district_counts is not None:
        # Create the bar plot for district accidents
        bar_fig = go.Figure(data=[go.Bar(
            x=district_counts.index,  # District indexes
            y=district_counts.values,  # Number of accidents
            marker_color='indianred'  # Bar color
        )])

        # Update layout for the bar chart
        bar_fig.update_layout(
            title=f'Accidents per District - {selected_year}',
            xaxis_title='District Index',
            yaxis_title='Number of Accidents',
            bargap=0.2,  # Gap between bars
            height=600,
            width=800

        )

        # Create traces for the line chart
        traces = []
        for district, counts in district_accidents_by_year.items():
            trace = go.Scatter(x=list(counts.keys()), y=list(counts.values()), mode='lines', name=f'District {district}')
            traces.append(trace)

        # Create layout for the line chart
        line_chart_layout = go.Layout(
            title='Accidents by District (2001-2021)',
            xaxis=dict(title='Year'),
            yaxis=dict(title='Total Accidents'),
            width=700,
            height=500
        )

        # Create figure for the line chart
        line_fig = go.Figure(data=traces, layout=line_chart_layout)



        return bar_fig, line_fig

    # If data is not available, return empty figures
    return go.Figure(), go.Figure()

@app.callback(
    Output('accident-map', 'figure'),
    [Input('toggle-accidents', 'value'),
     Input('toggle-speed-humps', 'value'),
     Input('year-slider', 'value'),
     Input('accident-map', 'clickData')]  # Add this input for clickData
)
def update_map(toggle_accidents, toggle_speed_humps, selected_years_range, click_data):
    global latest_triggered_year

    # Calculate the average of the selected range
    selected_year = int(sum(selected_years_range) / len(selected_years_range))

    # Get the triggered component ID
    ctx = dash.callback_context
    triggered_id = ctx.triggered_id

    # Check if the callback was triggered by the slider
    if triggered_id and 'year-slider' in triggered_id:
        # Update the latest triggered year
        latest_triggered_year = selected_year

    # Check if data for the selected year is already cached
    if latest_triggered_year not in cached_data or latest_triggered_year != selected_year:
        # Show loading indicator or update layout to indicate data processing
        return empty_fig

    # Call the process_data function to get the combined data for the selected year
    gdf_accidents, council_districts_geojson, district_counts, output_text = cached_data[latest_triggered_year]

    if gdf_accidents is not None:
        # Plotted the map for the selected year with accidents and other cities/towns
        fig = px.choropleth_mapbox(council_districts_geojson,
                                   geojson=council_districts_geojson.geometry,
                                   locations=council_districts_geojson.index,
                                   color=district_counts.values.astype(float),
                                   hover_name="Name",
                                   mapbox_style="open-street-map",
                                   zoom=9,
                                   center={"lat": 29.4201, "lon": -98.5721},
                                   opacity=0.5,
                                   color_continuous_scale="Jet",  # Set your desired color scale
                                   range_color=(0, 35),  # Set your desired range
                                   width=800,
                                   height=600,
                                   title=f'Map of Accidents in the Districts of San Antonio - {latest_triggered_year}',
                                   labels={'color': 'Number of Accidents'},
                                   )

        # Customizing hover text
        fig.update_traces(hovertemplate="District: %{customdata}<br>" +
                                        "Council Representative: %{hovertext}<br>" +
                                        "Number of Accidents: %{z}<extra></extra>",
                          customdata=council_districts_geojson['District'])

        # Reversed the color scale for the choropleth layer
        fig.update_traces(
            colorbar=dict(tickmode='array', tickvals=list(reversed(district_counts.values.astype(float))),
                          ticktext=list(reversed(district_counts.values.astype(float).astype(str))))
        )

        # Added Bexar County outline using GeoJSON file
        fig.update_geos(fitbounds="locations", visible=False)

        # Add a new choropleth_mapbox trace for other cities and towns
        fig.add_trace(px.choropleth_mapbox(filtered_other_cities_towns_geojson,
                                           geojson=filtered_other_cities_towns_geojson.geometry,
                                           locations=filtered_other_cities_towns_geojson.index,
                                           color_discrete_sequence=["#8B4513"],  # Set color to dark brown
                                           hover_name="Name",
                                           opacity=0.5,
                                           ).data[0])

        # Apply the modification to remove the legend entry for other cities and towns
        fig.update_traces(showlegend=False, selector=dict(type='choroplethmapbox'))

        if 'show_accidents' in toggle_accidents:
            # Adjusted marker size, opacity, and hover text in the Scattermapbox trace for accidents
            fig.add_trace(go.Scattermapbox(
                mode="markers",
                lon=gdf_accidents.geometry.x,
                lat=gdf_accidents.geometry.y,
                hoverinfo='text',
                hovertext=gdf_accidents[district_column_name].astype(str),  # Use district_column_name for hovertext
                marker=dict(
                    size=7,
                    opacity=0.8,
                    color='red',
                ),
                name='Accidents',
                showlegend=False  # Set showlegend to False for this trace
            ))

        # Load GeoJSON data for speed humps from file
        speed_humps_geojson = gpd.read_file("Traffic_Speed_Humps.geojson")

        # Check if 'geometry' column exists before creating GeoDataFrame for speed humps
        if 'geometry' in speed_humps_geojson.columns:
            # Create a GeoDataFrame from the speed humps data using the 'geometry' column
            gdf_speed_humps = gpd.GeoDataFrame(speed_humps_geojson, geometry='geometry', crs="EPSG:4326")

            # Check if 'show_speed_humps' is True
            if 'show_speed_humps' in toggle_speed_humps:
                # Adjusted marker size, opacity, and hover text in the Scattermapbox trace for speed humps
                fig.add_trace(go.Scattermapbox(
                    mode="markers",
                    lon=gdf_speed_humps.geometry.x,
                    lat=gdf_speed_humps.geometry.y,
                    hoverinfo='text',
                    hovertext=gdf_speed_humps['geometry'].apply(lambda geom: geom.coords[:]).astype(str),
                    marker=dict(
                        size=4,
                        opacity=0.6,
                        color='blue',
                    ),
                    name='Speed Humps',
                    showlegend=False  # showlegend to False for trace
                ))

            # Check if a point on the map was clicked
            if click_data and 'points' in click_data:
                clicked_point = click_data['points'][0]

                # Ensure that 'location' is present in the clicked_point dictionary
                if 'location' in clicked_point and clicked_point['location'] != '...':
                    # Extract the index of the clicked point
                    clicked_location_index = clicked_point['location']

                    # Check if the clicked location is a district or another city
                    if clicked_location_index in council_districts_geojson.index:
                        clicked_location = council_districts_geojson.loc[clicked_location_index, 'Name']
                    elif clicked_location_index in filtered_other_cities_towns_geojson.index:
                        clicked_location = filtered_other_cities_towns_geojson.loc[clicked_location_index, 'Name']
                    else:
                        clicked_location = None

                    # Print information for debugging
                    print(f"Clicked point: {clicked_point}")
                    print(f"Clicked location index: {clicked_location_index}")
                    print(f"Clicked location: {clicked_location}")

                    # Update the map center based on the clicked location
                    if clicked_location:
                        # Check if the clicked location is a district or another city
                        if clicked_location in council_districts_geojson['Name'].tolist():
                            # If the clicked location is a district, update map center to the district's coordinates
                            clicked_district = council_districts_geojson[
                                council_districts_geojson['Name'] == clicked_location]
                            mapbox_center = {"lat": clicked_district.geometry.centroid.y.values[0],
                                             "lon": clicked_district.geometry.centroid.x.values[0]}
                            # Set zoom level for the clicked district
                            zoom_level = 12

                            # Print additional information for debugging
                            print(f"Mapbox center: {mapbox_center}")
                            print(f"Zoom level: {zoom_level}")

                            # Update the map layout with the new center and zoom level
                            fig.update_layout(mapbox_center=mapbox_center, mapbox_zoom=zoom_level)

                        elif clicked_location in filtered_other_cities_towns_geojson['Name'].tolist():
                            # If the clicked location is another city, update map center to the city's coordinates
                            clicked_city = filtered_other_cities_towns_geojson[
                                filtered_other_cities_towns_geojson['Name'] == clicked_location]
                            mapbox_center = {"lat": clicked_city.geometry.centroid.y.values[0],
                                             "lon": clicked_city.geometry.centroid.x.values[0]}
                            # Set zoom level for the clicked city
                            zoom_level = 14

                            # Additional information for debugging
                            print(f"Mapbox center: {mapbox_center}")
                            print(f"Zoom level: {zoom_level}")

                            # Update the map layout with the new center and zoom level
                            fig.update_layout(mapbox_center=mapbox_center, mapbox_zoom=zoom_level)

            return fig

    # If selected year hasn't changed, return PreventUpdate to avoid unnecessary updates
    return dash.no_update

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)