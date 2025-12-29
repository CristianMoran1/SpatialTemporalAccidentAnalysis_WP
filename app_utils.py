"""
Small, testable helpers extracted from the dashboard logic.

These mirror key behaviors used in the app:
- accumulating district counts across years
- DBSCAN clustering on (lon, lat)
- building the micromorts bar chart figure
"""

from __future__ import annotations

from typing import Dict, Mapping, Any, Optional

import numpy as np
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go


def accumulate_district_counts(
    accumulated: Dict[Any, int],
    district_counts: Mapping[Any, int],
) -> Dict[Any, int]:
    """
    Pure version of the app's "update_accumulated_district_counts":
    returns an updated dict without relying on globals.
    """
    for district, count in district_counts.items():
        accumulated[district] = accumulated.get(district, 0) + int(count)
    return accumulated


def perform_dbscan_clustering(gdf_accidents, eps: float = 0.01, min_samples: int = 5):
    """
    Same clustering approach as the app: DBSCAN over point coordinates.
    Expects a GeoDataFrame-like object with geometry.x and geometry.y arrays.
    """
    coordinates = np.column_stack((gdf_accidents.geometry.x, gdf_accidents.geometry.y))
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    dbscan.fit(coordinates)
    return dbscan.labels_


def build_micromorts_bar_chart(data_table: Optional[dict] = None) -> go.Figure:
    """
    Build the micromorts bar chart figure (same idea as the Dash callback),
    but as a simple function we can test.
    """
    if data_table is None:
        # Default table matches the dashboard's micromorts section structure
        data_table = {
            "District #": [
                "District 1 (Central & North side)",
                "District 2 (Central & East side)",
                "District 3 (Central & Southeast side)",
                "District 4 (Southwest side)",
                "District 5 (Central and West side)",
                "District 6 (West side)",
                "District 7 (West & Northwest side)",
                "District 8 (Northwest side)",
                "District 9 (North side)",
                "District 10 (Northeast side)",
            ],
            "Micromorts": ["2325", "2475", "1250", "1397", "3065", "1933", "2194", "1605", "1682", "1550"],
        }

    labels = data_table["District #"]
    micromorts_values = [float(str(v).split()[0]) for v in data_table["Micromorts"]]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=micromorts_values,
                text=data_table["Micromorts"],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="Micromorts by District",
        xaxis_title="District",
        yaxis_title="Micromorts",
        template="plotly_white",
    )

    return fig
