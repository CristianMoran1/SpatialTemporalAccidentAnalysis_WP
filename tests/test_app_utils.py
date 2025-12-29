import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from app_utils import (
    accumulate_district_counts,
    perform_dbscan_clustering,
    build_micromorts_bar_chart,
)


def test_accumulate_district_counts_adds_and_sums():
    acc = {}
    acc = accumulate_district_counts(acc, {1: 10, 2: 5})
    acc = accumulate_district_counts(acc, {1: 3, 3: 7})

    assert acc[1] == 13
    assert acc[2] == 5
    assert acc[3] == 7


def test_accumulate_district_counts_accepts_series_like_counts():
    acc = {}
    series_counts = pd.Series({1: 2, 2: 4})
    acc = accumulate_district_counts(acc, series_counts)

    assert acc == {1: 2, 2: 4}


def test_perform_dbscan_clustering_finds_two_clusters():
    # Two tight clusters far apart
    cluster_a = [Point(x, y) for x, y in zip(np.random.normal(0, 0.01, 20), np.random.normal(0, 0.01, 20))]
    cluster_b = [Point(x, y) for x, y in zip(np.random.normal(5, 0.01, 20), np.random.normal(5, 0.01, 20))]
    gdf = gpd.GeoDataFrame(geometry=cluster_a + cluster_b, crs="EPSG:4326")

    labels = perform_dbscan_clustering(gdf, eps=0.1, min_samples=5)

    # Ignore noise (-1). We expect at least 2 real clusters.
    cluster_ids = set(labels)
    cluster_ids.discard(-1)

    assert len(labels) == len(gdf)
    assert len(cluster_ids) >= 2


def test_build_micromorts_bar_chart_returns_a_figure():
    fig = build_micromorts_bar_chart(
        {
            "District #": ["D1", "D2"],
            "Micromorts": ["100", "200"],
        }
    )

    assert fig.layout.title.text == "Micromorts by District"
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"
