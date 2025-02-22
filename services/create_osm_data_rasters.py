import os
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from pyrosm import OSM
from rasterio.features import rasterize
from rasterio.transform import from_origin, rowcol
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from geojson_by_way_id import geojson_to_gdf

from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.patches as patches


def determine_bounding_box(main_geom: shapely.geometry, target_epsg: int = 32632) -> tuple:
    """
    Bestimmt die quadratische Bounding Box basierend auf der gegebenen Gesamtgeometrie.

    Parameters:
    -----------
    main_geom : shapely.geometry
        Die Gesamtgeometrie (z.B. die Grundfläche).
    target_epsg : int, optional
        Die EPSG-Nummer des gewünschten projizierten CRS. Standard ist 32632 (UTM Zone 32N).

    Returns:
    --------
    tuple
        Quadratige Bounding Box (minx, miny, maxx, maxy).
    """
    # Erstellen eines GeoDataFrame für die Transformation
    gdf_main = gpd.GeoDataFrame({'geometry': [main_geom]}, crs=f"EPSG:{target_epsg}")

    # Schritt 1: Sicherstellen, dass das GeoDataFrame im gewünschten projizierten CRS ist
    if gdf_main.crs is None:
        raise ValueError("Die Geometrie hat kein definiertes CRS.")

    current_epsg = gdf_main.crs.to_epsg()
    if current_epsg != target_epsg:
        gdf_main = gdf_main.to_crs(epsg=target_epsg)

    # Schritt 2: Berechne die Bounding Box der Geometrie
    bounding_box = gdf_main.total_bounds  # (minx, miny, maxx, maxy)

    # Schritt 3: Quadratige Bounding Box erstellen, um Verzerrungen zu vermeiden
    square_bounding_box = make_square_bounding_box(bounding_box)

    return square_bounding_box


def make_square_bounding_box(bounding_box: tuple) -> tuple:
    """
    Macht eine Bounding Box quadratisch, indem die kleinere Dimension erweitert wird.

    Parameters:
    -----------
    bounding_box : tuple
        Bounding Box (minx, miny, maxx, maxy).

    Returns:
    --------
    tuple
        Quadratische Bounding Box (minx, miny, maxx, maxy).
    """
    minx, miny, maxx, maxy = bounding_box
    width = maxx - minx
    height = maxy - miny

    if width > height:
        # Erweiterung in y-Richtung
        delta = width - height
        miny -= delta / 2
        maxy += delta / 2
    else:
        # Erweiterung in x-Richtung
        delta = height - width
        minx -= delta / 2
        maxx += delta / 2

    return (minx, miny, maxx, maxy)


def rasterize_fixed_size(
        gdf: gpd.GeoDataFrame,
        width: int = 512,
        height: int = 512,
        bounding_box: tuple = None,
        all_touched: bool = False,
        fill: int = 0,
        burn_value: int = 255
) -> tuple:
    """
    Rasterisiert die Geometrien des GeoDataFrames auf eine feste Rastergröße.

    Parameters:
    -----------
    gdf : GeoDataFrame
        Das GeoDataFrame mit den geometrischen Objekten.
    width : int, optional
        Rasterbreite in Pixeln, Standard: 512.
    height : int, optional
        Rasterhöhe in Pixeln, Standard: 512.
    bounding_box : tuple, optional
        Bounding Box (minx, miny, maxx, maxy). Falls None, wird die Gesamt-Bounding-Box des GeoDataFrames verwendet.
    all_touched : bool, optional
        Gibt an, ob alle berührten Pixel einbezogen werden sollen.
    fill : int, optional
        Wert für Hintergrundpixel, Standard: 0.
    burn_value : int, optional
        Wert für gezeichnete Pixel, Standard: 255.

    Returns:
    --------
    tuple
        Ein Tuple bestehend aus:
        - raster_data (2D numpy array): Das gerasterte Bild.
        - transform: Die Transformationsmatrix.
    """
    if bounding_box is None:
        minx, miny, maxx, maxy = gdf.total_bounds
    else:
        minx, miny, maxx, maxy = bounding_box

    # Berechne Pixelgröße basierend auf quadratischer Bounding Box
    width_m = maxx - minx
    height_m = maxy - miny

    # Einheitliche Pixelgröße
    pixel_size = max(width_m / width, height_m / height)
    transform = from_origin(minx, maxy, pixel_size, pixel_size)

    # Rasterdaten erstellen mit fester Rastergröße (width x height)
    raster_data = rasterize(
        shapes=((geom, burn_value) for geom in gdf.geometry if geom is not None and not geom.is_empty),
        out_shape=(height, width),
        transform=transform,
        fill=fill,
        all_touched=all_touched,
        dtype=np.uint8
    )
    return raster_data, transform


def filter_layers(gdf: gpd.GeoDataFrame) -> dict:
    """
    Filtert das GeoDataFrame in separate GeoDataFrames für Gebäude, Straßen, Zugstrecken und Wasserflächen.

    Parameters:
    -----------
    gdf : GeoDataFrame
        Das GeoDataFrame mit den geometrischen Objekten und zugehörigen Tags.

    Returns:
    --------
    dict
        Ein Dictionary mit den gefilterten GeoDataFrames:
        - "buildings"
        - "roads"
        - "railways"
        - "water"
    """
    # Gebäude
    buildings_gdf = gdf[gdf['tags'].apply(lambda tags: tags.get('building') is not None if isinstance(tags, dict) else False)]

    # Straßen
    roads_gdf = gdf[
        gdf['tags'].apply(
            lambda tags: tags.get('highway') is not None if isinstance(tags, dict) else False
        )
    ]

    # Zugstrecken
    railways_gdf = gdf[
        gdf['tags'].apply(
            lambda tags: (tags.get('railway') is not None) if isinstance(tags, dict) else False
        )
    ]

    # Wasserflächen
    water_gdf = gdf[
        gdf['tags'].apply(
            lambda tags: ((tags.get('natural') == 'water') or
                          (tags.get('landuse') in ['basin', 'reservoir']) or
                          (tags.get('waterway') is not None))
            if isinstance(tags, dict) else False
        )
    ]

    return {
        "buildings": buildings_gdf,
        "roads": roads_gdf,
        "railways": railways_gdf,
        "water": water_gdf
    }


def create_rasters(layers: dict, width: int, height: int, bounding_box: tuple) -> dict:
    """
    Erstellt Raster für jede Klasse basierend auf den gefilterten GeoDataFrames.

    Parameters:
    -----------
    layers : dict
        Dictionary mit gefilterten GeoDataFrames.
    width : int
        Rasterbreite in Pixeln.
    height : int
        Rasterhöhe in Pixeln.
    bounding_box : tuple
        Quadratische Bounding Box (minx, miny, maxx, maxy).

    Returns:
    --------
    dict
        Dictionary mit Rasterdaten für jede Klasse.
    """
    def rasterize_layer(gdf_layer):
        if gdf_layer.empty:
            return np.zeros((height, width), dtype=np.uint8)
        else:
            raster, _ = rasterize_fixed_size(gdf_layer, width, height, bounding_box)
            return raster

    rail_raster = rasterize_layer(layers["railways"])
    roads_raster = rasterize_layer(layers["roads"])
    water_raster = rasterize_layer(layers["water"])
    build_raster = rasterize_layer(layers["buildings"])

    return {
        "railways": rail_raster,
        "roads": roads_raster,
        "water": water_raster,
        "buildings": build_raster
    }


def _save_overlaid_rasters(rasters: dict, save_path: str, main_geom=None, transform=None):
    """
    Visualisiert vier Raster-Layer, überlagert mit unterschiedlichen Farben und Transparenz,
    und speichert das Bild an einem angegebenen Pfad.

    Parameters:
    -----------
    rasters : dict
        Dictionary mit Rasterdaten für jede Klasse.
    save_path : str
        Pfad, unter dem das Bild gespeichert werden soll.
    main_geom : shapely.geometry, optional
        Die Gesamtgeometrie, deren Umriss gezeichnet werden soll.
    transform : affine.Affine, optional
        Transformationsmatrix, um Koordinaten in Pixelwerte umzuwandeln.

    Returns:
    --------
    None
    """
    # Definieren der Farben für jede Klasse
    class_colors = {
        "railways": "orange",
        "roads": "red",
        "water": "blue",
        "buildings": "green"
    }

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_aspect('equal')

    # Iteriere über jede Klasse und plotte sie
    for class_name, color in class_colors.items():
        raster = rasters[class_name]
        masked = np.ma.masked_where(raster != 255, raster)
        ax.imshow(masked, cmap=mcolors.ListedColormap([color]), alpha=0.5, origin='upper', aspect='equal')

    # Optional: Zeichnen des Umrisses der Gesamtfläche
    if main_geom is not None and transform is not None:
        if main_geom.geom_type == 'Polygon':
            geometries = [main_geom]
        elif main_geom.geom_type == 'MultiPolygon':
            geometries = main_geom.geoms
        else:
            geometries = []

        for geom in geometries:
            coords = list(geom.exterior.coords)
            pixel_coords = [rowcol(transform, x, y) for x, y in coords]
            pixel_coords_plot = [(col, row) for row, col in pixel_coords]
            polygon = patches.Polygon(pixel_coords_plot, linewidth=2, edgecolor='black', facecolor='none')
            ax.add_patch(polygon)

    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label=class_name.capitalize(),
               markerfacecolor=color, markersize=15, alpha=0.5)
        for class_name, color in class_colors.items()
    ]
    if main_geom is not None:
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w', label='Overall Area',
                   markerfacecolor='black', markersize=15, linewidth=2, markeredgecolor='black')
        )
    ax.legend(handles=legend_elements, loc='upper right')

    plt.title("Überlagerte Rasterdarstellung")
    plt.axis('off')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def process_dataframe(gdf: gpd.GeoDataFrame, current_boundary_way_id: int, current_label: str,
                      width: int, height: int, output_folder: str, target_epsg: int = 32632):
    """
    Verarbeitet ein GeoDataFrame, um Rasterdaten für die Klassen 'buildings', 'roads',
    'railways' und 'water' zu erstellen. Basierend auf der Hauptgeometrie wird die Bounding Box bestimmt.

    Parameters:
    -----------
    gdf : GeoDataFrame
        Das GeoDataFrame mit den OSM-Geometrien.
    current_boundary_way_id : int
        Die ID des Way, der die Hauptgeometrie (Boundary) repräsentiert.
    current_label : str
        Das Label bzw. die Kategorie der aktuellen Fläche.
    width : int
        Rasterbreite in Pixeln.
    height : int
        Rasterhöhe in Pixeln.
    output_folder : str
        Pfad zum Ordner, in dem die Ergebnisse gespeichert werden.
    target_epsg : int, optional
        Ziel-EPSG-Code für das Koordinatensystem, Standard: 32632.

    Returns:
    --------
    None
    """
    # Extrahiere die Hauptgeometrie
    main_geom = gdf[gdf["id"] == current_boundary_way_id]["geometry"].iloc[0]
    inner_gdf = gdf[gdf["id"] != current_boundary_way_id]

    # Bestimme die quadratische Bounding Box
    bounding_box = determine_bounding_box(main_geom, target_epsg=target_epsg)

    # Filtere die Layer basierend auf den inneren Geometrien
    layers = filter_layers(inner_gdf)

    # Erstelle Raster für jede Klasse
    rasters = create_rasters(layers, width, height, bounding_box)

    # Erstelle einen mehrkanaligen Raster (4 Kanäle: railways, roads, water, buildings)
    multi_raster = np.stack(
        [rasters["railways"], rasters["roads"], rasters["water"], rasters["buildings"]],
        axis=0
    )
    outfile = os.path.join(output_folder, f"way_{current_boundary_way_id}_{current_label}_{width}x{height}.npy")
    np.save(outfile, multi_raster)

    pixel_size = max((bounding_box[2] - bounding_box[0]) / width,
                     (bounding_box[3] - bounding_box[1]) / height)
    transform = from_origin(bounding_box[0], bounding_box[3], pixel_size, pixel_size)

    overlaid_raster_path = os.path.join(output_folder, f"way_{current_boundary_way_id}_{current_label}_overlay.png")
    _save_overlaid_rasters(rasters, overlaid_raster_path, main_geom, transform)


def generate_rasters_from_gdf(
    gpkg_path: str,
    base_output_folder: str = "data/dataset",
    width: int = 512,
    height: int = 512,
    geo_json_files_path: str = None,
    target_epsg: int = 32632
):
    """
    Liest aus jedem Eintrag in der GPKG (mit Spalten: 'osm_pbf_path' und 'label') den Pfad zu
    einer .pbf-Datei ein, erzeugt ggf. einen Unterordner nach 'label' und ruft process_dataframe(...) auf.

    Parameters:
    -----------
    gpkg_path : str
        Pfad zum GeoPackage, das die OSM-Daten enthält.
    base_output_folder : str, optional
        Basisordner für die Ausgabe, Standard: "data/dataset".
    width : int, optional
        Rasterbreite in Pixeln, Standard: 512.
    height : int, optional
        Rasterhöhe in Pixeln, Standard: 512.
    geo_json_files_path : str, optional
        Pfad zu den GeoJSON-Dateien.
    target_epsg : int, optional
        Ziel-EPSG-Code für das Koordinatensystem, Standard: 32632.

    Returns:
    --------
    None
    """
    gdf = gpd.read_file(gpkg_path)
    print(f"[INFO] GeoDataFrame mit {len(gdf)} Zeilen eingelesen.")

    os.makedirs(base_output_folder, exist_ok=True)

    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Verarbeite GPKG-Zeilen"):
        label = row["label"]
        geo_json_filename = row["geo_json_filename"]
        if geo_json_filename is None:
            print(f"[WARN] Kein GeoJSON-Dateiname für Zeile {idx}. Überspringe...")
            continue
        boundary_way_id = row["osm_id"]

        output_folder = os.path.join(base_output_folder, str(label))
        os.makedirs(output_folder, exist_ok=True)

        geo_json_path = os.path.join(geo_json_files_path, geo_json_filename)
        gdf_geojson = geojson_to_gdf(geo_json_path, target_epsg=target_epsg)

        process_dataframe(
            gdf=gdf_geojson,
            current_boundary_way_id=boundary_way_id,
            current_label=label,
            width=width,
            height=height,
            output_folder=output_folder,
            target_epsg=target_epsg
        )

    print("\n[DONE] Alle Zeilen aus dem GPKG verarbeitet.")