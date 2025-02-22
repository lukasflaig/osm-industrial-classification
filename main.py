import os

from services.create_labeled_database import create_labeled_database, filter_labeled_gpkg
from services.create_osm_data_rasters import generate_rasters_from_gdf
from services.geojson_by_way_id import get_geo_json_files_for_gpkg
from services.label_ways_with_llm import classify_osm_ways
from services.llama_client import Llama32Client


def run_preprocessing():
    """
    Führt die gesamte Preprocessing-Pipeline aus:
      1. Erzeugt das initiale labeled database GeoPackage aus der Input-PBF-Datei.
      2. Klassifiziert die OSM-Ways mithilfe des LLM-Clients.
      3. Filtert das GeoPackage, indem Zeilen ohne gültiges Label entfernt werden.
      4. Holt GeoJSON-Dateien für die gefilterten OSM-Ways.

    Returns:
    --------
    None
    """
    # Konfiguration – diese Parameter sollten an die tatsächlichen Gegebenheiten angepasst werden.
    folder_name = "output_folder"         # Ordnername für die Ausgabedateien
    input_pbf = "path/to/input_file.pbf"    # Pfad zur Input-PBF-Datei
    target_epsg = 4326

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_gpkg_dir = os.path.join(base_dir, f"../{folder_name}")

    gpkg_initial = os.path.join(output_gpkg_dir, "industrial_landuse.gpkg")
    gpkg_labeled = os.path.join(output_gpkg_dir, "industrial_landuse_labeled.gpkg")
    gpkg_filtered = os.path.join(output_gpkg_dir, "industrial_landuse_labeled_filtered.gpkg")
    gpkg_with_geojson = os.path.join(output_gpkg_dir, "industrial_landuse_labeled_filtered_geo_json_files.gpkg")

    FIXED_LABELS = [
        "Abwasserbehandlungsanlage",
        "Kohlekraftwerk",
        "Solarkraftwerk",
        "Schrottplatz",
        "Hafenanlage",
        "Windpark",
        "Biomassekraftwerk",
        "Müllverbrennungsanlage",
        "Chemiepark",
        "Automobilwerk",
        "Logistikzentrum",
        "Zementwerk",
        "Stahlwerk",
        "Raffinerie",
        "Recyclinganlage",
        "Gasspeicheranlage",
        "Großbrauerei",
    ]

    # Schritt 1: Erstelle das initiale labeled database GeoPackage aus der PBF-Datei.
    create_labeled_database(folder_name, input_pbf, target_epsg)

    # Schritt 2: Klassifiziere die OSM-Ways mithilfe des LLM-Clients.
    client = Llama32Client()
    classify_osm_ways(gpkg_initial, gpkg_labeled, client, FIXED_LABELS, batch_size=10)

    # Schritt 3: Filtere das GeoPackage, um Zeilen ohne gültiges Label zu entfernen.
    filter_labeled_gpkg(gpkg_labeled, gpkg_filtered)

    # Schritt 4: Hole GeoJSON-Dateien für die gefilterten OSM-Ways.
    get_geo_json_files_for_gpkg(folder_name, gpkg_filtered, target_epsg)

    print("[INFO] Preprocessing pipeline completed.")


def run_generate_rasters():
    """
    Führt die Rasterisierung der finalen GeoPackage-Datei aus, indem für jeden Eintrag
    GeoJSON-Dateien eingelesen und mittels process_dataframe(...) Rasterbilder erzeugt werden.

    Die Funktion ruft generate_rasters_from_gdf(...) mit den entsprechenden Parametern auf.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    gpkg_path = "path/to/industrial_landuse_labeled_filtered_geo_json_files.gpkg"
    base_output_folder = "data/dataset"
    width = 512
    height = 512
    geo_json_files_path = "database_germany_all/geo_json_files"
    target_epsg = 32632

    generate_rasters_from_gdf(
        gpkg_path=gpkg_path,
        base_output_folder=base_output_folder,
        width=width,
        height=height,
        geo_json_files_path=geo_json_files_path,
        target_epsg=target_epsg
    )

    print("[INFO] Raster generation completed.")

