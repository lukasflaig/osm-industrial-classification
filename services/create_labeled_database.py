import osmium
from shapely.geometry import Polygon
import geopandas as gpd
import pandas as pd
import ast
from typing import List, Dict
import shutil
import requests
import subprocess
import os
from tqdm import tqdm
import warnings

from geojson_by_way_id import get_ways_inside_boundary


class IndustrialLanduseHandler(osmium.SimpleHandler):
    """
    Liest Ways aus dem PBF, nutzt die interne Location-Auflösung (locations=True),
    baut Polygone (sofern alle Knoten vorhanden sind) und speichert die
    Ergebnisse, wenn der Way das Tag 'landuse=industrial' besitzt.
    """

    def __init__(self, progress_bar=None):
        """
        Initialisiert den Handler.

        Parameter:
        -----------
        progress_bar : tqdm, optional
            Eine tqdm-Instanz zur Anzeige des Fortschritts.
        """
        super().__init__()
        self.progress_bar = progress_bar
        self.filtered_data = []

    def way(self, w):
        """
        Verarbeitet einen Way aus der OSM-Datei.

        Parameter:
        -----------
        w : osmium.osm.Way
            Ein Way-Objekt aus der OSM-Datenstruktur.

        Returns:
        --------
        None
        """
        tags = {tag.k: tag.v for tag in w.tags}
        # Nur Ways, die 'landuse=industrial' haben
        if tags.get('landuse') == 'industrial':
            coords = []
            # Iteriere über alle Knoten im Way; dank locations=True sind diese verfügbar
            for n in w.nodes:
                try:
                    # Prüfe, ob die Location gültig ist
                    if n.location.valid():
                        coords.append((n.location.lon, n.location.lat))
                    else:
                        return  # Ein ungültiger Knoten: diesen Way überspringen
                except Exception as e:
                    print(f"Fehler bei Knoten in Way {w.id}: {e}")
                    return

            # Stelle sicher, dass das Polygon geschlossen ist
            if coords and (coords[0] != coords[-1]):
                coords.append(coords[0])

            try:
                polygon = Polygon(coords)
            except Exception as e:
                print(f"Fehler beim Erstellen des Polygons für Way {w.id}: {e}")
                return

            self.filtered_data.append({
                'geometry': polygon,
                'tags': tags,
                'osm_id': w.id
            })

        if self.progress_bar is not None:
            self.progress_bar.update(1)


def remove_specific_tag_rows(df: pd.DataFrame, target_conditions: list) -> pd.DataFrame:
    """
    Entfernt Zeilen, deren 'tags' exakt einer der Bedingungen in target_conditions entsprechen.
    Wird z. B. als Blacklist verwendet.

    Parameter:
    -----------
    df : pd.DataFrame
        Das DataFrame, das die OSM-Daten mit den 'tags' enthält.
    target_conditions : list
        Liste von Bedingungen (als Dictionaries), die zum Entfernen passender Zeilen verwendet werden.

    Returns:
    --------
    pd.DataFrame
        Das gefilterte DataFrame ohne die Zeilen, die den Zielbedingungen entsprechen.
    """
    df_filtered = df.copy()
    df_filtered['tags'] = df_filtered['tags'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    def matches_condition(tags, condition: dict) -> bool:
        if not isinstance(tags, dict):
            return False
        if len(tags) != len(condition):
            return False
        for cond_key, cond_value in condition.items():
            if cond_key not in tags:
                return False
            if cond_value is True:
                continue
            else:
                if tags[cond_key] != cond_value:
                    return False
        return True

    def matches_any_condition(tags):
        for cond in target_conditions:
            if matches_condition(tags, cond):
                return True
        return False

    df_filtered = df_filtered[~df_filtered['tags'].apply(matches_any_condition)]
    return df_filtered


def create_gpkg(input_pbf: str, output_gpkg: str, target_epsg: int) -> pd.DataFrame:
    """
    1) Liest 'landuse=industrial'-Ways mithilfe des internen Location-Index.
    2) Filtert und labelt diese anhand der Kategorien.
    3) Speichert das Ergebnis in einem GeoPackage.

    Parameter:
    -----------
    input_pbf : str
        Pfad zur Input-PBF-Datei.
    output_gpkg : str
        Pfad, unter dem das GeoPackage gespeichert werden soll.
    target_epsg : int
        Ziel-EPSG-Code für das Koordinatensystem.

    Returns:
    --------
    pd.DataFrame
        Ein GeoDataFrame mit den gefilterten industriellen Flächen.
    """
    print("Filtere industrielle Flächen...")
    with tqdm(desc="Processing ways", unit="ways") as ways_pbar:
        handler = IndustrialLanduseHandler(progress_bar=ways_pbar)
        handler.apply_file(input_pbf, locations=True)

    print("[OK] Industrielle Flächen gefiltert.")
    print(len(handler.filtered_data), "valid ways found.")

    if not handler.filtered_data:
        print("Keine validen Wege gefunden.")
        return gpd.GeoDataFrame(columns=['geometry', 'tags', 'osm_id'], crs=f"EPSG:{target_epsg}")

    # Erstelle ein GeoDataFrame
    gdf = gpd.GeoDataFrame(handler.filtered_data, geometry='geometry', crs=f"EPSG:{target_epsg}")

    # Filtere spezifische Tags heraus
    blacklist = [
        {'landuse': 'industrial'},
    ]

    # Entferne blacklisted Tags
    gdf_filtered = remove_specific_tag_rows(gdf, blacklist)

    # Speichere in ein GeoPackage
    gdf_filtered.to_file(output_gpkg, layer='industrieflaechen', driver='GPKG')
    print("Gefilterte Daten mit Labels gespeichert in", output_gpkg)
    return gdf_filtered


def create_labeled_database(folder_name: str, input_pbf: str, target_epsg: int = 4326):
    """
    Hauptfunktion: Liest den Input, extrahiert industrielle Flächen,
    filtert, labelt und holt zusätzlich Overpass-Daten für jedes Polygon.
    Speichert letztlich alle Ergebnisse in einem GeoPackage.

    Parameter:
    -----------
    folder_name : str
        Name des Ordners, in dem die Ergebnisse gespeichert werden.
    input_pbf : str
        Pfad zur Input-PBF-Datei.
    target_epsg : int, optional
        Ziel-EPSG-Code für das Koordinatensystem, Standard: 4326.

    Returns:
    --------
    None
    """
    current_working_dir = os.getcwd()
    print("Aktuelles Arbeitsverzeichnis:", current_working_dir)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("Basisverzeichnis des Skripts:", base_dir)

    output_gpkg_dir = os.path.join(base_dir, f"../{folder_name}")
    output_dir = os.path.join(output_gpkg_dir, "geo_json_files")
    output_gpkg = os.path.join(output_gpkg_dir, "industrial_landuse.gpkg")

    os.makedirs(output_gpkg_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("Erstelle GeoPackage mit gefilterten industriellen Flächen...")
    gdf_final = create_gpkg(input_pbf, output_gpkg, target_epsg)
    print("[OK] GeoPackage erstellt.")

    gdf_final.to_file(output_gpkg, layer='industrieflaechen', driver='GPKG')
    print("[OK] GeoPackage aktualisiert.")
    print("[DONE] Alle Schritte abgeschlossen.")


def filter_labeled_gpkg(input_gpkg_path: str, output_gpkg_path: str) -> gpd.GeoDataFrame:
    """
    Filtert ein GeoPackage, indem alle Zeilen entfernt werden, bei denen in der Spalte 'label'
    entweder None oder der String "None" steht, und speichert das gefilterte GeoPackage.

    Parameter:
    -----------
    input_gpkg_path : str
        Pfad zum ursprünglichen GeoPackage.
    output_gpkg_path : str
        Pfad, unter dem das gefilterte GeoPackage gespeichert werden soll.

    Returns:
    --------
    GeoDataFrame
        Das gefilterte GeoDataFrame.
    """
    gdf = gpd.read_file(input_gpkg_path)
    # Entferne Zeilen, bei denen 'label' None oder "None" ist.
    filtered_gdf = gdf[~gdf['label'].isnull() & (gdf['label'] != "None")]
    filtered_gdf.to_file(output_gpkg_path, driver="GPKG")
    print(f"[INFO] Filtered GeoPackage saved at: {output_gpkg_path}")
    return filtered_gdf

