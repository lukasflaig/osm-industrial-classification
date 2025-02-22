import json
import os
import time
from typing import Optional

import osmium
import requests
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from tqdm import tqdm


def _get_boundary_polygon(way_id: int) -> Polygon:
    """
    Ruft über Overpass einen OSM-Way (als Grenze) ab und erstellt daraus ein shapely Polygon.
    Der Way muss geschlossen sein.

    Parameter:
    -----------
    way_id : int
        Die OSM-Way-ID, die als Grenze verwendet wird.

    Returns:
    --------
    Polygon
        Ein shapely Polygon, aufgebaut aus den Knoten des Ways.

    Raises:
    -------
    ValueError
        Falls der Way nicht gefunden wird oder kein geschlossenes Polygon bildet.
    """
    # Overpass-Abfrage, um den Way und seine Knoten abzurufen.
    query = f"""
    [out:json];
    way({way_id});
    (._;>;);
    out body;
    """
    url = "https://overpass-api.de/api/interpreter"
    response = requests.post(url, data={'data': query})
    response.raise_for_status()
    data = response.json()

    elements = data.get("elements", [])

    # Erstelle ein Nachschlagewerk von Knoten-ID zu Koordinate (lon, lat).
    node_lookup = {}
    boundary_elem = None
    for elem in elements:
        if elem["type"] == "node":
            node_lookup[elem["id"]] = (elem["lon"], elem["lat"])
        elif elem["type"] == "way" and elem["id"] == way_id:
            boundary_elem = elem

    if boundary_elem is None:
        raise ValueError(f"Boundary way {way_id} not found in response.")

    # Erstelle eine Liste von Koordinaten anhand der Knoten-IDs des Ways.
    coords = []
    for nid in boundary_elem.get("nodes", []):
        if nid in node_lookup:
            coords.append(node_lookup[nid])
        else:
            print(f"Warning: Node {nid} not found for way {way_id}.")

    # Überprüfen, ob der Way ein geschlossenes Polygon bildet.
    if len(coords) < 4 or coords[0] != coords[-1]:
        raise ValueError("The provided way is not a closed polygon (or does not have enough nodes).")

    try:
        poly = Polygon(coords)
    except Exception as e:
        raise ValueError(f"Could not form a polygon: {e}")

    return poly


def _poly_to_poly_string(poly: Polygon) -> str:
    """
    Konvertiert ein shapely Polygon in einen String, der für den 'poly'-Filter von Overpass geeignet ist.
    Overpass erwartet Koordinaten im Format "lat lon" (beachte die Reihenfolge).

    Parameter:
    -----------
    poly : Polygon
        Ein shapely Polygon.

    Returns:
    --------
    str
        Ein String im Format "lat lon lat lon ...", der die Exterior-Koordinaten des Polygons auflistet.
    """
    # Extrahiere die Exterior-Koordinaten.
    coords = poly.exterior.coords
    # Overpass erwartet "lat lon"; die Polygon-Koordinaten liegen im Format (lon, lat) vor.
    coord_list = [f"{lat} {lon}" for lon, lat in coords]
    poly_str = " ".join(coord_list)
    return poly_str


def get_ways_inside_boundary(boundary_way_id: int, output_dir: str, retries: int = 3, target_epsg: int = 4326) -> Optional[str]:
    """
    Verarbeitet einen OSM-Way, der als Grenze dient, und führt folgende Schritte aus:
      1. Ruft das zugehörige Polygon der Grenze ab.
      2. Verwendet das Polygon, um über Overpass alle Ways innerhalb der Grenze abzufragen.
      3. Rekonstruiert die Geometrie für jeden Way.
      4. Speichert die rekonstruierten Geometrien als GeoJSON-Datei im angegebenen Ausgabeverzeichnis
         unter dem Dateinamen "way_{boundary_way_id}.geojson".
      5. Gibt den relativen Dateinamen der gespeicherten GeoJSON-Datei zurück.

    Parameter:
    -----------
    boundary_way_id : int
        Die OSM-Way-ID, die als Grenze verwendet wird.
    output_dir : str
        Das Verzeichnis, in dem die GeoJSON-Datei gespeichert werden soll.
    retries : int, optional
        Anzahl der Wiederholungsversuche bei der Overpass-Abfrage, Standard: 3.
    target_epsg : int, optional
        EPSG-Code des Ziel-Koordinatensystems, Standard: 4326.

    Returns:
    --------
    Optional[str]
        Der relative Dateiname der gespeicherten GeoJSON-Datei, oder None, falls ein Fehler auftritt.
    """
    # Schritt 1: Grenze als Polygon abrufen.
    try:
        boundary_poly = _get_boundary_polygon(boundary_way_id)
        boundary_poly.simplify(0.001, preserve_topology=True)  # Polygon vereinfachen
        poly_str = _poly_to_poly_string(boundary_poly)
    except Exception as e:
        print(f"Error getting boundary polygon: {e}")
        return None

    # Schritt 2: Verwende den Polygon-String in einer Overpass-Abfrage, um alle Ways innerhalb abzurufen.
    query = f"""
    [out:json];
    way(poly:"{poly_str}");
    (._;>;);
    out body;
    """
    overpass_url = "https://overpass-api.de/api/interpreter"
    data = None

    for i in range(retries):
        try:
            response = requests.post(overpass_url, data={'data': query}, timeout=60)
            response.raise_for_status()
            data = response.json()
            break  # Bei Erfolg Schleife beenden
        except requests.exceptions.RequestException as e:
            print(f"Error querying Overpass: {e}")
            if i < retries - 1:
                sleep_time = 2 ** i
                time.sleep(sleep_time)  # Exponentielles Backoff
            else:
                return None

    if not data:
        return None

    elements = data.get("elements", [])
    features = []

    # Erstelle ein Nachschlagewerk für Knoten-ID zu Koordinate.
    node_lookup = {}
    for elem in elements:
        if elem["type"] == "node":
            node_lookup[elem["id"]] = (elem["lon"], elem["lat"])

    # Verarbeite jedes Way-Element und rekonstruiere dessen Geometrie.
    for elem in elements:
        if elem["type"] == "way":
            coords = []
            for nid in elem.get("nodes", []):
                if nid in node_lookup:
                    coords.append(node_lookup[nid])
                else:
                    print(f"Warning: Node {nid} not found for way {elem['id']}.")
            if len(coords) < 2:
                continue
            # Wenn der Way geschlossen ist und mindestens 4 Knoten hat, versuche ein Polygon zu bilden, ansonsten eine LineString.
            if coords[0] == coords[-1] and len(coords) >= 4:
                try:
                    geom = Polygon(coords)
                except Exception as e:
                    print(f"Error forming polygon for way {elem['id']}: {e}")
                    geom = LineString(coords)
            else:
                geom = LineString(coords)

            features.append({
                "id": elem["id"],
                "osm_type": "way",
                "tags": elem.get("tags", {}),
                "geometry": geom
            })

    if not features:
        return None

    gdf = gpd.GeoDataFrame(features)
    gdf.set_crs(epsg=target_epsg, inplace=True)

    # Sicherstellen, dass das Ausgabeverzeichnis existiert.
    os.makedirs(output_dir, exist_ok=True)

    # Definiere den Ausgabedateipfad.
    filename = f"way_{boundary_way_id}.geojson"
    output_path = os.path.join(output_dir, filename)

    try:
        # Speichere das GeoDataFrame als GeoJSON-Datei.
        gdf.to_file(output_path, driver='GeoJSON')
    except Exception as e:
        return None

    # Gib den relativen Dateinamen der gespeicherten GeoJSON-Datei zurück.
    return filename


def geojson_to_gdf(geojson_path: str, target_epsg: int = 4326) -> gpd.GeoDataFrame:
    """
    Liest eine GeoJSON-Datei vom angegebenen Pfad ein und gibt ein GeoDataFrame zurück.
    Stellt sicher, dass das GeoDataFrame das gewünschte CRS besitzt und konvertiert die 'tags'-Spalte
    von JSON-Strings zu Dictionaries.

    Parameter:
    -----------
    geojson_path : str
        Der Dateipfad zur GeoJSON-Datei.
    target_epsg : int, optional
        EPSG-Code des gewünschten Koordinatensystems, Standard: 4326.

    Returns:
    --------
    geopandas.GeoDataFrame
        Ein GeoDataFrame, das die Daten aus der GeoJSON-Datei enthält, das gewünschte CRS hat und bei dem
        die 'tags'-Spalte als Dictionaries vorliegt.

    Raises:
    -------
    FileNotFoundError
        Falls die GeoJSON-Datei am angegebenen Pfad nicht existiert.
    ValueError
        Falls die Datei nicht als gültige GeoJSON gelesen werden kann oder kein CRS definiert ist.
    """
    try:
        # Lese die GeoJSON-Datei ein.
        gdf = gpd.read_file(geojson_path)

        # Stelle sicher, dass das GeoDataFrame ein definiertes CRS hat.
        if gdf.crs is None:
            raise ValueError("Das GeoDataFrame hat kein definiertes CRS.")

        current_epsg = gdf.crs.to_epsg()
        if current_epsg != target_epsg:
            gdf = gdf.to_crs(epsg=target_epsg)

        # Konvertiere die 'tags'-Spalte von JSON-Strings zu Dictionaries.
        def parse_tags(tags_str):
            try:
                return json.loads(tags_str)
            except (json.JSONDecodeError, TypeError):
                return {}

        if 'tags' in gdf.columns:
            gdf['tags'] = gdf['tags'].apply(parse_tags)
        else:
            print("Warnung: 'tags'-Spalte nicht im GeoDataFrame gefunden.")

        return gdf
    except FileNotFoundError:
        raise FileNotFoundError(f"Die Datei {geojson_path} existiert nicht.")
    except Exception as e:
        raise ValueError(f"Beim Lesen der GeoJSON-Datei ist ein Fehler aufgetreten: {e}")


def get_geo_json_files_for_gpkg(folder_name: str, gpkg_path: str, target_epsg: int = 4326):
    """
    Holt GeoJSON-Dateien für die in einem GeoPackage enthaltenen Ways und erstellt eine
    zweite GeoPackage-Datei, welche die Pfade zu den gespeicherten GeoJSON-Dateien enthält.

    Parameter:
    -----------
    folder_name : str
        Name des Ordners, in dem die Ergebnisse gespeichert werden sollen.
    gpkg_path : str
        Pfad zum ursprünglichen GeoPackage.
    target_epsg : int, optional
        EPSG-Code des gewünschten Koordinatensystems, Standard: 4326.

    Returns:
    --------
    None
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("Basisverzeichnis des Skripts:", base_dir)

    output_gpkg_dir = os.path.join(base_dir, f"../{folder_name}")
    output_dir = os.path.join(output_gpkg_dir, "geo_json_files")
    output_gpkg = os.path.join(output_gpkg_dir, "industrial_landuse_labeled_filtered_geo_json_files.gpkg")

    # Lade das GeoPackage
    gdf_final = gpd.read_file(gpkg_path)

    gdf_final['geo_json_filename'] = None
    total_requests = len(gdf_final)

    with tqdm(total=total_requests, desc="Verarbeite Overpass API Anfragen") as pbar:
        for idx, row in gdf_final.iterrows():
            boundary_way_id = row['osm_id']

            geo_json_filename = get_ways_inside_boundary(boundary_way_id, output_dir, target_epsg=target_epsg)

            # Speichere den relativen Pfad zur GeoJSON-Datei
            gdf_final.at[idx, 'geo_json_filename'] = geo_json_filename
            pbar.update(1)

    print("[OK] Alle OSM PBF Daten gesammelt und gespeichert.")
    print("Speichere GeoPackage mit OSM PBF Pfaden...")
    gdf_final.to_file(output_gpkg, layer='industrieflaechen', driver='GPKG')
    print("[OK] GeoPackage aktualisiert.")
    print("[DONE] Alle Schritte abgeschlossen.")

