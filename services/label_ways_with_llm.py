import geopandas as gpd
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
import math
import time
from tqdm import tqdm  # Für Fortschrittsanzeige
import pandas as pd  # Für pd.isnull

from services.llama_client import Llama32Client
from services.schemas import ChatMessage

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


# Pydantic-Schemas definieren
class LabelResponse(BaseModel):
    osm_id: int = Field(..., description="Eindeutige Kennung des OSM-Ways.")
    label: Optional[str] = Field(None, description="Zugewiesenes Label für den OSM-Way oder None, falls kein Label gewählt wurde.")


class BatchLabelResponse(BaseModel):
    responses: List[LabelResponse] = Field(..., description="Liste der Label-Antworten für einen Stapel von OSM-Ways.")


def process_single_batch(
        batch: gpd.GeoDataFrame,
        client: Llama32Client,
        fixed_labels: List[str],
) -> Optional[BatchLabelResponse]:
    """
    Verarbeitet einen einzelnen Batch und ruft den LLM zur Bestimmung des Labels auf.

    Parameter:
    -----------
    batch : GeoDataFrame
        Der aktuelle Batch als GeoDataFrame.
    client : Llama32Client
        Der initialisierte LLM-Client.
    fixed_labels : List[str]
        Liste der möglichen Labels.

    Returns:
    --------
    Optional[BatchLabelResponse]
        Ein BatchLabelResponse-Objekt oder None im Fehlerfall.
    """
    # Formatiere die Tags und osm_id als einen einzigen String
    batch_id_tags_str = batch.apply(
        lambda row: f"OSM ID: {row['osm_id']}\nTags: {row['tags']}",
        axis=1
    ).str.cat(sep="\n\n")

    system_message = "You are an expert in labeling OSM ways."  # Systemnachricht (kann bei Bedarf ins Deutsche übersetzt werden)

    user_message = f"""Please determine the appropriate label from the following list for each of the given tags for the following OSM ways. Use the provided tags, including the 'name' tag, to make an informed decision based on your expertise. If none of the labels fit, assign 'None'. Ensure high confidence in your decisions and only assign a label if you are certain.

The possible labels are: 
{', '.join(fixed_labels)}

Please label the following OSM ways:
{batch_id_tags_str}

Important: Only return the schema described in the instructions. Your are not allowed to return any additional information, text or comments."""
    # Hinweis: Die Benutzeranweisung bleibt in Englisch, da dies vermutlich vom LLM so erwartet wird.

    # Bereite die Nachrichten für den LLM vor
    messages = [
        ChatMessage(
            role="system",
            content=[{"type": "text", "text": system_message}]
        ),
        ChatMessage(
            role="user",
            content=[{"type": "text", "text": user_message}]
        )
    ]

    try:
        # Rufe die chat_completion-Methode auf
        response = client.chat_completion(
            messages=messages,
            temperature=0.0,
            output_schema=BatchLabelResponse
        )

        # Prüfe, ob response.content eine Instanz von BatchLabelResponse ist
        if isinstance(response.content, BatchLabelResponse):
            return response.content
        else:
            print(f"[WARN] Unerwarteter Antworttyp: {type(response.content)}. Antwortinhalt: {response.content}")
            return None

    except ValidationError as ve:
        # Fange Validierungsfehler von Pydantic ab
        print(f"[WARN] Validierungsfehler beim Verarbeiten des Batches: {ve}")
        return None
    except Exception as e:
        print(f"[WARN] Fehler beim Verarbeiten des Batches: {e}")
        return None


def classify_osm_ways(
        gpkg_path: str,
        output_gpkg_path: str,
        client: Llama32Client,
        fixed_labels: List[str],
        batch_size: int = 10,
        max_retries: Optional[int] = 3,  # Maximale Anzahl der Wiederholungsversuche
        retry_delay: int = 5  # Initiale Wartezeit in Sekunden zwischen den Wiederholungen
) -> gpd.GeoDataFrame:
    """
    Klassifiziert OSM-Ways mithilfe des LLM-Clients und aktualisiert das GeoDataFrame mit den zugewiesenen Labels.

    Parameter:
    -----------
    gpkg_path : str
        Pfad zur Eingabe-GeoPackage-Datei.
    output_gpkg_path : str
        Pfad zur Ausgabe-GeoPackage-Datei.
    client : Llama32Client
        Der initialisierte LLM-Client.
    fixed_labels : List[str]
        Liste der möglichen Labels.
    batch_size : int, optional
        Anzahl der Zeilen, die in jedem Batch verarbeitet werden sollen. Standard: 10.
    max_retries : Optional[int], optional
        Maximale Anzahl der Wiederholungsversuche (None bedeutet unendliche Wiederholungen).
    retry_delay : int, optional
        Anfangswartezeit in Sekunden zwischen Wiederholungen. Standard: 5.

    Returns:
    --------
    GeoDataFrame
        Das aktualisierte GeoDataFrame mit einer neuen 'label'-Spalte.
    """
    # Lade die GeoPackage-Datei
    gdf = gpd.read_file(gpkg_path)

    # Füge eine neue Spalte für die Labels hinzu, falls sie noch nicht existiert
    if 'label' not in gdf.columns:
        gdf['label'] = None
        print("[INFO] 'label'-Spalte zum GeoDataFrame hinzugefügt.")

    # Bestimme die Gesamtzahl der Zeilen
    total_rows = len(gdf)

    # Berechne die Anzahl der Batches
    num_batches = math.ceil(total_rows / batch_size)

    print(f"[INFO] Gesamtanzahl der Zeilen: {total_rows}")
    print(f"[INFO] Verarbeitung in {num_batches} Batches mit jeweils bis zu {batch_size} Zeilen.\n")

    # Iteriere über alle Batches mit tqdm für die Fortschrittsanzeige
    for batch_num in tqdm(range(num_batches), desc="Processing Batches", unit="batch"):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_rows)
        batch = gdf.iloc[start_idx:end_idx]

        attempt = 0
        success = False
        batch_response: Optional[BatchLabelResponse] = None

        # Versuche, den Batch zu verarbeiten (mit Wiederholungsversuchen)
        while not success:
            if max_retries is not None and attempt >= max_retries:
                print(f"[ERROR] Maximale Wiederholungen für Batch {batch_num + 1} überschritten. Überspringe diesen Batch.")
                # Optional: Setze die Labels auf None oder belasse vorhandene Werte
                gdf.loc[gdf['osm_id'].isin(batch['osm_id']), 'label'] = None
                break

            if attempt > 0:
                print(f"[WARN] Wiederhole Batch {batch_num + 1} (Versuch {attempt + 1})...")

            batch_response = process_single_batch(batch, client, fixed_labels)
            if batch_response and batch_response.responses:
                success = True
            else:
                attempt += 1
                print(f"[WARN] Batch {batch_num + 1} konnte im Versuch {attempt} nicht verarbeitet werden.")
                # Exponentielles Backoff
                sleep_time = retry_delay * (2 ** (attempt - 1))
                print(f"[INFO] Warte {sleep_time} Sekunden vor erneutem Versuch...")
                time.sleep(sleep_time)

        if success and isinstance(batch_response, BatchLabelResponse):
            # Aktualisiere das GeoDataFrame mit den erhaltenen Labels
            for label_resp in batch_response.responses:
                osm_id = label_resp.osm_id
                label = label_resp.label
                gdf.loc[gdf['osm_id'] == osm_id, 'label'] = label
        else:
            # Falls der Batch nach wiederholten Versuchen übersprungen wird
            print(f"[WARN] Batch {batch_num + 1} nach {attempt} Versuchen übersprungen.")
            gdf.loc[gdf['osm_id'].isin(batch['osm_id']), 'label'] = None

    print("[INFO] Alle Batches verarbeitet.")

    # Speichere das aktualisierte GeoDataFrame
    gdf.to_file(output_gpkg_path, driver="GPKG")
    print(f"[INFO] Aktualisiertes GeoDataFrame wurde gespeichert unter {output_gpkg_path}")

    return gdf

