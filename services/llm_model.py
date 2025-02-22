import math
import os
import random
import time

import numpy as np
import pandas as pd
from typing import List, Optional
from PIL import Image
import geopandas as gpd
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from services.llama_client import Llama32Client
from services.schemas import ChatMessage


# Pydantic-Schemas definieren
class LabelResponse(BaseModel):
    """
    Schema für die Label-Antwort eines einzelnen OSM-Ways.

    Parameter:
    -----------
    osm_id : int
        Eindeutige Kennung des OSM-Ways.
    label : Optional[str]
        Zugewiesenes Label. Es darf nur ein Label aus der vorgegebenen Liste ausgewählt werden.
    """
    osm_id: int = Field(..., description="Unique identifier of the OSM way.")
    label: Optional[str] = Field(None, description="Chosen label. Only provided labels are allowed.")


class BatchLabelResponse(BaseModel):
    """
    Schema für die Label-Antworten eines Batches von OSM-Ways.

    Parameter:
    -----------
    responses : List[LabelResponse]
        Liste der Label-Antworten für einen Batch von OSM-Ways.
    """
    responses: List[LabelResponse] = Field(..., description="List of label responses for a batch of OSM ways.")


def load_data_to_df_with_tags(
        dataset_path: str,
        gpkg_path: str,
        selected_labels: List[str] = None,
        limit_per_class: int = None,
        expected_shape: tuple = None
) -> pd.DataFrame:
    """
    Lädt die Pfade zu .png-Bildern aus Unterordnern (jeweils eine Klasse) und erzeugt ein DataFrame,
    das die Spalten "path", "id" und "ground_truth" enthält. Die 'id' wird aus dem Dateinamen extrahiert,
    der das Format "way_{id}_{name}_overlay.png" hat.

    Anschließend wird eine GeoPackage-Datei (gpkg) eingelesen, die mindestens die Spalten "osm_id" und "tags"
    enthält. Das resultierende DataFrame wird dann mit diesen Tags (Merge über id/osm_id) angereichert
    und abschließend durchmischt (shuffle).

    Parameter:
    -----------
    dataset_path : str
        Pfad zum Haupt-Dataset-Ordner, in dem jeder Unterordner eine Klasse repräsentiert.
    gpkg_path : str
        Pfad zur GeoPackage-Datei (.gpkg), die unter anderem die Spalten "osm_id" und "tags" enthält.
    selected_labels : List[str], optional
        Liste der Klassen (Unterordnernamen), die geladen werden sollen. Falls None, werden alle Klassen verwendet.
    limit_per_class : int, optional
        Maximale Anzahl an Dateien pro Klasse, Standard: None (unbegrenzt).
    expected_shape : tuple, optional
        Erwartete Form der PNG-Bilder, z.B. (512, 512, 4). Falls None, wird keine Formprüfung durchgeführt.

    Returns:
    --------
    pd.DataFrame
        DataFrame mit den Spalten:
            "path"         - Pfad zur Bilddatei,
            "id"           - Aus dem Dateinamen extrahierte ID,
            "ground_truth" - Klassenbezeichnung (Unterordnername),
            "tags"         - Aus der GeoPackage-Datei übernommene Tags (basierend auf osm_id).
    """
    paths = []
    ids = []
    labels = []

    # Falls keine spezifischen Labels angegeben sind, verwende alle Unterordner
    if selected_labels is None:
        selected_labels = [label for label in os.listdir(dataset_path)
                           if os.path.isdir(os.path.join(dataset_path, label))]
    else:
        # Nur vorhandene Unterordner verwenden
        selected_labels = [label for label in selected_labels
                           if os.path.isdir(os.path.join(dataset_path, label))]

    print(f"Verwendete Klassen: {selected_labels}")

    # Für jede ausgewählte Klasse
    for label_name in selected_labels:
        label_path = os.path.join(dataset_path, label_name)
        all_files = [f for f in os.listdir(label_path) if f.lower().endswith('.png')]

        # Mische die Dateien zufällig und setze ggf. ein Limit pro Klasse
        random.shuffle(all_files)
        if limit_per_class is not None:
            all_files = all_files[:limit_per_class]

        for file_name in all_files:
            file_path = os.path.join(label_path, file_name)
            try:
                # Falls expected_shape gesetzt ist, prüfe die Bildform
                if expected_shape is not None:
                    image = Image.open(file_path).convert("RGBA")
                    image_np = np.array(image)
                    if image_np.shape != expected_shape:
                        print(f"Überspringe {file_path}: Shape {image_np.shape} != {expected_shape}")
                        continue

                # Extrahiere die ID aus dem Dateinamen. Erwartetes Format: "way_{id}_{name}_overlay.png"
                if file_name.startswith("way_") and file_name.endswith("_overlay.png"):
                    id_part = file_name[len("way_"):-len("_overlay.png")]
                    id_extracted = id_part.split("_")[0]
                else:
                    print(f"Dateiname {file_name} entspricht nicht dem erwarteten Format.")
                    id_extracted = ""

                paths.append(file_path)
                ids.append(id_extracted)
                labels.append(label_name)

            except Exception as e:
                print(f"Fehler beim Verarbeiten von {file_path}: {e}")

    # Erstelle das DataFrame aus den gesammelten Listen
    df = pd.DataFrame({
        "path": paths,
        "id": ids,
        "ground_truth": labels
    })

    print(f"Erstelltes DataFrame mit {len(df)} Einträgen.")

    # Lese die GeoPackage-Datei als GeoDataFrame ein
    gdf = gpd.read_file(gpkg_path)

    # Konvertiere die Spalte osm_id in einen String (für den Merge)
    gdf["osm_id"] = gdf["osm_id"].astype(str)
    df["id"] = df["id"].astype(str)

    # Merge: Verbinde das DataFrame über df.id und gdf.osm_id
    df = df.merge(gdf[["osm_id", "tags"]], left_on="id", right_on="osm_id", how="left")
    df.drop("osm_id", axis=1, inplace=True)

    # Durchmische den finalen DataFrame
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def process_single_batch(
        batch: gpd.GeoDataFrame,
        client: Llama32Client,
        fixed_labels: List[str],
) -> Optional[BatchLabelResponse]:
    """
    Verarbeitet einen einzelnen Batch und ruft den LLM zur Bestimmung der Labels auf.

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
    # Formatierung: Verbinde osm_id und tags zu einem einzigen String
    batch_id_tags_str = batch.apply(
        lambda row: f"OSM ID: {row['id']}\nTags: {row['tags']}",
        axis=1
    ).str.cat(sep="\n\n")

    system_message = "You are an expert in labeling OSM ways."

    user_message = f"""For each provided sample, you will receive the tags from the boundary geometry of the industrial area. Based solely on the provided tags, classify the sample by selecting exactly one label from the following list:

{', '.join(fixed_labels)}

IMPORTANT: Only return the schema described in the instructions. You are not allowed to return any additional information, text or comments. Always assign a label, even if you are not sure. Never assign anything other than the provided labels.

Samples to label:
{batch_id_tags_str}"""

    # Erstelle die Nachrichten für den LLM
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
        print(f"[WARN] Validierungsfehler beim Verarbeiten des Batches: {ve}")
        return None
    except Exception as e:
        print(f"[WARN] Fehler beim Verarbeiten des Batches: {e}")
        return None


def classify_osm_ways(
        gdf: gpd.GeoDataFrame,
        client: Llama32Client,
        fixed_labels: List[str],
        batch_size: int = 10,
        max_retries: Optional[int] = None,  # None impliziert unendliche Wiederholungen
        retry_delay: int = 5  # Anfangswartezeit in Sekunden
) -> gpd.GeoDataFrame:
    """
    Klassifiziert OSM-Ways mithilfe des LLM-Clients und aktualisiert das GeoDataFrame
    mit einer neuen Spalte 'predicted_label'.

    Parameter:
    -----------
    gdf : GeoDataFrame
        GeoDataFrame, das mindestens die Spalten 'osm_id' und 'tags' enthält.
    client : Llama32Client
        Der initialisierte LLM-Client.
    fixed_labels : List[str]
        Liste der möglichen Labels.
    batch_size : int, optional
        Anzahl der Zeilen, die in jedem Batch verarbeitet werden sollen. Standard: 10.
    max_retries : Optional[int], optional
        Maximale Anzahl an Wiederholungsversuchen (None für unendliche Wiederholungen).
    retry_delay : int, optional
        Anfangswartezeit in Sekunden zwischen den Wiederholungen. Standard: 5.

    Returns:
    --------
    GeoDataFrame
        Das aktualisierte GeoDataFrame mit der neuen Spalte 'predicted_label'.
    """
    # Erstelle eine neue Spalte 'predicted_label' mit None-Werten
    gdf['predicted_label'] = None

    total_rows = len(gdf)
    num_batches = math.ceil(total_rows / batch_size)

    print(f"[INFO] Total rows: {total_rows}")
    print(f"[INFO] Processing in {num_batches} batches of up to {batch_size} rows each.\n")

    for batch_num in tqdm(range(num_batches), desc="Processing Batches", unit="batch"):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_rows)
        batch = gdf.iloc[start_idx:end_idx]

        attempt = 0
        success = False
        batch_response: Optional[BatchLabelResponse] = None

        while not success:
            if max_retries is not None and attempt >= max_retries:
                print(f"[ERROR] Exceeded maximum retries for batch {batch_num + 1}. Skipping this batch.")
                gdf.loc[gdf['id'].isin(batch['id']), 'predicted_label'] = None
                break

            if attempt > 0:
                print(f"[WARN] Retrying batch {batch_num + 1} (Attempt {attempt + 1})...")

            batch_response = process_single_batch(batch, client, fixed_labels)
            if batch_response and batch_response.responses:
                success = True
            else:
                attempt += 1
                print(f"[WARN] Failed to process batch {batch_num + 1} on attempt {attempt}.")
                sleep_time = retry_delay * (2 ** (attempt - 1))
                print(f"[INFO] Waiting for {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)

        if success and isinstance(batch_response, BatchLabelResponse):
            for label_resp in batch_response.responses:
                print(f"[INFO] OSM ID {label_resp.osm_id} classified as '{label_resp.label}'.")
                osm_id = label_resp.osm_id
                label = label_resp.label
                gdf.loc[gdf['id'] == str(osm_id), 'predicted_label'] = label
        else:
            print(f"[WARN] Skipped batch {batch_num + 1} after {attempt} attempts.")
            gdf.loc[gdf['id'].isin(batch['id']), 'predicted_label'] = None

    print("[INFO] All batches processed.")
    return gdf


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(gdf, true_col='ground_truth', pred_col='predicted_label', fixed_labels=None):
    """
    Berechnet und plottet die Confusion Matrix anhand eines GeoDataFrames.
    Es werden relative Werte (Prozentsätze) statt absoluten Werten angezeigt.

    Parameter:
    -----------
    gdf : GeoDataFrame
        Das GeoDataFrame, das mindestens die Spalten für die Ground Truth und die
        vorhergesagten Labels enthält.
    true_col : str, optional
        Name der Spalte mit den Ground Truth Labels. Standard: 'ground_truth'.
    pred_col : str, optional
        Name der Spalte mit den vorhergesagten Labels. Standard: 'predicted_label'.
    fixed_labels : List[str], optional
        Liste aller möglichen Labels in der gewünschten Reihenfolge.

    Returns:
    --------
    None
    """
    y_true = gdf[true_col].values
    y_pred = gdf[pred_col].values

    cm = confusion_matrix(y_true, y_pred, labels=fixed_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=fixed_labels, yticklabels=fixed_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Normalized Confusion Matrix")
    plt.show()

