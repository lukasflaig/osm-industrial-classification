# KI-basierte Klassifikation von Industrieflächen auf Basis von OpenStreetMap-Daten

Dieses Python-Projekt führt eine Klassifikation von Industrieflächen basierend auf OpenStreetMap-Daten durch. Es umfasst eine Datenverarbeitungspipeline sowie die Entwicklung und Evaluierung von KI-Modellen (CNN und LLM).

## Verwendung

Die Hauptfunktionen des Projekts werden über `main.py` ausgeführt:
```bash
python main.py
```

## Projektstruktur

```
├── main.py
├── requirements.txt
├── data/
│   ├── *.osm.pbf
│   └── dataset_germany/
├── database_germany_all/
│   ├── *.gpkg
│   ├── *.geojson
├── services/
│   ├── create_labeled_database.py
│   ├── label_ways_with_llm.py
│   ├── geojson_by_way_id.py
│   ├── cnn_model.py
│   ├── llm_model.py
├── notebooks/
│   ├── test_cnn.py
│   ├── test_llm.py
└── secrets/

```

## Hinweise

- Alle relevanten Verarbeitungsschritte sind im Ordner `services/` enthalten.
- Der Ordner `database_germany_all/` enthält die Datenbasis und Zwischenergebnisse der Pipeline.
- Der Ordner `secrets/` enthält die Zugangsschlüssel zum LLM, das in der Google Cloud gehostet wird. Diese Dateien werden nicht zur Verfügung gestellt.
- Die `.osm.pbf`-Dateien müssen manuell heruntergeladen und im Ordner `data/` abgelegt werden. Die Dateien können von [Geofabrik](https://download.geofabrik.de/) heruntergeladen werden.
- Da der Datensatz als `.zip`-Datei zu groß für GitHub ist, muss dieser manuell heruntergeladen und im Ordner `data/` entpackt werden. Der Datensatz kann hier heruntergeladen werden: [Industrieflächen-Datensatz](https://studfrauasde.sharepoint.com/:u:/s/GEOAI/EdJlRrPqymFLtP72uX8q7BYB_fl6EWkIhcMeMTUCkAFM7Q?e=QqMeqR)

## Abhängigkeiten

Die für das Projekt benötigten Python-Abhängigkeiten sind in der Datei `requirements.txt` hinterlegt.
