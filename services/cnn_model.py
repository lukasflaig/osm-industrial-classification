import os
import random
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.training.adam import AdamOptimizer
from tf_keras import layers, regularizers
from tf_keras.src import models
from tf_keras.src.callbacks import EarlyStopping
from tf_keras.src.utils import to_categorical
import seaborn as sns


def load_data(
        dataset_path: str,
        expected_shape: tuple = (4, 512, 512),
        selected_labels: List[str] = None,
        limit_per_class: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lädt .npy-Bilddaten (bereits im Bereich [0,1]) aus den Klassen, die in der Liste
    selected_labels angegeben sind. Falls selected_labels None ist, werden alle vorhandenen Klassen geladen.
    Optional pro Klasse limit_per_class.
    Transponiert von (4,512,512) auf (512,512,4) und wandelt in float32 um.

    Parameter:
    -----------
    dataset_path : str
        Pfad zum Haupt-Dataset-Ordner (pro Unterordner = eine Klasse).
    expected_shape : tuple, optional
        Erwartete Form der .npy-Dateien, Standard: (4, 512, 512).
    selected_labels : list, optional
        Liste der Klassen (Ordnernamen), die geladen werden sollen.
        Falls None, werden alle Klassen geladen.
    limit_per_class : int, optional
        Maximale Anzahl an Dateien pro Klasse, Standard: None = unbegrenzt.

    Returns:
    --------
    X : np.ndarray
        Array mit Shape (N, 512, 512, 4), falls expected_shape=(4,512,512).
    y : np.ndarray
        Array der Klassenlabels (Strings) mit Länge N.
    """
    data = []
    labels = []

    # Falls keine spezifischen Labels angegeben wurden, alle Unterordner als Klassen verwenden
    if selected_labels is None:
        selected_labels = [label for label in os.listdir(dataset_path)
                           if os.path.isdir(os.path.join(dataset_path, label))]
    else:
        # Nur Labels verwenden, die tatsächlich als Unterordner vorhanden sind
        selected_labels = [label for label in selected_labels
                           if os.path.isdir(os.path.join(dataset_path, label))]

    print(f"Verwendete Klassen: {selected_labels}")

    # Durchlaufe die ausgewählten Klassen
    for label_name in selected_labels:
        label_path = os.path.join(dataset_path, label_name)
        all_files = [f for f in os.listdir(label_path) if f.endswith('.npy')]

        # Zufällig mischen, falls pro Klasse ein Limit gesetzt werden soll
        random.shuffle(all_files)

        if limit_per_class is not None:
            all_files = all_files[:limit_per_class]

        for file_name in all_files:
            file_path = os.path.join(label_path, file_name)
            try:
                # Datei laden
                image = np.load(file_path)

                # Überprüfen, ob die Bildform wie erwartet ist
                if image.shape != expected_shape:
                    print(f"Überspringe {file_path}: Shape {image.shape} != {expected_shape}")
                    continue

                # Transponieren von (4,512,512) auf (512,512,4)
                image = np.transpose(image, (1, 2, 0))
                # In float32 konvertieren (Werte bleiben 0/1)
                image = image.astype(np.float32)

                data.append(image)
                labels.append(label_name)

            except Exception as e:
                print(f"Fehler beim Laden von {file_path}: {e}")

    # In numpy-Arrays packen
    X = np.array(data, dtype=np.float32)
    y = np.array(labels)

    print(f"Geladene Samples: {X.shape[0]}")
    if X.shape[0] > 0:
        print(f"Shape pro Sample: {X.shape[1:]}")
    return X, y


def preprocess_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, LabelEncoder]:
    """
    Kodiert die Labels, wandelt sie in eine kategorische Darstellung um und teilt die Daten in Trainings- und Testdaten auf.

    Parameter:
    -----------
    X : np.ndarray
        Array mit Bilddaten.
    y : np.ndarray
        Array mit den zugehörigen Labels.

    Returns:
    --------
    X_train : np.ndarray
        Trainingsdaten.
    X_test : np.ndarray
        Testdaten.
    y_train : np.ndarray
        One-Hot-encodierte Labels der Trainingsdaten.
    y_test : np.ndarray
        One-Hot-encodierte Labels der Testdaten.
    num_classes : int
        Anzahl der eindeutigen Klassen.
    label_encoder : LabelEncoder
        Fitted LabelEncoder zur späteren Decodierung.
    """
    print("Preprocessing data...")
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")

    # One-hot encode labels
    y_categorical = to_categorical(y_encoded, num_classes=num_classes)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=np.argmax(y_categorical, axis=1)
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, num_classes, label_encoder


def build_model(input_shape: Tuple[int, int, int], num_classes: int) -> models.Model:
    """
    Erstellt ein Convolutional Neural Network (CNN) Modell.

    Parameter:
    -----------
    input_shape : tuple
        Form der Eingabedaten (Höhe, Breite, Kanäle).
    num_classes : int
        Anzahl der Ausgabeklassen.

    Returns:
    --------
    model : tf.keras.Model
        Kompiliertes CNN-Modell.
    """
    print("Building the CNN model...")
    model = models.Sequential([
        # Erster Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        # Zweiter Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Dritter Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Flatten und Dense Layers
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.summary()
    return model


def compile_model(model: models.Model, lr: float = 0.0001) -> None:
    """
    Kompiliert das CNN-Modell mit Optimizer, Loss-Funktion und Metriken.

    Parameter:
    -----------
    model : tf.keras.Model
        Das zu kompillierende CNN-Modell.
    lr : float, optional
        Lernrate für den Adam-Optimizer, Standardwert: 0.0001.

    Returns:
    --------
    None
    """
    print("Compiling the model...")
    model.compile(
        optimizer=AdamOptimizer(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


def train_model(model: models.Model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, batch_size: int = 16, epochs: int = 50, patience: int = 5):
    """
    Trainiert das CNN-Modell mit den bereitgestellten Trainings- und Testdaten.

    Parameter:
    -----------
    model : tf.keras.Model
        Das kompilierte CNN-Modell.
    X_train : np.ndarray
        Trainingsdaten.
    y_train : np.ndarray
        One-Hot-encodierte Labels der Trainingsdaten.
    X_test : np.ndarray
        Testdaten.
    y_test : np.ndarray
        One-Hot-encodierte Labels der Testdaten.
    batch_size : int, optional
        Größe der Trainingsbatches, Standardwert: 16.
    epochs : int, optional
        Maximale Anzahl an Trainingsepochen, Standardwert: 50.
    patience : int, optional
        Anzahl der Epochen ohne Verbesserung, nach denen das Training abgebrochen wird, Standardwert: 5.

    Returns:
    --------
    history : History
        Das Trainingsverlauf-Objekt.
    """
    print("Setting up callbacks...")
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    print("Starting training...")
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )
    return history


def evaluate_model(model: models.Model, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
    """
    Bewertet das trainierte CNN-Modell anhand der Testdaten.

    Parameter:
    -----------
    model : tf.keras.Model
        Das trainierte CNN-Modell.
    X_test : np.ndarray
        Testdaten.
    y_test : np.ndarray
        One-Hot-encodierte Labels der Testdaten.

    Returns:
    --------
    test_loss : float
        Verlustwert auf dem Testset.
    test_accuracy : float
        Genauigkeit auf dem Testset.
    """
    print("Evaluating the model on the test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    return test_loss, test_accuracy


def save_model(model: models.Model, filepath='my_model.h5'):
    """
    Speichert das gegebene Modell als .h5-Datei (oder im SavedModel-Format, wenn gewünscht).

    Parameter:
    -----------
    model : tf.keras.Model
        Das zu speichernde Keras-Modell.
    filepath : str, optional
        Pfad/Dateiname, unter dem das Modell gespeichert werden soll. Standardwert: 'my_model.h5'.

    Returns:
    --------
    None
    """
    model.save(filepath)
    print(f"Modell erfolgreich unter '{filepath}' gespeichert.")


def plot_history(history):
    """
    Stellt den Trainingsverlauf (Genauigkeit und Verlust) über die Epochen grafisch dar.

    Parameter:
    -----------
    history : History
        Trainingsverlauf-Objekt, das Metriken wie 'accuracy', 'val_accuracy', 'loss' und 'val_loss' enthält.

    Returns:
    --------
    None
    """
    print("Plotting training history...")
    plt.figure(figsize=(12, 5))

    # Genauigkeits-Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Verlust-Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model: models.Model, X_test: np.ndarray, y_test: np.ndarray, label_encoder=None):
    """
    Erstellt und zeigt eine normalisierte Confusion Matrix basierend auf den Vorhersagen des Modells.

    Parameter:
    -----------
    model : tf.keras.Model
        Das trainierte Keras-Modell.
    X_test : np.ndarray
        Testdaten (Bilder).
    y_test : np.ndarray
        One-Hot-encodierte Labels zu den Testdaten.
    label_encoder : LabelEncoder, optional
        Falls übergeben, werden die Achsenbeschriftungen der Confusion Matrix
        mit den ursprünglichen Klassen-Namen versehen. Andernfalls werden Indizes (0,1,2,...) genutzt.

    Returns:
    --------
    None
    """
    # 1) Vorhersagen erhalten
    y_pred = model.predict(X_test)
    # 2) One-Hot zu Klassen-Indices umwandeln
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # 3) Confusion Matrix berechnen (absolute Werte)
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    # Normalisiere die Confusion Matrix zeilenweise
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if label_encoder is not None:
        class_names = label_encoder.classes_
    else:
        # Platzhalter-Namen erstellen, wenn kein LabelEncoder vorhanden ist
        unique_class_count = cm.shape[0]
        class_names = [str(i) for i in range(unique_class_count)]

    # 4) Plotten der normalisierten Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()