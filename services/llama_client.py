import json
import os

import requests
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from typing import List, Optional, Type, Union
from pydantic import ValidationError, BaseModel, Field

from services.exceptions import AuthenticationError, ResponseValidationError, APIRequestError
from services.schemas import ChatMessage, VertexAIResponse, RequestPayload

from dotenv import load_dotenv

# Laden der Umgebungsvariablen aus der .env-Datei
load_dotenv()


class Llama32Client:
    """
    Ein Client zur Interaktion mit dem Vertex AI Generative AI Endpoint für das Llama 3.2 Modell.

    Dieser Client übernimmt die Authentifizierung mittels eines Google Service Accounts,
    aktualisiert das Zugriffstoken falls nötig und sendet Chat-Anfragen an die API.

    Attribute:
    -----------
    project_id : str
        Google Projekt-ID.
    region : str
        Google Region.
    endpoint : str
        Endpoint-URL für den MAAS-Service.
    credentials_path : str
        Pfad zur Google Service Account JSON-Datei.
    model : str
        Modellname, das verwendet wird.
    access_token : str
        OAuth2-Zugriffstoken für die API-Authentifizierung.
    """

    def __init__(self):
        self.project_id = os.getenv("GOOGLE_PROJECT_ID")
        self.region = os.getenv("GOOGLE_REGION")
        self.endpoint = os.getenv("GOOGLE_MAAS_ENDPOINT")
        self.credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH")
        self.model = os.getenv("GOOGLE_MODEL")
        self.access_token = self._get_access_token()

    def _get_access_token(self) -> str:
        """
        Holt das Zugriffstoken über die Google Service Account Credentials.

        Parameter:
        -----------
        None

        Returns:
        --------
        str
            Das gültige Zugriffstoken.

        Raises:
        -------
        AuthenticationError
            Falls die Credentials-Datei nicht gefunden wird oder ein anderer Fehler auftritt.
        """
        try:
            credentials = service_account.Credentials.from_service_account_file(self.credentials_path)
            scoped_credentials = credentials.with_scopes(["https://www.googleapis.com/auth/cloud-platform"])
            scoped_credentials.refresh(Request())
            return scoped_credentials.token
        except FileNotFoundError:
            raise AuthenticationError(f"Credentials file not found at path: {self.credentials_path}")
        except Exception as e:
            raise AuthenticationError(f"Failed to obtain access token: {str(e)}") from e

    def _refresh_access_token(self):
        """
        Aktualisiert das Zugriffstoken, falls es abgelaufen ist.

        Parameter:
        -----------
        None

        Returns:
        --------
        None

        Raises:
        -------
        AuthenticationError
            Falls das Token nicht aktualisiert werden kann.
        """
        try:
            credentials = service_account.Credentials.from_service_account_file(self.credentials_path)
            scoped_credentials = credentials.with_scopes(["https://www.googleapis.com/auth/cloud-platform"])
            if not scoped_credentials.valid or scoped_credentials.expired:
                scoped_credentials.refresh(Request())  # Token aktualisieren
                self.access_token = scoped_credentials.token
        except Exception as e:
            raise AuthenticationError(f"Failed to refresh access token: {str(e)}") from e

    @staticmethod
    def _get_format_instructions(output_schema: Type[BaseModel]) -> str:
        """
        Erstellt Formatierungsanweisungen basierend auf dem angegebenen Pydantic-Output-Schema.

        Parameter:
        -----------
        output_schema : Type[BaseModel]
            Das Pydantic-Modell, das das erwartete Format der Ausgabe definiert.

        Returns:
        --------
        str
            Eine Zeichenkette mit den Formatierungsanweisungen als JSON-Schema.
        """
        schema = output_schema.schema()
        schema_str = json.dumps(schema, indent=4)
        instructions = f"""The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
{schema_str}"""
        return instructions

    @staticmethod
    def _parse_response_content(
        content: str,
        output_schema: Optional[Type[BaseModel]] = None
    ) -> Union[str, BaseModel]:
        """
        Parst den Inhalt der API-Antwort in das angegebene Pydantic-Modell, falls vorhanden.

        Parameter:
        -----------
        content : str
            Der Inhalt der API-Antwort als Zeichenkette.
        output_schema : Optional[Type[BaseModel]], optional
            Das Pydantic-Modell, in das der Inhalt geparst werden soll.
            Falls None, wird der Inhalt als String zurückgegeben.

        Returns:
        --------
        Union[str, BaseModel]
            Das geparste Objekt oder der ursprüngliche Inhalt als Zeichenkette, falls kein gültiges JSON vorliegt.

        Raises:
        -------
        ResponseValidationError
            Falls das JSON nicht geparst werden kann oder das Modell die Validierung nicht besteht.
        """
        if output_schema:
            try:
                # Versuchen, den Inhalt als JSON zu laden
                content_data = json.loads(content)
                # Parsen des Inhalts in das angegebene Pydantic-Modell
                parsed_content = output_schema.parse_obj(content_data)
                return parsed_content
            except json.JSONDecodeError:
                # Falls der Inhalt kein gültiges JSON ist, als String zurückgeben
                return content
            except ValidationError as ve:
                # Bei Validierungsfehlern den Fehler weitergeben
                raise ResponseValidationError(f"Validation error: {ve}") from ve
        return content

    def chat_completion(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 4000,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        output_schema: Optional[Type[BaseModel]] = None
    ) -> VertexAIResponse:
        """
        Sendet eine Chat-Anfrage an den Vertex AI Generative AI Endpoint für das Llama 3.2 Modell.

        Parameter:
        -----------
        messages : List[ChatMessage]
            Eine Liste von ChatMessage-Instanzen, die den Gesprächsverlauf enthalten.
        max_tokens : int, optional
            Maximale Anzahl von Tokens in der Antwort (Standard: 4000).
        temperature : float, optional
            Sampling-Temperatur (Standard: 0.0).
        top_k : Optional[int], optional
            Top-k Sampling (optional).
        top_p : Optional[float], optional
            Top-p Sampling (optional).
        output_schema : Optional[Type[BaseModel]], optional
            Optionales Pydantic-Modell, in das die Antwort geparst werden soll.

        Returns:
        --------
        VertexAIResponse
            Ein Objekt, das die Rolle, den Inhalt und den Erstellungszeitpunkt der Antwort enthält.

        Raises:
        -------
        APIRequestError
            Wenn die HTTP-Anfrage fehlschlägt oder einen Nicht-200-Statuscode zurückgibt.
        ResponseValidationError
            Wenn die Antwort nicht in das erwartete Modell geparst werden kann.
        """
        # Aktualisieren des Zugriffstokens, falls es abgelaufen ist
        self._refresh_access_token()

        # Falls ein output_schema angegeben ist, füge Formatierungsanweisungen als zusätzliche Nachricht hinzu
        if output_schema:
            format_instructions = self._get_format_instructions(output_schema)
            format_message = ChatMessage(
                role="system",
                content=[{"type": "text", "text": format_instructions}]
            )
            messages.append(format_message)

        # Vorbereitung der Nutzlast
        payload = RequestPayload(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Konstruktion der Anfrage-URL
        url = (
            f"https://{self.endpoint}/v1beta1/projects/{self.project_id}/"
            f"locations/{self.region}/endpoints/openapi/chat/completions"
        )

        # Einrichtung der Header
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

        try:
            # Senden der POST-Anfrage mit der Nutzlast
            response = requests.post(url, headers=headers, json=payload.dict())
            response.raise_for_status()  # Löst bei Fehlerstatuscodes eine HTTPError aus
        except requests.exceptions.RequestException as e:
            # Behandlung netzwerkbezogener Fehler
            raise APIRequestError(status_code=0, message=f"Request failed: {str(e)}") from e

        try:
            response_json = response.json()
        except ValueError as e:
            # Falls die Antwort kein gültiges JSON ist
            raise ResponseValidationError(f"Failed to parse response JSON: {e}") from e

        # Extraktion der relevanten Felder aus der Antwort
        try:
            choices = response_json.get('choices', [])
            if not choices:
                raise ResponseValidationError("No choices found in the response.")

            first_choice = choices[0]
            message = first_choice.get('message', {})
            content = message.get('content', '').strip()
            role = message.get('role', '')
            created_at = response_json.get('created', 0)
        except (KeyError, IndexError, TypeError) as e:
            # Behandlung fehlender Schlüssel oder falscher Datentypen
            raise ResponseValidationError(f"Error extracting data from response JSON: {e}") from e

        # Parsen des Inhalts gemäß dem output_schema
        parsed_content = self._parse_response_content(content, output_schema)

        # Erstellung und Rückgabe des VertexAIResponse-Modells
        return VertexAIResponse(
            role=role,
            content=parsed_content,
            created_at=created_at
        )
