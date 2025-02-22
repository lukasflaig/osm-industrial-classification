from typing import List, Union, Literal, Optional, TypeVar
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    """
    Modelliert eine Chat-Nachricht für den Dialog mit dem LLM.

    Parameter:
    -----------
    role : Literal['system', 'user', 'assistant']
        Die Rolle des Nachrichtenabsenders (System, Benutzer oder Assistent).
    content : List[dict]
        Eine Liste von Inhaltsobjekten. Jedes Objekt sollte typischerweise einen Text enthalten.
    """
    role: Literal['system', 'user', 'assistant']
    content: List[dict]

class RequestPayload(BaseModel):
    """
    Definiert die Nutzlast für eine Anfrage an den Vertex AI Generative AI Endpoint.

    Parameter:
    -----------
    model : str
        Der Name des zu verwendenden Modells.
    messages : List[ChatMessage]
        Eine Liste von ChatMessage-Objekten, die den Gesprächsverlauf darstellen.
    max_tokens : int
        Maximale Anzahl von Tokens, die in der Antwort generiert werden sollen.
    temperature : float
        Sampling-Temperatur zur Steuerung der Kreativität der Antwort.
    top_k : Optional[int], optional
        Top-k Sampling, um die Auswahlmöglichkeiten einzuschränken (optional).
    top_p : Optional[float], optional
        Top-p Sampling, um den Wahrscheinlichkeitsmassebereich zu begrenzen (optional).
    """
    model: str
    messages: List[ChatMessage]
    max_tokens: int
    temperature: float
    top_k: Optional[int] = Field(None, description="Top-k sampling")
    top_p: Optional[float] = Field(None, description="Top-p sampling")

class VertexAIResponse(BaseModel):
    """
    Modelliert die Antwort des Vertex AI Generative AI Endpoints.

    Parameter:
    -----------
    role : str
        Die Rolle des Antwortgebers (z.B. "assistant").
    content : Union[str, BaseModel]
        Der Inhalt der Antwort. Kann ein einfacher String sein oder ein Objekt, das einem definierten Schema entspricht.
    created_at : int
        Der Zeitstempel (Unix-Epoche) der Antworterstellung.
    """
    role: str
    content: Union[str, BaseModel]
    created_at: int