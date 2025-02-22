class Llama32ClientError(Exception):
    """
    Basis-Ausnahme für Fehler, die im Llama32Client auftreten.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    pass


class AuthenticationError(Llama32ClientError):
    """
    Ausnahme, die bei Authentifizierungsfehlern ausgelöst wird.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    pass


class APIRequestError(Llama32ClientError):
    """
    Ausnahme, die bei Fehlern während API-Anfragen ausgelöst wird.

    Parameters:
    -----------
    status_code : int
        HTTP-Statuscode der fehlgeschlagenen Anfrage.
    message : str
        Fehlermeldung, die den Grund des Fehlers beschreibt.

    Returns:
    --------
    None
    """
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"APIRequestError {status_code}: {message}")


class ResponseValidationError(Llama32ClientError):
    """
    Ausnahme, die bei Validierungsfehlern der API-Antwort ausgelöst wird.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    pass