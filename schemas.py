"""
schemas.py — Los "moldes" de datos con Pydantic.

Pydantic valida los datos automáticamente: si algo viene mal
(temperature fuera de rango, un campo faltante), explota con un
error claro en vez de dejar pasar el dato malo.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Un mensaje del chat: quién habla y qué dice."""
    # Literal = Pydantic sólo acepta estos tres valores; cualquier otro rol
    # (un typo como "usr") explota acá y no en medio de la llamada a la API.
    role: Literal["user", "assistant", "system"]
    content: str     # el texto del mensaje


class ModelConfig(BaseModel):
    """Configuración del modelo, con validación de rangos."""
    temperature: float = Field(default=1.0, ge=0, le=2)   # entre 0 y 2
    max_tokens: int = Field(default=1024, gt=0)           # mayor que 0


class ModelResponse(BaseModel):
    """La respuesta UNIFICADA que devuelve cualquier proveedor."""
    content: str     # el texto que contestó la IA
    model: str       # qué modelo respondió
    provider: str    # "openai" o "anthropic"
