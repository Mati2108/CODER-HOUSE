"""
schemas.py — Los "moldes" de datos con Pydantic.

Pydantic valida los datos automáticamente: si algo viene mal
(temperature fuera de rango, un campo faltante), explota con un
error claro en vez de dejar pasar el dato malo.
"""

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Un mensaje del chat: quién habla y qué dice."""
    role: str        # "user", "assistant" o "system"
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
