"""
manager.py — El "director" que elige qué hija usar.

Mira el texto `provider` y crea UNA sola hija (no gasta doble).
Además envuelve las llamadas en try/except para que un error de red
o de rate limit NO rompa todo el programa (error controlado).
"""

import anthropic
import openai
from google.genai import errors as gemini_errors

from clients import OpenAIClient, AnthropicClient, GeminiClient
from schemas import ModelResponse


# Cada SDK tiene sus propias clases de excepción, así que las agrupamos para
# poder distinguir los dos casos que pide la consigna: límite de tasa y red.
ERRORES_RATE_LIMIT = (openai.RateLimitError, anthropic.RateLimitError)
ERRORES_DE_RED = (openai.APIConnectionError, anthropic.APIConnectionError)
ERRORES_DE_API = (openai.APIError, anthropic.APIError, gemini_errors.APIError)


def _describir(e: Exception) -> str:
    """Traduce la excepción cruda a un mensaje entendible y accionable."""
    # Gemini no tiene una clase RateLimitError propia: manda un APIError con
    # code 429, así que lo detectamos por el código HTTP.
    es_rate_limit = isinstance(e, ERRORES_RATE_LIMIT) or getattr(e, "code", None) == 429

    if es_rate_limit:
        return f"Límite de tasa alcanzado (rate limit), esperá unos segundos: {e}"
    if isinstance(e, ERRORES_DE_RED):
        return f"No se pudo conectar con el proveedor (problema de red): {e}"
    if isinstance(e, ERRORES_DE_API):
        return f"El proveedor devolvió un error: {e}"
    return f"Error inesperado: {e}"


class AsyncLLMManager:
    def __init__(self, provider, api_key):
        if provider == "openai":
            self.client = OpenAIClient(api_key)
        elif provider == "anthropic":
            self.client = AnthropicClient(api_key)
        elif provider == "gemini":
            self.client = GeminiClient(api_key)
        else:
            raise ValueError(f"Proveedor desconocido: {provider}")

    async def generate(self, messages, config) -> ModelResponse:
        try:
            return await self.client.generate(messages, config)
        # El catch es amplio a propósito: la consigna pide que NADA rompa el
        # loop principal. La distinción entre rate limit, red y demás la hace
        # _describir(), que sí mira las clases de excepción de cada SDK.
        except Exception as e:
            return self._respuesta_de_error(e)

    async def stream(self, messages, config):
        try:
            async for fragmento in self.client.stream(messages, config):
                yield fragmento
        except Exception as e:
            yield f"[ERROR] Falló el streaming: {_describir(e)}"

    @staticmethod
    def _respuesta_de_error(e: Exception) -> ModelResponse:
        """Plan B: en vez de explotar, devolvemos una respuesta estructurada."""
        return ModelResponse(
            content=f"[ERROR] No se pudo generar: {_describir(e)}",
            model="error",
            provider="error",
        )
