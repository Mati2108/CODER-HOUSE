"""
manager.py — El "director" que elige qué hija usar.

Mira el texto `provider` y crea UNA sola hija (no gasta doble).
Además envuelve las llamadas en try/except para que un error de red
o de rate limit NO rompa todo el programa (error controlado).
"""

from clients import OpenAIClient, AnthropicClient, GeminiClient
from schemas import ModelResponse


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
        except Exception as e:
            # Plan B: en vez de explotar, devolvemos un error controlado
            return ModelResponse(
                content=f"[ERROR] No se pudo generar: {e}",
                model="error",
                provider="error",
            )

    async def stream(self, messages, config):
        try:
            async for fragmento in self.client.stream(messages, config):
                yield fragmento
        except Exception as e:
            yield f"[ERROR] Falló el streaming: {e}"
