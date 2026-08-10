"""
main.py — Script de prueba.

Carga el .env, crea el manager con el proveedor elegido, y hace una
pregunta corta en MODO NORMAL y en MODO STREAMING.

Correr con:  python main.py
"""

import asyncio
import os

from dotenv import load_dotenv

from schemas import ChatMessage, ModelConfig
from manager import AsyncLLMManager

load_dotenv()  # lee las variables del archivo .env


async def main():
    # Elegís el proveedor desde el .env: "openai" o "anthropic"
    provider = os.getenv("PROVIDER", "openai")

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    else:  # gemini
        api_key = os.getenv("GEMINI_API_KEY")

    manager = AsyncLLMManager(provider=provider, api_key=api_key)

    messages = [ChatMessage(role="user", content="¿Qué es la entropía? Respondé en 2 frases.")]
    config = ModelConfig(temperature=0.7, max_tokens=300)

    # ----- MODO NORMAL -----
    print(f"\n=== MODO NORMAL ({provider}) ===")
    respuesta = await manager.generate(messages, config)
    print(respuesta.content)

    # ----- MODO STREAMING -----
    print(f"\n=== MODO STREAMING ({provider}) ===")
    async for fragmento in manager.stream(messages, config):
        print(fragmento, end="", flush=True)   # aparece palabra por palabra
    print()


if __name__ == "__main__":
    asyncio.run(main())
