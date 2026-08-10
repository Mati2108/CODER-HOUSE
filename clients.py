"""
clients.py — La clase base abstracta y las dos hijas.

- BaseLLMClient: la MADRE abstracta. Pone la regla "todo cliente debe
  tener generate() y stream()". Deja los casilleros vacíos.
- OpenAIClient / AnthropicClient: las HIJAS. Cada una rellena esos
  casilleros con el código real de su proveedor.

Las dos hijas cumplen el mismo contrato -> intercambiabilidad.
"""

from abc import ABC, abstractmethod

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from google import genai
from google.genai import types

from schemas import ChatMessage, ModelConfig, ModelResponse


# ---------- LA MADRE ABSTRACTA ----------
class BaseLLMClient(ABC):
    """Contrato común: toda hija DEBE tener generate() y stream()."""

    @abstractmethod
    async def generate(self, messages, config) -> ModelResponse:
        ...  # casillero vacío: la madre no lo llena

    @abstractmethod
    def stream(self, messages, config):
        ...  # casillero vacío: la madre no lo llena


# ---------- HIJA 1: OPENAI ----------
class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key):
        # AsyncOpenAI = versión async (no bloquea el programa mientras espera)
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

    async def generate(self, messages, config) -> ModelResponse:
        respuesta = await self.client.chat.completions.create(
            model=self.model,
            messages=[m.model_dump() for m in messages],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        texto = respuesta.choices[0].message.content
        return ModelResponse(content=texto, model=self.model, provider="openai")

    async def stream(self, messages, config):
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[m.model_dump() for m in messages],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=True,                       # "mandámelo de a pedacitos"
        )
        async for chunk in stream:             # recorre los fragmentos según llegan
            # OpenAI a veces manda chunks con `choices` VACÍO (por ejemplo el
            # último, que sólo trae métricas de uso). Sin este chequeo eso sería
            # un IndexError que cortaría el stream a mitad de camino.
            if not chunk.choices:
                continue
            contenido = chunk.choices[0].delta.content
            if contenido:                      # salteamos fragmentos vacíos
                yield contenido                # entrega ese pedacito y sigue


# Anthropic NO acepta mensajes con role "system" dentro de `messages`: el system
# va aparte, como parámetro top-level. Este helper los separa para que nuestra
# interfaz común siga aceptando los mismos ChatMessage que OpenAI.
def _separar_system(messages):
    system = "\n".join(m.content for m in messages if m.role == "system")
    conversacion = [m.model_dump() for m in messages if m.role != "system"]
    return system, conversacion


# ---------- HIJA 2: ANTHROPIC ----------
class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = "claude-haiku-4-5"        # el más barato para probar

    def _kwargs(self, messages, config):
        """Arma los parámetros comunes a generate() y stream()."""
        system, conversacion = _separar_system(messages)
        kwargs = {
            "model": self.model,
            "messages": conversacion,
            "max_tokens": config.max_tokens,   # Anthropic lo EXIGE
            "temperature": config.temperature,
        }
        if system:                             # sólo lo mandamos si hay
            kwargs["system"] = system
        return kwargs

    async def generate(self, messages, config) -> ModelResponse:
        respuesta = await self.client.messages.create(**self._kwargs(messages, config))
        texto = respuesta.content[0].text      # Anthropic entierra el texto distinto
        return ModelResponse(content=texto, model=self.model, provider="anthropic")

    async def stream(self, messages, config):
        async with self.client.messages.stream(**self._kwargs(messages, config)) as stream:
            async for texto in stream.text_stream:   # el SDK ya te da solo el texto
                yield texto


# Gemini usa otro formato de mensajes: role "model" en vez de "assistant",
# y el texto va dentro de "parts". Este helper traduce nuestros ChatMessage.
# El system va aparte (system_instruction), igual que en Anthropic.
def _a_formato_gemini(messages):
    mapa_roles = {"user": "user", "assistant": "model"}
    return [
        {"role": mapa_roles.get(m.role, "user"), "parts": [{"text": m.content}]}
        for m in messages
        if m.role != "system"
    ]


# ---------- HIJA 3: GEMINI (tiene capa gratuita) ----------
class GeminiClient(BaseLLMClient):
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash"

    def _config(self, messages, config):
        system, _ = _separar_system(messages)
        return types.GenerateContentConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            system_instruction=system or None,
        )

    async def generate(self, messages, config) -> ModelResponse:
        respuesta = await self.client.aio.models.generate_content(
            model=self.model,
            contents=_a_formato_gemini(messages),
            config=self._config(messages, config),
        )
        texto = respuesta.text                 # Gemini entierra el texto en .text
        return ModelResponse(content=texto, model=self.model, provider="gemini")

    async def stream(self, messages, config):
        stream = await self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=_a_formato_gemini(messages),
            config=self._config(messages, config),
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text
