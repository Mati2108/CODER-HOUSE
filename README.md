# Unified Async LLM Client

Cliente asíncrono y unificado para consultar modelos de IA (OpenAI, Anthropic y
Gemini) bajo una interfaz común. Soporta respuesta normal y streaming, valida
datos con Pydantic y maneja errores sin caerse.

> 💡 **Gemini tiene capa gratuita** (https://aistudio.google.com/apikey), así que
> es el proveedor más cómodo para probar el proyecto sin gastar.

## Estructura

| Archivo        | Qué hace                                                        |
|----------------|-----------------------------------------------------------------|
| `schemas.py`   | Moldes de datos con Pydantic (`ChatMessage`, `ModelConfig`, `ModelResponse`) con validación (temperature 0–2, max_tokens > 0). |
| `clients.py`   | Clase base abstracta `BaseLLMClient` + hijas `OpenAIClient`, `AnthropicClient` y `GeminiClient` (métodos `generate` y `stream`, versiones async). |
| `manager.py`   | `AsyncLLMManager`: elige el proveedor según config y captura errores (error controlado). |
| `main.py`      | Script de prueba: una pregunta en modo normal y en streaming.   |

## Cómo correrlo

1. Crear el entorno (Python 3.12+) e instalar dependencias:

   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Configurar las variables de entorno:

   ```bash
   cp .env.example .env
   ```

   Editar `.env` y completar:

   - `PROVIDER` → `openai`, `anthropic` o `gemini`
   - `OPENAI_API_KEY` → tu key de OpenAI
   - `ANTHROPIC_API_KEY` → tu key de Anthropic
   - `GEMINI_API_KEY` → tu key de Gemini (gratis)

3. Ejecutar:

   ```bash
   python main.py
   ```

## Variables de entorno

| Variable            | Descripción                                  |
|---------------------|----------------------------------------------|
| `PROVIDER`          | Proveedor a usar: `openai`, `anthropic` o `gemini`. |
| `OPENAI_API_KEY`    | API key de OpenAI.                           |
| `ANTHROPIC_API_KEY` | API key de Anthropic.                        |
| `GEMINI_API_KEY`    | API key de Gemini (capa gratuita).           |

> La API key **nunca** va hardcodeada en el código: siempre en `.env`
> (que está ignorado por git en `.gitignore`).

## Decisiones de diseño

- **Interfaz común (clase abstracta):** `BaseLLMClient` obliga a cada proveedor a
  implementar `generate` y `stream`, así el resto del programa los trata igual.
- **Async real:** se usan `AsyncOpenAI` y `AsyncAnthropic` con `await`, nunca las
  versiones bloqueantes (que congelarían el event loop).
- **Streaming:** implementado con un generador asíncrono (`async for` + `yield`).
- **Validación:** Pydantic valida entradas y unifica la salida en `ModelResponse`.
- **Resiliencia:** el manager captura excepciones y devuelve un error controlado
  en lugar de romper el programa.
