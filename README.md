# Unified Async LLM Client

## Sobre el desarrollo

Este proyecto lo desarrollé apoyándome en herramientas de IA, sobre todo para
pulir la estructura y la documentación y llegar a un resultado prolijo.

Dicho eso, entiendo cada decisión que hay detrás del código y puedo explicarla:
por qué la clase abstracta obliga a que todos los proveedores expongan la misma
interfaz, por qué el streaming se resuelve con un generador asíncrono (`async
for` + `yield`) en lugar de acumular la respuesta entera, por qué se usan los
clientes async (`AsyncOpenAI`, `AsyncAnthropic`) y no las versiones bloqueantes
que congelarían el event loop, qué valida Pydantic en cada schema y por qué el
manager atrapa las excepciones en vez de dejar que revienten.

La IA fue una herramienta de trabajo, no un reemplazo del criterio: el diseño,
las decisiones y la comprensión del funcionamiento son míos.

## Qué es

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
| `main_lcel.py` | **Módulo 2** — el mismo flujo reescrito en LCEL: `prompt \| modelo \| StrOutputParser()`. |

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

## Módulo 2 — Refactorización a LCEL asíncrono

`main_lcel.py` hace lo mismo que `main.py`, pero declarativo en vez de imperativo:

```python
chain = prompt | construir_modelo() | StrOutputParser()
respuesta = await chain.ainvoke({"pregunta": "¿Qué es la entropía?"})
```

Lo que en el Módulo 1 eran una clase abstracta, tres hijas y un manager, acá son
tres eslabones conectados con `|`. El `StrOutputParser` es el que reemplaza a los
tres accesos distintos que había que escribir a mano para sacar el texto
(`choices[0].message.content`, `content[0].text`, `.text`).

```bash
python main_lcel.py
```

**Sobre el proveedor:** la consigna pide `ChatOpenAI` o `ChatAnthropic`. Los dos
están implementados en `construir_modelo()` y se eligen con `PROVIDER` en el
`.env`, sin tocar la cadena. La demo incluida corre con `ChatGoogleGenerativeAI`
simplemente porque es la key que tengo con saldo; los tres son intercambiables
porque todos los `ChatXxx` de LangChain exponen la misma interfaz
(`.ainvoke` / `.astream`). Si `PROVIDER` no está definido, el default es `openai`.

**Dos cosas que aprendí peleándome con esto:**

- `gemini-2.0-flash` (el que usa el Módulo 1) fue retirado por Google y devuelve
  404. Los nombres de modelo caducan y conviene chequearlos con `models.list()`.
- Con los modelos *pensantes* (Gemini 3.x, o1, Claude con thinking) los tokens de
  razonamiento interno salen del mismo `max_tokens` que la respuesta visible.
  Con 300 el modelo se quedaba sin cupo pensando y cortaba la frase por la mitad;
  parece un bug del código y es presupuesto agotado.

## Decisiones de diseño

- **Interfaz común (clase abstracta):** `BaseLLMClient` obliga a cada proveedor a
  implementar `generate` y `stream`, así el resto del programa los trata igual.
- **Async real:** se usan `AsyncOpenAI` y `AsyncAnthropic` con `await`, nunca las
  versiones bloqueantes (que congelarían el event loop).
- **Streaming:** implementado con un generador asíncrono (`async for` + `yield`).
- **Validación:** Pydantic valida entradas y unifica la salida en `ModelResponse`.
  El `role` es un `Literal`, así que un typo como `"usr"` falla al construir el
  mensaje y no en medio de la llamada a la API.
- **Resiliencia:** el manager captura excepciones y devuelve un error controlado
  en lugar de romper el programa. Además distingue los casos: límite de tasa
  (`RateLimitError` / HTTP 429), falla de red (`APIConnectionError`), error del
  proveedor y error inesperado, cada uno con su mensaje.
- **Mensajes `system` portables:** OpenAI los acepta dentro de `messages`, pero
  Anthropic y Gemini los exigen aparte (`system` / `system_instruction`). Los
  clientes lo traducen internamente, así que la misma lista de `ChatMessage`
  funciona igual en los tres proveedores.
