"""
main_lcel.py — Módulo 2: la misma llamada del Módulo 1, ahora con LCEL.

QUÉ CAMBIÓ RESPECTO DEL MÓDULO 1
--------------------------------
Módulo 1 (imperativo):   armabas el cliente del SDK a mano, construías el dict
                         de mensajes, llamabas a la API, y después buscabas el
                         texto adentro de la respuesta
                         (respuesta.choices[0].message.content, .content[0].text,
                         .text — uno distinto por proveedor).

Módulo 2 (declarativo):  describís el flujo con el operador `|` y LangChain se
                         encarga del resto:

                             prompt | modelo | StrOutputParser()

                         - `prompt`  convierte {"pregunta": "..."} en mensajes
                                     con roles System/Human.
                         - `modelo`  devuelve un AIMessage.
                         - `parser`  saca el .content y deja texto plano.

                         Eso es LCEL: no escribís los pasos, los conectás.

Correr con:  python main_lcel.py
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()  # lee las variables del archivo .env


# ---------- 1. EL MODELO ----------
def construir_modelo():
    """Devuelve un chat model de LangChain según PROVIDER del .env.

    Reemplaza al AsyncLLMManager del Módulo 1: ya no hay clase madre, hijas ni
    try/except por SDK. Cada ChatXxx de LangChain expone la misma interfaz
    (.invoke / .ainvoke / .astream) y por eso son intercambiables dentro de la
    cadena sin tocar una línea del resto del archivo.

    Nota: los tres son async-capable. Con `.ainvoke()` la llamada no bloquea
    el event loop mientras espera al proveedor.
    """
    provider = os.getenv("PROVIDER", "openai")
    temperature = 0.7   # parámetro base, igual que el ModelConfig del Módulo 1

    # OJO con los modelos "pensantes" (Gemini 3.x, o1, Claude con thinking):
    # los tokens que gastan razonando por dentro SALEN de este mismo
    # presupuesto. Con 300 el modelo piensa, se queda sin cupo y te corta la
    # respuesta a mitad de frase. Por eso vamos holgados: el límite de largo
    # real lo pone la instrucción "2 frases" del system prompt.
    max_tokens = 2048

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model="gpt-4o-mini",          # en versiones viejas: model_name=
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model="claude-haiku-4-5",
            temperature=temperature,
            max_tokens=max_tokens,        # Anthropic lo EXIGE
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",  # gemini-2.0-flash fue retirado por Google
            temperature=temperature,
            max_output_tokens=max_tokens,
            # El paquete busca GOOGLE_API_KEY por defecto; nuestro .env la
            # llama GEMINI_API_KEY, así que se la pasamos explícita.
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )

    raise ValueError(f"Proveedor desconocido: {provider}")


# ---------- 2. EL TEMPLATE ----------
# from_messages define los ROLES. El "system" fija la personalidad una sola vez;
# el "human" tiene el hueco {pregunta} que se rellena en cada ainvoke().
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sos un divulgador científico. Respondé en español rioplatense, "
            "claro y sin rodeos, en 2 frases como máximo.",
        ),
        ("human", "{pregunta}"),
    ]
)


# ---------- 3. LA CADENA (el pipeline LCEL) ----------
# El `|` acá NO es el "or" de Python: LangChain lo sobrecarga para que signifique
# "la salida de esto entra en aquello". La cadena entera es un solo objeto
# Runnable, con los mismos métodos que un modelo suelto.
#
# Sin StrOutputParser la cadena devolvería un AIMessage (el objeto completo);
# con él, devuelve el string pelado.
chain = prompt | construir_modelo() | StrOutputParser()


# ---------- 4. EJECUCIÓN ASÍNCRONA ----------
async def main():
    # La clave del dict tiene que coincidir EXACTO con la variable {pregunta}
    # del template. Si no, LangChain tira KeyError antes de llamar a la API.
    entrada = {"pregunta": "¿Qué es la entropía?"}

    # Sin `await`, ainvoke() devolvería una corrutina sin ejecutar,
    # no la respuesta.
    respuesta = await chain.ainvoke(entrada)

    print(f"\n=== ainvoke ({os.getenv('PROVIDER', 'openai')}) ===")
    print(respuesta)                 # texto plano, no un AIMessage

    # Chequeo del criterio de aceptación nº3: salió texto, no un BaseMessage.
    # (en langchain-core 1.x el parser devuelve un TextAccessor, que ES un str)
    print(f"\n¿es texto plano? {isinstance(respuesta, str)}")

    # EXTRA — la MISMA cadena, sin reconstruir nada, en modo streaming.
    # Es el equivalente LCEL del manager.stream() del Módulo 1.
    print("\n=== astream (misma cadena) ===")
    async for fragmento in chain.astream({"pregunta": "¿Qué es un agujero negro?"}):
        print(fragmento, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
