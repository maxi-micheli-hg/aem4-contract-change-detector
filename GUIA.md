# Guía de estudio — Contract Change Detector

Documento de referencia para entender el proyecto a fondo: arquitectura, tecnologías, cada archivo del código, y preguntas anticipadas para la defensa en vivo. Pensado para leerse de una sentada antes del 1:1 con el corrector.

---

## 1. ¿Qué hace este proyecto en una frase?

Recibe **2 imágenes escaneadas** (un contrato original + su enmienda), las "lee" con GPT-4o Vision, y devuelve un **JSON validado** que enumera todas las secciones modificadas, los temas legales afectados y un resumen detallado de los cambios.

Caso de uso simulado: una empresa legal (LegalMove) que hoy gasta 40+ horas semanales comparando contratos a mano. El sistema lo hace en ~20 segundos por par.

---

## 2. El flujo end-to-end paso a paso

Cuando corrés:

```bash
uv run python src/main.py data/test_contracts/contract_1_original.jpg data/test_contracts/contract_1_amendment.jpg
```

Pasa esto, en orden:

1. **`main.py` arranca** → parsea los argumentos (las 2 rutas a las imágenes) y carga las credenciales desde `.env`.

2. **Se construye un `ChatOpenAI(model="gpt-4o", temperature=0)`** con timeout=60s y 2 reintentos automáticos. Este mismo cliente LLM se usa para las 4 llamadas posteriores.

3. **Se inicializa Langfuse** → el `CallbackHandler` queda listo para "espiar" cada llamada al LLM.

4. **Se abre el span raíz `contract-analysis`** en Langfuse. Todo lo que pasa después queda registrado como hijo de este span.

5. **Etapa 1 — `parse_original_contract`** (span hijo):
   - La imagen del contrato original se lee como bytes.
   - Se codifica a **base64** y se mete en una *data URL* (`data:image/jpeg;base64,...`).
   - Se construye un `HumanMessage` multimodal con dos partes: un texto pidiendo "transcribime esto preservando jerarquía" + la imagen.
   - Se invoca el LLM. GPT-4o Vision lee la imagen y devuelve el texto extraído en Markdown.

6. **Etapa 2 — `parse_amendment_contract`** (span hijo): Mismo proceso para la imagen de la enmienda.

7. **Etapa 3 — `contextualization_agent`** (span hijo):
   - El **Analista Senior** (Agente 1) recibe los dos textos.
   - Su trabajo NO es enumerar cambios. Su trabajo es producir un **mapa estructural**: una tabla Markdown que alinea las secciones del original con las de la enmienda y marca cuáles cambiaron, cuáles son nuevas, y cuáles son iguales.
   - Esto sirve como contexto para el siguiente agente, así no tiene que "descubrir" la estructura desde cero.

8. **Etapa 4 — `extraction_agent`** (span hijo):
   - El **Auditor Legal Forense** (Agente 2) recibe el mapa + ambos textos.
   - Su trabajo es **enumerar cada cambio**: adiciones (cláusulas nuevas), eliminaciones, modificaciones (citando valores viejos vs nuevos).
   - Usa `with_structured_output(ContractChangeOutput)`: la API de OpenAI **garantiza a nivel de servidor** que el JSON cumple el schema de Pydantic. No hay parsing manual.

9. **Se imprime el JSON validado** por consola (con colores via Rich).

10. **`obs.flush()` en `finally`** → fuerza el envío de todas las trazas a Langfuse antes de que el proceso termine. Si no hicieras esto, las trazas se perderían si la app crashea.

---

## 3. Las tecnologías — qué son y por qué las usamos

### 3.1 OpenAI GPT-4o (Vision)

**Qué es:** El modelo multimodal de OpenAI — puede recibir texto + imágenes en el mismo input.

**Cómo lo usamos:**
- Para **leer las imágenes** del contrato (la única forma "viable" sin hacer OCR clásico, que perdería estructura).
- Para los **2 agentes de texto** también (mismo modelo, distinto prompt). Podríamos haber usado `gpt-4o-mini` para los agentes para abaratar, pero usamos GPT-4o para mantener calidad consistente.

**Por qué GPT-4o y no GPT-4o-mini para Vision:**
- GPT-4o tiene mejor precisión en español sobre documentos escaneados.
- Preserva mejor la jerarquía (numeración de cláusulas, sub-numeraciones).
- Cuesta más (~$0.02 por página) pero el rubric pide calidad.

**Concepto clave: multimodal input via data URL**
```python
HumanMessage(content=[
    {"type": "text", "text": "Transcribime esto..."},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<bytes>"}}
])
```

### 3.2 LangChain

**Qué es:** Framework para orquestar llamadas a LLMs. Provee abstracciones para chains, prompts, parsers, callbacks, etc.

**Qué usamos de LangChain:**
- `langchain_openai.ChatOpenAI` — wrapper sobre la API de OpenAI que se integra con el resto del ecosistema.
- `langchain_core.messages.HumanMessage` — clase para construir el mensaje multimodal.
- `langchain_core.prompts.ChatPromptTemplate` — templates para los system + user prompts de cada agente.
- `langchain_core.output_parsers.StrOutputParser` — parser que extrae el `.content` string de la respuesta.
- **LCEL (LangChain Expression Language)** — el operador `|` que encadena componentes: `prompt | llm | parser`. Esa "pipe" crea un objeto `Runnable` que tiene métodos `.invoke()`, `.stream()`, `.batch()`, etc.
- `with_structured_output(SchemaPydantic)` — hace que el output del LLM se valide automáticamente contra el schema de Pydantic.
- **Callbacks** — el mecanismo por el cual Langfuse "espía" cada llamada al LLM sin que tengamos que instrumentar manualmente.

**Concepto clave: LCEL pipes**

```python
# Esto:
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"variable": "valor"})

# Es equivalente a:
formatted = prompt.format_messages(variable="valor")
response = llm.invoke(formatted)
result = response.content
```

El LCEL pipe es solamente azúcar sintáctica, pero permite a LangChain pasar el `config={"callbacks": [...]}` automáticamente por toda la cadena.

### 3.3 Pydantic + Structured Outputs

**Qué es Pydantic:** Librería de Python para definir modelos de datos con validación de tipos en runtime. Es la base de FastAPI, instructor, etc.

**Cómo se usa acá:**
```python
class ContractChangeOutput(BaseModel):
    sections_changed: list[str] = Field(..., description="...")
    topics_touched: list[str] = Field(..., description="...")
    summary_of_the_change: str = Field(..., description="...")
```

**Truco clave — los `description=...` no son solo documentación**: cuando pasás este modelo a `llm.with_structured_output(ContractChangeOutput)`, LangChain genera un **JSON Schema** y lo envía a la API de OpenAI como parte del request (parámetro `response_format`). OpenAI **fuerza al modelo** a generar un JSON que cumpla ese schema, y los `description` aparecen en el schema como hints. **GPT-4o los lee como parte de la guía de generación**.

Por eso nuestros `description` son detallados — están instruyendo al modelo, no solo a un futuro lector del código.

**Concepto clave: structured outputs (OpenAI API)**

Antes (parsing manual, frágil):
```python
response = llm.invoke(prompt)
text = response.content
json_obj = json.loads(text)  # puede crashear si el LLM agrega prosa
parsed = ContractChangeOutput(**json_obj)  # puede fallar la validación
```

Ahora (server-side schema enforcement):
```python
chain = prompt | llm.with_structured_output(ContractChangeOutput)
parsed = chain.invoke(inputs)  # ya viene como Pydantic, garantizado válido
```

### 3.4 Langfuse + observability

**Qué es Langfuse:** Plataforma open-source de observabilidad para LLMs. Captura cada llamada que hace tu app y la muestra como una traza navegable con inputs, outputs, latencia, tokens y costo.

**Vocabulario de Langfuse:**

| Término | Qué es |
|---|---|
| **Trace** | Toda una ejecución de la app. En nuestro caso, una corrida de `main.py` = 1 trace. |
| **Span** | Un paso lógico dentro del trace. Tiene un nombre, duración, input, output. Nosotros creamos 5: el root `contract-analysis` + 4 hijos. |
| **Generation** | Un span especial que representa una llamada al LLM. Tiene tokens (prompt/completion), modelo, latencia, costo calculado. Los crea automáticamente el `CallbackHandler`. |
| **Observation** | Término paraguas — span y generation son los dos tipos de observation que vamos a ver. |

**Cómo se conecta a nuestro código:**

```python
# 1. Inicializás el cliente y el handler
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

client = Langfuse(public_key=..., secret_key=..., host=...)
handler = CallbackHandler()

# 2. Abrís un span manualmente con un context manager
with client.start_as_current_observation(name="contract-analysis", as_type="span") as root:
    # 3. Llamás al LLM pasando el handler como callback
    response = llm.invoke(message, config={"callbacks": [handler]})
    # ↑ Esta llamada aparece como `generation` ANIDADA bajo el span actual.

# 4. Antes de salir, flusheás
client.flush()
```

**Punto sutil pero importante:** cuando hacés `start_as_current_observation` dentro de otro span activo, **el nuevo span se vuelve HIJO automáticamente**. Por eso la jerarquía que ves en el dashboard se arma sola: vos solo abrís spans en orden, Langfuse arma el árbol.

### 3.5 Stack auxiliar

- **`python-dotenv`** — carga `.env` en variables de entorno. Línea única: `load_dotenv(".env")`.
- **`uv`** — gestor de dependencias rápido (10-100x más rápido que pip). Lockfile en `uv.lock`, manifest en `pyproject.toml`. Para CI/grader compatibility, exportamos un `requirements.txt` con `uv export`.
- **`Rich`** — librería para output colorido en terminal. Usamos `Console`, `console.print_json()` y un `RichHandler` en el logger.
- **`openai` (SDK)** — solo importamos sus clases de excepción (`APITimeoutError`, `RateLimitError`) para el `except`. El resto pasa por LangChain.

---

## 4. Recorrido del código archivo por archivo

### 4.1 `src/main.py` — el director de orquesta

**Responsabilidad:** parsear argumentos, construir el LLM + Observability, abrir el span raíz, secuenciar los 4 pasos, manejar errores, flushear trazas.

**Decisiones clave:**
- **Argparse con 2 positional args** — la consigna dice "entry point que acepta dos paths de imágenes como argumentos". Positional args es lo más limpio.
- **Una sola instancia de `ChatOpenAI`** compartida — el parser de imágenes y los 2 agentes usan el mismo cliente. Eso no rompe nada porque `ChatOpenAI` es stateless por invocación.
- **`temperature=0`** — outputs reproducibles. Crítico para que la defensa muestre el mismo resultado dos veces si el corrector quiere ver.
- **`max_retries=2, timeout=60`** — manejo de errores de red. `timeout=60s` reemplaza el default de OpenAI de 600s, que es absurdo para una sola llamada de visión.
- **Try/except con 3 ramas separadas** — input errors (exit 2), API errors (exit 2), validation errors (exit 1). Sin `except Exception: pass` que oculte bugs.
- **`finally: obs.flush()`** — siempre se ejecuta, incluso si hay excepción. Garantiza que las trazas lleguen a Langfuse aunque el proceso muera.

### 4.2 `src/image_parser.py` — la ventana de GPT-4o al mundo visual

**Responsabilidad:** convertir una imagen JPG/PNG en texto Markdown via GPT-4o Vision.

**Decisiones clave:**
- **Validación primero, encoding después** — chequea que el archivo exista y tenga extensión soportada ANTES de gastar memoria leyendo bytes.
- **Base64 data URL en lugar de subir la imagen a un servicio** — la API de OpenAI acepta tanto URLs como data URLs base64-encoded. Usar data URL evita el round-trip a un bucket externo y mantiene el proyecto autosuficiente.
- **El prompt explícito de "preservar jerarquía"** — sin esto, GPT-4o tiende a parafrasear y perder los identificadores de cláusula (`1.`, `2.1`). El prompt dice literalmente "transcribe, NO resumas, NO traduzcas".
- **Span con metadata útil** — guardamos `image_filename`, `mime_type`, y un `text_preview` (primeros 300 chars del output) para que en el dashboard se vea de qué se trata sin tener que abrir el output completo.

### 4.3 `src/models.py` — el contrato de datos

**Responsabilidad:** definir el schema `ContractChangeOutput` que el extraction agent debe producir.

**Decisiones clave:**
- **Los 3 campos exactos del consigna** — `sections_changed`, `topics_touched`, `summary_of_the_change`. Si renombrás cualquiera, fallás 1.3 del rubric.
- **`Field(..., description=...)` rico en cada campo** — como expliqué arriba, estos `description` se inyectan en el JSON Schema que recibe GPT-4o. Son **instrucciones al modelo**, no solo docs.
- **Sin valores default** — los 3 campos son obligatorios (`...` en Pydantic = required).

### 4.4 `src/agents/contextualization_agent.py` — el Analista Senior

**Responsabilidad:** producir el mapa estructural alineando secciones del original con las de la enmienda. **No enumera cambios.**

**Decisiones clave:**
- **System prompt declara el rol explícitamente**: "Eres un Analista Senior de Contratos en LegalMove". El rubric 2.1 premia exactamente esto.
- **5 responsabilidades enumeradas** + **una prohibición explícita**: "NO TE CORRESPONDE describir los cambios en detalle". Esa frase es la que evita que el agente "se entusiasme" y empiece a hacer el trabajo del siguiente.
- **Output format fijado**: párrafo intro + tabla Markdown. Eso le da estructura al input del próximo agente.
- **Chain LCEL: `prompt | llm | StrOutputParser()`** — el StrOutputParser convierte el `AIMessage` que devuelve el LLM en un string plano, listo para pasar al siguiente agente.
- **Span hijo abierto adentro del `run()`** — `name="contextualization_agent"`. Garantiza que el agente aparezca como nodo nombrado en el árbol del trace.

### 4.5 `src/agents/extraction_agent.py` — el Auditor Legal Forense

**Responsabilidad:** enumerar cada cambio en un `ContractChangeOutput` validado.

**Decisiones clave:**
- **Otro rol especializado**: "Eres un Auditor Legal Forense". Distinto al Analista Senior — el corrector va a notar y valorar la diferencia de personas.
- **3 tipos de cambios definidos** — adiciones, eliminaciones, modificaciones. Esa taxonomía explícita reduce ambigüedad en el output.
- **Instrucción de citar valores viejos vs nuevos** — "el plazo se extiende de 12 a 24 meses". Hace que el `summary_of_the_change` sea útil para un humano.
- **Anti-alucinación explícita**: "NO inventes cambios que no estén respaldados por los textos; si no estás seguro, omítelos."
- **Ejemplo one-shot en el prompt** — un mini JSON de ejemplo le ancla al modelo la forma y estilo del output. Crítico para que el `summary` venga en prosa fluida y no en bullets.
- **Chain LCEL: `prompt | llm.with_structured_output(ContractChangeOutput)`** — la versión "estructurada" del LLM. El parser ya viene incluido — devuelve directamente una instancia Pydantic.
- **`try/except ValidationError`** — captura el (raro) caso en que el structured output falle. Loggea los detalles del error y re-lanza.

### 4.6 `src/shared/observability.py` — el plumbing de Langfuse

**Responsabilidad:** un solo lugar donde se inicializa el cliente de Langfuse + el callback handler, encapsulados en una clase `Observability` que pasamos al resto de la app.

**Decisiones clave:**
- **Lazy import de Langfuse** — solo se importa si `enabled=True`. Permite correr el proyecto con `--no-langfuse` sin tener las keys configuradas (útil para debugging offline).
- **`Observability.callbacks` retorna una lista** — para pasar directamente a `config={"callbacks": obs.callbacks}` en cada invoke.
- **`flush()` opcional** — si el client es None, es no-op. Permite `obs.flush()` en `finally` sin condicionales.

### 4.7 `src/shared/config.py` — credenciales y paths

**Responsabilidad:** cargar `.env` y exponer accesores tipados para las keys.

**Decisiones clave:**
- **Funciones explícitas como `get_openai_api_key()`** en lugar de `os.getenv` directo — permite mensajes de error claros cuando falta una key.
- **Detección de placeholder**: `key.startswith("sk-...")` — si dejaste el valor del `.env.example`, falla loudly antes de gastar tiempo.
- **Path del `.env` resuelto relativamente al módulo** — no depende del current working directory, funciona desde cualquier carpeta.

### 4.8 `src/shared/logger.py` — Rich + theme

**Responsabilidad:** logger con colores. `[success]` verde, `[warn]` amarillo, `[error]` rojo en los logs.

---

## 5. Anatomía de un trace de Langfuse

Cuando abrís un trace en el dashboard, vas a ver algo como:

```
contract-analysis            (root span, 18s, $0.028, 7,002 tokens)
│   input:  {original_image, amendment_image}
│   output: {sections_changed, topics_touched, summary_chars}
│
├── parse_original_contract        (6s)
│   │   input:  {image_filename, mime_type, image_role}
│   │   output: {text_length, text_preview}
│   │
│   └── ChatOpenAI                 ← generation (auto)
│       prompt: 1,289 tokens
│       completion: 261 tokens
│       latency: 6.2s
│       model: gpt-4o
│
├── parse_amendment_contract       (4s)
│   └── ChatOpenAI                 ← generation (auto)
│
├── contextualization_agent        (6s)
│   │   input:  {original_chars, amendment_chars}
│   │   output: {map_chars, map_preview}
│   │
│   └── RunnableSequence
│       ├── ChatPromptTemplate     ← intermedio de LCEL
│       ├── ChatOpenAI             ← generation
│       └── StrOutputParser        ← intermedio de LCEL
│
└── extraction_agent               (2s)
    │   input:  {context_map_chars, original_chars, amendment_chars}
    │   output: {sections_changed, topics_touched, summary_chars}
    │
    └── RunnableSequence
        ├── ChatPromptTemplate
        ├── ChatOpenAI             ← generation (structured output)
        └── RunnableLambda         ← Pydantic instantiation
```

**¿De dónde sale cada uno?**

| Item | Quién lo crea |
|---|---|
| `contract-analysis` (root) | Manual, en `main.py` |
| `parse_*_contract` | Manual, en `image_parser.py` |
| `contextualization_agent` y `extraction_agent` | Manual, en cada clase de agente |
| `ChatOpenAI`, `ChatPromptTemplate`, `StrOutputParser`, `RunnableSequence`, `RunnableLambda` | **Automático** — los crea el `CallbackHandler` de Langfuse cuando ve que LangChain ejecuta cada paso del LCEL |

Eso explica por qué algunos nodos del árbol tienen nombres que no aparecen en tu código (`RunnableSequence` es el objeto que crea LCEL cuando hacés `prompt | llm | parser`).

---

## 6. Defense Q&A — preparación para el 1 a 1

### Las 4 preguntas estándar

**Q: ¿Por qué dos agentes en lugar de uno monolítico?**

Separar contextualización de extracción imita cómo trabajaría un equipo legal real:
1. Un analista senior primero entiende qué secciones del documento original corresponden a cuáles de la enmienda.
2. Un auditor recién entonces va sección por sección a enumerar las diferencias.

Beneficios concretos:
- **Prompts más cortos y focalizados** — cada agente tiene 4-5 responsabilidades enumeradas, no 10. Esto reduce drift de prompt y alucinaciones.
- **Handoff inspectable** — el output del Analista (mapa Markdown) es legible por humanos. Si algo sale mal, podés debuggear: "¿se equivocó el analista al alinear las secciones, o el auditor al enumerar cambios?".
- **Observabilidad más limpia** — cada agente tiene su propio span en Langfuse, con tokens y latencia atribuidos por etapa.

**Q: ¿Por qué GPT-4o para el parsing de imágenes?**

- **OCR en español sobre documentos escaneados** — GPT-4o es de los mejores en esto. Tesseract o servicios OCR clásicos pierden la jerarquía (numeración de cláusulas, sub-numeración).
- **Extracción jerárquica** — preserva `1.`, `2.`, `2.1`, etc. sin necesidad de un post-procesado.
- **Razonamiento + transcripción en una sola llamada** — el prompt de transcripción incluye "NO añadas comentarios, NO resumas". GPT-4o sigue esa instrucción mientras "ve" la imagen. Un pipeline OCR + LLM en serie costaría lo mismo en tokens y agregaría latencia.

**Q: ¿Cómo diseñaste los prompts?**

- **Role priming** — cada agente arranca con "Eres un Analista Senior..." / "Eres un Auditor Legal Forense...". El rol específico orienta el tono y rigor.
- **Responsabilidades numeradas y prohibiciones explícitas** — "Tu trabajo es X" + "NO TE CORRESPONDE Y". Las prohibiciones son tan importantes como las instrucciones.
- **Ejemplo one-shot en el Auditor** — un mini JSON que ancla la forma esperada (prosa fluida, valores viejos vs nuevos).
- **`temperature=0`** — runs reproducibles para la defensa.
- **`Field(..., description=...)` ricos en Pydantic** — esos `description` van al JSON Schema que GPT-4o recibe via `response_format`, así que también son parte del prompting.

**Q: ¿Cómo manejás los errores?**

4 clases nombradas, sin `except Exception` que oculte cosas:

| Clase | Dónde | Comportamiento |
|---|---|---|
| `FileNotFoundError` / `ValueError` | `image_parser._encode_image` y validación de path | Surface immediato, exit 2 |
| Base64 encoding error | `_encode_image` `try/except` | Capturado + re-lanzado con contexto |
| `APITimeoutError` / `RateLimitError` | OpenAI SDK | 2 retries automáticos con backoff exponencial (max_retries=2 en ChatOpenAI). Si siguen fallando, exit 2. Timeout 60s vs default de 600s |
| `pydantic.ValidationError` | `extraction_agent.run` | Loggea detalles + re-lanza. Exit 1. (Casi imposible que ocurra con structured outputs, pero es defensivo) |

Sin fallbacks silenciosos, sin retries beyond el del SDK. Principio: **un error visible al operador vale más que un "success" con basura adentro**.

### Preguntas adicionales que podría tirar el corrector

**Q: ¿Por qué `with_structured_output` y no parsing manual con `json.loads`?**

Porque OpenAI structured outputs garantiza a nivel de servidor que el JSON cumple el schema. El modelo no puede devolver malformed JSON ni un campo extra. Parsing manual sería defensivo contra problemas que ya están resueltos un nivel más abajo.

**Q: ¿Qué pasa si el grader le pasa una imagen que no existe?**

`parse_contract_image` chequea `if not p.exists()` y lanza `FileNotFoundError`. `main.py` la captura en el `except (FileNotFoundError, ValueError)` y sale con exit code 2. El usuario ve el error claro.

**Q: ¿Qué pasa si el contrato está en inglés en lugar de español?**

Los prompts están en español pero GPT-4o es multilingüe — extraería el texto en inglés y los agentes operarían igual. El `summary_of_the_change` saldría en español porque el system prompt del Auditor lo pide explícitamente en español.

**Q: ¿Y si la imagen tiene 5 páginas en lugar de 1?**

Con la implementación actual, una imagen = una llamada. Para multi-página tendrías que pre-procesar (split en N imágenes individuales) y concatenar los textos extraídos antes de pasarlos a los agentes. No está implementado porque la consigna especifica imágenes simples.

**Q: ¿Por qué usás `start_as_current_observation` y no `@observe` decorator?**

Porque el context manager me da control fino sobre cuándo abrir y cerrar el span, y permite anidar manualmente. El `@observe` es más limpio pero menos flexible — ideal si todo tu código sigue el patrón "función pura llamada una vez". En multi-agente con steps secuenciales, el context manager es más explícito.

**Q: ¿Cuál es el costo aproximado por par de contratos?**

Con GPT-4o a precios de Mayo 2026:
- Vision (2 llamadas): ~2,500-3,000 tokens prompt + ~500-600 completion = ~$0.012
- Contextualization: ~1,500 prompt + ~580 completion = ~$0.008
- Extraction: ~2,100 prompt + ~240 completion = ~$0.008
- **Total: ~$0.028 por par.**

A escala (1000 contratos/día): ~$28/día = $850/mes. Versus 40 horas/semana de un compliance analyst (≈$5k/mes), el ROI es claro.

**Q: ¿Por qué `temperature=0`?**

Reproducibilidad. Misma imagen + mismo prompt → mismo output. Crítico para que la defensa no tenga sorpresas y para que un grader pueda re-correr y obtener resultados comparables.

**Q: ¿Por qué `max_tokens` no está fijado?**

GPT-4o tiene un max_tokens default razonable (4096). Si lo fijara, podría truncar el `summary_of_the_change` en contratos largos. Lo dejo al default.

**Q: ¿Qué pasaría si pasara los 2 textos directamente al Auditor sin el Analista?**

Funcionaría peor en 2 dimensiones:
1. **Más alucinaciones** — el Auditor tendría que descubrir la alineación sección-a-sección al mismo tiempo que enumerar cambios. Más cosas en mente = más errores.
2. **Más tokens de contexto** — el Analista comprime ambos textos (~2k tokens cada uno = 4k) a un mapa estructural más corto. El Auditor recibe `mapa + textos` lo cual es ~5k en vez de tener que procesar ~4k crudos.

Hicimos el experimento mental, no lo tenemos benchmarkeado, pero la separación es defensible por principios.

**Q: ¿Por qué pusiste `data/test_contracts/README.md` con los ground-truth changes?**

Para que el corrector pueda **validar la calidad del output sin tener que abrir el .docx original**. Demuestra honestidad: "estos son los cambios que el sistema debería encontrar; corramos y veamos qué sale".

**Q: ¿Qué tan bien resiste el sistema a un escaneo de baja calidad?**

Depende. GPT-4o Vision tolera bastante ruido, pero si la imagen es muy pixelada o tiene rotación, podría fallar en preservar la numeración. Con las imágenes del bootcamp (alta calidad) no es un problema. En producción habría que agregar un paso de pre-procesado (deskew, denoise) antes del parsing.

**Q: ¿Cómo escalaría esto a 10,000 contratos/día?**

Las 2 llamadas de Vision son sequential en el código actual — podrían ir en paralelo con `asyncio.gather` (langchain tiene `.ainvoke`). Cuello de botella siguiente sería rate limits de OpenAI, que requeriría batch API o un pool de keys. La parte de los agentes se mantiene igual.

---

## 7. Glosario rápido

| Término | Significado |
|---|---|
| **LCEL** | LangChain Expression Language. El operador `|` que encadena Runnables (`prompt | llm | parser`). |
| **Runnable** | Cualquier objeto LangChain con `.invoke()`, `.stream()`, `.batch()`. Promp templates, modelos, parsers, retrievers son todos Runnables. |
| **Multimodal** | Input/output que combina más de una modalidad (texto + imagen, texto + audio, etc.). GPT-4o es multimodal en input (texto + imágenes). |
| **Structured output** | Feature de la API de OpenAI donde el server fuerza al modelo a generar JSON que cumple un schema dado. Garantizado válido. |
| **Data URL** | Esquema URI que mete los bytes del recurso inline: `data:image/jpeg;base64,<bytes>`. Permite pasar imágenes a APIs sin subirlas a un bucket. |
| **Base64** | Encoding que convierte bytes binarios a una string ASCII (sólo caracteres `A-Za-z0-9+/=`). Hace que los bytes sean transportables por protocolos texto-only (HTTP headers, JSON). |
| **Span** | En observabilidad: una unidad de trabajo dentro de un trace. Tiene start time, end time, nombre, input, output, metadata. |
| **Generation** | Span especial que representa una llamada al LLM. Captura prompt tokens, completion tokens, modelo. |
| **Trace** | Toda la ejecución end-to-end. Contiene un árbol de spans/generations. |
| **Callback handler** | Hook que LangChain expone para que herramientas como Langfuse intercepten cada llamada al LLM sin tocar el código de negocio. |
| **`functools.partial`** | Helper de Python para "pre-rellenar" argumentos de una función. No lo usé en M4 pero sí en M3. |
| **Pydantic v2** | La version actual (2024+) de Pydantic. Tiene `BaseModel`, `Field`, `model_dump_json()`, `model_validate()`, `ValidationError`. |
| **`response_format`** | Parámetro de la API de OpenAI Chat Completions que acepta un Pydantic model (via LangChain) o un dict JSON Schema. Activa structured outputs. |
| **Adenda** | Término legal para "enmienda" o "modificación" a un contrato. |
| **Compliance** | Función legal/regulatoria de una empresa — asegurarse de que los contratos cumplan políticas internas y externas. |

---

## 8. Comandos útiles de referencia

```bash
# Levantar todo de cero después de clonar el repo
uv sync
cp .env.example .env  # editar con las keys

# Correr en uno de los 3 pares
uv run python src/main.py data/test_contracts/contract_1_original.jpg data/test_contracts/contract_1_amendment.jpg
uv run python src/main.py data/test_contracts/contract_2_original.jpg data/test_contracts/contract_2_amendment.jpg
uv run python src/main.py data/test_contracts/contract_3_original.jpg data/test_contracts/contract_3_amendment.jpg

# Correr sin Langfuse (debugging offline)
uv run python src/main.py data/test_contracts/contract_1_original.jpg data/test_contracts/contract_1_amendment.jpg --no-langfuse

# Linter
uv run ruff check
uv run ruff format

# Exportar requirements.txt
uv export --format requirements-txt --no-dev --no-hashes > requirements.txt

# Ver el repo en GitHub
gh repo view --web
```

## 9. Si me preguntan "¿qué harías distinto si tuvieras 2 días más?"

1. **Multi-página** — pre-procesar PDFs/imágenes en N páginas individuales con `pdf2image` o `PyMuPDF`, parsear cada una, concatenar.
2. **Paralelizar las dos llamadas de Vision** — ahora son secuenciales (`parse_original` → `parse_amendment`). `asyncio.gather` cortaría la latencia total casi a la mitad.
3. **Score de confianza** — pedirle al Auditor que devuelva un campo `confidence: float` 0-1 sobre cuán seguro está de la extracción. Los casos < 0.7 irían a una cola de revisión humana.
4. **Test suite** — 10-20 pares más con expected outputs, y un script que corra el pipeline contra todos y mida accuracy.
5. **Streaming del output** — el `summary_of_the_change` puede ser largo. Streamearlo a stdout daría mejor UX.
6. **Cachear el text extraction** — si corrés sobre el mismo par 2 veces, no tendría que re-parsear las imágenes. Un cache key sobre el hash del archivo evitaría las 2 llamadas Vision en la 2da corrida.
