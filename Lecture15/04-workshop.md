# Workshop: Evalúa tu propio agente

En este workshop construirás un agente desde cero y lo evaluarás usando los mismos patrones que estudiamos en `agent_evals.ipynb`.

**⏱ Tiempo:** 1.5 horas  
**📓 Notebook:** `workshop_agent_evals.ipynb`  
**🔑 Requisito:** tener configurado `NVIDIA_API_KEY` o `GROQ_API_KEY` en tu `.env`

---

## Antes de empezar

Abre `workshop_agent_evals.ipynb` y corre las celdas de instalación y setup (Sección 0). Verifica que el LLM responde correctamente antes de continuar.

```python
print("LLM listo ✅")  # debe aparecer esto al correr la celda de setup
```

Si ves un error de API key, revisa tu archivo `.env` y vuelve a correr `load_dotenv()`.

---

## Parte 1 — Define tu agente y tools (20 min)

### ¿Qué tienes que hacer?

Tienes que definir **al menos 2 tools** para tu agente y construir el grafo LangGraph.

La tool `web_search` ya está implementada. Tu tarea es:

1. **Definir una segunda tool** con `@tool`
2. **Personalizar el system prompt** para que el agente sepa cuándo usar cada tool
3. **Verificar** que el agente funciona con 3 preguntas de prueba

### ¿Qué tool puedo crear?

Elige algo que tenga sentido como par con `web_search`. Algunas ideas:

| Tool | Descripción | Cuándo usarla |
|---|---|---|
| `calculator` | Evalúa expresiones matemáticas | Preguntas con cálculos numéricos |
| `unit_converter` | Convierte unidades (km↔millas, °C↔°F) | Conversiones de medidas |
| `days_between` | Calcula días entre dos fechas | Preguntas sobre fechas |
| `text_summarizer` | Resume un texto largo | Cuando el usuario pide un resumen |

Ejemplo de una tool de calculadora:

```python
@tool
def calculator(expression: str) -> str:
    """Evalúa una expresión matemática. Úsala para cualquier cálculo numérico
    como porcentajes, multiplicaciones, divisiones o fórmulas.
    Ejemplo: '15 * 340 / 100' para calcular el 15% de 340."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Resultado: {result}"
    except Exception as e:
        return f"Error al calcular: {e}"
```

> ⚠️ El docstring es crítico — es lo que el LLM lee para decidir cuándo llamar la tool. Un docstring vago hace que el agente no sepa cuándo usarla.

### Personaliza el system prompt

El system prompt debe decirle al agente **cuándo usar cada tool**. Sin esto, el agente puede usar `web_search` para todo, incluyendo cálculos simples.

```python
SYSTEM_PROMPT = SystemMessage(
    content="""Eres un asistente útil con acceso a búsqueda web y calculadora.

Reglas:
- Usa web_search para preguntas sobre eventos actuales, personas, empresas o datos que cambian.
- Usa calculator para cualquier operación matemática, sin importar qué tan simple sea.
- Para preguntas de conocimiento general (capitales, fórmulas, definiciones), responde directamente sin usar tools.
- Responde siempre en español, de forma concisa.
"""
)
```

### Construye el grafo LangGraph

El notebook tiene un bloque con comentarios que explican cada instrucción del grafo. Completa los argumentos marcados con `???`:

```python
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", call_model)   # nodo del LLM
graph_builder.add_node("tools", ToolNode(TOOLS))  # nodo ejecutor de tools
graph_builder.set_entry_point("agent")        # siempre empieza en el agente
graph_builder.add_conditional_edges("agent", should_continue)  # router
graph_builder.add_edge("tools", "agent")      # tras ejecutar tools → vuelve al agente
```

El flujo resultante es:
```
Usuario → agent → ¿tool_calls? → SÍ → tools → agent → ...
                               → NO → END
```

### Prueba manual

Antes de pasar a la evaluación, prueba el agente con al menos 3 preguntas que cubran todos los casos:

```python
test_questions = [
    "¿Cuánto es el 18% de 450?",          # debe usar calculator
    "¿Quién ganó el último Oscar?",         # debe usar web_search
    "¿Cuántos días tiene un año bisiesto?", # no debe usar tools
]
```

Verifica que cada pregunta activa (o no activa) la tool correcta antes de continuar.

---

## Parte 2 — Define el dataset de evaluación (20 min)

### ¿Qué tienes que hacer?

Construir un dataset de **mínimo 8 ejemplos** que cubra todos los casos de tu agente.

### Estructura de cada ejemplo

```python
{
    "question": "¿Cuánto es el 18% de 450?",
    "expected_answer": "El 18% de 450 es 81",
    "expected_tools": ["calculator"],        # tools que DEBE llamar
    "expected_content": [],                  # términos clave en los resultados de búsqueda
}
```

**`expected_content`** es la lista de términos que *deben aparecer* en los resultados devueltos por `web_search`. Es lo que usa el evaluador de retrieval para medir si la búsqueda fue útil.

- Solo aplica cuando `expected_tools` incluye `web_search`
- Para las demás tools (calculadora, conversor, etc.), déjalo `[]`
- Usa 2-4 términos concretos que casi seguramente aparecerán en los snippets

Ejemplos:
```python
# Pregunta con web_search → expected_content con términos esperados
{
    "question": "¿Quién ganó el último Oscar a mejor película?",
    "expected_answer": "...",
    "expected_tools": ["web_search"],
    "expected_content": ["Oscar", "mejor película", "Academy Awards"],
}

# Pregunta con calculator → expected_content vacío
{
    "question": "¿Cuánto es el 18% de 450?",
    "expected_answer": "El 18% de 450 es 81",
    "expected_tools": ["calculator"],
    "expected_content": [],
}
```

### Distribución mínima

| Tipo | Cantidad mínima | `expected_tools` |
|---|---|---|
| Usa `web_search` | 3 | `["web_search"]` |
| Usa tu segunda tool | 3 | `["tu_tool"]` |
| Sin tools | 2 | `[]` |

### Tips para diseñar buenos ejemplos

**Para preguntas con `web_search`:**
- Usa preguntas sobre eventos recientes o personas en cargos actuales
- El `expected_content` debe tener 2-4 términos clave que *deberían* aparecer en los snippets de DuckDuckGo
- Evita términos demasiado específicos (nombres propios con variantes ortográficas) — prefiere términos generales que casi siempre aparecen
- Evita preguntas cuya respuesta puede haber cambiado (pueden fallar por razones ajenas al agente)

**Para preguntas con tu segunda tool:**
- Asegúrate de que el `expected_answer` sea verificable exactamente
- Para calculadora: usa operaciones con resultado exacto, no aproximado

**Para preguntas sin tools:**
- Elige preguntas de conocimiento estable (ciencias, historia, geografía)
- Son las más útiles para detectar si el agente llama tools innecesariamente

### Validación automática

La celda de validación al final de la Parte 2 verifica que tu dataset tiene al menos 8 ejemplos y la distribución mínima de tools:

```python
assert len(eval_dataset) >= 8, "❌ El dataset debe tener al menos 8 ejemplos"
```

No avances a la Parte 3 hasta que esta celda corra sin errores.

---

## Parte 3 — Implementa los evaluadores (30 min)

### ¿Qué tienes que hacer?

Implementar la función `evaluate_tool_use` y configurar los otros tres evaluadores.

### 3.1 evaluate_tool_use (TODO obligatorio)

Esta es la única función que debes implementar desde cero. Recibe dos listas y retorna un diccionario con `score` y `comment`:

```python
def evaluate_tool_use(tools_called: list, expected_tools: list) -> dict:
    called_set = set(tools_called)
    expected_set = set(expected_tools)
    # TODO: implementa la lógica de scoring
```

**Criterios:**
- Si `expected_tools == []`: el agente **no debe** llamar ninguna tool → score 1.0 si `called_set` está vacío, 0.0 si no
- Si `expected_tools != []`: el agente **debe** llamar todas las tools esperadas → score 1.0 si `expected_set ⊆ called_set`, 0.0 si no

Los 4 tests al final de la celda deben pasar antes de continuar:

```python
assert evaluate_tool_use([], [])["score"] == 1.0
assert evaluate_tool_use(["web_search"], ["web_search"])["score"] == 1.0
assert evaluate_tool_use([], ["web_search"])["score"] == 0.0
assert evaluate_tool_use(["web_search"], [])["score"] == 0.0
```

### 3.2 evaluate_retrieval (TODO para preguntas con web_search)

Este evaluador mide si los resultados de `web_search` contienen la información necesaria para responder. Opera sobre los mensajes `role: tool` de la trayectoria.

La función ya está implementada en el notebook — tu tarea es **poblar `expected_content`** correctamente en el dataset (Parte 2) para que tenga sentido.

Cómo funciona internamente:

```python
def evaluate_retrieval(trajectory: list, expected_content: list) -> dict:
    # Extrae todo el texto devuelto por las tools
    tool_results = " ".join(
        msg["content"] for msg in trajectory if msg["role"] == "tool"
    ).lower()

    # Cuenta cuántos términos esperados aparecen
    found = [term for term in expected_content if term.lower() in tool_results]
    score = len(found) / len(expected_content)   # Recall@expected_content

    return {"score": score, "found": found, "missing": [...]}
```

**Score = 1.0** → todos los términos esperados aparecieron en los snippets  
**Score = 0.5** → solo la mitad → la respuesta puede ser incompleta  
**Score = 0.0** → no llamó la tool, o los resultados son irrelevantes

> ⚠️ Este evaluador retorna `None` para preguntas sin `web_search`. En la tabla de resultados aparecerá como `NaN` — es correcto, no es un error.

**Prueba directa:**
```python
sample = eval_dataset[1]  # una pregunta con web_search
result = run_agent_and_get_trajectory(sample["question"])
print(evaluate_retrieval(result["trajectory"], sample["expected_content"]))
# → {"score": 1.0, "found": ["Jensen Huang", "NVIDIA", "CEO"], "missing": []}
```

### 3.3 trajectory_match (configurar el modo)

Ya está implementado, pero debes elegir el `trajectory_match_mode` más apropiado para tu agente:

```python
trajectory_evaluator = create_trajectory_match_evaluator(
    trajectory_match_mode="subset",   # TODO: ¿es este el modo correcto?
    tool_args_match_mode="ignore",
)
```

Piensa: ¿qué quieres evaluar?
- Si no te importa que el agente llame tools extra → `"subset"`
- Si el agente nunca debe llamar tools fuera de las esperadas → `"superset"`
- Si necesitas el conjunto exacto → `"unordered"`

Consulta el documento `03-evaluators-deep-dive.md` si tienes dudas sobre los modos.

### 3.4 LLM-as-judge (personalizar el prompt)

El prompt ya tiene las 3 dimensiones (`correctness`, `efficiency`, `faithfulness`). Puedes dejarlo como está o agregar criterios específicos para tu dominio en la sección marcada con `TODO`:

```python
CUSTOM_EVAL_PROMPT = """...
TODO: Agrega criterios específicos para tu dominio aquí si lo necesitas.
...
"""
```

Por ejemplo, si tu agente incluye una calculadora:
```
- Para preguntas matemáticas: la respuesta debe incluir el resultado numérico exacto.
- Para preguntas de búsqueda: la respuesta debe citar explícitamente la fuente.
```

### Prueba individual antes del loop

Antes de correr la evaluación completa, prueba cada evaluador con un ejemplo del dataset:

```python
# Prueba evaluate_tool_use (pregunta sin tools)
test_result = run_agent_and_get_trajectory(eval_dataset[0]["question"])
print(evaluate_tool_use(test_result["tools_called"], eval_dataset[0]["expected_tools"]))

# Prueba evaluate_retrieval (pregunta con web_search)
search_sample = next(item for item in eval_dataset if item["expected_tools"] == ["web_search"])
search_result = run_agent_and_get_trajectory(search_sample["question"])
print(evaluate_retrieval(search_result["trajectory"], search_sample["expected_content"]))

# Prueba LLM-as-judge
print(multidim_judge(test_result["trajectory"]))
```

Si alguno falla aquí, corrígelo antes de correr el loop completo.

---

## Parte 4 — Corre la evaluación y analiza (20 min)

### ¿Qué tienes que hacer?

Correr el loop completo, analizar los resultados y responder las preguntas de reflexión.

### Ajusta el sleep

Antes de correr, ajusta `sleep_between` según tu proveedor:

```python
eval_results = run_full_eval(eval_dataset, sleep_between=8)  # NVIDIA
# eval_results = run_full_eval(eval_dataset, sleep_between=3)  # Groq
```

Si ves errores 429, sube el sleep. El loop puede tardar entre 3 y 10 minutos dependiendo del tamaño del dataset.

### Interpreta los resultados

Una vez que termina, analiza la tabla de resultados:

```
tool_use_score  retrieval_score  trajectory_match  correctness  efficiency  faithfulness
```

> `retrieval_score` aparece como `NaN` para preguntas que no usan `web_search` — es correcto. El promedio se calcula solo sobre las preguntas con búsqueda.

Busca patrones:

**Si `tool_use_score` es bajo:**
- El agente está llamando tools cuando no debe, o no las llama cuando debe
- Revisa el system prompt — ¿está claro cuándo usar cada tool?

**Si `efficiency` es bajo pero `correctness` es alto:**
- El agente llega a la respuesta correcta por un camino ineficiente
- Ejemplo clásico: buscar en web para responder "¿cuánto es 2+2?"

**Si `faithfulness` es bajo:**
- El agente está ignorando los resultados de las tools y respondiendo desde su conocimiento
- La respuesta puede ser correcta pero no está fundamentada en los datos obtenidos

**Si `retrieval_score` es bajo:**
- La búsqueda no está devolviendo los términos esperados en los snippets
- Dos causas posibles: (1) el LLM construyó un query poco específico, o (2) los términos en `expected_content` son demasiado restrictivos
- Para distinguirlas: imprime `search_result["trajectory"]` y lee los mensajes `role: tool` manualmente

**Si `retrieval_score` es alto pero `faithfulness` es bajo:**
- Los resultados de búsqueda sí llegaron, pero el LLM los ignoró y respondió desde su conocimiento
- Este es el caso más peligroso: la respuesta puede ser correcta hoy pero incorrecta mañana si el dato cambia

### LangSmith (opcional)

Si tienes configurado `LANGSMITH_API_KEY`, las secciones 4.3 y 4.4 del notebook suben tu experimento al dashboard automáticamente. Cada evaluador aparece como columna independiente y puedes comparar distintas versiones del agente sobre el mismo dataset.

El dataset se crea con nombre único por estudiante (`workshop-{USER}-evals`) para evitar colisiones.

### Preguntas de reflexión

Responde estas preguntas en una celda Markdown del notebook:

1. ¿Cuál fue el evaluador con el score más bajo? ¿Por qué crees que fue así?

2. ¿Hubo algún caso donde los scores del LLM-as-judge fueron altos pero los evaluadores determinísticos fueron bajos? ¿Qué significa eso?

3. Si `retrieval_score` fue bajo en alguna pregunta: ¿fue problema del query que construyó el LLM, o los términos en `expected_content` eran demasiado restrictivos? ¿Cómo lo distinguirías?

4. ¿Qué cambio específico harías en el system prompt para mejorar el score más bajo? Formula la hipótesis antes de implementarla.

5. Si `retrieval_score` fue alto pero `faithfulness` fue bajo: ¿qué implica eso? ¿Es un problema grave?

### Mejora iterativa (bonus)

Si tienes tiempo, implementa el cambio que propusiste en la pregunta 4, vuelve a correr la evaluación y compara los resultados:

```python
results_v1 = pd.DataFrame(eval_results)

SYSTEM_PROMPT = SystemMessage(content="... versión mejorada ...")
llm_with_tools = llm.bind_tools(TOOLS, parallel_tool_calls=False)
agent = graph_builder.compile()

eval_results_v2 = run_full_eval(eval_dataset, sleep_between=8)
results_v2 = pd.DataFrame(eval_results_v2)

compare_cols = ["tool_use_score", "trajectory_match", "correctness",
                "efficiency", "faithfulness"]
ok_v1 = results_v1[results_v1["status"] == "ok"]
ok_v2 = results_v2[results_v2["status"] == "ok"]
print("V1:", ok_v1[compare_cols].mean().round(2).to_dict())
print("V2:", ok_v2[compare_cols].mean().round(2).to_dict())
```

Este es el ciclo de *evaluation-driven development* aplicado en práctica.

---

## Checklist de entrega

Al finalizar el workshop, tu notebook debe tener:

- [ ] Al menos 2 tools definidas con docstrings claros
- [ ] Grafo LangGraph construido correctamente
- [ ] System prompt personalizado para tu dominio
- [ ] Dataset de mínimo 8 ejemplos con la distribución correcta
- [ ] `evaluate_tool_use` implementada y con los 4 tests pasando
- [ ] `expected_content` poblado correctamente para preguntas con `web_search`
- [ ] Loop completo corrido sin errores
- [ ] Dos tablas de resultados (determinísticos + LLM-as-judge)
- [ ] 5 preguntas de reflexión respondidas en una celda Markdown

---

## Recursos

- `agent_evals.ipynb` — el notebook de referencia que ya corriste
- `03-evaluators-deep-dive.md` — explicación detallada de todos los evaluadores de `agentevals`
- [Documentación de agentevals](https://github.com/langchain-ai/agentevals)
- [Documentación de LangGraph](https://langchain-ai.github.io/langgraph/)
