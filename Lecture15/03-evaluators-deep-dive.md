# Deep Dive: Evaluadores de agentevals

Este documento explora en detalle todos los evaluadores disponibles en `agentevals`, la librería oficial de LangChain para evaluar trayectorias de agentes. Para cada evaluador se explica cómo funciona internamente, cuándo usarlo, y cómo se relaciona con el Web Search Agent que implementamos.

---

## El formato de trayectoria

Todos los evaluadores de `agentevals` operan sobre **trayectorias** representadas como listas de mensajes en formato OpenAI:

```python
trajectory = [
    {"role": "user",      "content": "¿Quién es el CEO de NVIDIA?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "web_search", "arguments": '{"query": "CEO NVIDIA"}'}}
    ]},
    {"role": "tool",      "content": "Jensen Huang es el CEO de NVIDIA..."},
    {"role": "assistant", "content": "Jensen Huang es el CEO de NVIDIA."},
]
```

Cada mensaje tiene un `role`:

| Role | Quién habla | Cuándo aparece |
|---|---|---|
| `user` | El usuario | Siempre al inicio |
| `assistant` | El LLM | Después de cada ciclo de razonamiento |
| `tool` | El resultado de la tool | Después de cada `tool_call` |

Esta es la representación que construye `run_agent_and_get_trajectory` en `agent_evals.py`.

---

## Categorías de evaluadores

`agentevals` tiene dos familias principales:

```
agentevals/
├── trajectory/          ← operan sobre mensajes (lo que usamos)
│   ├── match.py         → trajectory match evaluators
│   └── llm.py           → trajectory LLM-as-judge
└── graph_trajectory/    ← operan sobre nodos del grafo LangGraph
    ├── strict.py        → graph trajectory strict match
    └── llm.py           → graph trajectory LLM-as-judge
```

---

## Familia 1: Trajectory Match (determinístico)

### ¿Cómo funciona internamente?

`create_trajectory_match_evaluator` compara la trayectoria del agente contra una trayectoria de referencia **extrayendo y comparando únicamente los tool calls**, ignorando el contenido de los mensajes de texto.

Internamente:
1. Extrae todos los `tool_calls` de los mensajes `assistant` en la trayectoria real
2. Extrae todos los `tool_calls` de los mensajes `assistant` en la referencia
3. Compara los dos conjuntos según el `trajectory_match_mode`

El contenido de los mensajes (`content`) **nunca se compara** — solo los nombres y argumentos de las tools.

---

### Modo: `strict`

**Mismas tools, mismo orden, mismos argumentos.**

```python
from agentevals.trajectory.match import create_trajectory_match_evaluator

evaluator = create_trajectory_match_evaluator(trajectory_match_mode="strict")
```

Ejemplo — falla porque el agente llamó dos tools pero la referencia espera solo una:

```python
outputs = [
    {"role": "user", "content": "¿Clima en SF?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "get_weather",        "arguments": '{"city": "SF"}'}},
        {"function": {"name": "accuweather_forecast","arguments": '{"city": "SF"}'}},
    ]},
    {"role": "tool",      "content": "80°F y soleado."},
    {"role": "assistant", "content": "Hace 80°F y está soleado en SF."},
]
reference_outputs = [
    {"role": "user", "content": "¿Clima en SF?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "get_weather", "arguments": '{"city": "SF"}'}},
    ]},
    {"role": "tool",      "content": "80°F y soleado."},
    {"role": "assistant", "content": "Hace 80°F y está soleado en SF."},
]
result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
# → {"key": "trajectory_strict_match", "score": False}
```

**Cuándo usarlo:** cuando el orden importa por razones de negocio — por ejemplo, un agente que **siempre** debe consultar una política de empresa antes de ejecutar una acción.

---

### Modo: `unordered`

**Mismas tools, cualquier orden.**

```python
evaluator = create_trajectory_match_evaluator(trajectory_match_mode="unordered")
```

Ejemplo — pasa aunque el agente llamó las tools en orden inverso al de la referencia:

```python
# Referencia: [get_fun_activities, get_weather]
# Agente:     [get_weather, get_fun_activities]
# → score: True
```

**Cuándo usarlo:** cuando necesitas que el agente obtenga cierta información, pero el orden no importa — por ejemplo, consultar clima y eventos de una ciudad en cualquier secuencia.

---

### Modo: `subset`

**El agente llamó AL MENOS las tools de la referencia** (puede llamar más).

```python
evaluator = create_trajectory_match_evaluator(trajectory_match_mode="subset")
```

```python
# Referencia: [web_search]
# Agente:     [web_search, web_search]  ← llamó más, pero incluye las esperadas
# → score: True
```

**En nuestro agente:** usamos `subset` porque queremos asegurar que el agente llame `web_search` cuando debe, pero no penalizamos si lo llama más veces.

**Cuándo usarlo:** evaluaciones permisivas donde lo importante es que el agente no omita tools críticas.

---

### Modo: `superset`

**El agente NO llamó tools fuera de las esperadas** (puede llamar menos).

```python
evaluator = create_trajectory_match_evaluator(trajectory_match_mode="superset")
```

```python
# Referencia: [web_search, calculator]
# Agente:     [web_search]  ← llamó solo una de las esperadas, pero ninguna extra
# → score: True

# Referencia: [web_search]
# Agente:     [web_search, calculator]  ← llamó una tool fuera del conjunto
# → score: False
```

**Cuándo usarlo:** evaluaciones restrictivas donde quieres asegurar que el agente no llame tools innecesarias o no autorizadas.

---

### Parámetro: `tool_args_match_mode`

Controla cómo se comparan los **argumentos** de cada tool call:

| Valor | Comportamiento |
|---|---|
| `"exact"` (default) | Los argumentos deben ser idénticos |
| `"ignore"` | Solo se compara el nombre de la tool, no los argumentos |
| `"subset"` | Los args del agente deben ser subconjunto de los de la referencia |
| `"superset"` | Los args de la referencia deben ser subconjunto de los del agente |

```python
# En nuestro agente usamos "ignore" porque no nos importa
# qué query exacto construye el LLM, solo que llame web_search
evaluator = create_trajectory_match_evaluator(
    trajectory_match_mode="subset",
    tool_args_match_mode="ignore",
)
```

---

### Parámetro: `tool_args_match_overrides`

Permite definir un comparador **custom por tool**:

```python
evaluator = create_trajectory_match_evaluator(
    trajectory_match_mode="strict",
    tool_args_match_overrides={
        # Permite case-insensitive para web_search
        "web_search": lambda x, y: x["query"].lower() == y["query"].lower(),
        # Exact match para calculator
        "calculator": "exact",
    }
)
```

Útil cuando diferentes tools necesitan niveles distintos de precisión en la comparación de argumentos.

---

### Resumen de modos

| Modo | Orden importa | Tools extra | Tools faltantes | Cuándo usar |
|---|---|---|---|---|
| `strict` | ✅ Sí | ❌ Falla | ❌ Falla | Secuencias obligatorias |
| `unordered` | ❌ No | ❌ Falla | ❌ Falla | Conjunto exacto, cualquier orden |
| `subset` | ❌ No | ✅ Permitido | ❌ Falla | No omitir tools críticas |
| `superset` | ❌ No | ❌ Falla | ✅ Permitido | No llamar tools extra |

---

## Familia 2: Trajectory LLM-as-judge

### ¿Cómo funciona internamente?

`create_trajectory_llm_as_judge` envía la trayectoria completa a un LLM juez con un prompt que le pide evaluar si el proceso fue razonable. El LLM retorna un score booleano (`True`/`False`) o numérico, más un comentario explicativo.

A diferencia del trajectory match, **no compara contra una referencia** — juzga la trayectoria de forma absoluta.

---

### Uso básico (sin referencia)

```python
from agentevals.trajectory.llm import (
    create_trajectory_llm_as_judge,
    TRAJECTORY_ACCURACY_PROMPT,
)

llm_judge = create_trajectory_llm_as_judge(
    prompt=TRAJECTORY_ACCURACY_PROMPT,
    judge=judge_llm,   # instancia de ChatOpenAI o similar
)

result = llm_judge(outputs=trajectory)
# → {"key": "trajectory_accuracy", "score": True, "comment": "..."}
```

El prompt `TRAJECTORY_ACCURACY_PROMPT` le pide al juez evaluar si la trayectoria es lógica, eficiente y correcta. No necesita saber cuál era la respuesta correcta.

---

### Uso con referencia

Cuando tienes una trayectoria de referencia, puedes pasarla para una evaluación más precisa:

```python
from agentevals.trajectory.llm import (
    create_trajectory_llm_as_judge,
    TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
)

llm_judge = create_trajectory_llm_as_judge(
    prompt=TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
    judge=judge_llm,
)

result = llm_judge(
    outputs=trajectory,
    reference_outputs=reference_trajectory,
)
```

El juez compara la trayectoria real contra la referencia y evalúa si son semánticamente equivalentes, aunque difieran en detalles como `"SF"` vs `"San Francisco"`.

---

### Score continuo vs binario

Por defecto el score es booleano. Para obtener un float entre 0 y 1:

```python
llm_judge = create_trajectory_llm_as_judge(
    prompt=TRAJECTORY_ACCURACY_PROMPT,
    judge=judge_llm,
    continuous=True,   # ← retorna float
)
# → {"score": 0.85, "comment": "..."}
```

O con choices específicos:

```python
llm_judge = create_trajectory_llm_as_judge(
    prompt=TRAJECTORY_ACCURACY_PROMPT,
    judge=judge_llm,
    choices=[0.0, 0.25, 0.5, 0.75, 1.0],
)
```

---

### Prompt personalizado con múltiples dimensiones

Esto es lo que implementamos en `agent_evals.py` — un prompt que evalúa tres dimensiones independientes:

```python
from pydantic import BaseModel

class EvalScores(BaseModel):
    correctness: float
    efficiency: float
    faithfulness: float
    comment: str

def multidim_judge(trajectory: list) -> dict:
    structured_judge = llm.with_structured_output(EvalScores)
    prompt = CUSTOM_EVAL_PROMPT.format(
        outputs=json.dumps(trajectory, indent=2, ensure_ascii=False)
    )
    result = structured_judge.invoke([HumanMessage(content=prompt)])
    return {
        "correctness": result.correctness,
        "efficiency": result.efficiency,
        "faithfulness": result.faithfulness,
        "comment": result.comment,
    }
```

| Dimensión | Pregunta | Cómo falla en nuestro agente |
|---|---|---|
| `correctness` | ¿La respuesta es correcta? | Rara vez — el LLM suele saber la respuesta |
| `efficiency` | ¿Usó las tools mínimas? | Cuando busca en web para preguntas de conocimiento general |
| `faithfulness` | ¿La respuesta viene de los resultados? | Cuando ignora los snippets y responde desde su conocimiento |

---

### Few-shot examples

Para calibrar el juez puedes proporcionar ejemplos:

```python
few_shot_examples = [
    {
        "inputs": "¿Cuántas lunas tiene Marte?",
        "outputs": "Marte tiene 2 lunas: Fobos y Deimos.",
        "reasoning": "Respondió correctamente sin necesidad de buscar.",
        "score": 1,
    },
    {
        "inputs": "¿Cuántas lunas tiene Marte?",
        "outputs": "Busqué en la web y encontré que Marte tiene 2 lunas.",
        "reasoning": "Buscó innecesariamente información de conocimiento general.",
        "score": 0,
    },
]

llm_judge = create_trajectory_llm_as_judge(
    prompt=TRAJECTORY_ACCURACY_PROMPT,
    judge=judge_llm,
    few_shot_examples=few_shot_examples,
)
```

Esto es especialmente útil para reducir inconsistencias entre ejecuciones.

---

## Familia 3: Graph Trajectory (específico de LangGraph)

Esta familia opera sobre **nodos del grafo** en lugar de mensajes. Es conveniente cuando usas LangGraph con checkpointer, porque puedes extraer la trayectoria directamente del thread sin construirla manualmente.

### Extracción automática de trayectoria

```python
from agentevals.graph_trajectory.utils import extract_langgraph_trajectory_from_thread
from langgraph.checkpoint.memory import MemorySaver

# El agente DEBE usar checkpointer para guardar el estado
checkpointer = MemorySaver()
agent = graph_builder.compile(checkpointer=checkpointer)

# Invocar con thread_id
agent.invoke(
    {"messages": [HumanMessage(content="¿Quién es el CEO de NVIDIA?")]},
    config={"configurable": {"thread_id": "1"}},
)

# Extraer trayectoria en formato GraphTrajectory
extracted = extract_langgraph_trajectory_from_thread(
    agent, {"configurable": {"thread_id": "1"}}
)
```

El resultado tiene el formato:

```python
{
    "inputs": [{"__start__": {"messages": [...]}}],
    "outputs": {
        "results": [...],
        "steps": [["__start__", "agent", "tools", "agent"]],  # nodos visitados
    }
}
```

---

### Graph trajectory LLM-as-judge

```python
from agentevals.graph_trajectory.llm import create_graph_trajectory_llm_as_judge

graph_judge = create_graph_trajectory_llm_as_judge(
    judge=judge_llm,
)

result = graph_judge(
    inputs=extracted["inputs"],
    outputs=extracted["outputs"],
)
# → {"key": "graph_trajectory_accuracy", "score": True, "comment": "..."}
```

La diferencia con `trajectory_llm_as_judge`: el juez ve los **nombres de los nodos** (`agent`, `tools`, `__interrupt__`) en lugar de los mensajes completos. Es más abstracto pero más rápido y barato.

---

### Graph trajectory strict match

Verifica que los nodos visitados sean exactamente los esperados:

```python
from agentevals.graph_trajectory.strict import graph_trajectory_strict_match

reference = {
    "results": [],
    "steps": [["__start__", "agent", "tools", "agent"]],  # secuencia esperada
}

result = graph_trajectory_strict_match(
    outputs=extracted["outputs"],
    reference_outputs=reference,
)
# → {"key": "graph_trajectory_strict_match", "score": True}
```

**Cuándo usarlo:** para verificar que el agente siempre sigue el mismo flujo de nodos para un tipo de pregunta dado.

**En nuestro agente:**
- Preguntas sin búsqueda: `["__start__", "agent"]`
- Preguntas con búsqueda: `["__start__", "agent", "tools", "agent"]`

---

## Comparación completa de evaluadores

| Evaluador | Familia | Necesita referencia | Output | Costo | Determinístico |
|---|---|---|---|---|---|
| `trajectory_match` (strict) | Trajectory | Sí | bool | Gratis | ✅ |
| `trajectory_match` (unordered) | Trajectory | Sí | bool | Gratis | ✅ |
| `trajectory_match` (subset) | Trajectory | Sí | bool | Gratis | ✅ |
| `trajectory_match` (superset) | Trajectory | Sí | bool | Gratis | ✅ |
| `trajectory_llm_as_judge` | Trajectory | Opcional | bool/float | 1 LLM call | ❌ |
| `graph_trajectory_strict_match` | Graph | Sí | bool | Gratis | ✅ |
| `graph_trajectory_llm_as_judge` | Graph | Opcional | bool/float | 1 LLM call | ❌ |
| `retrieval_quality`* | Custom | Sí | float | Gratis | ✅ |

> \* `retrieval_quality` no es de `agentevals` — es el evaluador custom que implementamos en `agent_evals.py`.

---

## ¿Cuándo usar cada evaluador?

```
¿Tienes una trayectoria de referencia?
├── SÍ → ¿El orden importa?
│         ├── SÍ → trajectory_match (strict)
│         └── NO → ¿Quieres permitir tools extra?
│                  ├── SÍ → trajectory_match (subset)
│                  └── NO → ¿Quieres permitir tools faltantes?
│                           ├── SÍ → trajectory_match (superset)
│                           └── NO → trajectory_match (unordered)
│
└── NO → ¿Usas LangGraph con checkpointer?
          ├── SÍ → graph_trajectory_llm_as_judge
          └── NO → trajectory_llm_as_judge
                   └── ¿Necesitas score numérico?
                       ├── SÍ → continuous=True o choices=[...]
                       └── NO → score booleano (default)
```

---

## Integración con LangSmith

Todos los evaluadores se pueden usar directamente con `client.evaluate()`:

```python
from langsmith import Client
from agentevals.trajectory.match import create_trajectory_match_evaluator
from agentevals.trajectory.llm import create_trajectory_llm_as_judge

client = Client()

# Evaluadores como funciones con firma (outputs, reference_outputs) → dict
def eval_trajectory_subset(outputs, reference_outputs):
    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode="subset",
        tool_args_match_mode="ignore",
    )
    result = evaluator(
        outputs=outputs["trajectory"],
        reference_outputs=build_reference_trajectory(...),
    )
    return {"key": "trajectory_subset", "score": float(result["score"])}


experiment = client.evaluate(
    agent_target,
    data="agent-evals-demo",
    evaluators=[
        eval_tool_use,
        eval_retrieval,
        eval_trajectory_subset,
        eval_llm_judge,
    ],
    experiment_prefix="llama-3.1-70b-v2",
    max_concurrency=1,
)
```

LangSmith muestra cada evaluador como una columna independiente, permitiendo comparar versiones del agente en el mismo dashboard.

---

## Async support

Todos los evaluadores tienen versiones async para usar en pipelines de evaluación concurrentes:

```python
from agentevals.trajectory.llm import create_async_trajectory_llm_as_judge
from agentevals.trajectory.match import create_async_trajectory_match_evaluator

async_judge = create_async_trajectory_llm_as_judge(
    prompt=TRAJECTORY_ACCURACY_PROMPT,
    judge=judge_llm,
)

# En un pipeline async
results = await asyncio.gather(*[
    async_judge(outputs=traj) for traj in trajectories
])
```

Útil cuando tienes datasets grandes y quieres reducir el tiempo total de evaluación.

---

## Limitaciones conocidas

**LLM-as-judge:**
- No es determinístico — el mismo input puede dar scores diferentes
- Tiene sesgos del modelo juez (verbosity bias, positivity bias)
- Es más caro — cada evaluación requiere una llamada adicional al LLM
- Puede "perdonar" errores que los evaluadores determinísticos detectarían

**Trajectory match:**
- Requiere construir trayectorias de referencia manualmente
- No evalúa la calidad del contenido — solo si se llamaron las tools correctas
- Sensible a cambios en los nombres de las tools

**Graph trajectory:**
- Requiere usar LangGraph con checkpointer (agrega complejidad)
- La evaluación por nodos es más abstracta — pierde el detalle de los mensajes

---

## ⏭️ Siguiente

➡️ [Workshop: evaluación de tu propio agente](03-workshop.md)
