# Métricas de evaluación y LLM-as-Judge

---

## ¿Por qué evaluar es difícil?

Los LLMs generan texto libre — no hay una respuesta binaria correcta/incorrecta como en software tradicional. Esto crea tres alternativas, cada una con sus trade-offs.

| Método | Velocidad | Costo | Calidad |
|---|---|---|---|
| Evaluación humana | 🐢 Lenta | 💸 Cara | ✅ Gold standard |
| Métricas basadas en reglas | 🐇 Rápida | 💰 Barata | ⚠️ Correlación limitada |
| LLM-as-judge | 🐇 Rápida | 💰 Moderada | ✅ Cercana a humanos |

> El flujo moderno: **LLM-as-judge** para iteración rápida → **humanos** para calibración y validación final.

---

## Sección 1 — Evaluación humana

La evaluación humana es el estándar de referencia (*gold standard*), pero tiene limitaciones prácticas.

### El problema de la subjetividad

Cuando dos evaluadores califican la misma respuesta con criterios distintos, ¿cómo medimos si están de acuerdo más allá del azar?

**Kappa de Cohen** — mide el acuerdo entre dos evaluadores descontando el acuerdo esperado por azar:

```
κ = (acuerdo observado - acuerdo esperado) / (1 - acuerdo esperado)
```

- κ = 1.0 → acuerdo perfecto
- κ = 0.0 → acuerdo al nivel del azar
- κ < 0 → desacuerdo sistemático

Variantes: **Kappa de Fleiss** (más de 2 evaluadores), **Alpha de Krippendorff** (escalas ordinales).

### Limitaciones prácticas

- **Subjetividad** — distintos evaluadores pueden tener criterios distintos
- **Lenta** — no escala para iterar rápido
- **Cara** — especialmente para dominios especializados

---

## Sección 2 — Métricas basadas en reglas

La idea: escribir etiquetas (*labels*) una sola vez como referencia y comparar automáticamente.

### Métricas clásicas

**BLEU** (*BiLingual Evaluation Understudy*) — compara n-gramas entre la respuesta generada y la referencia. Diseñado originalmente para traducción automática.

**ROUGE** (*Recall-Oriented Understudy for Gisting Evaluation*) — orientado a recall, útil para resúmenes. Variantes: ROUGE-N, ROUGE-L.

**METEOR** (*Metric for Evaluation of Translation with Explicit ORdering*) — considera sinónimos y paráfrasis, mejor correlación con humanos que BLEU.

### En nuestro agente: `evaluate_tool_use` y `evaluate_retrieval`

En lugar de comparar texto, nuestros evaluadores determinísticos comparan **estructura**:

```python
# evaluate_tool_use — compara conjuntos de tools
called_set = {"web_search"}
expected_set = {"web_search"}
score = 1.0 if expected_set.issubset(called_set) else 0.0

# evaluate_retrieval — Recall@expected_content
found = [term for term in expected_content if term.lower() in tool_results]
score = len(found) / len(expected_content)
```

`evaluate_retrieval` es conceptualmente idéntico a ROUGE — mide qué fracción de los términos de referencia aparecieron en el output.

### Limitaciones

- No capturan variaciones estilísticas — dos respuestas igualmente correctas pero con palabras distintas pueden tener scores muy diferentes
- Requieren ground truth — alguien tiene que escribir las etiquetas de referencia
- Correlación con evaluación humana es limitada para texto libre

---

## Sección 3 — LLM-as-Judge

### ¿Qué es?

Usar un LLM para calificar la calidad de la respuesta de otro LLM. El juez recibe el prompt original, la respuesta generada, y criterios de evaluación — y retorna un score con una justificación.

```
Prompt original → LLM agente → Respuesta
                                    ↓
                              LLM juez + criterios → Score + Rationale
```

### Ventajas sobre las alternativas

- **No necesita etiquetas de referencia** — el juez evalúa directamente
- **Interpretable** — retorna una justificación (rationale) además del score
- **Escalable** — tan rápido como hacer una llamada al LLM
- **Flexible** — puedes definir cualquier criterio de evaluación

### Las 3 dimensiones que evaluamos

En nuestro agente usamos un juez multidimensional que evalúa tres aspectos independientes de la trayectoria:

| Dimensión | Pregunta | Falla típica en nuestro agente |
|---|---|---|
| **Correctness** | ¿La respuesta final es factualmente correcta? | Rara — el LLM suele conocer la respuesta |
| **Efficiency** | ¿El agente usó las tools mínimas necesarias? | Llama `web_search` para preguntas de conocimiento general |
| **Faithfulness** | ¿La respuesta está fundamentada en los resultados de búsqueda? | Ignora los snippets y responde desde su conocimiento previo |

Estas tres dimensiones capturan aspectos que los evaluadores determinísticos no pueden medir:
- `tool_use` detecta si llamó la tool correcta, pero no si la respuesta la usó
- `retrieval_quality` detecta si los resultados llegaron, pero no si el LLM los incorporó
- `faithfulness` cierra ese gap — mide si el LLM realmente usó lo que buscó

#### Correctness

Mide si la respuesta final es factualmente correcta e completa, independientemente de cómo el agente llegó a ella.

- `1.0` → correcta y completa
- `0.5` → parcialmente correcta o incompleta
- `0.0` → incorrecta o ausente

En nuestro agente obtuvo `1.0` en todas las preguntas — el LLM conocía las respuestas incluso cuando no debería haber buscado. Esto ilustra por qué `correctness` solo no es suficiente: un agente puede ser correcto por las razones equivocadas.

#### Efficiency

Mide si el agente tomó el camino mínimo necesario — si usó exactamente las tools que necesitaba, sin llamadas innecesarias.

- `1.0` → solo llamó tools cuando era realmente necesario
- `0.5` → llamó algunas tools innecesarias
- `0.0` → patrón de uso de tools incorrecto

En nuestro agente obtuvo `0.40` — coincidiendo exactamente con `tool_use_score`. Esto valida ambos evaluadores: un determinístico y un LLM-as-judge llegaron a la misma conclusión de forma independiente.

#### Faithfulness

Mide si la respuesta final está fundamentada en los resultados obtenidos por las tools, o si el LLM los ignoró y respondió desde su conocimiento previo.

- `1.0` → la respuesta se basa completamente en los datos recuperados
- `0.5` → mezcla datos recuperados con suposiciones propias
- `0.0` → ignora los resultados de las tools o alucina

En nuestro agente obtuvo `0.70`. Las preguntas donde no se necesitaba búsqueda recibieron `0.5` — el LLM respondió correctamente pero sin fundamentarse en ninguna tool. Esto es aceptable aquí, pero en dominios donde los datos cambian frecuentemente sería un problema crítico: la respuesta puede ser correcta hoy e incorrecta mañana.

### Variaciones principales

**Pointwise** — evalúa una sola respuesta en una escala:
```
Evalúa la calidad de: [respuesta]
→ "Muy buena" / score: 0.9
```

**Pairwise** — compara dos respuestas y decide cuál es mejor:
```
¿Cuál es mejor: [respuesta A] o [respuesta B]?
→ "Respuesta A"
```

El modo pairwise se usa en **Chatbot Arena** y sistemas de RLHF para comparar modelos.

---

## Sección 4 — Implementación con Pydantic

Para garantizar que el juez retorne un formato consistente usamos salida estructurada con Pydantic.

### Paso 1: Definir el esquema

```python
from pydantic import BaseModel

class EvalScores(BaseModel):
    correctness: float   # ¿La respuesta es factualmente correcta?
    efficiency: float    # ¿Usó las tools mínimas necesarias?
    faithfulness: float  # ¿La respuesta viene de los resultados de búsqueda?
    comment: str         # Justificación del score
```

### Paso 2: Escribir el prompt del juez

El prompt define los criterios de evaluación. Siguiendo las best practices de la literatura:

```python
CUSTOM_EVAL_PROMPT = """You are an expert evaluator of AI agent trajectories.
Evaluate the following agent trajectory on THREE dimensions.

<Trajectory>
{outputs}
</Trajectory>

Score each dimension from 0.0 to 1.0:

1. **Correctness**: Is the final answer factually correct and complete?
   - 1.0: Correct and complete | 0.5: Partially correct | 0.0: Wrong or missing

2. **Efficiency**: Did the agent use the minimum tools necessary?
   - 1.0: Only called tools when truly needed
   - 0.5: Called some unnecessary tools
   - 0.0: Wrong tool usage pattern

3. **Faithfulness**: Is the final answer grounded in the tool results?
   - 1.0: Fully based on retrieved data
   - 0.5: Mixes retrieved data with assumptions
   - 0.0: Ignores tool results or hallucinates

Respond ONLY with JSON:
{{
  "correctness": <float 0.0-1.0>,
  "efficiency": <float 0.0-1.0>,
  "faithfulness": <float 0.0-1.0>,
  "comment": "<brief explanation>"
}}"""
```

### Paso 3: Invocar el juez con salida estructurada

```python
def multidim_judge(trajectory: list) -> dict:
    structured_judge = llm.with_structured_output(EvalScores)
    prompt = CUSTOM_EVAL_PROMPT.format(
        outputs=json.dumps(trajectory, indent=2, ensure_ascii=False)
    )
    result: EvalScores = structured_judge.invoke([HumanMessage(content=prompt)])
    return {
        "correctness": result.correctness,
        "efficiency": result.efficiency,
        "faithfulness": result.faithfulness,
        "comment": result.comment,
    }
```

`with_structured_output` garantiza que el LLM retorne siempre los tres scores como floats — sin necesidad de parsear texto libre ni manejar JSON mal formado.

---

## Sección 5 — Sesgos del LLM-as-Judge

La literatura identifica tres sesgos principales (Zheng et al., 2023):

### Position bias

El juez favorece la respuesta que aparece primero en una comparación pairwise.

```
[A, B] → elige A     vs     [B, A] → elige B   ← mismo par, resultado distinto
```

**Remedio:** promediar los scores con ambos órdenes, o usar evaluación pointwise.

### Verbosity bias

El juez favorece respuestas más largas, aunque sean menos útiles.

```
Respuesta A: corta y correcta         → score: 0.6
Respuesta B: larga, detallada, menos útil → score: 0.9  ← incorrecto
```

**Remedio:** criterios explícitos en el prompt sobre longitud, few-shot examples con respuestas cortas correctas.

### Self-enhancement bias

El juez favorece respuestas generadas por el mismo modelo.

```
Respuesta humana (perfecta)  → score: 0.7
Respuesta del propio LLM     → score: 0.9  ← incorrecto
```

**Remedio:** usar un modelo juez diferente al modelo evaluado.

---

## Sección 6 — Best practices

Basado en Zheng et al. (2023) y aplicado a nuestro agente:

| Práctica | Por qué | En nuestro agente |
|---|---|---|
| Criterios claros y específicos | Reduce ambigüedad | Definimos 3 dimensiones con escalas explícitas |
| Escribir rationale antes del score | Fuerza razonamiento (chain-of-thought) | El campo `comment` en `EvalScores` |
| Escala binaria o discreta | Más reproducible que escala continua libre | Usamos 0.0–1.0 con puntos ancla definidos |
| Mitigar sesgos | Scores más confiables | Evaluación pointwise, no pairwise |
| Temperatura baja | Reproducibilidad entre ejecuciones | `temperature=0` en el juez |
| Calibrar con humanos | Validar que el juez coincide con criterio humano | Pendiente — ejercicio del workshop |
| Juez ≠ modelo evaluado | Evitar self-enhancement bias | Podría usarse un modelo diferente como juez |

---

## Sección 7 — Los cuatro evaluadores en contexto

En `agent_evals.py` implementamos un stack de evaluación que cubre distintos niveles del agente:

```
Usuario → agent → tools → respuesta
   │          │       │         │
   │          │       │         └── correctness (LLM-as-judge)
   │          │       └──────────── retrieval_quality (determinístico)
   │          └──────────────────── tool_use + trajectory_match (determinísticos)
   └─────────────────────────────── (input del dataset)
```

| Evaluador | Tipo | Qué mide | Necesita referencia |
|---|---|---|---|
| `tool_use` | Determinístico | ¿Llamó las tools correctas? | Sí — `expected_tools` |
| `retrieval_quality` | Determinístico | ¿Los resultados de búsqueda son relevantes? | Sí — `expected_content` |
| `trajectory_match` | Determinístico | ¿El proceso fue el esperado? | Sí — trayectoria de referencia |
| `correctness` | LLM-as-judge | ¿La respuesta es correcta? | No |
| `efficiency` | LLM-as-judge | ¿Usó las tools mínimas? | No |
| `faithfulness` | LLM-as-judge | ¿La respuesta viene de los datos? | No |

### Resultado clave

Al correr los seis evaluadores sobre el Web Search Agent:

| Evaluador | Score | Interpretación |
|---|---|---|
| `tool_use` | 0.40 | Buscó en web para preguntas de conocimiento general |
| `retrieval_quality` | 0.84 | Cuando sí buscó, los resultados fueron relevantes |
| `trajectory_match` | 0.40 | El proceso no fue el esperado en 3/5 preguntas |
| `correctness` | 1.00 | Todas las respuestas finales fueron correctas |
| `efficiency` | 0.40 | Coincide exactamente con `tool_use` |
| `faithfulness` | 0.70 | En preguntas sin búsqueda, respuesta no fundamentada en tools |

> **Observación clave:** `tool_use` (determinístico) y `efficiency` (LLM-as-judge) llegaron al mismo score por caminos independientes. Cuando dos evaluadores de distinto tipo coinciden, la evidencia es más sólida.

> **Limitación detectada:** `correctness = 1.0` con `efficiency = 0.4` muestra que el agente compensa con su conocimiento previo cuando usa tools innecesariamente. Esto es frágil — funciona hoy, pero fallaría en dominios donde los datos cambian.

---

## Referencias

- Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*
- Banerjee et al. (2005). *METEOR: An Automatic Metric for MT Evaluation*
- Papineni et al. (2002). *BLEU: A Method for Automatic Evaluation of Machine Translation*
- Lin et al. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries*
- Wei et al. (2024). *Long-form factuality in large language models*
- Cohen (1960). *A coefficient of agreement for nominal scales*
- Fleiss (1971). *Measuring nominal scale agreement among many raters*
