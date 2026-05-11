# Tendencias en IA — 2025/2026

---

## Tendencia 1: RLVR — Reinforcement Learning with Verifiable Rewards

**¿Qué es?**
Evolución del paradigma RLHF que reemplaza el reward model (entrenado sobre preferencias humanas) por recompensas verificables automáticamente: corrección de código, soluciones matemáticas, ejecución de pruebas unitarias. El detonante fue DeepSeek-R1 (enero 2025), que popularizó GRPO como alternativa más eficiente a PPO. La clave: si la respuesta es correcta o incorrecta se puede verificar *sin humanos*, lo que permite escalar el entrenamiento masivamente.

**Preguntas para análisis**
- ¿En qué dominios aplica RLVR y en cuáles no? ¿Qué pasa cuando no hay un verificador confiable?
- ¿Qué ventajas tiene GRPO sobre PPO? ¿Qué pierde?
- ¿Cómo se relaciona RLVR con el fenómeno de "reasoning emergente" en modelos como DeepSeek-R1 o o1?

**Referencias**
- 📄 DeepSeek-R1 (paper fundacional): https://arxiv.org/abs/2501.12948
- 📄 GRPO original (DeepSeekMath): https://arxiv.org/abs/2402.03300
- 📚 Curated list actualizado — *awesome-RLVR*: https://github.com/opendilab/awesome-RLVR
- 📄 RISE — self-verification en el loop RL: https://arxiv.org/abs/2505.13445
- 📄 MR-RLVR — recompensas de proceso para razonamiento intermedio: https://arxiv.org/abs/2511.17473
- 📄 RLVR incentiva razonamiento correcto implícitamente: https://arxiv.org/abs/2506.14245
- 🎥 Explicación técnica (Yannic Kilcher): https://www.youtube.com/watch?v=rfS2T2bODMk

---

## Tendencia 2: Interpretabilidad Mecanicista

**¿Qué es?**
Área de investigación que busca entender *cómo* funcionan internamente los modelos de lenguaje, no solo evaluarlos desde fuera. Usando sparse autoencoders (SAE), se identifican "features" interpretables, circuitos neuronales y trayectorias causales desde el prompt hasta la respuesta. En 2025, Anthropic trazó secuencias completas de features en Claude (attribution graphs); DeepMind publicó Gemma Scope 2, el toolkit open-source más grande del campo. MIT Technology Review la nombró una de las "10 Breakthrough Technologies 2026".

**Preguntas para análisis**
- ¿Qué es un sparse autoencoder y por qué se usa para interpretabilidad?
- ¿Qué diferencia hay entre "circuit analysis" y "probing"? ¿Cuáles son las limitaciones de cada enfoque?
- ¿Puede la interpretabilidad mecanicista detectar comportamientos peligrosos antes del deployment? ¿Qué dice la evidencia actual?

**Referencias**
- 📄 Survey de referencia: *Mechanistic Interpretability for AI Safety — A Review*: https://arxiv.org/abs/2404.14082
- 📄 Open problems in mechanistic interpretability (roadmap del campo, enero 2025): https://arxiv.org/abs/2501.02035
- 🌐 Blog Transformer Circuits (Anthropic): https://transformer-circuits.pub
- 🌐 Gemma Scope 2 (Google DeepMind, open-source): https://deepmind.google/discover/blog/gemma-scope/
- 📄 Anthropic attribution graphs (marzo 2025): https://www.anthropic.com/research/mapping-mind-language-model
- 🌐 Status report 2026 — open problems: https://gist.github.com/bigsnarfdude/629f19f635981999c51a8bd44c6e2a54
- 🌐 Neuronpedia (explorador interactivo de features): https://www.neuronpedia.org

---

## Tendencia 3: Sistemas Multi-Agente y Protocolos de Estandarización

**¿Qué es?**
Los agentes de IA pasaron de experimentos aislados a infraestructura de producción. El cambio central: el Model Context Protocol (MCP), creado por Anthropic en noviembre 2024 y donado a la Linux Foundation, se convirtió en el estándar de facto para conectar agentes a herramientas externas (bases de datos, APIs, editores). En paralelo, Google lanzó A2A (Agent-to-Agent Protocol) para comunicación entre agentes de distintos vendors. Para principios de 2026, MCP registraba 97 millones de descargas mensuales. El concepto central que emergió: *context engineering* como la nueva disciplina crítica — los fallos de los agentes son principalmente fallos de contexto, no del modelo.

**Preguntas para análisis**
- ¿Qué problema técnico resuelve MCP? ¿Por qué se compara con TCP/IP?
- ¿Qué desafíos introduce la orquestación multi-agente que no existen en sistemas de un solo agente?
- Según el 2025 AI Agent Index, ¿qué nivel de transparencia tienen los developers de agentes sobre safety y evaluaciones?

**Referencias**
- 🌐 Especificación MCP (documentación oficial): https://modelcontextprotocol.io/introduction
- 📄 2025 AI Agent Index (MIT/Harvard/Stanford — 30 agentes analizados): https://arxiv.org/abs/2602.17753 | web: https://aiagentindex.mit.edu
- 📄 Survey de uso real — *How are AI agents used? Evidence from 177,000 MCP tools*: https://arxiv.org/abs/2603.23802
- 🌐 7 Agentic AI Trends 2026 (Machine Learning Mastery): https://machinelearningmastery.com/7-agentic-ai-trends-to-watch-in-2026/
- 🌐 Agentic AI Statistics 2026 (150+ data points): https://www.digitalapplied.com/blog/agentic-ai-statistics-2026-definitive-collection-150-data-points
- 📄 Google A2A Protocol: https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/
- 🌐 LangGraph (framework multi-agente open-source): https://langchain-ai.github.io/langgraph/

---

## Tendencia 4: Modelos Multimodales Omni

**¿Qué es?**
La arquitectura de los modelos está migrando de pipelines heterogéneos (visión y texto procesados por separado y luego "bridgeados") hacia modelos unificados que alinean texto, imagen, audio y video en un único espacio latente. Los modelos "omni" (GPT-4o, Gemini 2.0 Flash, LLaMA 4) procesan múltiples modalidades de forma nativa. En ICLR 2026 dominaron los trabajos sobre *multimodal alignment* y SAM 3. Un survey de 26,104 papers de CVPR/ICLR/NeurIPS 2023–2025 documenta la transición masiva del campo de CV clásico hacia entrenamiento multimodal y razonamiento general.

**Preguntas para análisis**
- ¿Cuál es la diferencia arquitectónica entre un VLM heterogéneo (ej: LLaVA) y un modelo omni (ej: GPT-4o)? ¿Qué trade-offs implica?
- ¿Qué es el "bridge bottleneck" en arquitecturas heterogéneas y cómo lo atacan los modelos unificados?
- ¿Qué benchmarks existen para evaluar razonamiento multimodal y cuáles son sus limitaciones?

**Referencias**
- 📄 Survey de VLMs — 26K papers (CVPR/ICLR/NeurIPS 2023–2025): https://arxiv.org/abs/2510.09586
- 📄 Survey state-of-the-art VLMs: https://arxiv.org/abs/2501.02189
- 📄 Unified modeling perspective (arquitecturas heterogéneas vs. unificadas): https://www.sciltp.com/journals/dmml/articles/2510001963
- 🌐 ICLR 2026 trends — multimodal & agentic: https://encord.com/iclr-2026/
- 🌐 Benchmark MMMU (razonamiento multimodal universitario): https://mmmu-benchmark.github.io
- 🌐 LLaVA (referencia open-source VLM heterogéneo): https://llava-vl.github.io
- 🌐 InternVL (modelo omni open-source competitivo): https://internvl.github.io

---

## Tendencia 5: IA para Matemáticas — Razonamiento Formal y Theorem Proving

**¿Qué es?**
Una de las fronteras más activas en 2025: sistemas de IA capaces de producir pruebas matemáticas formales verificables por computador (en Lean 4, Isabelle). El hito fue AlphaProof (Google DeepMind), que en IMO 2024 obtuvo nivel de medalla de plata — la primera vez que un sistema de IA alcanza ese nivel en una competencia olímpica de matemáticas. AlphaProof combina un LLM preentrenado con AlphaZero RL usando Lean como entorno verificable. En 2025, Aristotle (Harmonic) y Seed-Prover (ByteDance) superaron el nivel oro en IMO 2025. En paralelo, los modelos de razonamiento general (o1, DeepSeek-R1, Gemini) ahora superan el 85–95% en AIME 2025, benchmark que antes se consideraba fuera del alcance de la IA.

**Preguntas para análisis**
- ¿Qué ventaja tienen las pruebas formales (en Lean) sobre las pruebas en lenguaje natural generadas por LLMs?
- ¿Qué papel juega RLVR (Tendencia 1) en el éxito de AlphaProof y Aristotle?
- ¿Qué significa para la investigación matemática que un sistema pueda generar y verificar pruebas automáticamente? ¿Qué limitaciones persisten?

**Referencias**
- 📄 AlphaProof — *Olympiad-level formal mathematical reasoning with RL* (Nature 2025): https://www.nature.com/articles/s41586-025-09833-y
- 🌐 Blog técnico AlphaProof (DeepMind): https://deepmind.google/blog/ai-solves-imo-problems-at-silver-medal-level/
- 📄 Aristotle — gold medal en IMO 2025: https://arxiv.org/abs/2510.01346
- 📄 Seed-Prover (ByteDance — 5/6 problemas IMO 2025): https://arxiv.org/abs/2507.23726
- 📄 Proof or Bluff? — evaluando LLMs en pruebas reales (ETH Zurich): https://arxiv.org/abs/2503.21934
- 🌐 MathArena — leaderboard de razonamiento matemático: https://matharena.ai
- 🌐 AIME 2025 Benchmark leaderboard: https://artificialanalysis.ai/evaluations/aime-2025
- 🌐 Lean 4 (lenguaje de verificación formal): https://lean-lang.org

---

*Última actualización: mayo 2026*
