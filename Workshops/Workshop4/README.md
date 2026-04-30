# Tarea: Construcción de un Sistema RAG

## Descripción

Para esta tarea, deberás construir un sistema simple de **pregunta-respuesta** utilizando la técnica de **Retrieval-Augmented Generation (RAG)**.

Una vez termines el desarrollo, deberás compartir los resultados generados a 50 preguntas predefinidas.

---

## 1. Requisitos Previos

- Python 3.10 o superior
- Las librerías de los ejemplos en clase instaladas
- Una clave de API de **NVIDIA NIM** → https://build.nvidia.com (tier gratuito con 1,000 créditos; el API key empieza con `nvapi-`)

```bash
pip install openai langchain langchain-huggingface langchain-qdrant \
            qdrant-client sentence-transformers python-dotenv
```

---

## 2. Configuración

Crea el archivo `.env` en la raíz del proyecto:

```bash
NVIDIA_API_KEY=nvapi-...
```

---

## 3. Implementación del Sistema RAG

Implementarás el pipeline en **dos fases separadas** para usar una base de datos vectorial **persistente en disco**.

**Stack tecnológico** (el mismo del curso):

| Componente    | Tecnología                                  |
|---------------|---------------------------------------------|
| LLM           | NVIDIA NIM — `meta/llama-3.3-70b-instruct`  |
| Embeddings    | HuggingFace — `all-mpnet-base-v2` (local)   |
| Vector store  | Qdrant persistente en disco                 |
| Pipeline      | LangChain LCEL                              |

> **¿Por qué dos fases?**
> La ingesta (embed + indexar) es costosa y se ejecuta una sola vez.
> La consulta carga el índice ya construido y responde en tiempo real.

---

### 3.1. Fase A — Ingesta (`ingest.py`)

Carga la transcripción, la divide en chunks, genera embeddings y guarda el índice en disco. **Ejecutar una sola vez.**

```python
# ingest.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# 1. Cargar y dividir el documento
loader = TextLoader("docs/intro-to-llms-karpathy.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
docs = text_splitter.split_documents(documents)
print(f"✅ {len(docs)} chunks generados")

# 2. Embeddings locales (sin API key)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# 3. Crear colección Qdrant persistente en disco
#    all-mpnet-base-v2 produce vectores de dimensión 768
COLLECTION   = "karpathy_qdrant"
PERSIST_PATH = "./db/karpathy_qdrant"

qdrant_client = QdrantClient(path=PERSIST_PATH)
qdrant_client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION,
    embedding=embeddings,
)
vectorstore.add_documents(docs)
print(f"✅ Base de datos vectorial creada en {PERSIST_PATH}")
```

> **Nota:** `size=768` corresponde a la dimensión de `all-mpnet-base-v2`.
> Si usaras otro modelo de embeddings este valor cambiaría.

---

### 3.2. Fase B — Pipeline RAG (`rag_pipeline.py`)

Carga el índice persistente, construye el pipeline LCEL y responde preguntas.

```python
# rag_pipeline.py
import os
from dotenv import load_dotenv
from openai import OpenAI

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models.llms import LLM
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from qdrant_client import QdrantClient
from typing import List

load_dotenv()

# ── 1. LLM: NVIDIA NIM via cliente OpenAI-compatible ──────────
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)
MODEL = "meta/llama-3.3-70b-instruct"

class NvidiaLLM(LLM):
    """
    Wrapper mínimo sobre el cliente OpenAI apuntando a NVIDIA NIM.
    Hereda de LangChain LLM para ser compatible con LCEL ( | ).
    Mismo patrón que ChatGoogleGenerativeAI pero sin dependencias de Google.
    """
    model: str = MODEL

    @property
    def _llm_type(self) -> str:
        return "nvidia-nim"

    def _call(
        self,
        prompt: str,
        stop=None,
        run_manager: CallbackManagerForLLMRun = None,
        **kwargs,
    ) -> str:
        response = nvidia_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content

llm = NvidiaLLM()
print(f"✅ LLM: NVIDIA NIM / {MODEL}")

# ── 2. Embeddings — mismo modelo que en la ingesta ─────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# ── 3. Cargar Qdrant desde disco ───────────────────────────────
COLLECTION   = "karpathy_qdrant"
PERSIST_PATH = "./db/karpathy_qdrant"

qdrant_client = QdrantClient(path=PERSIST_PATH)
vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION,
    embedding=embeddings,
)
print("✅ Vector store cargado desde disco")

# ── 4. Retriever ───────────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.3},
)

# ── 5. Pipeline LCEL ───────────────────────────────────────────
def format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[Fragmento {i+1}]\n{d.page_content}"
        for i, d in enumerate(docs)
    )

prompt = ChatPromptTemplate.from_template("""
Eres un asistente experto. Responde la pregunta basándote ÚNICAMENTE
en el contexto proporcionado. Si la información no está en el contexto,
di explícitamente que no la tienes.

CONTEXTO:
{context}

PREGUNTA: {question}

RESPUESTA:
""")

rag_chain = (
    {
        "context":  retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ── 6. Prueba rápida ───────────────────────────────────────────
question = "¿Qué es retrieval augmented generation?"
answer   = rag_chain.invoke(question)
print(f"\nPregunta: {question}")
print(f"Respuesta: {answer.strip()}")
```

---

## 4. Generación de Respuestas a las 50 Preguntas

Escribe un script que itere sobre `docs/questions.json`, invoque el pipeline para cada pregunta y guarde los resultados.

```python
# generate_answers.py
import json
from rag_pipeline import rag_chain, retriever  # importar desde Fase B

with open("docs/questions.json", encoding="utf-8") as f:
    test_questions = json.load(f)

print(f"📋 Respondiendo {len(test_questions)} preguntas...\n")

results = []
for i, question in enumerate(test_questions, 1):
    print(f"  [{i}/{len(test_questions)}] {question[:70]}...")

    source_docs = retriever.invoke(question)
    answer      = rag_chain.invoke(question)

    results.append({
        "question": question,
        "answer":   answer,
        "contexts": [doc.page_content for doc in source_docs],
    })

with open("my_rag_output.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("\n✅ Resultados guardados en my_rag_output.json")
```

El archivo de salida tendrá este formato:

```json
[
  {
    "question": "¿Qué es RAG?",
    "answer": "RAG es...",
    "contexts": [
      "Fragmento del documento 1...",
      "Fragmento del documento 2..."
    ]
  }
]
```

---

## 5. Estructura de Archivos Esperada

```
proyecto/
├── .env                        ← NVIDIA_API_KEY=nvapi-...
├── ingest.py                   ← Fase A (ejecutar una vez)
├── rag_pipeline.py             ← Fase B (pipeline RAG)
├── generate_answers.py         ← Generación de las 50 respuestas
├── db/
│   └── karpathy_qdrant/        ← Índice Qdrant persistente (generado)
├── docs/
│   ├── intro-to-llms-karpathy.txt
│   └── questions.json
└── my_rag_output.json          ← Entregable (generado)
```

---
## Notas Finales

- No alteres las preguntas del archivo `questions.json`.
- Asegúrate de ejecutar la celda de ingesta **antes** de la celda de consulta.
- Entregables:
  - Un **notebook** (`.ipynb`) con las dos fases implementadas (ingesta y pipeline RAG) y la generación de las 50 respuestas. Los scripts `.py` también son válidos.
  - El archivo `my_rag_output.json` con las 50 preguntas, contextos y respuestas.

**Atención**: El conjunto de datos utilizado proviene de la transcripción del video ["[1hr Talk] Intro to Large Language Models"](https://www.youtube.com/watch?v=zjkBMFhNj_g) de Andrej Karpathy, bajo licencia Creative Commons Attribution (CC-BY).

