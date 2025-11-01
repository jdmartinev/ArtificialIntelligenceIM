
# 📚 Tarea: Construcción de un Sistema RAG

## Descripción

Para esta tarea, deberás construir un sistema simple de **pregunta-respuesta** utilizando la técnica de **Retrieval-Augmented Generation (RAG)**.

Una vez termines el desarrollo, deberás compartir los resultados generados a 50 preguntas predefinidas.

Por favor sigue atentamente las instrucciones de configuración y desarrollo que encontrarás a continuación.

---

## 1. Requisitos Previos

Para completar esta tarea necesitarás:

- Python 3.10 o superior con las librerías de los ejemplos en clase instaladas.
- Una clave de API de **Google (Gemini)**.

---

## 2. Configuración

- Crea el archivo `api_keys.env` (o `.env`).
- Configura tu clave `GOOGLE_API_KEY` en el archivo `.env`:
  ```bash
  GOOGLE_API_KEY=...

---

## 3. Implementación de un Sistema de Pregunta-Respuesta usando RAG

En esta tarea implementarás un sistema RAG que responda preguntas basadas en la transcripción de la famosa charla de Andrey Karpathy: ["[1hr Talk] Intro to Large Language Models"](https://www.youtube.com/watch?v=zjkBMFhNj_g).

La transcripción se encuentra en [intro-to-llms-karpathy.txt](docs/intro-to-llms-karpathy.txt).

Para esta tarea, usaremos el stack tecnológico que practicamos en nuestros ejemplos:
- **LLM**: Google Gemini (`ChatGoogleGenerativeAI`)
- **Embeddings**: Hugging Face (`all-mpnet-base-v2`)
- **Vector Store**: `Chroma` Persistente

Es **requisito** implementar el pipeline en dos fases (separando la ingesta de la consulta) para usar una base de datos vectorial **persistente**.

---

### 3.1. Fase A: Ingesta y Creación del Vector Store

Crea un script o notebook para cargar, procesar y guardar los documentos en disco. **Esto solo se ejecuta una vez.**

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# 1. Cargar y Dividir el Documento
loader = TextLoader("docs/intro-to-llms-karpathy.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# 2. Inicializar Embeddings (los de nuestro ejemplo)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# 3. Crear y Guardar el Vector Store
persist_directory = "db/karpathy_chroma"
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory
)

vector_store.persist()
print("✅ Base de datos vectorial creada.")
```
### 3.2. Fase B: Pipeline RAG y Consultas

Crea un segundo script que **cargue** la base de datos persistente y construya el pipeline para responder preguntas.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# 1. Cargar Variables de Entorno
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")

# 2. Cargar LLM (como en nuestro ejemplo)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# 3. Cargar Embeddings (el mismo modelo)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# 4. Cargar el Vector Store desde el disco
persist_directory = "db/karpathy_chroma"
vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

# 5. Crear el Pipeline RAG
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vector_store.as_retris_ever(),
    return_source_documents=True,
)

# 6. Probar
question = "¿Qué es retrieval augmented generation?"
result = qa_chain.invoke({"query": question})
print(result["result"])
**Recursos recomendados**:

- Python:
  - [Tutorial de RAG con LangChain](https://python.langchain.com/v0.2/docs/tutorials/rag/)
  - [Q&A RAG con LLamaIndex](https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/q_and_a/#semantic-search)
---

## 4. Generación de Respuestas a la Lista de Preguntas

Se ha preparado un conjunto de **50 preguntas** que tu pipeline RAG deberá responder.

Puedes encontrar estas preguntas en el archivo [questions.json](docs/questions.json).

Debes escribir un script que:

- Recupere los contextos relevantes para cada pregunta.
- Genere una respuesta usando tu pipeline RAG.
- Guarde las preguntas, los contextos recuperados y las respuestas en un archivo JSON con el siguiente formato:

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

Un ejemplo simple en Python podría ser:

```python
import json

json_results = []
for question in test_questions:
    response = qa_chain.invoke({"query": question})
    
    json_results.append({
        "question": question,
        "answer": response["result"],
        "contexts": [context.page_content for context in response["source_documents"]]
    })

with open('./my_rag_output.json', mode='w', encoding='utf-8') as f:
    f.write(json.dumps(json_results, indent=4))
```

---

## Notas Finales

- No debes alterar las preguntas.
- Asegúrate de enviar:
  - El código de tu pipeline RAG.
  - El archivo JSON con las preguntas, contextos y respuestas.



  

**Atención**: El conjunto de datos utilizado proviene de la transcripción del video ["[1hr Talk] Intro to Large Language Models"](https://www.youtube.com/watch?v=zjkBMFhNj_g) de Andrej Karpathy, bajo licencia Creative Commons Attribution (CC-BY).

---
