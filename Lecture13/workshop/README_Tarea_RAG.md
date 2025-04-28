
# 📚 Tarea: Construcción de un Sistema RAG

## Descripción

Para esta tarea, deberás construir un sistema simple de **pregunta-respuesta** utilizando la técnica de **Retrieval-Augmented Generation (RAG)**.

Una vez termines el desarrollo, deberás compartir los resultados generados a 50 preguntas predefinidas.

Por favor sigue atentamente las instrucciones de configuración y desarrollo que encontrarás a continuación.

---

## 1. Requisitos Previos

Para completar esta tarea necesitarás:

- Python 3.10 o superior con las librerías de los ejemplos en clase instaladas
- Una clave de API de Groq

---

## 2. Configuración

- Crea el archivo `api_keys.env.
- Configura tu clave `GROQ_API_KEY` en el archivo `.env`:
  ```bash
  GROQ_API_KEY=...
  ```

---

## 3. Implementación de un Sistema de Pregunta-Respuesta usando RAG

En esta tarea implementarás un sistema RAG que responda preguntas basadas en la transcripción de la famosa charla de Andrey Karpathy: ["[1hr Talk] Intro to Large Language Models"](https://www.youtube.com/watch?v=zjkBMFhNj_g).

La transcripción se encuentra en [intro-to-llms-karpathy.txt](docs/intro-to-llms-karpathy.txt).

Tu pipeline RAG debe incluir los siguientes pasos:

- **Ingesta de datos**:
  - Divide el documento en fragmentos pequeños.
  - Genera embeddings para cada fragmento.
  - Almacena los embeddings en una base de datos vectorial.
- **RAG**:
  - Genera el embedding de una pregunta.
  - Recupera los fragmentos relevantes de la base de datos.
  - Envía la pregunta y los fragmentos al LLM para generar una respuesta.

Un ejemplo básico usando LangChain en Python podría verse así:

```python
hfEmbeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
loader = TextLoader("intro-to-llms-karpathy.txt")
index = VectorstoreIndexCreator(embedding=hfEmbeddings).from_loaders([loader])
llm = init_chat_model("llama3-8b-8192", model_provider="groq")
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=index.vectorstore.as_retriever(),
    return_source_documents=True,
)

question = "¿Qué es retrieval augmented generation y cómo mejora las capacidades de los modelos de lenguaje?"
result = qa_chain({"query": question})
```

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



  

**Atención**: El conjunto de datos utilizado proviene de la transcripción del video "[1hr Talk] Intro to Large Language Models" de Andrej Karpathy, bajo licencia Creative Commons Attribution (CC-BY).
---
