import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

import chromadb
import os
from dotenv import load_dotenv


load_dotenv('api_keys.env')


# Configurar la API Key (puedes establecerla como variable de entorno)
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Inicializar la base de datos ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Crear una colección en ChromaDB
collection = chroma_client.get_or_create_collection(name="chatbot_knowledge")


# Inicializar embeddings con OpenAI
embeddings = OpenAIEmbeddings()

# Crear el VectorStore con ChromaDB
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Crear el retriever para buscar documentos relevantes
retriever = vectorstore.as_retriever()

#os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
# bash: export OPENAI_API_KEY="your-openai-api-key"

# Inicializar el modelo GPT-4
llm = ChatOpenAI(model_name="gpt-4")

# Crear memoria de la conversación
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Integrar memoria en la cadena conversacional
chat_chain = ConversationalRetrievalChain.from_llm(llm, memory=memory, retriever=retriever)

# Interfaz en Streamlit
st.title("🤖 Chatbot con GPT-4 y LangChain")

# Mostrar el historial de la conversación
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada del usuario
user_input = st.text_input("Escribe tu mensaje:")

if user_input:
    # Obtener respuesta del chatbot
    response = chat_chain({"question": user_input})["answer"]
    
    # Guardar la conversación
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.session_state["messages"].append({"role": "assistant", "content": response})
    
    # Mostrar respuesta
    with st.chat_message("assistant"):
        st.markdown(response)
