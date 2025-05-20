import getpass
import os
from dotenv import load_dotenv
from langsmith import utils
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chat_models import init_chat_model

load_dotenv(dotenv_path=".env", override=True)
print(utils.tracing_is_enabled())

model = ChatOllama(model="qwen3")
llm = init_chat_model("qwen3", model_provider="ollama")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = InMemoryVectorStore(embeddings)

# Run a simple prompt
response = llm.invoke("Are you there?")
print(response)