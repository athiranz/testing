import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

app = FastAPI(title="NAVASARA AI Kiosk Backend")

# Inisialisasi Vector DB
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
vector_db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_model
)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# Inisialisasi LLM
llm = ChatOpenAI(
    model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

SYSTEM_PROMPT = """You are NAVASARA, an interactive AI kiosk that bridges cultural heritage (Javanese folklore of Nyi Roro Kidul) and scientific reality (coastal oceanography and geomorphology).

Your Core Mission:
- "Preserve": Respectfully present folklore and local traditions.
- "Experience": Engage visitors naturally in a friendly kiosk persona.
- "Interpretation": Explain the scientific phenomena behind the myths (e.g., Rip currents, wave dynamics, ocean optics, megathrust subduction).

CRITICAL LANGUAGE INSTRUCTION:
- Always respond in clear, fluent, and engaging ENGLISH.
- Context data may be provided in Indonesian, but your final answer MUST be in English.

Context Information:
{context}

User Question: {question}
NAVASARA Response:"""

prompt_template = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

class QueryRequest(BaseModel):
    question: str

class TriggerRequest(BaseModel):
    event: str
    person_id: int

@app.get("/")
def root():
    return {"status": "active", "system": "NAVASARA AI Backend"}

@app.post("/trigger_greeting")
def trigger_greeting(payload: TriggerRequest):
    """Endpoint Menerima Sinyal Melambai dari Camera Tracker (Port 8000)"""
    print(f"👋 [TRIGGER] Person {payload.person_id} melambai!")
    greeting_text = (
        "Welcome to NAVASARA! I am your AI guide bridging Javanese cultural heritage "
        "and marine science. Please ask me anything about the legends of the Southern Coast "
        "or the oceanographic science behind them!"
    )
    return {
        "status": "success",
        "event": payload.event,
        "person_id": payload.person_id,
        "greeting": greeting_text
    }

@app.post("/chat")
def chat_endpoint(payload: QueryRequest):
    """Endpoint RAG Chatbot untuk Streamlit UI Kiosk"""
    try:
        docs = retriever.invoke(payload.question)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        chain = prompt_template | llm
        response = chain.invoke({
            "context": context_text,
            "question": payload.question
        })
        
        sources = [doc.metadata.get("source_file", doc.metadata.get("sources", "JSON Data")) for doc in docs]
        
        return {
            "answer": response.content,
            "sources": list(set(sources))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)