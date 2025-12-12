# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import psycopg2
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import List, Dict, Any

load_dotenv()

# Config
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
DB_URL = os.getenv("DB_URL")

if not DB_URL or not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("Set GOOGLE_API_KEY and DB_URL in .env")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

app = FastAPI(
    title="Merchant Price Forecast Agent API",
    description="Ask Gemini anything about price drops, best time to buy, etc.",
    version="1.0"
)

# In-memory conversation store (use Redis/DB in production)
sessions: Dict[str, List[Any]] = {}

def get_merchant(merchant_id: int) -> Dict[str, Any]:
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute(
        "SELECT merchant_name, product_name, price FROM merchants WHERE merchant_id = %s",
        (merchant_id,)
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Merchant not found")
    return {"name": row[0], "product": row[1], "price": float(row[2])}

def get_system_prompt(data: dict) -> str:
    return f"""You are an expert retail price analyst with 15 years of experience.
Analyzing one product only:

Merchant: {data['name']}
Product: {data['product']}
Current Price: ${data['price']:.2f}
Today is December 2025.

For any price-related question (drops, deals, "should I wait?", Black Friday, etc.):
- Think step-by-step
- Consider seasonality, brand strategy, product age, upcoming events
- Give realistic probability + expected discount
- Never invent fake sales

Answer naturally and conversationally. Never say you lack real-time data."""
    

class ChatRequest(BaseModel):
    merchant_id: int
    message: str
    session_id: str = "default"  # optional, for conversation memory

class ChatResponse(BaseModel):
    response: str
    merchant_info: str

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    merchant = get_merchant(req.merchant_id)
    info = f"{merchant['name']} — {merchant['product']} — ${merchant['price']:.2f}"

    # Initialize or load conversation
    session_key = f"{req.merchant_id}_{req.session_id}"
    if session_key not in sessions:
        sessions[session_key] = [SystemMessage(content=get_system_prompt(merchant))]
    
    messages = sessions[session_key]
    messages.append(HumanMessage(content=req.message))

    # Get response from Gemini
    resp = llm.invoke(messages)
    messages.append(AIMessage(content=resp.content))

    return ChatResponse(
        response=resp.content,
        merchant_info=info
    )

@app.get("/merchant/{merchant_id}")
async def merchant_info(merchant_id: int):
    data = get_merchant(merchant_id)
    return {
        "merchant_id": merchant_id,
        "display": f"{data['name']} — {data['product']} — ${data['price']:.2f}"
    }

# Optional: clear session
@app.delete("/session/{merchant_id}/{session_id}")
async def clear_session(merchant_id: int, session_id: str = "default"):
    key = f"{merchant_id}_{session_id}"
    sessions.pop(key, None)
    return {"detail": "Session cleared"}