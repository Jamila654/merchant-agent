import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import psycopg2
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Merchant Chat")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    merchant_id: int
    messages: List[Message]

def get_merchant(merchant_id: int):
    try:
        conn = psycopg2.connect(os.getenv("DB_URL"))
        cur = conn.cursor()
        cur.execute("SELECT merchant_name, product_name, price FROM merchants WHERE merchant_id = %s", (merchant_id,))
        row = cur.fetchone()
        conn.close()
        if row:
            return {"name": row[0], "product": row[1], "price": float(row[2])}
    except Exception as e:
        print("DB error:", e)
    return None

@app.get("/")
def home():
    return {"message": "Welcome! POST to /chat with merchant_id and messages. Example in README."}

@app.post("/chat")
def chat(request: ChatRequest):
    data = get_merchant(request.merchant_id)
    if not data:
        raise HTTPException(status_code=404, detail="Merchant not found (ID 1-200)")

    system_prompt = f"""You are an expert retail price analyst with 15 years of experience.
You are analyzing exactly one product:

Merchant: {data['name']}
Product: {data['product']}
Current Price: ${data['price']:.2f}
Today is December 2025.

When the user asks anything about price movement, deals, timing, "should I wait", "will it drop", Black Friday, etc. — you must:
1. Think step-by-step (show your reasoning)
2. Consider: brand pricing strategy, product age, category patterns, seasonality, upcoming events
3. Give a realistic forecast with probability and expected discount
4. Never make up fake sales — base it on real-world patterns

Answer naturally and conversationally."""

    messages = [SystemMessage(content=system_prompt)]
    
    for msg in request.messages:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(SystemMessage(content=msg.content))

    try:
        response = llm.invoke(messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")
    
    return {
        "merchant": f"{data['name']} — {data['product']} — ${data['price']:.2f}",
        "reply": response.content
    }