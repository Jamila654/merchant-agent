import os
import psycopg2
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

DB_URL = os.getenv("DB_URL")

def get_merchant(merchant_id: int) -> dict:
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute("SELECT merchant_name, product_name, price FROM merchants WHERE merchant_id = %s", (merchant_id,))
        row = cur.fetchone()
        conn.close()
        if row:
            return {"name": row[0], "product": row[1], "price": float(row[2])}
    except Exception as e:
        print("DB error:", e)
    return None

def chat_with_merchant(merchant_id: int):
    data = get_merchant(merchant_id)
    if not data:
        print("Merchant not found.")
        return

    info = f"{data['name']} — {data['product']} — ${data['price']:.2f}"
    print(f"\nLoaded → {info}\n")
    print("Ask anything — the agent will think and forecast dynamically!\nType 'new' or 'quit'\n")

    # This single prompt makes Gemini actually reason
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

Answer naturally and conversationally. Never say "I don't have real-time data" — just reason like a pro.

For non-forecast questions, answer normally and concisely."""

    messages = [SystemMessage(content=system_prompt)]

    while True:
        try:
            q = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            return

        if q.lower() in {"quit", "q", "exit"}:
            print("Bye!")
            return
        if q.lower() == "new":
            return
        if not q:
            continue

        messages.append(HumanMessage(content=q))

        response = llm.invoke(messages)
        print(f"Agent: {response.content}\n")

        messages.append(response)  # memory

def main():
    print("Merchant Agent – Dynamic Forecasting (Gemini Thinks)\n")
    while True:
        try:
            inp = input("Enter merchant ID (1–200) → ").strip()
            if inp.lower() in {"quit", "q", "exit"}:
                break
            mid = int(inp)
            chat_with_merchant(mid)
            print("-" * 70 + "\n")
        except ValueError:
            print("Enter a number")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()