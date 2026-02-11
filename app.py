import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# -----------------------------
# APP SETUP
# -----------------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

MONDAY_API_KEY = os.getenv("MONDAY_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEALS_BOARD_ID = int(os.getenv("DEALS_BOARD_ID"))
WORK_BOARD_ID = int(os.getenv("WORK_BOARD_ID"))

MONDAY_URL = "https://api.monday.com/v2"

# -----------------------------
# MODELS
# -----------------------------
class Query(BaseModel):
    question: str

# -----------------------------
# MONDAY API
# -----------------------------
def fetch_board_data(board_id):
    query = f"""
    {{
      boards(ids: {board_id}) {{
        items_page(limit: 500) {{
          items {{
            name
            column_values {{
              id
              text
            }}
          }}
        }}
      }}
    }}
    """
    headers = {"Authorization": MONDAY_API_KEY}
    response = requests.post(MONDAY_URL, json={"query": query}, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Monday API Error: {response.text}")

    return response.json()

def convert_board_to_dataframe(raw_data):
    if not raw_data.get("data") or not raw_data["data"]["boards"]:
        return pd.DataFrame()

    items = raw_data["data"]["boards"][0]["items_page"]["items"]
    rows = []

    for item in items:
        row = {"Item Name": item["name"]}
        for col in item["column_values"]:
            row[col["id"]] = col["text"]
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.replace("", np.nan)

# -----------------------------
# UTILITIES
# -----------------------------
def normalize_probability(prob):
    mapping = {"High": 0.8, "Medium": 0.5, "Low": 0.2}
    return mapping.get(str(prob).strip(), 0.3)

def safe_parse_date(date_str):
    try:
        return parser.parse(date_str)
    except:
        return None

# -----------------------------
# BUSINESS LOGIC
# -----------------------------
def compute_deal_metrics(df):
    if df.empty:
        return {"error": "No deal data available."}

    df["value"] = pd.to_numeric(df["numeric_mm0f6tc2"], errors="coerce")
    df["prob"] = df["color_mm0f30w0"].apply(normalize_probability)
    df["weighted"] = df["value"] * df["prob"]

    # Only open deals
    df = df[df["color_mm0fqvp6"] == "Open"]

    return {
        "total_pipeline": float(df["value"].sum(skipna=True)),
        "weighted_pipeline": float(df["weighted"].sum(skipna=True)),
        "sector_breakdown": df.groupby("color_mm0fe66m")["value"].sum().to_dict(),
        "stage_distribution": df["color_mm0f24qr"].value_counts().to_dict()
    }

def compute_work_order_metrics(df):
    if df.empty:
        return {"error": "No work order data available."}

    df["planned_date"] = df["date_mm0fpdes"].apply(safe_parse_date)
    df["actual_date"] = df["date_mm0fz6jw"].apply(safe_parse_date)
    df["delay_days"] = (df["actual_date"] - df["planned_date"]).dt.days

    return {
        "average_delay_days": float(df["delay_days"].mean()) if df["delay_days"].notna().any() else 0,
        "sector_delay": df.groupby("color_mm0f5e45")["delay_days"].mean().dropna().to_dict(),
        "execution_status_distribution": df["color_mm0fcx9e"].value_counts().to_dict()
    }

# -----------------------------
# OPENROUTER SUMMARY
# -----------------------------
def generate_summary(structured_data, question):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": f"""
You are a founder-level business intelligence assistant.

User Question:
{question}

Structured Data:
{structured_data}

Provide:
1. Executive Summary
2. Key Risks
3. Opportunities
3. Data Caveats

Be concise and strategic.
"""
            }
        ],
        "temperature": 0.3
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        return "Error generating summary from OpenRouter."

    return response.json()["choices"][0]["message"]["content"]

# -----------------------------
# ROUTES
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
def ask(query: Query):
    try:
        deals_raw = fetch_board_data(DEALS_BOARD_ID)
        work_raw = fetch_board_data(WORK_BOARD_ID)

        deals_df = convert_board_to_dataframe(deals_raw)
        work_df = convert_board_to_dataframe(work_raw)

        deal_metrics = compute_deal_metrics(deals_df)
        work_metrics = compute_work_order_metrics(work_df)

        structured_data = {
            "deal_metrics": deal_metrics,
            "work_order_metrics": work_metrics
        }

        answer = generate_summary(structured_data, query.question)

        return {"response": answer}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )