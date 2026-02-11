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
def generate_summary(structured_data, question, context=None, format_preference=None):
    """
    Generate dynamic, context-aware business intelligence summaries.
    
    Args:
        structured_data: The data to analyze
        question: User's question
        context: Optional dict with user_role, industry, urgency_level, previous_context
        format_preference: 'executive', 'detailed', 'action-oriented', or 'technical'
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Build dynamic system prompt based on context
    system_prompt = _build_system_prompt(context)
    
    # Build dynamic user prompt with adaptive sections
    user_prompt = _build_user_prompt(
        structured_data, 
        question, 
        context, 
        format_preference
    )
    
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1500  # Add token limit for cost control
    }
    
    # Add streaming for better UX (optional)
    if context and context.get('stream', False):
        payload['stream'] = True
        return _stream_response(url, headers, payload)
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        return {
            "error": True,
            "message": f"Error generating summary: {response.status_code}",
            "raw_response": response.text
        }
    
    return _format_response(response.json(), format_preference)


def _build_system_prompt(context):
    """Build adaptive system prompt based on user context."""
    base_prompt = "You are a founder-level business intelligence assistant."
    
    if not context:
        return base_prompt
    
    role_context = {
        'ceo': 'Focus on strategic implications and board-level insights.',
        'cfo': 'Emphasize financial metrics, ROI, and risk management.',
        'product': 'Highlight user impact, feature opportunities, and product-market fit.',
        'sales': 'Focus on revenue opportunities, customer insights, and pipeline health.',
        'ops': 'Emphasize efficiency, scalability, and operational bottlenecks.'
    }
    
    urgency_context = {
        'critical': 'This is time-sensitive. Prioritize immediate actions and risks.',
        'high': 'Provide actionable insights with clear next steps.',
        'normal': 'Balance strategic thinking with practical recommendations.'
    }
    
    role = context.get('user_role', '').lower()
    urgency = context.get('urgency_level', 'normal').lower()
    industry = context.get('industry', '')
    
    enhanced_prompt = base_prompt
    
    if role in role_context:
        enhanced_prompt += f"\n{role_context[role]}"
    
    if urgency in urgency_context:
        enhanced_prompt += f"\n{urgency_context[urgency]}"
    
    if industry:
        enhanced_prompt += f"\nIndustry context: {industry}. Use relevant benchmarks and terminology."
    
    return enhanced_prompt


def _build_user_prompt(structured_data, question, context, format_preference):
    """Build dynamic user prompt with adaptive sections."""
    
    # Determine which sections to include
    sections = _determine_sections(question, context, format_preference)
    
    prompt_parts = [
        f"**User Question:**\n{question}\n",
        f"**Data:**\n{structured_data}\n"
    ]
    
    # Add context if available
    if context and context.get('previous_context'):
        prompt_parts.append(
            f"**Previous Context:**\n{context['previous_context']}\n"
        )
    
    # Build dynamic analysis request
    prompt_parts.append("**Provide:**")
    
    section_templates = {
        'executive_summary': "1. Executive Summary (2-3 sentences)",
        'key_metrics': "2. Key Metrics & Trends",
        'risks': "3. Critical Risks & Mitigation Strategies",
        'opportunities': "4. Opportunities & Quick Wins",
        'recommendations': "5. Prioritized Recommendations",
        'comparisons': "6. Benchmark Comparisons",
        'data_quality': "7. Data Quality & Caveats",
        'next_steps': "8. Immediate Next Steps"
    }
    
    for i, section in enumerate(sections, 1):
        if section in section_templates:
            prompt_parts.append(f"{i}. {section_templates[section].split('. ', 1)[1]}")
    
    # Add format instructions
    format_instructions = {
        'executive': "\n\nFormat: Brief bullet points, no more than 200 words total.",
        'detailed': "\n\nFormat: Comprehensive analysis with supporting data.",
        'action-oriented': "\n\nFormat: Focus on concrete actions with owners and timelines.",
        'technical': "\n\nFormat: Include methodologies, data sources, and confidence levels."
    }
    
    if format_preference in format_instructions:
        prompt_parts.append(format_instructions[format_preference])
    else:
        prompt_parts.append("\n\nBe concise, strategic, and data-driven.")
    
    return "\n".join(prompt_parts)


def _determine_sections(question, context, format_preference):
    """Intelligently determine which sections to include."""
    
    # Default sections
    sections = ['executive_summary', 'risks', 'opportunities', 'data_quality']
    
    # Add sections based on question keywords
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['metric', 'kpi', 'performance', 'growth']):
        sections.insert(1, 'key_metrics')
    
    if any(word in question_lower for word in ['action', 'next', 'should', 'recommend']):
        sections.append('recommendations')
        sections.append('next_steps')
    
    if any(word in question_lower for word in ['compare', 'benchmark', 'vs', 'versus']):
        sections.insert(2, 'comparisons')
    
    # Adjust based on format preference
    if format_preference == 'executive':
        sections = ['executive_summary', 'recommendations']
    elif format_preference == 'action-oriented':
        sections = ['executive_summary', 'recommendations', 'next_steps']
    
    # Adjust based on urgency
    if context and context.get('urgency_level') == 'critical':
        sections = ['executive_summary', 'risks', 'next_steps']
    
    return sections


def _format_response(api_response, format_preference):
    """Format the API response into a structured output."""
    content = api_response["choices"][0]["message"]["content"]
    
    return {
        "success": True,
        "summary": content,
        "metadata": {
            "model": api_response.get("model"),
            "tokens_used": api_response.get("usage", {}),
            "format": format_preference
        }
    }


def _stream_response(url, headers, payload):
    """Stream response for better UX (optional implementation)."""
    response = requests.post(url, headers=headers, json=payload, stream=True)
    
    for line in response.iter_lines():
        if line:
            # Process streaming chunks
            yield line.decode('utf-8')


# Usage examples:
"""
# Basic usage
summary = generate_summary(data, "What are our top risks?")

# With context
summary = generate_summary(
    data, 
    "Should we expand to EMEA?",
    context={
        'user_role': 'ceo',
        'industry': 'SaaS',
        'urgency_level': 'high',
        'previous_context': 'We discussed ARR growth last week'
    },
    format_preference='action-oriented'
)

# Executive brief
summary = generate_summary(
    data,
    "Board meeting prep",
    context={'user_role': 'ceo'},
    format_preference='executive'
)
"""

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
