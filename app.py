"""
SpendWise AI — Personal Expense Tracker
Stack: Streamlit · LangChain · OpenAI · Supabase · Plotly
"""

import json
import re
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ── LangChain ──────────────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# ── Supabase ───────────────────────────────────────────────────────────────
from supabase import create_client, Client

# ═══════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SpendWise AI",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* gradient header */
  .spend-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    color: white;
    margin-bottom: 1.5rem;
  }
  .spend-header h1 { margin: 0; font-size: 2rem; }
  .spend-header p  { margin: 0.25rem 0 0; opacity: .85; font-size: 0.95rem; }

  /* metric cards */
  [data-testid="metric-container"] {
    background: #f8f9ff;
    border: 1px solid #e2e5ff;
    border-radius: 10px;
    padding: 1rem;
  }

  /* chat bubble */
  .chat-tip {
    background: #eef2ff;
    border-left: 4px solid #667eea;
    padding: 0.75rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.88rem;
    color: #4338ca;
    margin-bottom: 1rem;
  }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 2. SUPABASE CONNECTION
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_supabase() -> Client:
    url  = st.secrets["supabase"]["url"]
    key  = st.secrets["supabase"]["key"]
    return create_client(url, key)

try:
    supabase: Client = get_supabase()
except Exception as e:
    st.error(f"❌ Supabase connection failed: {e}")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# 3. LangChain AI PARSER
# ═══════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are an expense parsing assistant for SpendWise AI.

Given a natural-language expense description, return ONLY a valid JSON object (no markdown, no explanation) with:
  - "item"        : short descriptive name of the expense (string)
  - "amount"      : numeric cost as a float (infer currency symbol but output only the number)
  - "category"    : one of exactly: Housing | Food | Transport | Health | Personal | Entertainment | Financial
  - "subcategory" : the most appropriate subcategory from the lists below

Category → allowed subcategories:
  Housing      : Rent, Utilities, Maintenance, Insurance
  Food         : Groceries, Restaurants, Delivery, Coffee/Snacks
  Transport    : Fuel, Public Transit, Uber/Taxi, Repairs
  Health       : Doctor, Pharmacy, Gym, Insurance
  Personal     : Shopping, Clothing, Grooming, Laundry
  Entertainment: Streaming, Gaming, Hobbies, AI Tools
  Financial    : Debt, Savings, Investments

Categorization rules — follow these strictly:
  - Fruits, vegetables, produce, meat, dairy, eggs, bread, any raw/fresh food item → Food > Groceries
  - Supermarket or kirana store visits → Food > Groceries
  - Any food or drink consumed at a café or restaurant → Food > Restaurants
  - Any food ordered online (Swiggy, Zomato, etc.) → Food > Delivery
  - Tea, coffee, snacks, street food → Food > Coffee/Snacks
  - Never put food, fruits, or groceries under Personal.

If the amount cannot be determined, set "amount" to 0.
Example input : "Bought mangoes and bananas for ₹120"
Example output: {{"item":"Fruits","amount":120,"category":"Food","subcategory":"Groceries"}}
"""

@st.cache_resource(show_spinner=False)
def get_llm():
    api_key = st.secrets["openai"]["api_key"]
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key,
    )

def parse_expense(description: str) -> dict:
    """Use LangChain + OpenAI to parse a natural-language expense."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{description}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"description": description})
    raw = response.content.strip()
    # Strip possible ```json fences
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)

# ═══════════════════════════════════════════════════════════════════════════
# 4. DATABASE HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def insert_expense(email: str, parsed: dict) -> None:
    row = {
        "user_email"  : email,
        "item"        : parsed["item"],
        "amount"      : float(parsed["amount"]),
        "category"    : parsed["category"],
        "subcategory" : parsed["subcategory"],
    }
    supabase.table("expenses").insert(row).execute()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_expenses(email: str) -> pd.DataFrame:
    res = (
        supabase.table("expenses")
        .select("*")
        .eq("user_email", email)
        .order("created_at", desc=True)
        .execute()
    )
    if not res.data:
        return pd.DataFrame()
    df = pd.DataFrame(res.data)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


def get_month_df(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    return df[(df["created_at"].dt.year == year) & (df["created_at"].dt.month == month)]

# ═══════════════════════════════════════════════════════════════════════════
# 5. SIDEBAR — USER SESSION
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 👤 Your Session")
    user_email = st.text_input(
        "Email address",
        placeholder="you@example.com",
        help="Used to scope your personal expense data.",
    )
    st.divider()
    st.markdown("### 📋 Category Reference")
    categories = {
        "🏠 Housing"      : "Rent · Utilities · Maintenance · Insurance",
        "🍔 Food"         : "Groceries · Restaurants · Delivery · Coffee/Snacks",
        "🚗 Transport"    : "Fuel · Public Transit · Uber/Taxi · Repairs",
        "❤️ Health"       : "Doctor · Pharmacy · Gym · Insurance",
        "🛍️ Personal"    : "Shopping · Clothing · Grooming · Laundry",
        "🎮 Entertainment": "Streaming · Gaming · Hobbies · AI Tools",
        "💰 Financial"    : "Debt · Savings · Investments",
    }
    for cat, subs in categories.items():
        with st.expander(cat):
            st.caption(subs)

# ═══════════════════════════════════════════════════════════════════════════
# 6. HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="spend-header">
  <h1>💸 SpendWise AI</h1>
  <p>Tell me what you spent — I'll categorize, save, and visualize it automatically.</p>
</div>
""", unsafe_allow_html=True)

# ── Guard: require email ───────────────────────────────────────────────────
if not user_email or "@" not in user_email:
    st.info("👈 Enter your email in the sidebar to get started.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# 7. CHAT INPUT + AI PARSE + AUTO-SAVE
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="chat-tip">💡 Try: "Uber ride ₹120", "Netflix subscription 199", "Grocery run for 500 rupees", "Bought mangoes ₹80"</div>', unsafe_allow_html=True)

user_input = st.chat_input("Describe your expense…")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Parsing your expense…"):
            try:
                parsed = parse_expense(user_input)

                # Validate required keys
                for key in ("item", "amount", "category", "subcategory"):
                    if key not in parsed:
                        raise ValueError(f"Missing key in AI response: {key}")

                insert_expense(user_email, parsed)
                fetch_expenses.clear()  # bust cache

                st.success(
                    f"✅ **{parsed['item']}** — **₹{parsed['amount']:.2f}** "
                    f"saved under **{parsed['category']} › {parsed['subcategory']}**"
                )

                # Mini detail card
                col1, col2, col3 = st.columns(3)
                col1.metric("Item",        parsed["item"])
                col2.metric("Amount",      f"₹{parsed['amount']:.2f}")
                col3.metric("Category",    f"{parsed['category']} › {parsed['subcategory']}")

            except json.JSONDecodeError:
                st.error("🚨 AI returned invalid JSON. Please rephrase and try again.")
            except Exception as exc:
                st.error(f"🚨 Error: {exc}")

# ═══════════════════════════════════════════════════════════════════════════
# 8. VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════
df = fetch_expenses(user_email)

if df.empty:
    st.info("No expenses yet — add one above to see your dashboard!")
    st.stop()

now       = datetime.now(timezone.utc)
cur_year  = now.year
cur_month = now.month
prev_dt   = now - relativedelta(months=1)
prev_year = prev_dt.year
prev_month= prev_dt.month

df_cur  = get_month_df(df, cur_year, cur_month)
df_prev = get_month_df(df, prev_year, prev_month)

total_cur  = df_cur["amount"].sum()
total_prev = df_prev["amount"].sum()
delta      = total_cur - total_prev

st.divider()
st.subheader("📊 Your Dashboard")

# ── Row 1: KPI Metrics ─────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "This Month's Total",
    f"₹{total_cur:,.2f}",
    delta=f"{delta:+.2f} vs last month",
    delta_color="inverse",
)
m2.metric(
    "Last Month's Total",
    f"₹{total_prev:,.2f}",
)
m3.metric(
    "Transactions (this month)",
    len(df_cur),
)
m4.metric(
    "Avg per Transaction",
    f"₹{df_cur['amount'].mean():.2f}" if len(df_cur) else "—",
)

st.divider()

# ── Row 2: Pie + Bar ───────────────────────────────────────────────────────
col_pie, col_bar = st.columns(2)

# Pie — Category breakdown (current month)
with col_pie:
    st.markdown("#### 🍕 Category Breakdown — This Month")
    if df_cur.empty:
        st.info("No data for the current month yet.")
    else:
        cat_sum = df_cur.groupby("category")["amount"].sum().reset_index()
        fig_pie = px.pie(
            cat_sum,
            names="category",
            values="amount",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(
            showlegend=True,
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# Bar — Month-over-month comparison
with col_bar:
    st.markdown("#### 📅 Month-over-Month Comparison")
    cur_label  = now.strftime("%B %Y")
    prev_label = prev_dt.strftime("%B %Y")

    bar_data = pd.DataFrame({
        "Month" : [prev_label, cur_label],
        "Total" : [total_prev, total_cur],
    })
    fig_bar = px.bar(
        bar_data,
        x="Month",
        y="Total",
        text_auto=".2f",
        color="Month",
        color_discrete_sequence=["#a78bfa", "#667eea"],
    )
    fig_bar.update_layout(
        showlegend=False,
        yaxis_title="Total Spending (₹)",
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Row 3: Subcategory breakdown (current month) ───────────────────────────
st.divider()
st.markdown("#### 🔍 Subcategory Breakdown — This Month")
if not df_cur.empty:
    sub_sum = (
        df_cur.groupby(["category", "subcategory"])["amount"]
        .sum()
        .reset_index()
        .sort_values("amount", ascending=False)
    )
    fig_sub = px.bar(
        sub_sum,
        x="amount",
        y="subcategory",
        color="category",
        orientation="h",
        text_auto=".2f",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_sub.update_layout(
        yaxis_title="",
        xaxis_title="Amount (₹)",
        margin=dict(t=10, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_sub, use_container_width=True)

# ── Row 4: Recent Transactions Table ──────────────────────────────────────
st.divider()
st.markdown("#### 🧾 Recent Transactions")
display_df = df[["created_at", "item", "amount", "category", "subcategory"]].copy()
display_df["created_at"] = display_df["created_at"].dt.strftime("%Y-%m-%d %H:%M")
display_df.columns = ["Date", "Item", "Amount (₹)", "Category", "Subcategory"]
st.dataframe(display_df.head(50), use_container_width=True, hide_index=True)
