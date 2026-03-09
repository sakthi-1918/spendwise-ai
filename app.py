"""
SpendWise AI — Personal Expense Tracker
Stack: Streamlit · LangChain · Groq (Llama 4 Scout) · Supabase · Plotly
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import json, re, hashlib, uuid, secrets
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from supabase import create_client, Client

st.set_page_config(
    page_title="SpendWise AI",
    page_icon="💸",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── RESET & BASE ─────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html {
  background-color: #07070f !important;
  min-height: 100% !important;
  min-height: 100dvh !important;
  overscroll-behavior: none !important;
}
html, body, [class*="css"], .stApp {
  font-family: 'DM Sans', sans-serif !important;
  background-color: #07070f !important;
  color: #ede9ff !important;
  font-size: 16px !important;
  -webkit-font-smoothing: antialiased !important;
  min-height: 100dvh !important;
  overscroll-behavior: none !important;
}
#MainMenu, footer, header { visibility: hidden; display: none !important; height: 0 !important; }
[data-testid="stSidebar"] { display: none !important; }

/* ─── KILL WHITE SPACE / WHITE BG AT BOTTOM ────────────── */
.stApp, .stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"] {
  background-color: #07070f !important;
  background: #07070f !important;
}
body {
  background-color: #07070f !important;
  overscroll-behavior: none !important;
}
[data-testid="stBottom"] {
  background: #07070f !important;
  border-top: none !important;
  padding-bottom: env(safe-area-inset-bottom, 0px) !important;
}
.block-container {
  padding: 0 16px 130px 16px !important;
  max-width: min(520px, 100%) !important;
  width: 100% !important;
  margin: 0 auto !important;
}
/* Make the sticky header and tab bar span full width ignoring the container padding */
.app-header,
div[data-testid="stRadio"] {
  margin-left: -16px !important;
  margin-right: -16px !important;
  width: calc(100% + 32px) !important;
}
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2a4a; border-radius: 99px; }

/* ─── APP HEADER ───────────────────────────────────────── */
.app-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 15px 20px 13px;
  background: rgba(7,7,15,.98);
  backdrop-filter: blur(24px);
  border-bottom: 1px solid rgba(255,255,255,.06);
  position: sticky; top: 0; z-index: 1000;
  height: 58px;
}
.app-logo { display: flex; align-items: center; gap: 9px; }
.app-logo-icon { font-size: 22px; line-height: 1; }
.app-logo-text {
  font-family: 'Inter', sans-serif; font-weight: 800; font-size: 19px;
  background: linear-gradient(120deg, #ede9ff 30%, #9d7dff);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  letter-spacing: -.3px;
}
.avatar {
  width: 36px; height: 36px; border-radius: 11px;
  background: linear-gradient(135deg, #6c56f5, #9d7dff);
  display: flex; align-items: center; justify-content: center;
  font-family: 'Inter', sans-serif; font-weight: 800; font-size: 15px; color: #fff;
  box-shadow: 0 4px 16px rgba(108,86,245,.45);
}

/* ─── TAB BAR (st.radio) ───────────────────────────────── */
div[data-testid="stRadio"] {
  position: sticky !important;
  top: 58px !important;
  z-index: 998 !important;
  background: rgba(7,7,15,.98) !important;
  border-bottom: 1px solid rgba(255,255,255,.06) !important;
  margin: 0 !important; padding: 0 !important;
  backdrop-filter: blur(24px) !important;
}
div[data-testid="stRadio"] > label { display: none !important; }
div[data-testid="stRadio"] > div {
  display: flex !important; flex-direction: row !important;
  gap: 0 !important; margin: 0 !important; padding: 0 !important; width: 100% !important;
}
/* each tab — ALWAYS row, icon left of text, never wrap */
div[data-testid="stRadio"] > div > label {
  flex: 1 !important;
  display: flex !important;
  flex-direction: row !important;
  align-items: center !important;
  justify-content: center !important;
  padding: 0 4px !important;
  margin: 0 !important;
  cursor: pointer !important;
  gap: 5px !important;
  border-bottom: 3px solid transparent !important;
  transition: background .15s, border-color .15s !important;
  height: 52px !important;
  min-width: 0 !important;
  position: relative !important;
  white-space: nowrap !important;
}
div[data-testid="stRadio"] > div > label:not(:last-child)::after {
  content: ''; position: absolute; right: 0; top: 20%; height: 60%;
  width: 1px; background: rgba(255,255,255,.05);
}
div[data-testid="stRadio"] > div > label:hover {
  background: rgba(108,86,245,.06) !important;
}
/* hide radio circle */
div[data-testid="stRadio"] > div > label > div:first-child { display: none !important; }
/* tab label text */
div[data-testid="stRadio"] > div > label > div:last-child {
  font-family: 'DM Sans', sans-serif !important;
  font-size: clamp(10px, 3vw, 13px) !important;
  font-weight: 600 !important;
  color: #4a4a6a !important;
  letter-spacing: 0 !important;
  line-height: 1 !important;
  white-space: nowrap !important;
}
/* ACTIVE */
div[data-testid="stRadio"] > div > label[data-checked="true"] {
  border-bottom: 3px solid #7c5ef7 !important;
  background: rgba(108,86,245,.08) !important;
}
div[data-testid="stRadio"] > div > label[data-checked="true"] > div:last-child {
  color: #b39dff !important; font-weight: 800 !important;
}

/* ─── SECTION ──────────────────────────────────────────── */
.section { padding: 10px 0; }

/* ─── PAGE TITLE ───────────────────────────────────────── */
.page-title { padding: 20px 0 8px; }
.page-title h2 {
  font-family: 'Inter', sans-serif; font-weight: 800; font-size: 22px;
  color: #ede9ff; letter-spacing: -.4px; line-height: 1.2;
}
.page-title p { font-size: 13px; color: #4a4a6a; margin-top: 3px; font-weight: 400; }

/* ─── HERO CARD ────────────────────────────────────────── */
.hero-card {
  background: linear-gradient(140deg, #6c56f5 0%, #4830c8 60%, #3620a8 100%);
  border-radius: 22px; padding: 22px 20px 20px;
  box-shadow: 0 16px 48px rgba(108,86,245,.4);
  position: relative; overflow: hidden;
}
.hero-card::before {
  content: ''; position: absolute; top: -40px; right: -40px;
  width: 160px; height: 160px; border-radius: 50%;
  background: rgba(255,255,255,.07); pointer-events: none;
}
.hero-card::after {
  content: ''; position: absolute; bottom: -40px; right: 16px;
  width: 100px; height: 100px; border-radius: 50%;
  background: rgba(255,255,255,.04); pointer-events: none;
}
.hero-label {
  color: rgba(255,255,255,.65); font-size: 11px; font-weight: 700;
  text-transform: uppercase; letter-spacing: .08em; margin-bottom: 6px;
}
.hero-amount {
  font-family: 'Inter', sans-serif;
  font-size: clamp(32px, 8vw, 40px);
  font-weight: 800; color: #fff;
  letter-spacing: -2px; margin-bottom: 14px; line-height: 1;
}
.hero-pills { display: flex; gap: 8px; flex-wrap: wrap; }
.hero-pill {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 4px 11px; border-radius: 99px;
  font-size: 11px; font-weight: 700; white-space: nowrap; letter-spacing: .02em;
}

/* ─── STAT GRID ────────────────────────────────────────── */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.stat-card {
  background: #111124; border: 1px solid rgba(255,255,255,.07);
  border-radius: 18px; padding: 16px 14px; display: flex; flex-direction: column; gap: 2px;
}
.stat-icon { font-size: 20px; margin-bottom: 8px; line-height: 1; }
.stat-value {
  font-family: 'Inter', sans-serif; font-size: 20px; font-weight: 800;
  color: #ede9ff; line-height: 1.1; letter-spacing: -.5px;
}
.stat-label { font-size: 11px; color: #4a4a6a; margin-top: 3px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }

/* ─── CHART CARD ───────────────────────────────────────── */
.chart-card {
  background: #111124; border: 1px solid rgba(255,255,255,.07);
  border-radius: 20px; padding: 18px 16px;
}
.chart-title {
  font-family: 'Inter', sans-serif; font-weight: 800; font-size: 15px;
  color: #ede9ff; margin-bottom: 12px; letter-spacing: -.2px;
}
.legend-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; justify-content: center; }
.legend-dot { display: flex; align-items: center; gap: 5px; font-size: 11px; color: #6a6a8a; font-weight: 500; }
.legend-circle { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }

/* ─── RECENT ROWS ──────────────────────────────────────── */
.recent-row {
  display: flex; align-items: center; gap: 12px;
  padding: 11px 12px; background: #0c0c1e;
  border-radius: 14px; margin-bottom: 7px;
  border: 1px solid rgba(255,255,255,.05);
}
.cat-icon-box {
  width: 38px; height: 38px; border-radius: 11px; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center; font-size: 19px;
}
.recent-name { font-size: 14px; font-weight: 600; color: #ede9ff; line-height: 1.2; }
.recent-sub { font-size: 11px; color: #4a4a6a; margin-top: 2px; font-weight: 500; }
.recent-amt {
  font-family: 'Inter', sans-serif; font-size: 15px; font-weight: 800;
  margin-left: auto; flex-shrink: 0; letter-spacing: -.3px;
}

/* ─── CHAT TIP ─────────────────────────────────────────── */
.chat-tip {
  background: rgba(108,86,245,.08);
  border: 1px solid rgba(108,86,245,.2);
  border-left: 3px solid #7c5ef7;
  padding: 11px 14px; border-radius: 0 14px 14px 0;
  font-size: 13px; color: #8a74cc; font-style: italic; line-height: 1.5;
}

/* ─── CHAT MESSAGES ────────────────────────────────────── */
/* user bubble */
[data-testid="stChatMessage"] {
  background: #111124 !important;
  border: 1px solid rgba(255,255,255,.06) !important;
  border-radius: 18px !important;
  padding: 14px 16px !important;
  margin-bottom: 10px !important;
}
[data-testid="stChatMessage"] p {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 15px !important; line-height: 1.65 !important;
  color: #d8d2ff !important; font-weight: 400 !important;
}
/* assistant bubble — premium card */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
  background: linear-gradient(145deg, #161630, #0f0f28) !important;
  border: 1px solid rgba(124,94,247,.25) !important;
  box-shadow: 0 4px 24px rgba(108,86,245,.12) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) p {
  color: #ede9ff !important;
  font-size: 15px !important;
  line-height: 1.7 !important;
}
/* success text */
[data-testid="stChatMessage"] [data-testid="stAlert"] {
  border-radius: 12px !important;
}
[data-testid="stChatMessage"] [data-testid="stAlert"] p {
  font-size: 14px !important; font-weight: 500 !important;
}

/* ─── CHAT METRICS ─────────────────────────────────────── */
[data-testid="stChatMessage"] [data-testid="stMetricContainer"],
[data-testid="stChatMessage"] [data-testid="metric-container"] {
  background: rgba(12,12,30,.9) !important;
  border: 1px solid rgba(108,86,245,.18) !important;
  border-radius: 14px !important;
  padding: 14px 12px !important;
  text-align: center !important;
}
[data-testid="stMetricLabel"] > div {
  color: #4a4a6a !important; font-size: 10px !important;
  text-transform: uppercase !important; letter-spacing: .08em !important; font-weight: 700 !important;
}
[data-testid="stMetricValue"] > div {
  font-family: 'Inter', sans-serif !important; color: #ede9ff !important;
  font-size: 16px !important; font-weight: 800 !important; letter-spacing: -.3px !important;
}
[data-testid="metric-container"] {
  background: #111124 !important; border: 1px solid rgba(255,255,255,.07) !important;
  border-radius: 16px !important; padding: 16px 14px !important;
}

/* ─── CHAT INPUT ───────────────────────────────────────── */
[data-testid="stChatInput"] > div {
  background: #111124 !important;
  border: 1px solid rgba(255,255,255,.1) !important;
  border-radius: 16px !important;
  box-shadow: 0 4px 20px rgba(0,0,0,.4) !important;
}
[data-testid="stChatInput"] > div:focus-within {
  border-color: rgba(124,94,247,.5) !important;
  box-shadow: 0 4px 24px rgba(108,86,245,.2) !important;
}
[data-testid="stChatInput"] textarea {
  color: #ede9ff !important; font-family: 'DM Sans', sans-serif !important; font-size: 15px !important;
}

/* ─── EXPENSE CARDS ────────────────────────────────────── */
.exp-card {
  background: #111124; border: 1px solid rgba(255,255,255,.06);
  border-radius: 18px; padding: 14px 14px;
  display: flex; gap: 12px; align-items: center; margin-bottom: 8px;
  transition: background .15s;
}
.exp-left {
  width: 42px; height: 42px; border-radius: 13px; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center; font-size: 21px;
}
.exp-name {
  font-size: 15px; font-weight: 600; color: #ede9ff; line-height: 1.2;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 175px;
}
.exp-sub { font-size: 11px; color: #4a4a6a; margin-top: 3px; font-weight: 500; letter-spacing: .02em; }
.exp-right { margin-left: auto; display: flex; flex-direction: column; align-items: flex-end; gap: 6px; flex-shrink: 0; }
.exp-amt { font-family: 'Inter', sans-serif; font-size: 17px; font-weight: 800; letter-spacing: -.5px; }

/* ─── PROFILE ──────────────────────────────────────────── */
.profile-card {
  background: linear-gradient(140deg, #111124, #0c0c1e);
  border: 1px solid rgba(255,255,255,.07); border-radius: 22px; padding: 22px 18px;
  display: flex; align-items: center; gap: 16px;
}
.profile-avatar {
  width: 62px; height: 62px; border-radius: 19px; flex-shrink: 0;
  background: linear-gradient(135deg, #6c56f5, #9d7dff);
  display: flex; align-items: center; justify-content: center;
  font-family: 'Inter', sans-serif; font-weight: 800; font-size: 28px; color: #fff;
  box-shadow: 0 8px 28px rgba(108,86,245,.5);
}
.profile-name { font-family: 'Inter', sans-serif; font-size: 21px; font-weight: 800; color: #ede9ff; letter-spacing: -.4px; }
.profile-email { font-size: 13px; color: #4a4a6a; margin-top: 3px; font-weight: 400; }
.prog-row { margin-bottom: 14px; }
.prog-label { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.prog-name { font-size: 13px; color: #b8b0e0; font-weight: 500; }
.prog-val { font-size: 13px; font-weight: 800; font-family: 'Inter', sans-serif; letter-spacing: -.2px; }
.prog-track { height: 5px; background: #0c0c1e; border-radius: 99px; }
.prog-fill { height: 100%; border-radius: 99px; }

/* ─── INPUTS ───────────────────────────────────────────── */
.stTextInput > label, .stNumberInput > label, .stSelectbox > label {
  color: #4a4a6a !important; font-size: 10px !important;
  font-weight: 700 !important; letter-spacing: .08em !important; text-transform: uppercase !important;
}
.stTextInput > div > div > input, .stNumberInput > div > div > input {
  background: #0c0c1e !important; border: 1px solid rgba(255,255,255,.08) !important;
  border-radius: 13px !important; color: #ede9ff !important;
  font-family: 'DM Sans', sans-serif !important; font-size: 16px !important;
  padding: 12px 15px !important;
}
.stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus {
  border-color: #7c5ef7 !important; box-shadow: 0 0 0 3px rgba(124,94,247,.18) !important;
}
[data-baseweb="select"] > div {
  background: #0c0c1e !important; border: 1px solid rgba(255,255,255,.08) !important;
  border-radius: 13px !important; color: #ede9ff !important; font-size: 15px !important;
}
[data-baseweb="popover"], [data-baseweb="menu"] {
  background: #111124 !important; border: 1px solid rgba(255,255,255,.08) !important; border-radius: 14px !important;
}
[role="option"] { color: #ede9ff !important; font-size: 14px !important; padding: 11px 14px !important; }
[role="option"]:hover { background: #1a1a36 !important; }

/* ─── BUTTONS ──────────────────────────────────────────── */
.stButton > button {
  font-family: 'DM Sans', sans-serif !important; font-weight: 700 !important;
  border-radius: 13px !important; transition: all .18s !important;
  font-size: 14px !important; padding: 10px 18px !important; min-height: 44px !important;
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background: linear-gradient(135deg, #6c56f5, #9d7dff) !important;
  border: none !important; color: #fff !important;
  box-shadow: 0 4px 20px rgba(108,86,245,.45) !important;
}
.stButton > button[kind="primary"]:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 28px rgba(108,86,245,.6) !important; }
.stButton > button[kind="secondary"], .stFormSubmitButton > button[kind="secondary"] {
  background: #111124 !important; border: 1px solid rgba(255,255,255,.09) !important; color: #9d7dff !important;
}

/* ─── EXPENSE ACTION BUTTONS (Edit / Delete) ───────────── */
div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) {
  gap: 6px !important;
  margin-top: 2px !important;
  margin-bottom: 14px !important;
}
div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) button {
  min-height: 34px !important;
  height: 34px !important;
  font-size: 12px !important;
  padding: 0 8px !important;
  border-radius: 9px !important;
  font-weight: 600 !important;
}
/* Edit button — 2nd column (index 1) */
div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) [data-testid="column"]:nth-child(2) button {
  background: rgba(108,86,245,.1) !important;
  border: 1px solid rgba(108,86,245,.25) !important;
  color: #a990ff !important;
}
div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) [data-testid="column"]:nth-child(2) button:hover {
  background: rgba(108,86,245,.2) !important;
}
/* Delete button — 3rd column (index 2) */
div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) [data-testid="column"]:nth-child(3) button {
  background: rgba(239,68,68,.07) !important;
  border: 1px solid rgba(239,68,68,.22) !important;
  color: #f87171 !important;
}
div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) [data-testid="column"]:nth-child(3) button:hover {
  background: rgba(239,68,68,.15) !important;
}

/* ─── FORM ─────────────────────────────────────────────── */
[data-testid="stForm"] {
  background: #111124 !important; border: 1px solid rgba(255,255,255,.07) !important;
  border-radius: 18px !important; padding: 18px !important;
}

/* ─── AUTH ─────────────────────────────────────────────── */
.auth-wrap {
  width: 100%; display: flex; flex-direction: column; align-items: center;
  padding: 28px 18px 80px;
  background-image: radial-gradient(ellipse 110% 45% at 50% 0%, rgba(108,86,245,.14) 0%, transparent 60%);
}
.auth-logo { text-align: center; margin-bottom: 26px; }
.auth-logo-icon { font-size: 54px; filter: drop-shadow(0 0 28px rgba(108,86,245,.7)); display: block; margin-bottom: 10px; }
.auth-logo-name {
  font-family: 'Inter', sans-serif; font-weight: 800; font-size: 28px;
  background: linear-gradient(135deg, #ede9ff, #9d7dff);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -.5px;
}
.auth-logo-tagline { color: #4a4a6a; font-size: 13px; margin-top: 5px; font-weight: 400; }
.auth-card {
  width: 100%; max-width: 420px;
  background: #111124; border: 1px solid rgba(255,255,255,.07); border-radius: 24px;
  padding: 28px 24px;
  box-shadow: 0 32px 80px rgba(0,0,0,.7), 0 0 0 1px rgba(108,86,245,.08);
}
.auth-heading { font-family: 'Inter', sans-serif; font-size: 22px; font-weight: 800; color: #ede9ff; margin-bottom: 3px; letter-spacing: -.4px; }
.auth-subhead { font-size: 13px; color: #4a4a6a; margin-bottom: 22px; font-weight: 400; }

/* ─── MISC ─────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,.06) !important; }
.stMarkdown p { color: #d8d2ff; font-size: 15px; line-height: 1.65; }
[data-testid="stAlert"] { border-radius: 14px !important; }
[data-testid="stAlert"] p { font-size: 14px !important; line-height: 1.55 !important; }
[data-testid="stCaptionContainer"] { color: #4a4a6a !important; font-size: 12px !important; }
h3, h4, h5 { font-family: 'Inter', sans-serif !important; color: #ede9ff !important; }
code, .stCode {
  font-family: 'JetBrains Mono', monospace !important;
  background: #0c0c1e !important; border: 1px solid rgba(255,255,255,.07) !important;
  border-radius: 10px !important; font-size: 13px !important; color: #9d7dff !important;
}
.stSpinner > div { border-top-color: #7c5ef7 !important; }

/* ─── RESPONSIVE ───────────────────────────────────────── */
@media (max-width: 400px) {
  .block-container { padding-left: 12px !important; padding-right: 12px !important; }
  .app-header, div[data-testid="stRadio"] {
    margin-left: -12px !important; margin-right: -12px !important;
    width: calc(100% + 24px) !important;
  }
  .page-title h2 { font-size: 20px !important; }
  .hero-amount { font-size: clamp(28px, 9vw, 36px) !important; }
  .stat-value { font-size: 18px !important; }
  .exp-name { font-size: 14px !important; max-width: 140px !important; }
  .exp-amt { font-size: 15px !important; }
  .app-logo-text { font-size: 17px !important; }
  .auth-card { padding: 22px 16px !important; border-radius: 18px !important; }
  .auth-heading { font-size: 20px !important; }
  .profile-name { font-size: 18px !important; }
  .profile-avatar { width: 52px !important; height: 52px !important; font-size: 24px !important; }
}
@media (min-width: 600px) {
  .block-container { max-width: 520px !important; }
  .hero-amount { font-size: 44px !important; }
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
CAT_COLORS = {
    "Housing": "#6366f1", "Food": "#f59e0b", "Transport": "#10b981",
    "Health": "#ef4444", "Personal": "#ec4899", "Entertainment": "#8b5cf6", "Financial": "#14b8a6",
}
CAT_ICONS = {
    "Housing": "🏠", "Food": "🍔", "Transport": "🚗",
    "Health": "❤️", "Personal": "🛍️", "Entertainment": "🎮", "Financial": "💰",
}
CATEGORIES = {
    "Housing":       ["Rent", "Utilities", "Maintenance", "Insurance"],
    "Food":          ["Groceries", "Restaurants", "Delivery", "Coffee/Snacks"],
    "Transport":     ["Fuel", "Public Transit", "Uber/Taxi", "Repairs"],
    "Health":        ["Doctor", "Pharmacy", "Gym", "Insurance"],
    "Personal":      ["Shopping", "Clothing", "Grooming", "Laundry"],
    "Entertainment": ["Streaming", "Gaming", "Hobbies", "AI Tools"],
    "Financial":     ["Debt", "Savings", "Investments"],
}
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ═══════════════════════════════════════════════════════════════════════════
# SUPABASE
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_supabase() -> Client:
    return create_client(st.secrets["supabase"]["url"], st.secrets["supabase"]["key"])
try:
    supabase: Client = get_supabase()
except Exception as e:
    st.error(f"❌ Supabase connection failed: {e}"); st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# AUTH HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()

def signup_user(name, email, password):
    try:
        if supabase.table("users").select("id").eq("email", email.strip().lower()).execute().data:
            return False, "An account with this email already exists."
        supabase.table("users").insert({
            "id": str(uuid.uuid4()), "name": name.strip(),
            "email": email.strip().lower(), "password_hash": hash_password(password), "reset_token": None,
        }).execute()
        return True, "Account created! Please log in."
    except Exception as e:
        return False, f"Signup error: {e}"

def login_user(email, password):
    try:
        res = supabase.table("users").select("*").eq("email", email.strip().lower()).execute()
        if not res.data: return False, "No account found with this email.", {}
        u = res.data[0]
        if u["password_hash"] != hash_password(password): return False, "Incorrect password.", {}
        return True, f"Welcome back, {u['name']}!", u
    except Exception as e:
        return False, f"Login error: {e}", {}

def generate_reset_token(email):
    try:
        if not supabase.table("users").select("id").eq("email", email.strip().lower()).execute().data:
            return False, "No account found with this email."
        token = secrets.token_urlsafe(32)
        supabase.table("users").update({"reset_token": token}).eq("email", email.strip().lower()).execute()
        return True, token
    except Exception as e:
        return False, str(e)

def reset_password_with_token(email, token, new_password):
    try:
        res = supabase.table("users").select("*").eq("email", email.strip().lower()).execute()
        if not res.data: return False, "No account found."
        u = res.data[0]
        if u.get("reset_token") != token: return False, "Invalid or expired reset token."
        supabase.table("users").update({
            "password_hash": hash_password(new_password), "reset_token": None,
        }).eq("email", email.strip().lower()).execute()
        return True, "Password reset successfully! Please log in."
    except Exception as e:
        return False, str(e)

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
defaults = {
    "authenticated": False, "user": {}, "edit_id": None,
    "auth_screen": "login", "reset_email": "",
    "active_tab": "home", "filter_cat": "All",
    "last_activity": None,
    "login_time": None,
    "manage_year": None, "manage_month": None,
    "reset_token_generated": None,
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# ── Persistent login via Supabase session_token column ────────────────────
TIMEOUT_MINUTES = 10

def save_session_token(user_id: str, token: str):
    try:
        supabase.table("users").update({"session_token": token}).eq("id", user_id).execute()
    except Exception:
        pass

def clear_session_token(user_id: str):
    try:
        supabase.table("users").update({"session_token": None}).eq("id", user_id).execute()
    except Exception:
        pass

def lookup_session_token(token: str):
    try:
        res = supabase.table("users").select("*").eq("session_token", token).execute()
        if res.data:
            return res.data[0]
    except Exception:
        pass
    return None

now_ts = datetime.now(timezone.utc)

# ── On refresh: restore session from URL token ────────────────────────────
if not st.session_state.authenticated:
    _tok = st.query_params.get("sid", None)
    if _tok:
        _u = lookup_session_token(_tok)
        if _u:
            st.session_state.authenticated = True
            st.session_state.user = _u
            # Restore login_time from session or set now (will timeout from first restore)
            if st.session_state.login_time is None:
                st.session_state.login_time = now_ts

# ── Timeout: check elapsed since login_time, NOT last page interaction ────
# This means: 10 min after you logged in, you are logged out regardless
if st.session_state.authenticated:
    if st.session_state.login_time is not None:
        elapsed = (now_ts - st.session_state.login_time).total_seconds() / 60
        if elapsed > TIMEOUT_MINUTES:
            _uid = st.session_state.user.get("id")
            if _uid:
                clear_session_token(_uid)
            st.query_params.clear()
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# LLM
# ═══════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are an expense validation and parsing assistant for SpendWise AI.

FIRST — decide if the input is a genuine expense description.
A genuine expense must describe:
  - Something a person actually spent money on, paid, lent, or transferred, AND
  - An amount (explicit like "₹200" or "200 rupees", OR clearly implied by context)

Valid expenses include: food, travel, trips, tours, loans given to friends/family,
EMIs, subscriptions, bills, shopping, medical, transport, fuel, rent, fees,
recharges, gifts, entertainment, and any personal payment or money given out.

If the input is NOT a genuine expense (e.g. it's a question, a greeting, or
completely unrelated to money), respond with EXACTLY:
  {{"error": "not_an_expense"}}

If it IS a genuine expense, return ONLY a valid JSON object with:
  - "item"        : short descriptive name (string, title-cased)
  - "amount"      : numeric cost as a float (numbers only, no symbols)
  - "category"    : one of exactly: Housing | Food | Transport | Health | Personal | Entertainment | Financial
  - "subcategory" : most appropriate from the lists below

Category -> subcategories:
  Housing      : Rent, Utilities, Maintenance, Insurance
  Food         : Groceries, Restaurants, Delivery, Coffee/Snacks
  Transport    : Fuel, Public Transit, Uber/Taxi, Repairs
  Health       : Doctor, Pharmacy, Gym, Insurance
  Personal     : Shopping, Clothing, Grooming, Laundry
  Entertainment: Streaming, Gaming, Hobbies, AI Tools
  Financial    : Debt, Savings, Investments

STRICT categorisation rules:
  - Cake, pastry, sweets, dessert, bakery → Food > Groceries (NOT Personal)
  - Fruits, vegetables, produce, meat, dairy → Food > Groceries
  - Kirana / supermarket / BigBasket / Blinkit → Food > Groceries
  - Restaurant, cafe, dhaba, hotel food → Food > Restaurants
  - Swiggy, Zomato, online food order → Food > Delivery
  - Tea, coffee, chai, snacks, juice → Food > Coffee/Snacks
  - NEVER put food, cake, sweets, fruits under Personal
  - Trip / tour / travel / vacation / flight / hotel stay → Entertainment > Hobbies
  - Loan given to friend / family / lent money → Financial > Debt
  - Beer / wine / alcohol → Food > Restaurants
  - Recharge / mobile bill → Housing > Utilities
  - EMI / credit card payment → Financial > Debt
"""

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0,
                    groq_api_key=st.secrets["groq"]["api_key"])

def parse_expense(description):
    chain = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{description}")]) | get_llm()
    raw = chain.invoke({"description": description}).content.strip()
    # Strip markdown code fences
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    # Extract only the first JSON object to avoid "extra data" errors
    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if not match:
        raise ValueError("That doesn't look like an expense. Try: \"Ola ride 280\" or \"Groceries 1500\".")
    data = json.loads(match.group())
    if data.get("error") == "not_an_expense":
        raise ValueError("That doesn't look like an expense. Try: \"Ola ride 280\" or \"Groceries 1500\".")
    return data

# ═══════════════════════════════════════════════════════════════════════════
# DB HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def insert_expense(user_id, parsed):
    supabase.table("expenses").insert({
        "user_id": user_id, "item": parsed["item"],
        "amount": float(parsed["amount"]), "category": parsed["category"], "subcategory": parsed["subcategory"],
    }).execute()

def delete_expense(expense_id):
    supabase.table("expenses").delete().eq("id", expense_id).execute()

def update_expense(expense_id, item, amount, category, subcategory):
    supabase.table("expenses").update({
        "item": item, "amount": amount, "category": category, "subcategory": subcategory,
    }).eq("id", expense_id).execute()

@st.cache_data(ttl=30, show_spinner=False)
def fetch_expenses(user_id):
    res = supabase.table("expenses").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    if not res.data: return pd.DataFrame()
    df = pd.DataFrame(res.data)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df

def get_month_df(df, year, month):
    return df[(df["created_at"].dt.year == year) & (df["created_at"].dt.month == month)]

# ═══════════════════════════════════════════════════════════════════════════
# AUTH SCREENS
# ═══════════════════════════════════════════════════════════════════════════
if not st.session_state.authenticated:
    st.markdown('<div class="auth-wrap">', unsafe_allow_html=True)
    st.markdown("""
    <div class="auth-logo">
      <span class="auth-logo-icon">💸</span>
      <div class="auth-logo-name">SpendWise AI</div>
      <div class="auth-logo-tagline">Track smarter. Spend wiser.</div>
    </div>""", unsafe_allow_html=True)
    # st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    scr = st.session_state.auth_screen

    if scr == "login":
        st.markdown('<div class="auth-heading" style="text-align: center;">Welcome back 👋</div><div class="auth-subhead" style="text-align: center;">Log in to your account</div>', unsafe_allow_html=True)
        with st.form("login_form", border=False):
            l_email = st.text_input("Email", placeholder="you@example.com")
            l_pwd   = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Log In →", type="primary", use_container_width=True)
        if submitted:
            if not l_email or not l_pwd: st.error("Please fill in all fields.")
            else:
                ok, msg, user = login_user(l_email, l_pwd)
                if ok:
                    _tok = secrets.token_urlsafe(32)
                    save_session_token(user["id"], _tok)
                    st.query_params["sid"] = _tok
                    st.session_state.authenticated = True
                    st.session_state.user = user
                    st.session_state.login_time = datetime.now(timezone.utc)
                    st.rerun()
                else: st.error(msg)
        c1, c2 = st.columns(2)
        if c1.button("✨ Create account", use_container_width=True): st.session_state.auth_screen = "signup"; st.rerun()
        if c2.button("🔒 Forgot password?", use_container_width=True): st.session_state.auth_screen = "forgot"; st.rerun()

    elif scr == "signup":
        st.markdown('<div class="auth-heading">Create account ✨</div><div class="auth-subhead">Free forever. No credit card needed.</div>', unsafe_allow_html=True)
        with st.form("signup_form", border=False):
            s_name  = st.text_input("Full Name", placeholder="Arjun Sharma")
            s_email = st.text_input("Email", placeholder="you@example.com")
            s_pwd   = st.text_input("Password", type="password", placeholder="min. 6 characters")
            s_pwd2  = st.text_input("Confirm Password", type="password", placeholder="repeat password")
            submitted_s = st.form_submit_button("Create Account →", type="primary", use_container_width=True)
        if submitted_s:
            if not all([s_name, s_email, s_pwd, s_pwd2]): st.error("Please fill in all fields.")
            elif len(s_pwd) < 6: st.error("Password must be at least 6 characters.")
            elif s_pwd != s_pwd2: st.error("Passwords do not match.")
            elif "@" not in s_email: st.error("Please enter a valid email address.")
            else:
                ok, msg = signup_user(s_name, s_email, s_pwd)
                if ok: st.success(msg)
                else: st.error(msg)
        if st.button("← Back to Login", use_container_width=True): st.session_state.auth_screen = "login"; st.rerun()

    elif scr == "forgot":
        st.markdown('<div class="auth-heading">Forgot password 🔒</div><div class="auth-subhead">Enter your email to get a reset token.</div>', unsafe_allow_html=True)
        with st.form("forgot_form", border=False):
            f_email = st.text_input("Registered Email", placeholder="you@example.com")
            submitted_f = st.form_submit_button("Generate Token →", type="primary", use_container_width=True)
        if submitted_f:
            if not f_email or "@" not in f_email:
                st.error("Please enter a valid email.")
            else:
                ok, result = generate_reset_token(f_email)
                if ok:
                    st.session_state.reset_email = f_email
                    st.session_state.reset_token_generated = result
                    st.rerun()
                else:
                    st.error(result)
        # Show token + Proceed button persistently after generation
        if st.session_state.get("reset_token_generated"):
            st.success("Token generated — copy it below!")
            st.code(st.session_state.reset_token_generated, language=None)
            st.warning("⚠️ Copy this token before proceeding.")
            if st.button("Proceed to Reset →", type="primary", use_container_width=True):
                st.session_state.auth_screen = "reset"
                st.session_state.reset_token_generated = None
                st.rerun()
        if st.button("← Back to Login", use_container_width=True):
            st.session_state.reset_token_generated = None
            st.session_state.auth_screen = "login"
            st.rerun()

    elif scr == "reset":
        st.markdown('<div class="auth-heading">Reset password 🔑</div><div class="auth-subhead">Enter your token and new password.</div>', unsafe_allow_html=True)
        with st.form("reset_form", border=False):
            r_email = st.text_input("Email", value=st.session_state.reset_email, placeholder="you@example.com")
            r_token = st.text_input("Reset Token", placeholder="paste token here")
            r_pwd   = st.text_input("New Password", type="password", placeholder="min. 6 characters")
            submitted_r = st.form_submit_button("Reset Password →", type="primary", use_container_width=True)
        if submitted_r:
            if not all([r_email, r_token, r_pwd]): st.error("Please fill in all fields.")
            elif len(r_pwd) < 6: st.error("New password must be at least 6 characters.")
            else:
                ok, msg = reset_password_with_token(r_email, r_token, r_pwd)
                if ok:
                    st.success(msg); st.session_state.auth_screen = "login"; st.session_state.reset_email = ""; st.rerun()
                else: st.error(msg)
        if st.button("← Back to Login", use_container_width=True): st.session_state.auth_screen = "login"; st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# LOGGED-IN APP
# ═══════════════════════════════════════════════════════════════════════════
user     = st.session_state.user
tab      = st.session_state.active_tab
initials = user.get("name", "U")[0].upper()
now_dt   = datetime.now(timezone.utc)
prev_dt  = now_dt - relativedelta(months=1)

# ── Header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
  <div class="app-logo">
    <span class="app-logo-icon">💸</span>
    <span class="app-logo-text">SpendWise AI</span>
  </div>
  <div class="avatar">{initials}</div>
</div>""", unsafe_allow_html=True)

# ── Tab navigation ────────────────────────────────────────────────────────
NAV_LABELS = ["📊 Home", "💬 Chat", "🧾 Manage", "👤 Profile"]
NAV_IDS    = ["home", "chat", "manage", "profile"]
_cur_idx   = NAV_IDS.index(tab) if tab in NAV_IDS else 0
_selected  = st.radio("nav", NAV_LABELS, index=_cur_idx, horizontal=True,
                       label_visibility="collapsed", key="nav_radio")
_new_tab   = NAV_IDS[NAV_LABELS.index(_selected)]
if _new_tab != tab:
    st.session_state.active_tab = _new_tab
    st.session_state.edit_id = None
    tab = _new_tab; st.rerun()

# ── Page title ────────────────────────────────────────────────────────────
page_meta = {
    "home":    ("Dashboard",   f"{MONTHS[now_dt.month-1]} {now_dt.year} overview"),
    "chat":    ("Add Expense", "Describe what you spent in plain English"),
    "manage":  ("My Expenses", "View, edit or delete entries"),
    "profile": ("Profile",     user.get("email", "")),
}
pt, ps = page_meta[tab]
st.markdown(f'<div class="page-title"><h2>{pt}</h2><p>{ps}</p></div>', unsafe_allow_html=True)

# ── Fetch data ────────────────────────────────────────────────────────────
df         = fetch_expenses(user["id"])
df_cur     = get_month_df(df, now_dt.year,  now_dt.month)  if not df.empty else pd.DataFrame()
df_prev    = get_month_df(df, prev_dt.year, prev_dt.month) if not df.empty else pd.DataFrame()
total_cur  = float(df_cur["amount"].sum())  if not df_cur.empty  else 0.0
total_prev = float(df_prev["amount"].sum()) if not df_prev.empty else 0.0
delta      = total_cur - total_prev

# ═══════════════════════════════════════════════════════════════════════════
# HOME TAB
# ═══════════════════════════════════════════════════════════════════════════
if tab == "home":
    dc      = "#f87171" if delta > 0 else "#34d399"
    ds      = "▲" if delta > 0 else "▼"
    avg_str = f"&#x20B9;{df_cur['amount'].mean():,.0f}" if len(df_cur) else "—"

    st.markdown(f"""
    <div class="section">
      <div class="hero-card">
        <div class="hero-label">{MONTHS[now_dt.month-1]} {now_dt.year} &mdash; Total Spent</div>
        <div class="hero-amount">&#x20B9;{total_cur:,.0f}</div>
        <div class="hero-pills">
          <span class="hero-pill" style="background:{dc}22;color:{dc};border:1px solid {dc}33">{ds} &#x20B9;{abs(delta):,.0f} vs {MONTHS[prev_dt.month-1]}</span>
          <span class="hero-pill" style="background:rgba(255,255,255,.12);color:#fff;border:1px solid rgba(255,255,255,.18)">{len(df_cur)} transactions</span>
        </div>
      </div>
      <br>
      <div class="stat-grid">
        <div class="stat-card"><div class="stat-icon">🗓️</div><div class="stat-value">&#x20B9;{total_prev:,.0f}</div><div class="stat-label">Last Month</div></div>
        <div class="stat-card"><div class="stat-icon">📈</div><div class="stat-value">{avg_str}</div><div class="stat-label">Avg / Entry</div></div>
      </div>
    </div>""", unsafe_allow_html=True)

    if not df_cur.empty:
        cat_sum = df_cur.groupby("category")["amount"].sum().reset_index()

        st.markdown('<div class="section"><div class="chart-card"><div class="chart-title">Category Breakdown</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=cat_sum["category"], values=cat_sum["amount"], hole=0.46,
            marker_colors=[CAT_COLORS.get(c, "#7c5ef7") for c in cat_sum["category"]],
            textinfo="percent", textfont=dict(color="#fff", family="DM Sans", size=11),
        ))
        fig_pie.update_layout(showlegend=False, margin=dict(t=4,b=4,l=4,r=4),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=195)
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
        leg = '<div class="legend-row">'
        for _, r in cat_sum.iterrows():
            c = CAT_COLORS.get(r["category"], "#7c5ef7")
            leg += f'<div class="legend-dot"><div class="legend-circle" style="background:{c}"></div>{r["category"]}</div>'
        st.markdown(leg + "</div></div></div>", unsafe_allow_html=True)

        st.markdown('<div class="section"><div class="chart-card"><div class="chart-title">3-Month Comparison</div>', unsafe_allow_html=True)
        prev2_dt    = now_dt - relativedelta(months=2)
        df_prev2    = get_month_df(df, prev2_dt.year, prev2_dt.month) if not df.empty else pd.DataFrame()
        total_prev2 = float(df_prev2["amount"].sum()) if not df_prev2.empty else 0.0
        bar_months  = [MONTHS[prev2_dt.month-1], MONTHS[prev_dt.month-1], MONTHS[now_dt.month-1]]
        bar_totals  = [total_prev2, total_prev, total_cur]
        bar_colors  = ["#161630", "#2e2060", "#6c56f5"]
        fig_bar = go.Figure(go.Bar(
            x=bar_months, y=bar_totals,
            marker_color=bar_colors, marker_line_width=0,
            text=[f"₹{v:,.0f}" for v in bar_totals],
            textposition="outside", textfont=dict(color="#9d7dff", family="DM Sans", size=11),
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=28,b=4,l=4,r=4), height=185, bargap=0.45,
            xaxis=dict(color="#4a4a6a", showgrid=False, tickfont=dict(size=13, family="DM Sans")),
            yaxis=dict(color="#4a4a6a", gridcolor="rgba(255,255,255,.04)", tickprefix="₹", tickformat=",.0f", tickfont=dict(size=11)),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div></div>", unsafe_allow_html=True)

    if not df.empty:
        st.markdown('<div class="section"><div class="chart-card"><div class="chart-title">Recent Entries</div>', unsafe_allow_html=True)
        rows_html = ""
        for _, r in df.head(4).iterrows():
            c  = CAT_COLORS.get(r["category"], "#7c5ef7")
            ic = CAT_ICONS.get(r["category"], "💰")
            rows_html += f"""
            <div class="recent-row">
              <div class="cat-icon-box" style="background:{c}22">{ic}</div>
              <div style="flex:1;min-width:0">
                <div class="recent-name">{r['item']}</div>
                <div class="recent-sub">{r['subcategory']}</div>
              </div>
              <div class="recent-amt" style="color:{c}">&#x20B9;{r['amount']:,.0f}</div>
            </div>"""
        st.markdown(rows_html + "</div></div>", unsafe_allow_html=True)

    if df.empty:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.info("No expenses yet — go to Chat tab to add your first one!")
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# CHAT TAB
# ═══════════════════════════════════════════════════════════════════════════
elif tab == "chat":
    st.markdown('<div class="section"><div class="chat-tip">💡 Try: "Ola ride 280" · "Netflix 199" · "Birthday cake 450" · "Grocery run 1500"</div></div>', unsafe_allow_html=True)

    user_input = st.chat_input("e.g. Swiggy biryani 320…")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Parsing your expense…"):
                try:
                    parsed = parse_expense(user_input)
                    for k in ("item", "amount", "category", "subcategory"):
                        if k not in parsed: raise ValueError(f"Missing key: {k}")
                    insert_expense(user["id"], parsed)
                    fetch_expenses.clear()
                    # Premium AI response card
                    st.markdown(f"""
                    <div style="margin-bottom:12px">
                      <div style="font-size:13px;color:#6a6a9a;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">✅ Expense Saved</div>
                      <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:800;color:#ede9ff;letter-spacing:-.5px;margin-bottom:4px">&#x20B9;{parsed['amount']:,.2f}</div>
                      <div style="font-size:15px;color:#b8b0e0;font-weight:500;margin-bottom:10px">{parsed['item']}</div>
                      <div style="display:flex;gap:8px;flex-wrap:wrap">
                        <span style="background:rgba(108,86,245,.15);color:#c4b0ff;border:1px solid rgba(108,86,245,.25);padding:4px 12px;border-radius:99px;font-size:12px;font-weight:700">{parsed['category']}</span>
                        <span style="background:rgba(255,255,255,.05);color:#8a8ab0;border:1px solid rgba(255,255,255,.08);padding:4px 12px;border-radius:99px;font-size:12px;font-weight:600">{parsed['subcategory']}</span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Item",     parsed["item"])
                    c2.metric("Amount",   f"₹{parsed['amount']:.0f}")
                    c3.metric("Category", parsed["category"])
                except ValueError as ve:
                    st.warning(f"⚠️ {ve}")
                except json.JSONDecodeError:
                    st.error("🚨 AI returned invalid JSON. Please rephrase and try again.")
                except Exception as exc:
                    st.error(f"🚨 Error: {exc}")

# ═══════════════════════════════════════════════════════════════════════════
# MANAGE TAB
# ═══════════════════════════════════════════════════════════════════════════
elif tab == "manage":
    st.markdown('<div class="section">', unsafe_allow_html=True)

    # ── Month picker ──────────────────────────────────────────────────────
    # Build list of months that have data, plus current month always shown
    if not df.empty:
        df["_ym"] = df["created_at"].dt.to_period("M")
        available_periods = sorted(df["_ym"].unique(), reverse=True)
    else:
        available_periods = []

    # Always include current month
    cur_period = pd.Period(f"{now_dt.year}-{now_dt.month:02d}", freq="M")
    if cur_period not in available_periods:
        available_periods = [cur_period] + list(available_periods)

    month_labels = [p.strftime("%b %Y") for p in available_periods]

    # Default to current month
    if st.session_state.manage_year is None:
        st.session_state.manage_year = now_dt.year
        st.session_state.manage_month = now_dt.month

    # Find selected index
    try:
        sel_period = pd.Period(f"{st.session_state.manage_year}-{st.session_state.manage_month:02d}", freq="M")
        sel_idx = list(available_periods).index(sel_period)
    except (ValueError, KeyError):
        sel_idx = 0

    # Month selector styled as segmented pill row
    st.markdown("""
    <div style="font-size:11px;font-weight:700;color:#4a4a6a;text-transform:uppercase;
                letter-spacing:.06em;margin-bottom:8px">📅 Month</div>
    """, unsafe_allow_html=True)

    chosen_month_label = st.selectbox(
        "Select month", month_labels, index=sel_idx,
        label_visibility="collapsed", key="month_picker"
    )
    chosen_period = available_periods[month_labels.index(chosen_month_label)]
    if (chosen_period.year != st.session_state.manage_year or
            chosen_period.month != st.session_state.manage_month):
        st.session_state.manage_year  = chosen_period.year
        st.session_state.manage_month = chosen_period.month
        st.rerun()

    # Filter df to selected month
    sel_year  = st.session_state.manage_year
    sel_month = st.session_state.manage_month
    df_month  = get_month_df(df, sel_year, sel_month) if not df.empty else pd.DataFrame()

    # Month summary pill
    month_total = float(df_month["amount"].sum()) if not df_month.empty else 0.0
    n_entries   = len(df_month)
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                background:#111124;border:1px solid rgba(255,255,255,.07);
                border-radius:14px;padding:12px 16px;margin:8px 0 14px">
      <div>
        <div style="font-size:12px;color:#4a4a6a;font-weight:600;text-transform:uppercase;letter-spacing:.05em">{chosen_month_label}</div>
        <div style="font-family:'Inter',sans-serif;font-size:22px;font-weight:800;color:#ede9ff;letter-spacing:-.5px;margin-top:2px">&#x20B9;{month_total:,.0f}</div>
      </div>
      <div style="background:rgba(108,86,245,.12);color:#b39dff;border:1px solid rgba(108,86,245,.2);
                  padding:5px 14px;border-radius:99px;font-size:12px;font-weight:700">
        {n_entries} entries
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Category filter ───────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:11px;font-weight:700;color:#4a4a6a;text-transform:uppercase;
                letter-spacing:.06em;margin-bottom:8px">🗂 Category</div>
    """, unsafe_allow_html=True)

    all_cats    = ["All"] + list(CATEGORIES.keys())
    cat_options = ["🔍 All"] + [f"{CAT_ICONS.get(c,'')} {c}" for c in CATEGORIES.keys()]
    cur_idx     = all_cats.index(st.session_state.filter_cat) if st.session_state.filter_cat in all_cats else 0
    chosen_cat  = st.selectbox("Filter", cat_options, index=cur_idx, label_visibility="collapsed", key="cat_filter")
    new_cat     = all_cats[cat_options.index(chosen_cat)]
    if new_cat != st.session_state.filter_cat:
        st.session_state.filter_cat = new_cat; st.rerun()

    # ── Edit form ─────────────────────────────────────────────────────────
    if st.session_state.edit_id is not None and not df.empty:
        row = df[df["id"] == st.session_state.edit_id]
        if not row.empty:
            r = row.iloc[0]
            st.markdown("##### ✏️ Edit Expense")
            with st.form("edit_form"):
                e_item   = st.text_input("Item Name", value=r["item"])
                e_amount = st.number_input("Amount (₹)", value=float(r["amount"]), min_value=0.01, step=1.0, format="%.2f")
                e_cat    = st.selectbox("Category", list(CATEGORIES.keys()),
                                        index=list(CATEGORIES.keys()).index(r["category"]) if r["category"] in CATEGORIES else 0)
                e_sub    = st.selectbox("Subcategory", CATEGORIES[e_cat],
                                        index=CATEGORIES[e_cat].index(r["subcategory"]) if r["subcategory"] in CATEGORIES[e_cat] else 0)
                cs, cx = st.columns(2)
                save   = cs.form_submit_button("💾 Save", type="primary", use_container_width=True)
                cancel = cx.form_submit_button("✖ Cancel", use_container_width=True)
            if save:
                if not e_item.strip(): st.error("Item name cannot be empty.")
                else:
                    update_expense(int(st.session_state.edit_id), e_item.strip(), e_amount, e_cat, e_sub)
                    fetch_expenses.clear(); st.session_state.edit_id = None
                    st.success("✅ Updated!"); st.rerun()
            if cancel: st.session_state.edit_id = None; st.rerun()

    # ── Expense list ──────────────────────────────────────────────────────
    if df_month.empty:
        st.info(f"No expenses recorded for {chosen_month_label}.")
    else:
        fc       = st.session_state.filter_cat
        filtered = df_month if fc == "All" else df_month[df_month["category"] == fc]
        if filtered.empty:
            st.info(f"No {fc} expenses in {chosen_month_label}.")
        else:
            for _, row in filtered.iterrows():
                c  = CAT_COLORS.get(row["category"], "#7c5ef7")
                ic = CAT_ICONS.get(row["category"], "💰")
                dt = row["created_at"].strftime("%d %b %Y")
                st.markdown(f"""
                <div class="exp-card" style="border-left:3px solid {c};flex-direction:column;gap:10px">
                  <div style="display:flex;align-items:center;gap:12px;width:100%">
                    <div class="exp-left" style="background:{c}20">{ic}</div>
                    <div style="flex:1;min-width:0">
                      <div class="exp-name">{row['item']}</div>
                      <div class="exp-sub">{row['subcategory']} · {dt}</div>
                    </div>
                    <div class="exp-amt" style="color:{c};font-family:'Inter',sans-serif;font-size:17px;font-weight:800;letter-spacing:-.5px;flex-shrink:0">&#x20B9;{row['amount']:,.0f}</div>
                  </div>
                </div>""", unsafe_allow_html=True)
                _, ba, bb = st.columns([3, 1, 1])
                if ba.button("✏️ Edit", key=f"edit_{row['id']}", use_container_width=True):
                    st.session_state.edit_id = int(row["id"]); st.rerun()
                if bb.button("🗑️ Del", key=f"del_{row['id']}", use_container_width=True):
                    delete_expense(int(row["id"])); fetch_expenses.clear()
                    st.success(f"Deleted: {row['item']}"); st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PROFILE TAB
# ═══════════════════════════════════════════════════════════════════════════
elif tab == "profile":
    total_all = float(df["amount"].sum()) if not df.empty else 0.0
    top_cat   = ""
    if not df.empty:
        tc      = df.groupby("category")["amount"].sum().idxmax()
        top_cat = f"{CAT_ICONS.get(tc, '')} {tc}"

    st.markdown(f"""
    <div class="section">
      <div class="profile-card">
        <div class="profile-avatar">{initials}</div>
        <div>
          <div class="profile-name">{user.get('name', 'User')}</div>
          <div class="profile-email">{user.get('email', '')}</div>
        </div>
      </div>
      <br>
      <div class="stat-grid">
        <div class="stat-card"><div class="stat-icon">💳</div><div class="stat-value">&#x20B9;{total_all:,.0f}</div><div class="stat-label">Total Spent</div></div>
        <div class="stat-card"><div class="stat-icon">📝</div><div class="stat-value">{len(df)}</div><div class="stat-label">Total Entries</div></div>
        <div class="stat-card"><div class="stat-icon">🏆</div><div class="stat-value" style="font-size:14px;letter-spacing:0">{top_cat or "—"}</div><div class="stat-label">Top Category</div></div>
        <div class="stat-card"><div class="stat-icon">📅</div><div class="stat-value">&#x20B9;{total_cur:,.0f}</div><div class="stat-label">This Month</div></div>
      </div>
      <br>
      <div class="chart-card"><div class="chart-title">Spending by Category</div>
    """, unsafe_allow_html=True)

    if not df.empty:
        prog = ""
        for cat in CATEGORIES:
            v   = float(df[df["category"] == cat]["amount"].sum())
            pct = (v / total_all * 100) if total_all > 0 else 0
            c   = CAT_COLORS.get(cat, "#7c5ef7")
            ic  = CAT_ICONS.get(cat, "")
            prog += f"""
            <div class="prog-row">
              <div class="prog-label">
                <span class="prog-name">{ic} {cat}</span>
                <span class="prog-val" style="color:{c}">&#x20B9;{v:,.0f}</span>
              </div>
              <div class="prog-track"><div class="prog-fill" style="width:{pct:.1f}%;background:{c}"></div></div>
            </div>"""
        st.markdown(prog, unsafe_allow_html=True)

    st.markdown("</div><br></div>", unsafe_allow_html=True)

    if st.button("🚪 Log Out", use_container_width=True, type="secondary"):
        _uid = st.session_state.user.get("id")
        if _uid:
            clear_session_token(_uid)
        st.query_params.clear()
        for k, v in defaults.items(): st.session_state[k] = v
        fetch_expenses.clear(); st.rerun()
