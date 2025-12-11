# app.py
import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta

# Optional: Prophet forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# Config from .env
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "").strip()
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "").strip()
DEFAULT_EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "").strip()
STOCK_THRESHOLD = int(os.getenv("STOCK_THRESHOLD", "20"))

# App state keys
PENDING_EMAIL_KEY = "pending_stock_email"  # stores dict with 'df' and 'recipients' and 'message'

# Simple local DB for logging (actions)
DB_PATH = "retailsense_actions.db"
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS actions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,
        action TEXT,
        detail TEXT
    )
    """)
    conn.commit()
    conn.close()

def log_action(action, detail):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT INTO actions (ts, action, detail) VALUES (?, ?, ?)",
                    (datetime.utcnow().isoformat(), action, str(detail)))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Logging failed: {e}")

init_db()

st.set_page_config(page_title="RetailSense Agentic MVP", layout="wide")
st.title("🛒 RetailSense — Agentic MVP (stocks column = `stocks`)")
st.caption("Uploads → analysis → agent actions (safe confirmations)")

# -----------------------
# File upload & load
# -----------------------
uploaded_file = st.file_uploader("Upload CSV (must include: product, stocks, date/sales optional)", type=["csv"])
df = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("CSV loaded.")
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

# -----------------------
# Utility: safe date parsing
# -----------------------
def ensure_datetime_column(df, col="date"):
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception:
            # if parsing fails, ignore
            pass
    return df

# -----------------------
# Tools
# -----------------------
def find_low_stock(df, threshold=STOCK_THRESHOLD):
    """Return DataFrame of items with stocks < threshold. Requires 'product' and 'stocks' columns."""
    if "product" not in df.columns or "stocks" not in df.columns:
        raise ValueError("CSV must include 'product' and 'stocks' columns for stock checks.")
    low_df = df.loc[df["stocks"] < threshold].copy()
    # keep unique product rows — aggregate by product if multiple rows per product
    agg = low_df.groupby("product", as_index=False).agg({
        "stocks": "sum",
        **({c: "sum" for c in df.columns if c not in ["product", "stocks", "date"] and df[c].dtype != object} if True else {})
    })
    # if date exists, show latest date per product
    if "date" in df.columns:
        ensure_datetime_column(df, "date")
        latest_dates = df.groupby("product")["date"].max().reset_index()
        agg = agg.merge(latest_dates, on="product", how="left")
    return agg.sort_values("stocks")

def top_products(df, n=1, period_days=None):
    """Return top n products by sum(sales). If period_days is set, filter by recent date window."""
    if "product" not in df.columns or "sales" not in df.columns:
        raise ValueError("CSV must include 'product' and 'sales' columns for top_products.")
    if period_days is not None and "date" in df.columns:
        ensure_datetime_column(df, "date")
        cutoff = pd.to_datetime("today") - pd.Timedelta(days=period_days)
        sub = df[df["date"] >= cutoff]
    else:
        sub = df
    agg = sub.groupby("product", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    return agg.head(n)

def forecast_sales(df, product=None, days=7):
    """Return matplotlib figure for forecast using Prophet. Requires 'date' and 'sales' columns."""
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet not installed.")
    if "date" not in df.columns or "sales" not in df.columns:
        raise ValueError("CSV must include 'date' and 'sales' columns for forecasting.")
    d = df.copy()
    d = ensure_datetime_column(d, "date")
    if product:
        d = d[d["product"] == product]
    if d.empty:
        raise ValueError("No data available for requested product/time window.")
    prop_df = d[["date", "sales"]].rename(columns={"date": "ds", "sales": "y"})
    m = Prophet()
    m.fit(prop_df)
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(prop_df["ds"], prop_df["y"], label="Actual")
    ax.plot(forecast["ds"], forecast["yhat"], label="Forecast")
    ax.legend()
    ax.set_title(f"Forecast for {'all products' if not product else product} (next {days} days)")
    return fig, forecast

def send_email(recipients, subject, body):
    """Send email using SMTP_SSL (Gmail). Requires EMAIL_SENDER & EMAIL_PASSWORD."""
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        raise RuntimeError("Email sender or password not configured in .env.")
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = ", ".join(recipients if isinstance(recipients, (list,tuple)) else [recipients])
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.sendmail(EMAIL_SENDER, recipients, msg.as_string())
        return True, "Email sent"
    except Exception as e:
        return False, str(e)

# -----------------------
# Agent (improved local agent)
# -----------------------
def agent_handle_query(df, question):
    """
    Interpret question (simple rule-based planner).
    Returns dict with keys:
      - mode: 'display'|'confirm_action'|'info'|'error'
      - content: depends on mode (DataFrame, text, figure, etc.)
      - action_needed: dict describing pending action (for confirmation)
    """
    q = question.lower()
    # low stock request
    if any(k in q for k in ["low stock", "stock deficiency", "stock alert", "check stock", "low on stock"]):
        try:
            low = find_low_stock(df)
        except Exception as e:
            return {"mode":"error", "content":str(e)}
        if low.empty:
            log_action("check_low_stock", "no deficiency")
            return {"mode":"info", "content":"No low-stock items found."}
        # Prepare email body sample
        body = "Low Stock Alert generated by RetailSense:\n\n" + low.to_string(index=False)
        subject = "RetailSense - Low Stock Alert"
        # store pending email in session_state for confirmation
        st.session_state[PENDING_EMAIL_KEY] = {
            "df": low.to_dict(orient="records"),
            "subject": subject,
            "body": body,
            "recipients": [DEFAULT_EMAIL_RECEIVER] if DEFAULT_EMAIL_RECEIVER else []
        }
        log_action("check_low_stock", f"found {len(low)} items")
        return {"mode":"confirm_action", "content": low, "message": f"Found {len(low)} low-stock items. Send alert email to {DEFAULT_EMAIL_RECEIVER or 'enter recipient below'}?"}

    # top products - last N days detection
    if "top product" in q or "top products" in q or ("best" in q and "product" in q):
        # try to detect period keywords: month, 30 days, 7 days
        period = None
        if "month" in q or "30" in q:
            period = 30
        elif "week" in q or "7" in q:
            period = 7
        try:
            top = top_products(df, n=5 if "products" in q else 1, period_days=period)
        except Exception as e:
            return {"mode":"error", "content":str(e)}
        log_action("top_products", f"period={period}, rows={len(top)}")
        return {"mode":"display", "content": top}

    # forecast
    if "forecast" in q or "predict" in q:
        # try to extract product name by naive approach:
        product = None
        for p in df["product"].unique() if "product" in df.columns else []:
            if p.lower() in q:
                product = p
                break
        try:
            fig, forecast = forecast_sales(df, product=product, days=7)
            log_action("forecast", f"product={product}")
            return {"mode":"display_figure", "content": fig, "message": f"Forecast for {'all products' if product is None else product}"}
        except Exception as e:
            return {"mode":"error", "content": str(e)}

    # total sales
    if "total sales" in q or ("total" in q and "sales" in q):
        if "sales" in df.columns:
            s = df["sales"].sum()
            log_action("total_sales", f"sum={s}")
            return {"mode":"info", "content": f"Total sales = {s}"}
        else:
            return {"mode":"error", "content": "No sales column in dataset."}

    # fallback summary
    try:
        summary = df.describe().to_string()
        return {"mode":"info", "content": "Dataset summary:\n\n" + summary}
    except Exception as e:
        return {"mode":"error", "content": str(e)}

# -----------------------
# UI: Main
# -----------------------
if df is not None:
    # ensure date parsed when present
    df = ensure_datetime_column(df, "date")

    # Basic charts & preview
    st.subheader("Data preview & charts")
    st.dataframe(df.head())

    if "date" in df.columns and "sales" in df.columns:
        fig, ax = plt.subplots(figsize=(9,3))
        d_sorted = df.sort_values("date")
        ax.plot(d_sorted["date"], d_sorted["sales"])
        ax.set_title("Sales over time")
        st.pyplot(fig)

    # Agent chat input
    st.markdown("---")
    st.header("🤖 RetailSense Agent (local)")
    user_q = st.text_input("Ask (examples: 'Which products are low on stock?', 'Top product this month', 'Forecast next 7 days')")

    if user_q:
        with st.spinner("Analyzing..."):
            result = agent_handle_query(df, user_q)

        if result["mode"] == "error":
            st.error(result["content"])
        elif result["mode"] == "info":
            st.info(result["content"])
        elif result["mode"] == "display":
            st.write("Result:")
            st.dataframe(result["content"])
        elif result["mode"] == "display_figure":
            st.write(result.get("message", ""))
            st.pyplot(result["content"])
        elif result["mode"] == "confirm_action":
            st.warning(result["message"])
            st.dataframe(result["content"])
            # recipients input and confirm buttons
            rec_input = st.text_input("Recipient email (edit if needed):", value=(DEFAULT_EMAIL_RECEIVER or ""))
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Confirm Send Alert"):
                    # perform send email using stored pending content (but update recipients from input)
                    pending = st.session_state.get(PENDING_EMAIL_KEY)
                    if not pending:
                        st.error("No pending email found.")
                    else:
                        pending["recipients"] = [rec_input] if rec_input else pending.get("recipients", [])
                        # call send_email
                        success, msg = send_email(pending["recipients"], pending["subject"], pending["body"])
                        if success:
                            st.success("Email sent successfully.")
                            log_action("send_email", {"to": pending["recipients"], "subject": pending["subject"]})
                            # clear pending
                            st.session_state[PENDING_EMAIL_KEY] = None
                        else:
                            st.error(f"Email failed: {msg}")
            with col2:
                if st.button("Cancel"):
                    st.info("Alert cancelled.")
                    st.session_state[PENDING_EMAIL_KEY] = None
else:
    st.info("Upload a CSV (with 'product' and 'stocks' columns) to begin.")

# -----------------------
# Sidebar: Quick actions & logs
# -----------------------
st.sidebar.header("Quick actions")
if st.sidebar.button("Run low-stock check now"):
    if df is None:
        st.sidebar.warning("Upload CSV first.")
    else:
        try:
            lowdf = find_low_stock(df)
            if lowdf.empty:
                st.sidebar.success("No low-stock items found.")
            else:
                st.sidebar.write(lowdf)
                st.sidebar.info("Use the main UI to confirm and send alerts.")
                log_action("manual_low_check", f"found {len(lowdf)} items")
        except Exception as e:
            st.sidebar.error(str(e))

st.sidebar.markdown("---")
st.sidebar.header("Audit log (last 10)")
try:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    rows = cur.execute("SELECT ts, action, detail FROM actions ORDER BY id DESC LIMIT 10").fetchall()
    conn.close()
    for r in rows:
        st.sidebar.write(f"- {r[0]} — {r[1]} — {r[2]}")
except Exception:
    st.sidebar.write("No logs yet.")

