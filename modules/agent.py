# agents.py
import pandas as pd

def simple_local_agent(df, question):
    q = question.lower().strip()
    response_texts = []

    try:
        if "sales" in df.columns:
            total_sales = df["sales"].sum()
            mean_sales = df["sales"].mean()
            max_idx = df["sales"].idxmax()
            max_row = df.loc[max_idx].to_dict() if pd.notna(max_idx) else None
            response_texts.append(f"Total sales: {total_sales}")
            response_texts.append(f"Average sales: {mean_sales:.2f}")
            if max_row:
                if "date" in df.columns:
                    response_texts.append(f"Highest sales: {max_row.get('sales')} on {max_row.get('date')}")
                else:
                    response_texts.append(f"Highest sales: {max_row.get('sales')} (row {max_idx})")

        if "product" in df.columns:
            top = df.groupby("product")["sales"].sum().sort_values(ascending=False)
            if not top.empty:
                prod, val = top.index[0], top.iloc[0]
                response_texts.append(f"Top product: {prod} (total {val})")
    except:
        pass

    # simple question parsing
    if "top" in q or "best" in q:
        if "product" in df.columns:
            top = df.groupby("product")["sales"].sum().sort_values(ascending=False)
            if not top.empty:
                prod, val = top.index[0], top.iloc[0]
                return f"Top product: {prod} (total {val})"
        return "No product data available."

    if "total" in q and "sales" in q:
        return f"Total sales = {df['sales'].sum()}"

    if "highest" in q:
        idx = df["sales"].idxmax()
        row = df.loc[idx]
        if "date" in df.columns:
            return f"Highest sales = {row['sales']} on {row['date']}"
        return f"Highest sales = {row['sales']} at row {idx}"

    return "Quick stats:\n- " + "\n- ".join(response_texts)
