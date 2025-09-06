from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.tools import tool

DATA_DIR = Path(__file__).resolve().parent / "data"
SALES_CSV = DATA_DIR / "sales.csv"
INV_CSV = DATA_DIR / "inventory.csv"


def _load_sales() -> pd.DataFrame:
    df = pd.read_csv(SALES_CSV, parse_dates=["date"]) if SALES_CSV.exists() else pd.DataFrame()
    # computed column
    if not df.empty:
        df["revenue"] = df["unit_price"] * df["quantity"]
    return df


def _load_inventory() -> pd.DataFrame:
    return pd.read_csv(INV_CSV) if INV_CSV.exists() else pd.DataFrame()


@tool("retail_sales_summary")
def retail_sales_summary(params_json: str) -> str:
    """Summarize sales for a date range. Input JSON fields:
    {"start": "YYYY-MM-DD" (optional), "end": "YYYY-MM-DD" (optional), "top_n": int}
    Returns JSON with totals and top SKUs/categories.
    """
    try:
        params = json.loads(params_json or "{}")
    except json.JSONDecodeError:
        params = {}
    top_n = int(params.get("top_n", 5))
    start = pd.to_datetime(params.get("start")) if params.get("start") else None
    end = pd.to_datetime(params.get("end")) if params.get("end") else None

    df = _load_sales()
    if df.empty:
        return json.dumps({"error": "no_sales_data"})
    if start is not None:
        df = df[df["date"] >= start]
    if end is not None:
        df = df[df["date"] <= end]

    totals = {
        "orders": int(df["order_id"].nunique()),
        "units": int(df["quantity"].sum()),
        "revenue": float(df["revenue"].sum()),
    }
    top_skus = (
        df.groupby(["sku"], as_index=False)[["revenue", "quantity"]].sum()
          .sort_values("revenue", ascending=False).head(top_n)
          .to_dict(orient="records")
    )
    top_categories = (
        df.groupby(["category"], as_index=False)[["revenue", "quantity"]].sum()
          .sort_values("revenue", ascending=False).head(top_n)
          .to_dict(orient="records")
    )
    result = {
        "totals": totals,
        "top_skus": top_skus,
        "top_categories": top_categories,
    }
    return json.dumps(result)


@tool("retail_inventory_status")
def retail_inventory_status(_: str = "") -> str:
    """Return low-stock items (on_hand <= reorder_point). Input ignored. Returns JSON."""
    inv = _load_inventory()
    if inv.empty:
        return json.dumps({"error": "no_inventory_data"})
    low = inv[inv["on_hand"] <= inv["reorder_point"]].copy()
    result = {
        "low_stock": low.to_dict(orient="records"),
        "total_skus": int(inv.shape[0]),
        "low_count": int(low.shape[0]),
    }
    return json.dumps(result)


@tool("retail_price_optimize")
def retail_price_optimize(params_json: str) -> str:
    """Suggest price within ±10% that maximizes revenue using simple elasticity.
    Input JSON: {"skus": ["SKU-001", ...], "elasticity": -1.2}
    Returns JSON with suggested price and expected revenue delta per SKU.
    """
    try:
        params = json.loads(params_json or "{}")
    except json.JSONDecodeError:
        params = {}
    elasticity = float(params.get("elasticity", -1.2))
    sel = set(params.get("skus") or [])

    inv = _load_inventory()
    if inv.empty:
        return json.dumps({"error": "no_inventory_data"})

    # If no selection, optimize top 5 by price
    if not sel:
        sel = set(inv.sort_values("unit_price", ascending=False).head(5)["sku"].tolist())

    results = []
    # Baseline demand proxy: average daily quantity from sales data if available, else heuristic
    sales = _load_sales()
    for _, row in inv[inv["sku"].isin(sel)].iterrows():
        sku = row["sku"]
        p0 = float(row["unit_price"])
        # approximate q0 from sales data
        if not sales.empty:
            q0 = max(1.0, sales[sales["sku"] == sku]["quantity"].mean())
        else:
            q0 = 5.0

        # Grid search ±10%
        grid = np.linspace(0.9 * p0, 1.1 * p0, 21)
        best = {"price": p0, "revenue": p0 * q0}
        for p in grid:
            # Constant elasticity demand: q = q0 * (p/p0)^elasticity
            q = q0 * (p / p0) ** elasticity
            r = p * q
            if r > best["revenue"]:
                best = {"price": float(p), "revenue": float(r)}
        results.append({
            "sku": sku,
            "current_price": p0,
            "suggested_price": round(best["price"], 2),
            "revenue_baseline": round(p0 * q0, 2),
            "revenue_suggested": round(best["revenue"], 2),
            "delta": round(best["revenue"] - (p0 * q0), 2),
        })
    return json.dumps({"pricing": results, "assumptions": {"elasticity": elasticity, "band": "+/-10%"}})


@tool("retail_markdown_report")
def retail_markdown_report(params_json: str) -> str:
    """Build a markdown report from gathered findings.
    Input JSON keys can include: totals, top_skus, top_categories, low_stock, pricing, assumptions.
    Returns a markdown string.
    """
    try:
        p = json.loads(params_json or "{}")
    except json.JSONDecodeError:
        p = {}
    md = ["# Retail Summary Report\n"]
    if p.get("totals"):
        t = p["totals"]
        md += ["## Overview", f"- Orders: {t.get('orders')}", f"- Units: {t.get('units')}", f"- Revenue: ${t.get('revenue', 0):,.2f}", ""]
    if p.get("top_skus"):
        md += ["## Top SKUs"]
        for r in p["top_skus"][:10]:
            md.append(f"- {r['sku']}: ${r['revenue']:,.2f} | units={int(r['quantity'])}")
        md.append("")
    if p.get("top_categories"):
        md += ["## Top Categories"]
        for r in p["top_categories"][:10]:
            md.append(f"- {r['category']}: ${r['revenue']:,.2f} | units={int(r['quantity'])}")
        md.append("")
    if p.get("low_stock"):
        md += ["## Low Stock Alerts"]
        for r in p["low_stock"][:15]:
            md.append(f"- {r['sku']} (on_hand={r['on_hand']}, ROP={r['reorder_point']})")
        md.append("")
    if p.get("pricing"):
        md += ["## Pricing Suggestions"]
        for r in p["pricing"][:10]:
            md.append(
                f"- {r['sku']}: {r['current_price']} -> {r['suggested_price']} | "
                f"ΔRev=${r['delta']:,.2f}")
        if p.get("assumptions"):
            md.append(f"Assumptions: elasticity={p['assumptions'].get('elasticity')} within {p['assumptions'].get('band')}")
        md.append("")
    if len(md) == 1:
        md.append("(No findings)")
    return "\n".join(md)


__all__ = [
    "retail_sales_summary",
    "retail_inventory_status",
    "retail_price_optimize",
    "retail_markdown_report",
]

