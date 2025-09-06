from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import json
from langchain_core.tools import tool


# ----------------------- Calculator (safe) -----------------------

_ALLOWED_AST_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.FloorDiv,
    ast.Num,  # Py<3.8
    ast.Load,
    ast.Constant,  # Py>=3.8
    ast.Call,  # allow specific functions like round(...)
    ast.Name,
}

_ALLOWED_NAMES = {
    "pi": 3.141592653589793,
    "e": 2.718281828459045,
    "round": round,
    "abs": abs,
    "min": min,
    "max": max,
}


def _safe_eval(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_AST_NODES:
            raise ValueError(f"Disallowed expression node: {type(node).__name__}")
        if isinstance(node, ast.Call):
            # Only allow calling known safe names
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_NAMES:
                raise ValueError("Only safe builtins allowed: round, abs, min, max")
    return eval(compile(tree, filename="<expr>", mode="eval"), {"__builtins__": {}}, _ALLOWED_NAMES)


@tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate a basic math expression safely (supports +,-,*,/,**,%, floor division, round, abs, min, max).

    Args:
        expression: arithmetic expression, e.g., "(2+3*4)/5" or "round(1.234, 2)".
    """
    try:
        value = _safe_eval(expression)
        return str(value)
    except Exception as e:
        return f"Calculator error: {e}"


# ----------------------- FAQ Lookup -----------------------

_FAQ_PATH = Path(__file__).parent / "data" / "faq.json"


def _load_faq() -> List[Dict[str, Any]]:
    if not _FAQ_PATH.exists():
        return []
    with open(_FAQ_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _score(query: str, text: str) -> float:
    """Very simple keyword overlap score for demo purposes."""
    q = {t for t in query.lower().split() if len(t) > 2}
    d = {t for t in text.lower().split() if len(t) > 2}
    if not q:
        return 0.0
    inter = len(q & d)
    return inter / len(q)


@tool("faq_lookup")
def faq_lookup(question: str) -> str:
    """Answer from a curated FAQ knowledge base. Input should be a short question.

    Returns an answer with a brief rationale and the matched question when available.
    """
    faqs = _load_faq()
    if not faqs:
        return "FAQ not available."
    best = None
    best_score = 0.0
    for item in faqs:
        s = max(_score(question, item.get("q", "")), _score(question, item.get("a", "")))
        if s > best_score:
            best_score = s
            best = item
    if best and best_score >= 0.2:
        return f"MatchScore={best_score:.2f}\nQ: {best.get('q')}\nA: {best.get('a')}"
    return "No good match found in FAQ. Try rephrasing or provide more context."


__all__ = ["calculator", "faq_lookup"]

