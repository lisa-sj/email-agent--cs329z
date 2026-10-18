"""Helpers shared by the Part 1 pipelines."""

from __future__ import annotations

from datetime import datetime

from cs329z_hw1.llm import OutputTruncated
from cs329z_hw1.types import Email, LMCallable


def call_lm(lm: LMCallable, messages: list[dict]) -> str:
    """Call the model and always return a string.

    OutputTruncated (the model hit max_tokens) is treated as a reply made of
    whatever text was written so far; the parsers downstream then decide
    whether that text is usable. Other exceptions (budget, network) propagate.
    """
    try:
        reply = lm(messages)
    except OutputTruncated as exc:
        reply = exc.text or ""
    return reply if isinstance(reply, str) else ""


def weekday_date(date: str) -> str:
    """'2001-06-22T13:42:00Z' -> 'Friday 2001-06-22 13:42 UTC' (falls back to the raw string)."""
    try:
        dt = datetime.strptime(date[:19], "%Y-%m-%dT%H:%M:%S")
        return dt.strftime("%A %Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError):
        return str(date or "unknown")


def clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "\n[... truncated ...]"


def render_email(email: Email, body_chars: int = 4000, with_id: bool = False) -> str:
    """A readable header + body block for one email, for use inside a prompt."""
    to = email.get("to") or []
    to_line = ", ".join(to[:8]) + (f", ... ({len(to)} recipients in total)" if len(to) > 8 else "")
    lines = []
    if with_id:
        lines.append(f"Email id: {email.get('id', '')}")
    lines += [
        f"From: {email.get('from') or '(unknown)'}",
        f"To: {to_line or '(no recipients listed)'}",
        f"Date: {weekday_date(email.get('date', ''))}",
        f"Subject: {(email.get('subject') or '').strip() or '(no subject)'}",
        "",
        clip(email.get("body", ""), body_chars) or "(empty body)",
    ]
    return "\n".join(lines)
