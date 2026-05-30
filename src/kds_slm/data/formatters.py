"""Dataset schema normalization utilities."""

from __future__ import annotations

import json
from typing import Any


def _first_user_message(messages: list[dict[str, Any]]) -> str | None:
    for msg in messages:
        role = msg.get("role", "").lower()
        if role in ("user", "patient", "customer", "human"):
            content = msg.get("content") or msg.get("text") or msg.get("message")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def _last_assistant_message(messages: list[dict[str, Any]]) -> str | None:
    answer = None
    for msg in messages:
        role = msg.get("role", "").lower()
        if role in ("assistant", "doctor", "agent", "gpt", "bot"):
            content = msg.get("content") or msg.get("text") or msg.get("message")
            if isinstance(content, str) and content.strip():
                answer = content.strip()
    return answer


def _parse_messages_field(value: Any) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value.strip(), None
    if isinstance(value, list):
        return _first_user_message(value), _last_assistant_message(value)
    return None, None


def _parse_dialogue_field(value: Any) -> tuple[str | None, str | None]:
    if not isinstance(value, list):
        return None, None
    prompt = _first_user_message(value)
    answer = _last_assistant_message(value)
    return prompt, answer


def normalize_row(row: dict[str, Any], row_id: str) -> dict[str, str] | None:
    """Normalize a dataset row into {id, prompt, reference_answer}."""
    prompt: str | None = None
    reference: str | None = None

    if "messages" in row:
        prompt, reference = _parse_messages_field(row["messages"])
    elif "dialogue" in row:
        prompt, reference = _parse_dialogue_field(row["dialogue"])
    elif "conversation" in row:
        prompt, reference = _parse_messages_field(row["conversation"])

    field_pairs = [
        ("patient_message", "doctor_response"),
        ("question", "answer"),
        ("input", "output"),
        ("prompt", "response"),
        ("query", "response"),
        ("instruction", "output"),
        ("user", "assistant"),
    ]
    for p_key, r_key in field_pairs:
        if prompt is None and p_key in row and isinstance(row[p_key], str):
            prompt = row[p_key].strip()
        if reference is None and r_key in row and isinstance(row[r_key], str):
            reference = row[r_key].strip()

    if prompt is None:
        for key in ("text", "content", "description", "patient", "Description", "Patient"):
            if key in row and isinstance(row[key], str) and row[key].strip():
                prompt = row[key].strip()
                break

    if reference is None:
        for key in ("response", "doctor", "Doctor", "label", "target"):
            if key in row and isinstance(row[key], str) and row[key].strip():
                reference = row[key].strip()
                break

    if not prompt:
        return None

    return {
        "id": row_id,
        "prompt": prompt,
        "reference_answer": reference or "",
    }


def build_chat_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages
