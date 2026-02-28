"""
Wang & Demszky replication API calls.

Replicates a study originally done with one LLM across multiple LLMs.
Supports Together, Google (Gemini), HuggingFace, and Anthropic (Claude) APIs.
"""

import os
import re
import sys

import pandas as pd
from tqdm import tqdm

# --- Configuration (from command line: python script.py TOGETHER_API_KEY MODEL) ---
together_api = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TOGETHER_API_KEY")
model = mod_name = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("MODEL", "default")

# --- Data loading ---
PREVIOUSLY_SAVED_TRANSCRIPTS = "all_transcripts_results.csv"
# OBSIDs without prior prompts per original study exclusions
EXCLUDED_OBSIDS = [706, 6, 4471, 2976]

df = pd.read_csv(PREVIOUSLY_SAVED_TRANSCRIPTS)
df = df[~df.OBSID.isin(EXCLUDED_OBSIDS)]
df["response"] = None
df["score"] = None
df["model"] = mod_name

# --- API clients (lazy imports to avoid errors when keys are not set) ---
from together import Together

client = Together(api_key=together_api)

# Optional: Uncomment and set env vars to enable alternative providers
# from openai import OpenAI
# from google import genai
# from google.genai import types
# import anthropic
# client_goog = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
# client_hf = OpenAI(base_url="https://router.huggingface.co/v1", api_key=os.environ.get("HF_API_KEY"))
# client_claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Response text extraction
# ---------------------------------------------------------------------------


def extract_response_text(response):
    """
    Extract plain text from an API response.

    Handles OpenAI-compatible format (Together, HuggingFace, etc.) and raw strings.
    Returns the content string or None if extraction fails.
    """
    if response is None:
        return None
    if isinstance(response, str):
        return response.strip()
    try:
        return response.choices[0].message.content.strip()
    except (AttributeError, IndexError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Score parsing (most robust logic consolidated from all variants)
# ---------------------------------------------------------------------------

# Rating instruction markers used in prompts; we seek the first number after these
RATING_MARKERS = [
    "Rating (only specify a number between 1-3)",
    "Rating (only specify a number between 1-7)",
    "Rating:",
]


def _parse_score_from_text(text: str, use_last_number: bool = False) -> float | None:
    """
    Parse a numeric score from text using robust heuristics.

    Args:
        text: Raw response text (should be normalized: asterisks removed, stripped)
        use_last_number: If True, prefer the last number in the text (for NR-style
            responses with reasoning). If False, prefer the first number.

    Returns:
        Parsed score as float, or None if no valid number found.
    """
    if not text or not text.strip():
        return None

    text = text.replace("*", "").strip()
    nums = re.findall(r"\d+", text)

    # If first character is a digit, that is often the intended rating
    if text[0].isdigit():
        return float(text[0])

    # Check for explicit rating markers (from prompt instructions)
    for marker in RATING_MARKERS:
        parts = text.split(marker)
        if len(parts) > 1:
            after_marker = parts[1].strip()
            found = re.findall(r"\d+", after_marker)
            if found:
                return float(found[0])

    if not nums:
        return None
    idx = -1 if use_last_number else 0
    return float(nums[idx])


def get_score_NR(text: str) -> float | None:
    """
    Parse score for Numerical Reasoning (NR) prompt style.

    NR prompts elicit longer reasoning; the rating is typically at the end or
    immediately after an explicit "Rating:"-style marker. Uses last-number
    fallback when no marker is found.
    """
    try:
        return _parse_score_from_text(text, use_last_number=True)
    except (ValueError, IndexError, TypeError):
        return None


def get_score_for_NR(response) -> float | None:
    """
    Parse NR score from an API response object or raw string.
    """
    text = extract_response_text(response)
    return get_score_NR(text) if text else None


def get_score(response) -> float | None:
    """
    Parse score for standard (non-NR) prompts.

    Prefers first number. If the response has multiple numbers and is long
    (>15 chars), falls back to NR-style parsing in case the model included
    reasoning before the score.
    """
    try:
        text = extract_response_text(response)
        if not text:
            return None

        # Optionally restrict to text after </think> if present
        if "</think>" in text:
            parts = text.split("</think>")
            if len(parts) > 1:
                text = parts[1].strip()

        text = text.replace("*", "").strip()
        nums = re.findall(r"\d+", text)

        if not nums:
            return None
        if text and text[0].isdigit():
            return float(text[0])
        if len(nums) > 1 and len(text) > 15:
            return get_score_NR(text)
        return float(nums[0])
    except (ValueError, IndexError, TypeError, AttributeError):
        return None


def get_score_str(response: str) -> float | None:
    """
    Parse score from a raw string (convenience wrapper for get_score).
    """
    return get_score(response)


# ---------------------------------------------------------------------------
# Optional: alternative API completion helpers (for multi-LLM replication)
# ---------------------------------------------------------------------------


def get_goog_completion(
    prompt,
    api_key=None,
    model="gemini-2.5-pro",
    temperature=0.0,
    max_tokens=100,
):
    """Get completion from Google Gemini API."""
    if not prompt:
        return None
    from google import genai
    from google.genai import types

    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(temperature=temperature)
    return client.models.generate_content(model=model, contents=prompt, config=config)


def get_hf_completion(
    prompt,
    base_url="https://router.huggingface.co/v1",
    api_key=None,
    model=None,
    temperature=0.0,
    max_tokens=8000,
):
    """Get completion from HuggingFace Inference API (OpenAI-compatible)."""
    if not prompt:
        return None
    from openai import OpenAI

    api_key = api_key or os.environ.get("HF_API_KEY")
    model = model or os.environ.get("MODEL", "default")
    client = OpenAI(base_url=base_url, api_key=api_key)
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=0.7,
        max_tokens=max_tokens,
        stream=False,
    )


# ---------------------------------------------------------------------------
# Main execution: run replication over dataframe
# ---------------------------------------------------------------------------


def run_replication(df, client, model, output_path=None):
    """
    Run LLM replication over rows with missing scores.

    Uses Together client by default. For other providers, swap the completion
    call and ensure responses are OpenAI-compatible or use extract_response_text.
    """
    output_path = output_path or f"all_transcripts_{model}_results.csv"
    responses = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Replicating with {model}"):
        if row["score"] is not None:
            continue

        max_tokens = 10000 if row["prompt_style"] == "NR" else 8000
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": row["prompt"]}],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=False,
        )

        if response is not None:
            text_response = extract_response_text(response)
            df.at[index, "response"] = text_response
            score_fn = get_score_for_NR if row["prompt_style"] == "NR" else get_score
            df.at[index, "score"] = score_fn(response)
            responses.append(response)

    df.to_csv(output_path, index=False)
    return df, responses


if __name__ == "__main__":
    if not together_api or not model:
        sys.exit(
            "Usage: python wd_replication_api_calls.py TOGETHER_API_KEY MODEL\n"
            "Or set TOGETHER_API_KEY and MODEL environment variables."
        )
    run_replication(df, client, model)
