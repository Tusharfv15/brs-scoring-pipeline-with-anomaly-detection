"""
tools/run_python.py
-------------------
run_python — Open-ended code execution sandbox

Allows the anomaly agent to write and execute arbitrary Python code
against the review data and scoring results.

Pre-injected variables:
    df         — pandas DataFrame of reviews (all columns)
    biz        — places JSON dict
    score      — full scoring output JSON dict
    CHARTS_DIR — Path to outputs/charts/ for saving charts

Allowed libraries:
    pandas, numpy, matplotlib, seaborn, sklearn, difflib,
    json, re, math, collections, itertools

Blocked:
    os, sys, subprocess, requests, importlib, open() outside charts dir

Returns:
    dict with stdout output, any saved chart paths, and error if any.
"""

import io
import re
import math
import json
import difflib
import traceback
import collections
import itertools
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

CHARTS_DIR = Path("outputs/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Safe open — only allows writing to outputs/charts/
# ---------------------------------------------------------------------------

def _safe_open(path, mode="r", *args, **kwargs):
    resolved = Path(path).resolve()
    charts   = CHARTS_DIR.resolve()
    if "w" in mode or "a" in mode:
        if not str(resolved).startswith(str(charts)):
            raise PermissionError(
                f"run_python can only write to {CHARTS_DIR}. "
                f"Attempted: {resolved}"
            )
    return open(resolved, mode, *args, **kwargs)


# ---------------------------------------------------------------------------
# Allowed globals for exec
# ---------------------------------------------------------------------------

def _build_globals(df: pd.DataFrame, biz: dict, score: dict) -> dict:
    return {
        # Data
        "df"        : df.copy(),
        "biz"       : biz,
        "score"     : score,
        "CHARTS_DIR": CHARTS_DIR,

        # Libraries
        "pd"        : pd,
        "np"        : np,
        "plt"       : plt,
        "sns"       : sns,
        "sklearn"   : sklearn,
        "TfidfVectorizer"   : __import__('sklearn.feature_extraction.text', fromlist=['TfidfVectorizer']).TfidfVectorizer,
        "cosine_similarity" : __import__('sklearn.metrics.pairwise', fromlist=['cosine_similarity']).cosine_similarity,
        "difflib"   : difflib,
        "json"      : json,
        "re"        : re,
        "math"      : math,
        "collections": collections,
        "itertools" : itertools,
        "Path"      : Path,

        # Safe builtins
        "print"     : print,
        "len"       : len,
        "range"     : range,
        "enumerate" : enumerate,
        "zip"       : zip,
        "map"       : map,
        "filter"    : filter,
        "sorted"    : sorted,
        "sum"       : sum,
        "min"       : min,
        "max"       : max,
        "abs"       : abs,
        "round"     : round,
        "list"      : list,
        "dict"      : dict,
        "set"       : set,
        "tuple"     : tuple,
        "str"       : str,
        "int"       : int,
        "float"     : float,
        "bool"      : bool,
        "isinstance": isinstance,
        "open"      : _safe_open,

        # Blocked — explicitly set to None so any attempt raises clear error
        "os"        : None,
        "sys"       : None,
        "subprocess": None,
        "requests"  : None,
        "importlib" : None,
        "__import__": None,
        "__builtins__": {},
    }


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def run_python(
    code: str,
    reasoning: str,
    df: pd.DataFrame,
    biz: dict,
    score: dict,
    stem: str,
    timeout: int = 30,
) -> dict:
    """
    Execute arbitrary Python code in a restricted sandbox.

    Args:
        code      : Python code string to execute
        reasoning : Agent's explanation of why it is running this code
        df        : review DataFrame
        biz       : places JSON dict
        score     : scoring output dict
        stem      : filename stem for chart naming context
        timeout   : max execution time in seconds (default 30)

    Returns:
        dict with:
            reasoning   — why the agent ran this code
            output      — captured stdout
            charts      — list of chart paths saved during execution
            error       — error message if execution failed, else None
            success     — bool
    """
    import signal

    # Capture stdout
    stdout_capture = io.StringIO()

    # Track charts saved before and after
    charts_before = set(CHARTS_DIR.glob("*.png"))

    # Timeout handler
    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Code execution exceeded {timeout}s timeout.")

    globals_dict = _build_globals(df, biz, score)
    globals_dict["stem"] = stem

    output = ""
    error  = None
    success = True

    try:
        import sys as _sys
        old_stdout = _sys.stdout
        _sys.stdout = stdout_capture

        # Set timeout (Unix only)
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)
        except (AttributeError, OSError):
            pass  # Windows doesn't support SIGALRM

        exec(code, globals_dict)

        # Cancel timeout
        try:
            signal.alarm(0)
        except (AttributeError, OSError):
            pass

        _sys.stdout = old_stdout
        output = stdout_capture.getvalue()

    except TimeoutError as e:
        _sys.stdout = old_stdout
        error   = f"TimeoutError: {str(e)}"
        success = False

    except PermissionError as e:
        _sys.stdout = old_stdout
        error   = f"PermissionError: {str(e)}"
        success = False

    except Exception as e:
        _sys.stdout = old_stdout
        if isinstance(e, (ImportError, ModuleNotFoundError)):
            error = (
                "ImportError: Do not use import statements or 'from X import Y'. "
                "Use only the pre-injected variables: df, score, biz, pd, np, plt, "
                "sns, sklearn, TfidfVectorizer, cosine_similarity, difflib, json, "
                "re, math, collections, itertools, Path, CHARTS_DIR."
            )
        else:
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        success = False

    finally:
        plt.close("all")

    # Detect any new charts saved
    charts_after = set(CHARTS_DIR.glob("*.png"))
    new_charts   = [str(p) for p in charts_after - charts_before]

    return {
        "reasoning": reasoning,
        "output"   : output.strip(),
        "charts"   : new_charts,
        "error"    : error,
        "success"  : success,
    }