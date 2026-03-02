#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Advanced benchmark plot and report generator for IME performance suites.
#
# Usage:
#   python plot_metrics.py --run-dir <run_dir> --out-dir <out_dir>
#
# Example:
#   python plot_metrics.py --run-dir . --out-dir plots

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit("matplotlib import failed: {0}. Install with: pip install matplotlib".format(exc))


SCI_BLUE = "#0B5FA5"
SCI_ORANGE = "#E68613"
SCI_BRICK = "#B24745"
SCI_GREEN = "#1B9E77"
SCI_PURPLE = "#7B61A8"
SCI_TEAL = "#2A9D8F"
SCI_GRAY = "#6B6B6B"
SCI_TEXT = "#222222"
SCI_TEXT_MUTED = "#4F4F4F"
SCI_GRID = "#E5E5E5"
SCI_BG = "#FFFFFF"

# Color-blind friendly palette (Okabe-Ito inspired), anchored by deep science blue/orange.
COLORS = [
    SCI_BLUE,
    SCI_ORANGE,
    SCI_BRICK,
    SCI_GREEN,
    SCI_PURPLE,
    SCI_TEAL,
    SCI_GRAY,
]
MARKERS = ["o", "s", "^", "D", "P", "X", "v"]
DEFAULT_CENTER_STAT = "median"
DEFAULT_BAND_STAT = "iqr"  # P25-P75
DEFAULT_INVALID_DECODE_TOKEN_RATIO = 0.60
DEFAULT_INVALID_DECODE_MIN_BUDGET = 12


def setup_style():
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "axes.titleweight": "bold",
            "figure.facecolor": SCI_BG,
            "axes.facecolor": SCI_BG,
            "savefig.facecolor": SCI_BG,
            "axes.edgecolor": SCI_TEXT,
            "axes.labelcolor": SCI_TEXT,
            "xtick.color": SCI_TEXT,
            "ytick.color": SCI_TEXT,
            "axes.linewidth": 0.9,
            "grid.color": SCI_GRID,
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.55,
            "legend.frameon": False,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )


def style_axis(ax):
    ax.set_facecolor(SCI_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(SCI_TEXT)
        ax.spines[side].set_alpha(0.35)
        ax.spines[side].set_linewidth(0.9)
    ax.grid(True, axis="y", color=SCI_GRID, linewidth=0.8, alpha=0.55)
    ax.grid(True, axis="x", color=SCI_GRID, linewidth=0.6, alpha=0.30)
    ax.tick_params(axis="both", which="major", length=4, width=0.8, colors=SCI_TEXT)
    ax.title.set_color(SCI_TEXT)
    ax.xaxis.label.set_color(SCI_TEXT)
    ax.yaxis.label.set_color(SCI_TEXT)


def cycle_colors(n, offset=0):
    return [COLORS[(i + offset) % len(COLORS)] for i in range(max(0, n))]


def is_number(v):
    if isinstance(v, bool):
        return False
    if not isinstance(v, (int, float)):
        return False
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return False
    return True


def to_float(v):
    if not is_number(v):
        return None
    return float(v)


def to_int(v):
    fv = to_float(v)
    if fv is None:
        return None
    return int(round(fv))


def format_num(v, digits=2):
    if v is None:
        return "-"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "-"
    fmt = "{0:." + str(digits) + "f}"
    return fmt.format(float(v))


def parse_bucket_num(text):
    m = re.search(r"(\d+)", str(text))
    return int(m.group(1)) if m else None


def read_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            try:
                rows.append(json.loads(t))
            except Exception:
                continue
    return rows


def save_figure(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def annotate_points(ax, xs, ys, color=SCI_TEXT_MUTED, digits=1, yoffset=6):
    for x, y in zip(xs, ys):
        if not is_number(y):
            continue
        ax.annotate(
            format_num(y, digits),
            (x, y),
            textcoords="offset points",
            xytext=(0, yoffset),
            ha="center",
            fontsize=9,
            color=color,
        )


def percentile(values, q):
    vals = sorted(float(v) for v in values if is_number(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    idx = (len(vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return vals[lo]
    frac = idx - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def rolling_median(values, window=3):
    vals = list(values or [])
    if not vals:
        return []
    w = max(1, int(window))
    if (w % 2) == 0:
        w += 1
    half = w // 2
    out = []
    for i in range(len(vals)):
        lo = max(0, i - half)
        hi = min(len(vals), i + half + 1)
        seg = [float(v) for v in vals[lo:hi] if is_number(v)]
        out.append(percentile(seg, 0.50) if seg else None)
    return out


def rolling_mean(values, window=3):
    vals = list(values or [])
    if not vals:
        return []
    w = max(1, int(window))
    if (w % 2) == 0:
        w += 1
    half = w // 2
    out = []
    for i in range(len(vals)):
        lo = max(0, i - half)
        hi = min(len(vals), i + half + 1)
        seg = [float(v) for v in vals[lo:hi] if is_number(v)]
        out.append((sum(seg) / len(seg)) if seg else None)
    return out


def calc_stats(values):
    vals = [float(v) for v in values if is_number(v)]
    if not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    else:
        var = 0.0
    std = math.sqrt(max(0.0, var))
    return {
        "n": n,
        "avg": mean,
        "std": std,
        "min": min(vals),
        "max": max(vals),
        "p50": percentile(vals, 0.50),
        "p90": percentile(vals, 0.90),
        "p99": percentile(vals, 0.99),
    }


def get_styles(summary):
    styles = summary.get("results_by_style", {})
    return styles if isinstance(styles, dict) else {}


def collect_prefill(summary):
    out = {}
    for style, payload in get_styles(summary).items():
        rows = []
        for item in payload.get("prefill_scaling") or payload.get("prefill_buckets") or []:
            x = parse_bucket_num(item.get("bucket"))
            ms = to_float(item.get("prefill_ms"))
            tps = to_float(item.get("prefill_tps"))
            tokens = to_float(item.get("prefill_tokens"))
            if x is None or ms is None:
                continue
            rows.append(
                {
                    "x": x,
                    "ms": ms,
                    "tps": tps,
                    "tokens": tokens,
                    "ms_per_token": (ms / tokens) if (tokens is not None and tokens > 0.0) else None,
                }
            )
        rows.sort(key=lambda r: r["x"])
        if rows:
            out[style] = rows
    return out


def resolve_decode_invalid_rule(summary):
    cfg = summary.get("decode") if isinstance(summary, dict) else {}
    if not isinstance(cfg, dict):
        cfg = {}
    ratio = to_float(cfg.get("invalid_token_ratio"))
    if ratio is None or ratio <= 0.0:
        ratio = DEFAULT_INVALID_DECODE_TOKEN_RATIO
    min_budget = to_int(cfg.get("invalid_min_budget"))
    if min_budget is None or min_budget <= 0:
        min_budget = DEFAULT_INVALID_DECODE_MIN_BUDGET
    return {
        "ratio": float(ratio),
        "min_budget": int(min_budget),
        "rule_text": "decode_tokens < {0:.2f} * step_budget (step_budget >= {1})".format(float(ratio), int(min_budget)),
    }


def decode_row_invalid_info(row, ratio, min_budget):
    explicit_invalid = row.get("decode_sample_invalid")
    if isinstance(explicit_invalid, bool):
        reason = row.get("decode_invalid_reason")
        threshold = to_float(row.get("decode_invalid_threshold_tokens"))
        budget = to_int(row.get("decode_step_budget"))
        if budget is None:
            budget = parse_bucket_num(row.get("bucket"))
        tokens = to_float(row.get("decode_tokens"))
        return {
            "invalid": explicit_invalid,
            "reason": str(reason) if reason not in (None, "", "null") else None,
            "budget": budget,
            "tokens": tokens,
            "threshold": threshold,
        }

    budget = to_int(row.get("decode_step_budget"))
    if budget is None:
        budget = parse_bucket_num(row.get("bucket"))
    tokens = to_float(row.get("decode_tokens"))
    if budget is None or tokens is None or budget < int(min_budget):
        return {
            "invalid": False,
            "reason": None,
            "budget": budget,
            "tokens": tokens,
            "threshold": (float(budget) * ratio) if budget is not None else None,
        }
    threshold = float(budget) * float(ratio)
    invalid = float(tokens) < threshold
    reason = None
    if invalid:
        reason = "decode_tokens({0:.1f}) < {1:.2f}*step_budget({2})".format(float(tokens), float(ratio), int(budget))
    return {
        "invalid": invalid,
        "reason": reason,
        "budget": budget,
        "tokens": tokens,
        "threshold": threshold,
    }


def collect_decode_validity(raw_rows, ratio, min_budget):
    by_style = defaultdict(lambda: {"total": 0, "valid": 0, "invalid": 0, "rows": []})
    for row in raw_rows:
        if row.get("tag") != "decode":
            continue
        style = str(row.get("style_mode") or "unknown")
        info = decode_row_invalid_info(row, ratio=ratio, min_budget=min_budget)
        stat = by_style[style]
        stat["total"] += 1
        if info["invalid"]:
            stat["invalid"] += 1
            stat["rows"].append(
                {
                    "style": style,
                    "bucket": str(row.get("bucket") or ""),
                    "iteration": to_int(row.get("iteration")) or 0,
                    "decode_tokens": info["tokens"],
                    "step_budget": info["budget"],
                    "threshold": info["threshold"],
                    "decode_control_mode": str(row.get("decode_control_mode") or ""),
                    "reason": info["reason"] or "invalid_decode_sample",
                }
            )
        else:
            stat["valid"] += 1

    all_total = sum(v["total"] for v in by_style.values())
    all_valid = sum(v["valid"] for v in by_style.values())
    all_invalid = sum(v["invalid"] for v in by_style.values())
    all_rows = []
    for v in by_style.values():
        all_rows.extend(v["rows"])
    all_rows.sort(
        key=lambda r: (
            (float(r.get("decode_tokens", 0.0)) / float(r.get("step_budget", 1.0))) if (is_number(r.get("decode_tokens")) and is_number(r.get("step_budget")) and float(r.get("step_budget", 0.0)) > 0.0) else 1.0,
            int(r.get("step_budget") or 0),
            int(r.get("iteration") or 0),
        )
    )
    return {
        "by_style": by_style,
        "all": {
            "total": all_total,
            "valid": all_valid,
            "invalid": all_invalid,
            "rows": all_rows,
        },
    }


def collect_decode(raw_rows):
    out = defaultdict(list)
    for row in raw_rows:
        if row.get("tag") != "decode":
            continue
        style = str(row.get("style_mode") or "unknown")
        x = None
        budget = to_float(row.get("decode_step_budget"))
        if budget is not None and budget > 0:
            x = int(round(budget))
        if x is None:
            x = parse_bucket_num(row.get("bucket"))
        if x is None:
            continue

        ms = to_float(row.get("decode_ms"))
        tps = to_float(row.get("decode_tps"))
        tokens = to_float(row.get("decode_tokens"))
        if ms is not None and ms < 0.0:
            ms = None
        if tps is not None and tps < 0.0:
            tps = None
        if tokens is not None and tokens <= 0.0:
            tokens = None

        per_token = None
        if ms is not None and tokens is not None:
            per_token = ms / tokens
        else:
            per_token_raw = to_float(row.get("decode_per_token_ms"))
            if per_token_raw is not None and per_token_raw >= 0.0:
                per_token = per_token_raw

        # Drop failed/empty decode samples that carry no usable metric value.
        if ms is None and tps is None and per_token is None:
            continue
        out[style].append({"x": x, "ms": ms, "tps": tps, "ms_per_token": per_token})

    normalized = {}
    for style, rows in out.items():
        rows.sort(key=lambda r: r["x"])
        if rows:
            normalized[style] = rows
    return normalized


def collect_decode_by_token_bucket(raw_rows):
    out = defaultdict(list)
    for row in raw_rows:
        if row.get("tag") != "decode":
            continue
        style = str(row.get("style_mode") or "unknown")
        tokens = to_float(row.get("decode_tokens"))
        tps = to_float(row.get("decode_tps"))
        if tokens is None or tokens <= 0.0:
            continue
        if tps is None or tps < 0.0:
            continue
        x = int(round(tokens))
        out[style].append({"x": x, "tps": float(tps)})

    normalized = {}
    for style, rows in out.items():
        rows.sort(key=lambda r: r["x"])
        if rows:
            normalized[style] = rows
    return normalized


def extract_ttft_value(row):
    ttft = to_float(row.get("ttft_ms"))
    if ttft is not None and ttft >= 0.0:
        return ttft
    prefill = to_float(row.get("prefill_ms"))
    lat = row.get("decode_latencies")
    if prefill is not None and isinstance(lat, list) and lat:
        d0 = to_float(lat[0])
        if d0 is not None and d0 >= 0.0:
            return prefill + d0
    return None


def extract_ttft_from_row(row):
    return extract_ttft_value(row)


def collect_e2e_ttft_by_style(raw_rows, bucket_filter=None):
    by_style = defaultdict(list)
    bucket_order = {"cold": 0, "warm": 1}
    for row in raw_rows:
        if row.get("tag") != "e2e":
            continue
        style = str(row.get("style_mode") or "unknown")
        bucket = str(row.get("bucket") or "")
        if bucket_filter and bucket != bucket_filter:
            continue
        it = int(row.get("iteration") or 0)
        e2e = to_float(row.get("e2e_ms"))
        ttft = extract_ttft_from_row(row)
        by_style[style].append(
            {
                "bucket": bucket,
                "iter": it,
                "e2e_ms": e2e,
                "ttft_ms": ttft,
                "sample_label": "{0}_{1}".format(bucket if bucket else "sample", it + 1),
            }
        )
    for style in list(by_style.keys()):
        by_style[style].sort(key=lambda r: (bucket_order.get(r["bucket"], 99), r["iter"]))
    return by_style


def collect_context_sync_by_style(summary):
    out = {}
    for style, payload in get_styles(summary).items():
        rows = []
        for item in payload.get("ime_first") or payload.get("context_sync") or []:
            x = parse_bucket_num(item.get("bucket"))
            first_candidate_ms = to_float(item.get("ime_first_candidate_ms"))
            if first_candidate_ms is None:
                ttft_ms = to_float(item.get("ttft_ms"))
                first_candidate_ms = ttft_ms
            if first_candidate_ms is None:
                # Backward-compat with old context_sync payload.
                load_ms = to_float(item.get("context_sync_load_ms"))
                infer_ttft_ms = to_float(item.get("context_sync_infer_ttft_ms"))
                if load_ms is not None and infer_ttft_ms is not None and load_ms >= 0.0 and infer_ttft_ms >= 0.0:
                    first_candidate_ms = load_ms + infer_ttft_ms
            if first_candidate_ms is not None and first_candidate_ms < 0.0:
                first_candidate_ms = None
            rows.append(
                {
                    "x": x,
                    "bucket": str(item.get("bucket") or ""),
                    "first_candidate_ms": first_candidate_ms,
                }
            )
        rows = [r for r in rows if r["x"] is not None]
        rows.sort(key=lambda r: r["x"])
        if rows:
            out[style] = rows
    return out


def collect_ttft_history_by_style(raw_rows, summary):
    out = defaultdict(list)
    for row in raw_rows:
        if row.get("tag") != "ime_first":
            continue
        style = str(row.get("style_mode") or "unknown")
        x = parse_bucket_num(row.get("bucket"))
        if x is None:
            continue
        ttft_ms = extract_ttft_value(row)
        if ttft_ms is None or ttft_ms < 0.0:
            continue
        out[style].append({"x": x, "bucket": str(row.get("bucket") or ""), "ttft_ms": ttft_ms})

    normalized = {}
    for style, rows in out.items():
        rows.sort(key=lambda r: r["x"])
        if rows:
            normalized[style] = rows
    if normalized:
        return normalized

    for style, payload in get_styles(summary).items():
        rows = []
        for item in payload.get("ime_first") or payload.get("context_sync") or []:
            x = parse_bucket_num(item.get("bucket"))
            if x is None:
                continue
            ttft_ms = extract_ttft_value(item)
            if ttft_ms is None:
                load_ms = to_float(item.get("context_sync_load_ms"))
                infer_ttft_ms = to_float(item.get("context_sync_infer_ttft_ms"))
                if load_ms is not None and infer_ttft_ms is not None and load_ms >= 0.0 and infer_ttft_ms >= 0.0:
                    ttft_ms = load_ms + infer_ttft_ms
            if ttft_ms is None or ttft_ms < 0.0:
                continue
            rows.append({"x": x, "bucket": str(item.get("bucket") or ""), "ttft_ms": ttft_ms})
        rows.sort(key=lambda r: r["x"])
        if rows:
            normalized[style] = rows
    return normalized


def build_envelope_from_points(points_by_x, higher_is_better):
    envelope = []
    for x in sorted(points_by_x):
        vals = [float(v) for v in points_by_x[x] if is_number(v)]
        if not vals:
            continue
        n = len(vals)
        mean = sum(vals) / n
        median = percentile(vals, 0.50)
        p25 = percentile(vals, 0.25)
        p75 = percentile(vals, 0.75)
        if median is None:
            median = mean

        if DEFAULT_BAND_STAT == "std":
            std = 0.0
            if n > 1:
                var = sum((v - mean) ** 2 for v in vals) / (n - 1)
                std = math.sqrt(max(var, 0.0))
            lower = mean - std
            upper = mean + std
            center = mean if DEFAULT_CENTER_STAT == "mean" else median
        else:
            lower = p25 if p25 is not None else min(vals)
            upper = p75 if p75 is not None else max(vals)
            center = mean if DEFAULT_CENTER_STAT == "mean" else median
        if lower > upper:
            lower, upper = upper, lower
        envelope.append(
            {
                "x": x,
                "center": center,
                "lower": lower,
                "upper": upper,
                "n": n,
                "mean": mean,
                "median": median,
                "min": min(vals),
                "max": max(vals),
            }
        )
    return envelope


def envelope_from_grouped_rows(grouped_rows, value_key, higher_is_better):
    points_by_x = defaultdict(list)
    for rows in grouped_rows.values():
        for row in rows:
            x = row.get("x")
            v = row.get(value_key)
            if x is None or not is_number(v):
                continue
            points_by_x[x].append(float(v))
    return build_envelope_from_points(points_by_x, higher_is_better)


def plot_envelope_band(ax, envelope, digits, higher_is_better, center_stat=DEFAULT_CENTER_STAT):
    if not envelope:
        return False
    xs = [p["x"] for p in envelope]
    center_mode = "mean" if str(center_stat).strip().lower() == "mean" else "median"
    center_ys = []
    for p in envelope:
        v = p.get("mean") if center_mode == "mean" else p.get("median")
        if not is_number(v):
            v = p.get("center")
        center_ys.append(v)
    lower = [p["lower"] for p in envelope]
    upper = [p["upper"] for p in envelope]
    band_label = "P25-P75" if DEFAULT_BAND_STAT == "iqr" else "Mean +/- Std"
    center_label = "Mean" if center_mode == "mean" else "Median"
    if center_mode == "mean":
        smooth_center = rolling_mean(center_ys, window=3)
        smooth_label = "Rolling Mean (w=3, visual aid)"
    else:
        smooth_center = rolling_median(center_ys, window=3)
        smooth_label = "Rolling Median (w=3, visual aid)"

    ax.fill_between(xs, lower, upper, color=SCI_BLUE, alpha=0.10, label=band_label)
    ax.plot(xs, center_ys, color=SCI_BLUE, linestyle="-", marker="o", linewidth=2.3, markersize=6.0, label=center_label)
    if len(smooth_center) == len(center_ys) and len(center_ys) >= 3:
        ax.plot(
            xs,
            smooth_center,
            color=SCI_ORANGE,
            linestyle="--",
            linewidth=1.6,
            alpha=0.95,
            label=smooth_label,
        )
    annotate_points(ax, xs, center_ys, color=SCI_BLUE, digits=digits, yoffset=7)

    sample_counts = [int(p.get("n", 0)) for p in envelope if is_number(p.get("n"))]
    if sample_counts:
        min_n = min(sample_counts)
        max_n = max(sample_counts)
        med_n = int(round(percentile(sample_counts, 0.50) or sample_counts[0]))
        ax.text(
            0.99,
            0.98,
            "n per x: median={0}, min={1}, max={2}".format(med_n, min_n, max_n),
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=9,
            color=SCI_TEXT_MUTED,
        )

    if higher_is_better:
        ax.text(
            0.01,
            0.98,
            "{0} with {1}; higher is better".format(center_label, band_label),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color=SCI_TEXT_MUTED,
        )
    else:
        ax.text(
            0.01,
            0.98,
            "{0} with {1}; lower is better".format(center_label, band_label),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color=SCI_TEXT_MUTED,
        )
    return True


def build_percentile_envelope(values_by_style, percentiles, higher_is_better):
    points_by_q = defaultdict(list)
    for vals in values_by_style.values():
        clean = [float(v) for v in vals if is_number(v)]
        if not clean:
            continue
        for q in percentiles:
            pv = percentile(clean, q / 100.0)
            if is_number(pv):
                points_by_q[q].append(float(pv))
    return build_envelope_from_points(points_by_q, higher_is_better)


def detect_global_extreme_points(grouped_rows, value_key, robust_z_threshold=3.5):
    points = []
    values = []
    for rows in grouped_rows.values():
        for row in rows:
            x = row.get("x")
            v = row.get(value_key)
            if x is None or not is_number(v):
                continue
            fv = float(v)
            points.append({"x": x, "value": fv, "bucket": row.get("bucket")})
            values.append(fv)
    if not values:
        return [], {}
    med = percentile(values, 0.50)
    abs_dev = [abs(v - med) for v in values] if med is not None else []
    mad = percentile(abs_dev, 0.50) if abs_dev else None
    p99 = percentile(values, 0.99)

    outliers = []
    for p in points:
        rz = 0.0
        if mad is not None and mad > 0:
            rz = 0.6745 * (p["value"] - med) / mad
        if abs(rz) >= robust_z_threshold or (is_number(p99) and p["value"] >= p99):
            outliers.append(
                {
                    "x": p["x"],
                    "value": p["value"],
                    "bucket": p.get("bucket"),
                    "robust_z": rz,
                }
            )
    dist = {"median": med, "mad": mad, "p99": p99, "max": max(values), "n": len(values)}
    return outliers, dist


def plot_prefill_latency(summary, out_dir, dpi, center_stat=DEFAULT_CENTER_STAT, filename="prefill_latency_ms.png"):
    data = collect_prefill(summary)
    envelope = envelope_from_grouped_rows(data, "ms", higher_is_better=False)
    if not envelope:
        return None
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_envelope_band(ax, envelope, digits=1, higher_is_better=False, center_stat=center_stat)
    center_mode = "mean" if str(center_stat).strip().lower() == "mean" else "median"
    if center_mode == "mean":
        ax.set_title("Prefill Latency Mean (ms)")
    else:
        ax.set_title("Prefill Latency (ms)")
    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("Latency (ms)")
    ax.legend(loc="best")
    return save_figure(fig, out_dir / filename, dpi)


def plot_prefill_latency_mean(summary, out_dir, dpi):
    return plot_prefill_latency(
        summary=summary,
        out_dir=out_dir,
        dpi=dpi,
        center_stat="mean",
        filename="prefill_latency_ms_mean.png",
    )


def plot_prefill_throughput(summary, out_dir, dpi, center_stat=DEFAULT_CENTER_STAT, filename="prefill_throughput_tps.png"):
    data = collect_prefill(summary)
    envelope = envelope_from_grouped_rows(data, "tps", higher_is_better=True)
    if not envelope:
        return None
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_envelope_band(ax, envelope, digits=2, higher_is_better=True, center_stat=center_stat)
    center_mode = "mean" if str(center_stat).strip().lower() == "mean" else "median"
    if center_mode == "mean":
        ax.set_title("Prefill Throughput Mean (tokens/s)")
    else:
        ax.set_title("Prefill Throughput (tokens/s)")
    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.legend(loc="best")
    return save_figure(fig, out_dir / filename, dpi)


def plot_prefill_throughput_mean(summary, out_dir, dpi):
    return plot_prefill_throughput(
        summary=summary,
        out_dir=out_dir,
        dpi=dpi,
        center_stat="mean",
        filename="prefill_throughput_tps_mean.png",
    )


def plot_prefill_scaling(summary, out_dir, dpi, center_stat=DEFAULT_CENTER_STAT, filename="prefill_scaling.png"):
    data = collect_prefill(summary)
    envelope = envelope_from_grouped_rows(data, "ms_per_token", higher_is_better=False)
    if not envelope:
        return None
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_envelope_band(ax, envelope, digits=3, higher_is_better=False, center_stat=center_stat)
    center_mode = "mean" if str(center_stat).strip().lower() == "mean" else "median"
    if center_mode == "mean":
        ax.set_title("Prefill Scaling Mean (ms/token)")
    else:
        ax.set_title("Prefill Scaling (ms/token)")
    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("Latency per Token (ms/token)")
    ax.legend(loc="best")
    return save_figure(fig, out_dir / filename, dpi)


def plot_prefill_scaling_mean(summary, out_dir, dpi):
    return plot_prefill_scaling(
        summary=summary,
        out_dir=out_dir,
        dpi=dpi,
        center_stat="mean",
        filename="prefill_scaling_mean.png",
    )


def plot_decode_throughput(raw_rows, out_dir, dpi):
    data = collect_decode(raw_rows)
    envelope = envelope_from_grouped_rows(data, "tps", higher_is_better=True)
    if not envelope:
        return None
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_envelope_band(ax, envelope, digits=2, higher_is_better=True)
    ax.set_title("Decode Throughput (tokens/s)")
    ax.set_xlabel("Max Decode Steps")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.legend(loc="best")
    return save_figure(fig, out_dir / "decode_throughput_tps.png", dpi)


def plot_decode_throughput_token_bucket(raw_rows, out_dir, dpi):
    data = collect_decode_by_token_bucket(raw_rows)
    envelope = envelope_from_grouped_rows(data, "tps", higher_is_better=True)
    if not envelope:
        return None
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_envelope_band(ax, envelope, digits=2, higher_is_better=True)
    ax.set_title("Decode Throughput by Token Bucket (tokens/s)")
    ax.set_xlabel("Decode Token Bucket (tokens)")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.legend(loc="best")
    return save_figure(fig, out_dir / "decode_throughput_tps_token_bucket.png", dpi)


def plot_decode_per_token_latency(raw_rows, out_dir, dpi):
    data = collect_decode(raw_rows)
    envelope = envelope_from_grouped_rows(data, "ms_per_token", higher_is_better=False)
    if not envelope:
        return None
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_envelope_band(ax, envelope, digits=3, higher_is_better=False)
    ax.set_title("Decode Per-token Latency (ms/token)")
    ax.set_xlabel("Max Decode Steps")
    ax.set_ylabel("Latency per Token (ms/token)")
    ax.legend(loc="best")
    return save_figure(fig, out_dir / "decode_per_token_latency.png", dpi)


def plot_decode_scaling(raw_rows, out_dir, dpi, center_stat=DEFAULT_CENTER_STAT, filename="decode_scaling.png"):
    data = collect_decode(raw_rows)
    envelope = envelope_from_grouped_rows(data, "ms", higher_is_better=False)
    if not envelope:
        return None
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_envelope_band(ax, envelope, digits=1, higher_is_better=False, center_stat=center_stat)
    center_mode = "mean" if str(center_stat).strip().lower() == "mean" else "median"
    if center_mode == "mean":
        ax.set_title("Decode Scaling: Decode Latency Mean (ms)")
    else:
        ax.set_title("Decode Scaling: Decode Latency (ms)")
    ax.set_xlabel("Max Decode Steps")
    ax.set_ylabel("Latency (ms)")
    ax.legend(loc="best")
    return save_figure(fig, out_dir / filename, dpi)


def plot_decode_scaling_mean(raw_rows, out_dir, dpi):
    return plot_decode_scaling(
        raw_rows=raw_rows,
        out_dir=out_dir,
        dpi=dpi,
        center_stat="mean",
        filename="decode_scaling_mean.png",
    )


def plot_latency_by_bucket(raw_rows, out_dir, dpi, bucket, value_key, title, ylabel, filename):
    by_style = collect_e2e_ttft_by_style(raw_rows, bucket_filter=bucket)
    if not by_style:
        return None

    points_by_x = defaultdict(list)
    xtick_labels = {}
    for rows in by_style.values():
        valid_rows = [r for r in rows if is_number(r.get(value_key))]
        for idx, row in enumerate(valid_rows, start=1):
            points_by_x[idx].append(float(row[value_key]))
            if idx not in xtick_labels:
                xtick_labels[idx] = row.get("sample_label") or "sample_{0}".format(idx)

    envelope = build_envelope_from_points(points_by_x, higher_is_better=False)
    if not envelope:
        return None

    fig, ax = plt.subplots(figsize=(11, 6))
    plot_envelope_band(ax, envelope, digits=1, higher_is_better=False)
    xs = [p["x"] for p in envelope]
    ax.set_title(title)
    bucket_label = "hot" if bucket == "warm" else bucket
    ax.set_xlabel("Sample Index ({0}_1...{0}_n)".format(bucket_label))
    ax.set_ylabel(ylabel)
    if xs:
        labels = [xtick_labels.get(x, "{0}_{1}".format(bucket_label, x)) for x in xs]
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend(loc="best")
    return save_figure(fig, out_dir / filename, dpi)


def plot_e2e_latency_cold(raw_rows, out_dir, dpi):
    return plot_latency_by_bucket(
        raw_rows=raw_rows,
        out_dir=out_dir,
        dpi=dpi,
        bucket="cold",
        value_key="e2e_ms",
        title="E2E Latency (Cold Start)",
        ylabel="E2E Latency (ms)",
        filename="e2e_latency_cold.png",
    )


def plot_e2e_latency_hot(raw_rows, out_dir, dpi):
    return plot_latency_by_bucket(
        raw_rows=raw_rows,
        out_dir=out_dir,
        dpi=dpi,
        bucket="warm",
        value_key="e2e_ms",
        title="E2E Latency (Hot Start)",
        ylabel="E2E Latency (ms)",
        filename="e2e_latency_hot.png",
    )


def plot_ttft_cold(raw_rows, out_dir, dpi):
    return plot_latency_by_bucket(
        raw_rows=raw_rows,
        out_dir=out_dir,
        dpi=dpi,
        bucket="cold",
        value_key="ttft_ms",
        title="TTFT (Cold Start)",
        ylabel="TTFT (ms)",
        filename="ttft_latency_cold.png",
    )


def plot_ttft_hot(raw_rows, out_dir, dpi):
    return plot_latency_by_bucket(
        raw_rows=raw_rows,
        out_dir=out_dir,
        dpi=dpi,
        bucket="warm",
        value_key="ttft_ms",
        title="TTFT (Hot Start)",
        ylabel="TTFT (ms)",
        filename="ttft_latency_hot.png",
    )


def plot_e2e_latency(raw_rows, out_dir, dpi):
    cold = plot_e2e_latency_cold(raw_rows, out_dir, dpi)
    hot = plot_e2e_latency_hot(raw_rows, out_dir, dpi)
    return hot or cold


def plot_ttft(raw_rows, out_dir, dpi):
    cold = plot_ttft_cold(raw_rows, out_dir, dpi)
    hot = plot_ttft_hot(raw_rows, out_dir, dpi)
    return hot or cold


def plot_tail_latency(raw_rows, out_dir, dpi):
    by_style = collect_e2e_ttft_by_style(raw_rows)
    if not by_style:
        return None

    e2e_by_style = {}
    ttft_by_style = {}
    for style, rows in by_style.items():
        e2e_vals = [r["e2e_ms"] for r in rows if is_number(r.get("e2e_ms"))]
        ttft_vals = [r["ttft_ms"] for r in rows if is_number(r.get("ttft_ms"))]
        if e2e_vals:
            e2e_by_style[style] = e2e_vals
        if ttft_vals:
            ttft_by_style[style] = ttft_vals

    qx = [50, 90, 99]
    e2e_envelope = build_percentile_envelope(e2e_by_style, qx, higher_is_better=False)
    ttft_envelope = build_percentile_envelope(ttft_by_style, qx, higher_is_better=False)
    if not e2e_envelope and not ttft_envelope:
        return None

    fig, (ax_e2e, ax_ttft) = plt.subplots(1, 2, figsize=(15, 5.8))
    if e2e_envelope:
        plot_envelope_band(ax_e2e, e2e_envelope, digits=1, higher_is_better=False)
        ax_e2e.legend(loc="best")
    else:
        ax_e2e.text(0.5, 0.5, "No data", transform=ax_e2e.transAxes, ha="center", va="center")
    ax_e2e.set_title("Tail Latency: E2E")
    ax_e2e.set_xlabel("Percentile")
    ax_e2e.set_ylabel("Latency (ms)")
    ax_e2e.set_xticks(qx)

    if ttft_envelope:
        plot_envelope_band(ax_ttft, ttft_envelope, digits=1, higher_is_better=False)
        ax_ttft.legend(loc="best")
    else:
        ax_ttft.text(0.5, 0.5, "No data", transform=ax_ttft.transAxes, ha="center", va="center")
    ax_ttft.set_title("Tail Latency: TTFT")
    ax_ttft.set_xlabel("Percentile")
    ax_ttft.set_ylabel("Latency (ms)")
    ax_ttft.set_xticks(qx)
    return save_figure(fig, out_dir / "tail_latency_p50_p90_p99.png", dpi)


def plot_tail_latency_prefill_decode_ime(summary, raw_rows, out_dir, dpi):
    metrics = collect_metric_values(raw_rows)
    if not metrics:
        return None

    key_defs = [
        ("prefill_latency_ms", "Prefill Latency (ms)"),
        ("decode_ms_per_token", "Decode Per-token Latency (ms/token)"),
        ("ime_first_candidate_ms", "IME First Candidate Latency (ms)"),
    ]
    qx = [50, 90, 99]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.0))
    plotted = False
    for ax, (key, title) in zip(axes, key_defs):
        values_by_style = {}
        for style, style_metrics in metrics.items():
            vals = style_metrics.get(key, [])
            if vals:
                values_by_style[style] = vals
        envelope = build_percentile_envelope(values_by_style, qx, higher_is_better=False)
        if envelope:
            plot_envelope_band(ax, envelope, digits=2, higher_is_better=False)
            ax.legend(loc="best")
            plotted = True
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")

        ax.set_title("Tail Latency: {0}".format(title))
        ax.set_xlabel("Percentile")
        ax.set_ylabel("Latency")
        ax.set_xticks(qx)

    if not plotted:
        plt.close(fig)
        return None
    return save_figure(fig, out_dir / "tail_latency_prefill_decode_ime.png", dpi)


def plot_ime_first_candidate(summary, out_dir, dpi, center_stat=DEFAULT_CENTER_STAT, filename="ime_first_candidate_latency.png"):
    data = collect_context_sync_by_style(summary)
    envelope = envelope_from_grouped_rows(data, "first_candidate_ms", higher_is_better=False)
    if not envelope:
        return None
    outliers, dist = detect_global_extreme_points(data, "first_candidate_ms", robust_z_threshold=3.5)
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_envelope_band(ax, envelope, digits=1, higher_is_better=False, center_stat=center_stat)
    if outliers:
        ox = [p["x"] for p in outliers]
        oy = [p["value"] for p in outliers]
        ax.scatter(ox, oy, color=SCI_BRICK, marker="x", s=64, linewidths=1.8, zorder=5, label="Extreme outliers")
        for p in outliers[:4]:
            ax.annotate(
                "x={0}, {1}ms".format(p["x"], format_num(p["value"], 0)),
                (p["x"], p["value"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8.5,
                color=SCI_BRICK,
            )

    center_mode = "mean" if str(center_stat).strip().lower() == "mean" else "median"
    if center_mode == "mean":
        ax.set_title("IME First Candidate Latency (first-token ready, mean)")
    else:
        ax.set_title("IME First Candidate Latency (first-token ready)")
    ax.set_xlabel("History Bucket")
    ax.set_ylabel("Latency (ms)")
    if dist:
        ax.text(
            0.99,
            0.02,
            "outliers={0}, p99={1}, max={2}".format(
                len(outliers),
                format_num(dist.get("p99"), 1),
                format_num(dist.get("max"), 1),
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color=SCI_TEXT_MUTED,
        )
    ax.legend(loc="best")
    return save_figure(fig, out_dir / filename, dpi)


def plot_ime_first_candidate_mean(summary, out_dir, dpi):
    return plot_ime_first_candidate(
        summary=summary,
        out_dir=out_dir,
        dpi=dpi,
        center_stat="mean",
        filename="ime_first_candidate_latency_mean.png",
    )


def plot_ttft_history_bucket(summary, raw_rows, out_dir, dpi, center_stat=DEFAULT_CENTER_STAT, filename="ttft_history_bucket.png"):
    data = collect_ttft_history_by_style(raw_rows, summary)
    envelope = envelope_from_grouped_rows(data, "ttft_ms", higher_is_better=False)
    if not envelope:
        return None
    outliers, dist = detect_global_extreme_points(data, "ttft_ms", robust_z_threshold=3.5)
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_envelope_band(ax, envelope, digits=1, higher_is_better=False, center_stat=center_stat)
    if outliers:
        ox = [p["x"] for p in outliers]
        oy = [p["value"] for p in outliers]
        ax.scatter(ox, oy, color=SCI_BRICK, marker="x", s=64, linewidths=1.8, zorder=5, label="Extreme outliers")
        for p in outliers[:4]:
            ax.annotate(
                "x={0}, {1}ms".format(p["x"], format_num(p["value"], 0)),
                (p["x"], p["value"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8.5,
                color=SCI_BRICK,
            )

    center_mode = "mean" if str(center_stat).strip().lower() == "mean" else "median"
    if center_mode == "mean":
        ax.set_title("TTFT by History Bucket (mean)")
    else:
        ax.set_title("TTFT by History Bucket")
    ax.set_xlabel("History Bucket")
    ax.set_ylabel("TTFT (ms)")
    if dist:
        ax.text(
            0.99,
            0.02,
            "outliers={0}, p99={1}, max={2}".format(
                len(outliers),
                format_num(dist.get("p99"), 1),
                format_num(dist.get("max"), 1),
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color=SCI_TEXT_MUTED,
        )
    ax.legend(loc="best")
    return save_figure(fig, out_dir / filename, dpi)


def plot_system(summary, device_snapshot, out_dir, dpi):
    system = summary.get("system") if isinstance(summary, dict) else None
    if not isinstance(system, dict):
        system = {}
    src = dict(device_snapshot or {})
    src.update(system)

    model_size_bytes = to_float(src.get("model_file_size_bytes"))
    kv_obj = src.get("kv_cache_size_bytes")
    kv_avg = None
    if isinstance(kv_obj, dict):
        kv_avg = to_float(kv_obj.get("avg_bytes"))
    elif is_number(kv_obj):
        kv_avg = float(kv_obj)
    peak_kb = to_float(src.get("peak_pss_kb"))
    idle_kb = to_float(src.get("idle_pss_kb"))
    if model_size_bytes is None and kv_avg is None and peak_kb is None and idle_kb is None:
        return None

    model_mb = (model_size_bytes / (1024.0 * 1024.0)) if model_size_bytes is not None else 0.0
    kv_mb = (kv_avg / (1024.0 * 1024.0)) if kv_avg is not None else 0.0
    peak_mb = (peak_kb / 1024.0) if peak_kb is not None else 0.0
    idle_mb = (idle_kb / 1024.0) if idle_kb is not None else 0.0

    labels = ["Model Size", "KV Cache", "Idle RAM", "Peak RAM"]
    totals = [max(0.0, model_mb), max(0.0, kv_mb), max(0.0, idle_mb), max(0.0, peak_mb)]
    if max(totals) <= 0.0:
        return None

    model_ref = max(0.0, model_mb)
    kv_ref = max(0.0, kv_mb)
    model_seg = [totals[0], 0.0, min(model_ref, totals[2]), min(model_ref, totals[3])]
    kv_seg = [
        0.0,
        totals[1],
        min(kv_ref, max(0.0, totals[2] - model_seg[2])),
        min(kv_ref, max(0.0, totals[3] - model_seg[3])),
    ]
    other_seg = [
        0.0,
        0.0,
        max(0.0, totals[2] - model_seg[2] - kv_seg[2]),
        max(0.0, totals[3] - model_seg[3] - kv_seg[3]),
    ]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    xs = list(range(len(labels)))
    ax.bar(xs, model_seg, color=SCI_BLUE, alpha=0.88, label="Model")
    ax.bar(xs, kv_seg, bottom=model_seg, color=SCI_ORANGE, alpha=0.88, label="KV Cache")
    bottoms = [m + k for m, k in zip(model_seg, kv_seg)]
    ax.bar(xs, other_seg, bottom=bottoms, color=SCI_BRICK, alpha=0.88, label="Other RAM")

    y_pad = max(totals) * 0.015
    for x, total in zip(xs, totals):
        ax.text(x, total + y_pad, format_num(total, 2), ha="center", va="bottom", fontsize=9)

    # Highlight RAM composition for Idle/Peak bars only.
    seg_threshold = max(totals) * 0.055
    for x in [2, 3]:
        if model_seg[x] > seg_threshold:
            ax.text(x, model_seg[x] * 0.5, "Model\n{0}".format(format_num(model_seg[x], 0)), ha="center", va="center", fontsize=8, color="white")
        if kv_seg[x] > seg_threshold:
            ax.text(
                x,
                model_seg[x] + kv_seg[x] * 0.5,
                "KV\n{0}".format(format_num(kv_seg[x], 0)),
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )
        if other_seg[x] > seg_threshold:
            ax.text(
                x,
                model_seg[x] + kv_seg[x] + other_seg[x] * 0.5,
                "Other\n{0}".format(format_num(other_seg[x], 0)),
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )

    variant = src.get("model_variant") or "UNKNOWN"
    ax.set_title("System Metrics (Model Variant: {0})".format(variant))
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Size (MB)")
    ax.legend(loc="upper left")
    return save_figure(fig, out_dir / "system_overview.png", dpi)


def plot_cpu(summary, device_snapshot, out_dir, dpi):
    cpu = summary.get("cpu_usage_percent") if isinstance(summary, dict) else None
    if not isinstance(cpu, dict):
        cpu = {}
    dev_cpu = (device_snapshot or {}).get("cpu_usage_percent")
    if isinstance(dev_cpu, dict):
        merged = dict(dev_cpu)
        merged.update(cpu)
        cpu = merged
    avg = to_float(cpu.get("avg_percent"))
    peak = to_float(cpu.get("peak_percent"))
    if avg is None and peak is None:
        return None
    avg = 0.0 if avg is None else avg
    peak = 0.0 if peak is None else peak
    labels = ["CPU Avg", "CPU Peak"]
    values = [avg, peak]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(range(2), values, color=[SCI_BLUE, SCI_ORANGE], alpha=0.88)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), format_num(val, 2), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Usage (%)")
    ax.set_title("Compute Utilization: CPU avg / peak")
    return save_figure(fig, out_dir / "cpu_usage.png", dpi)


def stats_from_summary_obj(obj):
    if not isinstance(obj, dict):
        return None
    n = obj.get("n")
    if not isinstance(n, int):
        n = 0
    out = {"n": n}
    for key in ["avg", "std", "min", "max", "p50", "p90", "p99"]:
        v = to_float(obj.get(key))
        out[key] = v
    if out["n"] <= 0 and not any(is_number(out[k]) for k in ["avg", "p50", "p90", "p99"]):
        return None
    return out


def collect_memory_eval_rows(run_dir, summary):
    files = summary.get("files") if isinstance(summary, dict) else {}
    if not isinstance(files, dict):
        files = {}
    eval_name = files.get("eval_records") or "memory_eval_records.jsonl"
    return read_jsonl(run_dir / str(eval_name))


def get_memory_stage_stats(summary, eval_rows):
    mp = summary.get("memory_processing") if isinstance(summary, dict) else {}
    trigger = summary.get("trigger") if isinstance(summary, dict) else {}
    retrieval = summary.get("retrieval") if isinstance(summary, dict) else {}
    generation = summary.get("generation") if isinstance(summary, dict) else {}
    case_latency = summary.get("case_latency_ms") if isinstance(summary, dict) else {}

    stats = {}
    s = stats_from_summary_obj((mp.get("batch_latency_ms") if isinstance(mp, dict) else {}))
    if s:
        stats["processing_batch"] = s
    s = stats_from_summary_obj((trigger.get("latency_ms") if isinstance(trigger, dict) else {}))
    if s:
        stats["trigger_base"] = s
    s = stats_from_summary_obj((retrieval.get("latency_ms") if isinstance(retrieval, dict) else {}))
    if s:
        stats["retrieval"] = s
    s = stats_from_summary_obj((generation.get("rerun_latency_ms") if isinstance(generation, dict) else {}))
    if s:
        stats["rerun"] = s
    s = stats_from_summary_obj(case_latency if isinstance(case_latency, dict) else {})
    if s:
        stats["case_e2e"] = s

    # Fallback from eval rows when summary stats are missing.
    if "trigger_base" not in stats:
        vals = [to_float(r.get("trigger_latency_ms")) for r in eval_rows]
        vals = [v for v in vals if is_number(v) and v >= 0.0]
        calc = calc_stats(vals)
        if calc:
            stats["trigger_base"] = calc
    if "retrieval" not in stats:
        vals = [to_float(r.get("retrieval_elapsed_ms")) for r in eval_rows]
        vals = [v for v in vals if is_number(v) and v >= 0.0]
        calc = calc_stats(vals)
        if calc:
            stats["retrieval"] = calc
    if "rerun" not in stats:
        vals = [to_float(r.get("rerun_latency_ms_measured")) for r in eval_rows]
        vals = [v for v in vals if is_number(v) and v >= 0.0]
        calc = calc_stats(vals)
        if calc:
            stats["rerun"] = calc
    if "case_e2e" not in stats:
        vals = [to_float(r.get("base_latency_ms")) for r in eval_rows]
        vals = [v for v in vals if is_number(v) and v >= 0.0]
        calc = calc_stats(vals)
        if calc:
            stats["case_e2e"] = calc
    return stats


def plot_memory_stage_tails(summary, eval_rows, out_dir, dpi):
    stats = get_memory_stage_stats(summary, eval_rows)
    stage_order = [
        ("processing_batch", "Processing Batch"),
        ("trigger_base", "Trigger/Base"),
        ("retrieval", "Retrieval"),
        ("rerun", "Rerun"),
        ("case_e2e", "Case E2E"),
    ]
    labels = []
    p50 = []
    p90 = []
    p99 = []
    for key, label in stage_order:
        s = stats.get(key)
        if not s:
            continue
        if not all(is_number(s.get(k)) for k in ["p50", "p90", "p99"]):
            continue
        labels.append(label)
        p50.append(s["p50"])
        p90.append(s["p90"])
        p99.append(s["p99"])
    if not labels:
        return None
    fig, ax = plt.subplots(figsize=(11.5, 6))
    xs = list(range(1, len(labels) + 1))
    ax.plot(xs, p50, color=SCI_BLUE, marker="o", linewidth=2.2, markersize=6.2, label="P50")
    ax.plot(xs, p90, color=SCI_ORANGE, marker="s", linewidth=2.2, markersize=6.2, label="P90")
    ax.plot(xs, p99, color=SCI_BRICK, marker="^", linewidth=2.2, markersize=6.2, label="P99")
    annotate_points(ax, xs, p50, color=SCI_BLUE, digits=1)
    annotate_points(ax, xs, p90, color=SCI_ORANGE, digits=1)
    annotate_points(ax, xs, p99, color=SCI_BRICK, digits=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Memory Stage Tail Latency (P50/P90/P99)")
    ax.set_ylabel("Latency (ms)")
    ax.legend(loc="best")
    return save_figure(fig, out_dir / "memory_stage_tail_latency.png", dpi)


def plot_memory_stage_avg_std(summary, eval_rows, out_dir, dpi):
    stats = get_memory_stage_stats(summary, eval_rows)
    stage_order = [
        ("processing_batch", "Processing Batch"),
        ("trigger_base", "Trigger/Base"),
        ("retrieval", "Retrieval"),
        ("rerun", "Rerun"),
        ("case_e2e", "Case E2E"),
    ]
    labels = []
    avgs = []
    stds = []
    for key, label in stage_order:
        s = stats.get(key)
        if not s:
            continue
        if not is_number(s.get("avg")):
            continue
        labels.append(label)
        avgs.append(s["avg"])
        stds.append(s["std"] if is_number(s.get("std")) else 0.0)
    if not labels:
        return None
    fig, ax = plt.subplots(figsize=(11.5, 6))
    xs = list(range(len(labels)))
    bars = ax.bar(xs, avgs, yerr=stds, capsize=5, color=cycle_colors(len(labels)), alpha=0.88)
    for x, bar, val in zip(xs, bars, avgs):
        ax.text(x, bar.get_height(), format_num(val, 1), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Memory Stage Average Latency (with Std)")
    return save_figure(fig, out_dir / "memory_stage_avg_std_latency.png", dpi)


def plot_memory_success_rates(summary, out_dir, dpi):
    mp = summary.get("memory_processing") if isinstance(summary, dict) else {}
    trigger = summary.get("trigger") if isinstance(summary, dict) else {}
    retrieval = summary.get("retrieval") if isinstance(summary, dict) else {}
    recall = summary.get("recall") if isinstance(summary, dict) else {}
    generation = summary.get("generation") if isinstance(summary, dict) else {}

    rates = [
        ("Process Success", to_float(mp.get("success_rate")) if isinstance(mp, dict) else None),
        ("Trigger Accuracy", to_float(trigger.get("accuracy")) if isinstance(trigger, dict) else None),
        ("Trigger Recall", to_float(trigger.get("estimated_recall")) if isinstance(trigger, dict) else None),
        ("Retrieval Hit", to_float(retrieval.get("hit_rate")) if isinstance(retrieval, dict) else None),
        ("Recall Success", to_float(recall.get("success_rate")) if isinstance(recall, dict) else None),
        ("Generation Success", to_float(generation.get("success_rate")) if isinstance(generation, dict) else None),
        ("Rerun Success", to_float(generation.get("rerun_success_rate")) if isinstance(generation, dict) else None),
        ("Ref Match", to_float(generation.get("reference_match_rate")) if isinstance(generation, dict) else None),
    ]
    labels = [k for k, v in rates if is_number(v)]
    values = [v * 100.0 for _, v in rates if is_number(v)]
    if not labels:
        return None
    fig, ax = plt.subplots(figsize=(12, 6))
    xs = list(range(len(labels)))
    bars = ax.bar(xs, values, color=cycle_colors(len(labels)), alpha=0.88)
    for x, bar, val in zip(xs, bars, values):
        ax.text(x, bar.get_height(), "{0:.1f}%".format(val), ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0.0, 105.0)
    ax.set_title("Memory Pipeline Success Rates")
    return save_figure(fig, out_dir / "memory_success_rates.png", dpi)


def plot_memory_trigger_confusion(summary, out_dir, dpi):
    trigger = summary.get("trigger") if isinstance(summary, dict) else {}
    total = int(summary.get("dialog_case_count") or 0) if isinstance(summary, dict) else 0
    if not isinstance(trigger, dict):
        trigger = {}
    expected = int(trigger.get("expected") or 0)
    predicted = int(trigger.get("predicted") or 0)
    matched = int(trigger.get("matched") or 0)
    tp = int(trigger.get("true_positive") or 0)
    tn = max(0, matched - tp)
    fn = max(0, expected - tp)
    fp = max(0, predicted - tp)
    other = max(0, total - (tp + tn + fn + fp))
    labels = ["TP", "FN", "FP", "TN"]
    values = [tp, fn, fp, tn]
    if other > 0:
        labels.append("Other")
        values.append(other)
    if sum(values) <= 0:
        return None
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    xs = list(range(len(labels)))
    bars = ax.bar(xs, values, color=[SCI_GREEN, SCI_BRICK, SCI_ORANGE, SCI_BLUE, SCI_GRAY][: len(labels)], alpha=0.88)
    for x, bar, val in zip(xs, bars, values):
        ax.text(x, bar.get_height(), str(val), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title("Memory Trigger Confusion Breakdown")
    return save_figure(fig, out_dir / "memory_trigger_confusion.png", dpi)


def plot_memory_case_timeline(eval_rows, out_dir, dpi):
    if not eval_rows:
        return None
    points = []
    for row in eval_rows:
        base = to_float(row.get("base_latency_ms"))
        trig = to_float(row.get("trigger_latency_ms"))
        rerun = to_float(row.get("rerun_latency_ms_measured"))
        pred = bool(row.get("predicted_trigger"))
        points.append({"base": base, "trigger": trig, "rerun": rerun, "pred": pred})
    valid = [p for p in points if is_number(p["base"]) or is_number(p["trigger"])]
    if not valid:
        return None
    xs = list(range(1, len(valid) + 1))
    base_ys = [p["base"] if is_number(p["base"]) else None for p in valid]
    trig_ys = [p["trigger"] if is_number(p["trigger"]) else None for p in valid]
    rerun_x = [x for x, p in zip(xs, valid) if is_number(p["rerun"])]
    rerun_y = [p["rerun"] for p in valid if is_number(p["rerun"])]
    pred_x = [x for x, p in zip(xs, valid) if p["pred"] and is_number(p["base"])]
    pred_y = [p["base"] for p in valid if p["pred"] and is_number(p["base"])]

    fig, ax = plt.subplots(figsize=(12, 6.2))
    if any(is_number(v) for v in base_ys):
        xs_b = [x for x, y in zip(xs, base_ys) if is_number(y)]
        ys_b = [y for y in base_ys if is_number(y)]
        ax.plot(xs_b, ys_b, color=SCI_BLUE, marker="o", markersize=4.4, linewidth=1.9, label="Base E2E")
    if any(is_number(v) for v in trig_ys):
        xs_t = [x for x, y in zip(xs, trig_ys) if is_number(y)]
        ys_t = [y for y in trig_ys if is_number(y)]
        ax.plot(xs_t, ys_t, color=SCI_ORANGE, marker="s", markersize=4.4, linewidth=1.8, alpha=0.88, label="Trigger Latency")
    if rerun_x:
        ax.scatter(rerun_x, rerun_y, color=SCI_GREEN, marker="^", s=30, label="Rerun Latency", zorder=4)
    if pred_x:
        ax.scatter(pred_x, pred_y, facecolors="none", edgecolors=SCI_BRICK, s=54, linewidths=1.5, label="Predicted Trigger", zorder=5)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Memory Case Timeline (Base / Trigger / Rerun)")
    ax.legend(loc="best")
    return save_figure(fig, out_dir / "memory_case_timeline.png", dpi)


def plot_memory_retrieval_scatter(eval_rows, out_dir, dpi):
    xs = []
    ys = []
    colors = []
    for row in eval_rows:
        elapsed = to_float(row.get("retrieval_elapsed_ms"))
        selected = row.get("retrieval_selected")
        sim01 = to_float(selected.get("sim01")) if isinstance(selected, dict) else None
        if not is_number(elapsed) or not is_number(sim01):
            continue
        xs.append(sim01)
        ys.append(elapsed)
        pred = bool(row.get("predicted_trigger"))
        colors.append(SCI_BRICK if pred else SCI_BLUE)
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.scatter(xs, ys, c=colors, s=42, alpha=0.85, edgecolors="white", linewidths=0.5)
    ax.set_xlabel("Selected Similarity (sim01)")
    ax.set_ylabel("Retrieval Latency (ms)")
    ax.set_title("Memory Retrieval Quality vs Latency")
    return save_figure(fig, out_dir / "memory_retrieval_quality_scatter.png", dpi)


def run_memory_plots(summary, eval_rows, device_snapshot, out_dir, dpi):
    generated = []
    for fn in [
        lambda: plot_memory_stage_tails(summary, eval_rows, out_dir, dpi),
        lambda: plot_memory_stage_avg_std(summary, eval_rows, out_dir, dpi),
        lambda: plot_memory_success_rates(summary, out_dir, dpi),
        lambda: plot_memory_trigger_confusion(summary, out_dir, dpi),
        lambda: plot_memory_case_timeline(eval_rows, out_dir, dpi),
        lambda: plot_memory_retrieval_scatter(eval_rows, out_dir, dpi),
        lambda: plot_system(summary, device_snapshot, out_dir, dpi),
        lambda: plot_cpu(summary, device_snapshot, out_dir, dpi),
    ]:
        p = fn()
        if p is not None:
            generated.append(p)
    return generated


def write_memory_markdown_summary(summary, eval_rows, out_dir, markdown_name):
    lines = []
    lines.append("# Memory Test Research Summary")
    lines.append("")
    lines.append("## Experiment Metadata")
    lines.append("")
    lines.append("- Test Name: `{0}`".format(summary.get("test_name", "unknown")))
    lines.append("- Duration (ms): `{0}`".format(summary.get("duration_ms", "unknown")))
    lines.append("- Success: `{0}`".format(summary.get("success", "unknown")))
    lines.append("- Sample Size: `{0}`".format(summary.get("sample_size", "unknown")))
    lines.append("- Dialog Cases: `{0}`".format(summary.get("dialog_case_count", "unknown")))
    lines.append("")

    mp = summary.get("memory_processing") if isinstance(summary, dict) else {}
    trigger = summary.get("trigger") if isinstance(summary, dict) else {}
    retrieval = summary.get("retrieval") if isinstance(summary, dict) else {}
    recall = summary.get("recall") if isinstance(summary, dict) else {}
    generation = summary.get("generation") if isinstance(summary, dict) else {}
    if not isinstance(mp, dict):
        mp = {}
    if not isinstance(trigger, dict):
        trigger = {}
    if not isinstance(retrieval, dict):
        retrieval = {}
    if not isinstance(recall, dict):
        recall = {}
    if not isinstance(generation, dict):
        generation = {}

    lines.append("## Memory Success Metrics")
    lines.append("")
    lines.append("| Module | Metric | Value |")
    lines.append("| --- | --- | ---: |")
    lines.append("| Processing | Success Rate | {0}% |".format(format_num(to_float(mp.get("success_rate")) * 100.0 if is_number(to_float(mp.get("success_rate"))) else None, 2)))
    lines.append("| Trigger | Accuracy | {0}% |".format(format_num(to_float(trigger.get("accuracy")) * 100.0 if is_number(to_float(trigger.get("accuracy"))) else None, 2)))
    lines.append("| Trigger | Estimated Recall | {0}% |".format(format_num(to_float(trigger.get("estimated_recall")) * 100.0 if is_number(to_float(trigger.get("estimated_recall"))) else None, 2)))
    lines.append("| Retrieval | Hit Rate | {0}% |".format(format_num(to_float(retrieval.get("hit_rate")) * 100.0 if is_number(to_float(retrieval.get("hit_rate"))) else None, 2)))
    lines.append("| Recall | Success Rate | {0}% |".format(format_num(to_float(recall.get("success_rate")) * 100.0 if is_number(to_float(recall.get("success_rate"))) else None, 2)))
    lines.append("| Generation | Success Rate | {0}% |".format(format_num(to_float(generation.get("success_rate")) * 100.0 if is_number(to_float(generation.get("success_rate"))) else None, 2)))
    lines.append("| Generation | Rerun Success Rate | {0}% |".format(format_num(to_float(generation.get("rerun_success_rate")) * 100.0 if is_number(to_float(generation.get("rerun_success_rate"))) else None, 2)))
    lines.append("")

    stage_stats = get_memory_stage_stats(summary, eval_rows)
    lines.append("## Memory Latency Stats")
    lines.append("")
    lines.append("| Stage | N | Avg | Std | P50 | P90 | P99 | Min | Max |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    stage_labels = [
        ("processing_batch", "Processing Batch"),
        ("trigger_base", "Trigger/Base"),
        ("retrieval", "Retrieval"),
        ("rerun", "Rerun"),
        ("case_e2e", "Case E2E"),
    ]
    for key, label in stage_labels:
        s = stage_stats.get(key)
        if not s:
            continue
        lines.append(
            "| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} |".format(
                label,
                s.get("n", 0),
                format_num(s.get("avg"), 2),
                format_num(s.get("std"), 2),
                format_num(s.get("p50"), 2),
                format_num(s.get("p90"), 2),
                format_num(s.get("p99"), 2),
                format_num(s.get("min"), 2),
                format_num(s.get("max"), 2),
            )
        )
    lines.append("")

    out_path = out_dir / markdown_name
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out_path


def collect_metric_values(raw_rows):
    data = defaultdict(lambda: defaultdict(list))
    for row in raw_rows:
        style = str(row.get("style_mode") or "unknown")
        tag = str(row.get("tag") or "")
        if tag == "prefill_scaling":
            v = to_float(row.get("prefill_ms"))
            if v is not None:
                data[style]["prefill_latency_ms"].append(v)
            v = to_float(row.get("prefill_tps"))
            if v is not None:
                data[style]["prefill_tps"].append(v)
        elif tag == "decode":
            v = to_float(row.get("decode_tps"))
            if v is not None and v >= 0.0:
                data[style]["decode_tps"].append(v)
            ms = to_float(row.get("decode_ms"))
            tok = to_float(row.get("decode_tokens"))
            if ms is not None and tok is not None and tok > 0.0:
                data[style]["decode_ms_per_token"].append(ms / tok)
        elif tag == "ime_first":
            ttft_ms = extract_ttft_value(row)
            if ttft_ms is not None and ttft_ms >= 0.0:
                data[style]["ttft_ms"].append(ttft_ms)
            ttft_1token_ms = to_float(row.get("ttft_1token_ms"))
            if ttft_1token_ms is not None and ttft_1token_ms >= 0.0:
                data[style]["ttft_1token_ms"].append(ttft_1token_ms)
            ttft_generate_ms = to_float(row.get("ttft_generate_ms"))
            if ttft_generate_ms is not None and ttft_generate_ms >= 0.0:
                data[style]["ttft_generate_ms"].append(ttft_generate_ms)
            first_ms = to_float(row.get("ime_first_candidate_ms"))
            if first_ms is None:
                first_ms = to_float(row.get("ttft_ms"))
            if first_ms is not None and first_ms >= 0.0:
                data[style]["ime_first_candidate_ms"].append(first_ms)
        elif tag == "context_sync":
            # Backward-compat for old runs only.
            load_ms = to_float(row.get("context_sync_load_ms"))
            infer_ttft_ms = to_float(row.get("context_sync_infer_ttft_ms"))
            if load_ms is not None and infer_ttft_ms is not None and load_ms >= 0.0 and infer_ttft_ms >= 0.0:
                ttft_ms = load_ms + infer_ttft_ms
                data[style]["ttft_ms"].append(ttft_ms)
                data[style]["ime_first_candidate_ms"].append(ttft_ms)
    return data


def write_markdown_summary(summary, raw_rows, out_dir, markdown_name):
    metrics = collect_metric_values(raw_rows)
    decode_rule = resolve_decode_invalid_rule(summary)
    decode_validity = collect_decode_validity(
        raw_rows=raw_rows,
        ratio=decode_rule["ratio"],
        min_budget=decode_rule["min_budget"],
    )
    lines = []
    lines.append("# IME Performance Research Summary")
    lines.append("")
    lines.append("## Experiment Metadata")
    lines.append("")
    lines.append("- Test Name: `{0}`".format(summary.get("test_name", "unknown")))
    lines.append("- Duration (ms): `{0}`".format(summary.get("duration_ms", "unknown")))
    lines.append("- Success: `{0}`".format(summary.get("success", "unknown")))
    lines.append("")

    system = summary.get("system") if isinstance(summary, dict) else {}
    if not isinstance(system, dict):
        system = {}
    cpu = summary.get("cpu_usage_percent") if isinstance(summary, dict) else {}
    if not isinstance(cpu, dict):
        cpu = {}
    kv = system.get("kv_cache_size_bytes") if isinstance(system, dict) else {}
    model_bytes = to_float(system.get("model_file_size_bytes")) if isinstance(system, dict) else None
    model_mb = (model_bytes / (1024.0 * 1024.0)) if model_bytes is not None else None
    kv_min = to_float(kv.get("min_bytes")) if isinstance(kv, dict) else None
    kv_avg = to_float(kv.get("avg_bytes")) if isinstance(kv, dict) else None
    kv_max = to_float(kv.get("max_bytes")) if isinstance(kv, dict) else None
    kv_min_mb = (kv_min / (1024.0 * 1024.0)) if kv_min is not None else None
    kv_avg_mb = (kv_avg / (1024.0 * 1024.0)) if kv_avg is not None else None
    kv_max_mb = (kv_max / (1024.0 * 1024.0)) if kv_max is not None else None
    peak_mb = to_float(system.get("peak_pss_kb"))
    idle_mb = to_float(system.get("idle_pss_kb"))
    peak_mb = (peak_mb / 1024.0) if peak_mb is not None else None
    idle_mb = (idle_mb / 1024.0) if idle_mb is not None else None
    cpu_avg = to_float(cpu.get("avg_percent")) if isinstance(cpu, dict) else None
    cpu_peak = to_float(cpu.get("peak_percent")) if isinstance(cpu, dict) else None
    variant = system.get("model_variant", "UNKNOWN") if isinstance(system, dict) else "UNKNOWN"

    lines.append("## System Summary")
    lines.append("")
    lines.append("| Model Variant | Model Size (MB) | KV Cache Avg (MB) | Peak RAM (MB) | Idle RAM (MB) | CPU Avg (%) | CPU Peak (%) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        "| {0} | {1} | {2} | {3} | {4} | {5} | {6} |".format(
            variant,
            format_num(model_mb, 2),
            format_num(kv_avg_mb, 2),
            format_num(peak_mb, 2),
            format_num(idle_mb, 2),
            format_num(cpu_avg, 2),
            format_num(cpu_peak, 2),
        )
    )
    lines.append("")

    lines.append("## System Metric Stats (min/avg/max)")
    lines.append("")
    lines.append("| Metric | Min | Avg | Max |")
    lines.append("| --- | ---: | ---: | ---: |")
    lines.append("| Model Size (MB) | {0} | {0} | {0} |".format(format_num(model_mb, 2)))
    lines.append(
        "| KV Cache Size (MB) | {0} | {1} | {2} |".format(
            format_num(kv_min_mb, 2),
            format_num(kv_avg_mb, 2),
            format_num(kv_max_mb, 2),
        )
    )
    lines.append("| Idle RAM (MB) | {0} | {0} | {0} |".format(format_num(idle_mb, 2)))
    lines.append("| Peak RAM (MB) | {0} | {0} | {0} |".format(format_num(peak_mb, 2)))
    lines.append(
        "| CPU Usage (%) | {0} | {1} | {2} |".format(
            format_num(cpu_avg, 2),
            format_num(cpu_avg, 2),
            format_num(cpu_peak, 2),
        )
    )
    lines.append("")

    lines.append("## Scientific Metric Table (avg/std + tails)")
    lines.append("")
    lines.append("| Style | Metric | N | Avg | Std | P50 | P90 | P99 | Min | Max |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    metric_order = [
        ("prefill_latency_ms", "Prefill Latency (ms)"),
        ("prefill_tps", "Prefill Throughput (tokens/s)"),
        ("decode_tps", "Decode Throughput (tokens/s)"),
        ("decode_ms_per_token", "Decode Per-token Latency (ms/token)"),
        ("ttft_ms", "TTFT (ms)"),
        ("ttft_1token_ms", "TTFT 1-Token (ms)"),
        ("ttft_generate_ms", "TTFT Generate-Path (ms)"),
        ("ime_first_candidate_ms", "IME First Candidate Latency (ms)"),
    ]

    all_style_values = defaultdict(list)
    for style, style_metrics in metrics.items():
        for key, _ in metric_order:
            all_style_values[key].extend(style_metrics.get(key, []))

    style_names = sorted(set(metrics.keys()) | set(decode_validity["by_style"].keys()))
    for style in style_names + ["ALL"]:
        src = metrics.get(style, {}) if style != "ALL" else all_style_values
        for key, metric_name in metric_order:
            stats = calc_stats(src.get(key, []))
            if not stats:
                continue
            lines.append(
                "| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} |".format(
                    style,
                    metric_name,
                    stats["n"],
                    format_num(stats["avg"], 3),
                    format_num(stats["std"], 3),
                    format_num(stats["p50"], 3),
                    format_num(stats["p90"], 3),
                    format_num(stats["p99"], 3),
                    format_num(stats["min"], 3),
                    format_num(stats["max"], 3),
                )
            )

    lines.append("")
    lines.append("## Decode Sample Validity")
    lines.append("")
    lines.append("- Rule: `{0}`".format(decode_rule["rule_text"]))
    lines.append("")
    lines.append("| Style | Decode Samples | Valid | Invalid | Invalid Rate (%) |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for style in style_names + ["ALL"]:
        if style == "ALL":
            st = decode_validity["all"]
        else:
            st = decode_validity["by_style"].get(style, {"total": 0, "valid": 0, "invalid": 0, "rows": []})
        total = int(st.get("total", 0))
        valid = int(st.get("valid", 0))
        invalid = int(st.get("invalid", 0))
        invalid_rate = (invalid * 100.0 / total) if total > 0 else 0.0
        lines.append(
            "| {0} | {1} | {2} | {3} | {4} |".format(
                style,
                total,
                valid,
                invalid,
                format_num(invalid_rate, 2),
            )
        )
    lines.append("")

    invalid_rows = decode_validity["all"].get("rows", [])
    lines.append("### Invalid Decode Samples (Top 20)")
    lines.append("")
    if not invalid_rows:
        lines.append("- None")
    else:
        lines.append("| Style | Bucket | Iter | decode_tokens | step_budget | threshold | control_mode | Reason |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
        for row in invalid_rows[:20]:
            lines.append(
                "| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} |".format(
                    row.get("style", "unknown"),
                    row.get("bucket", "-"),
                    row.get("iteration", 0),
                    format_num(row.get("decode_tokens"), 2),
                    row.get("step_budget", "-"),
                    format_num(row.get("threshold"), 2),
                    row.get("decode_control_mode", "-"),
                    str(row.get("reason", "-")).replace("|", "/"),
                )
            )
    lines.append("")

    lines.append("## Tail Latency Focus (P50/P90/P99)")
    lines.append("")
    lines.append("| Style | Metric | N | P50 | P90 | P99 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    tail_metric_order = [
        ("prefill_latency_ms", "Prefill Latency (ms)"),
        ("decode_ms_per_token", "Decode Per-token Latency (ms/token)"),
        ("ttft_ms", "TTFT (ms)"),
        ("ttft_1token_ms", "TTFT 1-Token (ms)"),
        ("ttft_generate_ms", "TTFT Generate-Path (ms)"),
        ("ime_first_candidate_ms", "IME First Candidate Latency (ms)"),
    ]
    for style in style_names + ["ALL"]:
        src = metrics.get(style, {}) if style != "ALL" else all_style_values
        for key, metric_name in tail_metric_order:
            stats = calc_stats(src.get(key, []))
            if not stats:
                continue
            lines.append(
                "| {0} | {1} | {2} | {3} | {4} | {5} |".format(
                    style,
                    metric_name,
                    stats["n"],
                    format_num(stats["p50"], 3),
                    format_num(stats["p90"], 3),
                    format_num(stats["p99"], 3),
                )
            )

    out_path = out_dir / markdown_name
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark figures and markdown summary.")
    parser.add_argument("--run-dir", type=str, default=".", help="Run directory containing summary.json")
    parser.add_argument("--out-dir", type=str, default="plots", help="Output directory for png and markdown files")
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI")
    parser.add_argument(
        "--summary-md",
        type=str,
        default="research_summary.md",
        help="Markdown summary filename under out-dir; set empty string to disable",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = (run_dir / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir).resolve()
    summary_path = run_dir / "summary.json"
    raw_path = run_dir / "raw_llm_metrics.jsonl"
    device_path = run_dir / "device_snapshot.json"

    if not summary_path.exists():
        raise SystemExit("summary.json not found under run dir: {0}".format(summary_path))

    setup_style()
    summary = read_json(summary_path)
    device_snapshot = read_json(device_path) if device_path.exists() else {}
    out_dir.mkdir(parents=True, exist_ok=True)

    test_name = str(summary.get("test_name", "")).strip().lower()
    generated = []
    md_file = None

    if test_name == "memory_test":
        eval_rows = collect_memory_eval_rows(run_dir, summary)
        generated = run_memory_plots(summary, eval_rows, device_snapshot, out_dir, args.dpi)
        if args.summary_md and args.summary_md.strip():
            md_file = write_memory_markdown_summary(summary, eval_rows, out_dir, args.summary_md.strip())
    else:
        raw_rows = read_jsonl(raw_path)
        for plot_fn in [
            lambda: plot_prefill_latency(summary, out_dir, args.dpi),
            lambda: plot_prefill_latency_mean(summary, out_dir, args.dpi),
            lambda: plot_prefill_throughput(summary, out_dir, args.dpi),
            lambda: plot_prefill_throughput_mean(summary, out_dir, args.dpi),
            lambda: plot_prefill_scaling(summary, out_dir, args.dpi),
            lambda: plot_prefill_scaling_mean(summary, out_dir, args.dpi),
            lambda: plot_decode_throughput(raw_rows, out_dir, args.dpi),
            lambda: plot_decode_throughput_token_bucket(raw_rows, out_dir, args.dpi),
            lambda: plot_decode_per_token_latency(raw_rows, out_dir, args.dpi),
            lambda: plot_decode_scaling(raw_rows, out_dir, args.dpi),
            lambda: plot_decode_scaling_mean(raw_rows, out_dir, args.dpi),
            lambda: plot_tail_latency_prefill_decode_ime(summary, raw_rows, out_dir, args.dpi),
            lambda: plot_ttft_history_bucket(summary, raw_rows, out_dir, args.dpi),
            lambda: plot_ime_first_candidate(summary, out_dir, args.dpi),
            lambda: plot_ime_first_candidate_mean(summary, out_dir, args.dpi),
            lambda: plot_system(summary, device_snapshot, out_dir, args.dpi),
            lambda: plot_cpu(summary, device_snapshot, out_dir, args.dpi),
        ]:
            p = plot_fn()
            if p is not None:
                generated.append(p)
        if args.summary_md and args.summary_md.strip():
            md_file = write_markdown_summary(summary, raw_rows, out_dir, args.summary_md.strip())

    if not generated and md_file is None:
        print("No figures or markdown summary were generated.")
        return

    print("Generated {0} figure(s) in: {1}".format(len(generated), out_dir))
    for item in generated:
        print(" - {0}".format(item))
    if md_file is not None:
        print("Markdown summary:")
        print(" - {0}".format(md_file))


if __name__ == "__main__":
    main()
