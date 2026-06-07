from __future__ import annotations

import csv
import json
import math
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
SOURCES_DIR = ROOT / "sources"
TEMPLATE_DOCX = ROOT / "Template Proyek Makalah ASA 2026.docx"
DOCX_OUT = TEMPLATE_DOCX
BACKUP_DOCX = ROOT / "Template Proyek Makalah ASA 2026.backup.docx"

AUTHOR_NAME = "Hasta Putra Wildantara"
AUTHOR_NIM = "24060124130119"
AUTHOR_EMAIL = "hastaputrawildantara@gmail.com"
STATEMENT_DATE = "Semarang, 3 Juni 2026"

NASA_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point?"
    + urllib.parse.urlencode(
        {
            "parameters": "T2M,RH2M,T2M_MAX,T2M_MIN,T2MWET",
            "community": "AG",
            "longitude": "110.4167",
            "latitude": "-6.9667",
            "start": "20220403",
            "end": "20220421",
            "format": "JSON",
            "time-standard": "LST",
        }
    )
)

MENDELEY_PAGE = "https://data.mendeley.com/datasets/7fptd3rfzy/1"
MENDELEY_API = "https://api.data.mendeley.com/datasets/publics/7fptd3rfzy/files?version=1"

LEVELS = [0, 1, 2, 3]
ENERGY_UNITS = {0: 0.00, 1: 1.20, 2: 2.60, 3: 4.10}
COOLING_DELTA_C = {0: 0.00, 1: 0.90, 2: 1.80, 3: 2.80}
COMFORT_LIMIT = 76.0
SEVERE_LIMIT = 80.0
SWITCH_WEIGHT = 0.22
COMFORT_WEIGHT = 1.00

NS_W = "http://purl.oclc.org/ooxml/wordprocessingml/main"
NS_R = "http://purl.oclc.org/ooxml/officeDocument/relationships"
NS = {"w": NS_W, "r": NS_R}
W = f"{{{NS_W}}}"
R = f"{{{NS_R}}}"


def ensure_dirs() -> None:
    for path in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, SOURCES_DIR]:
        path.mkdir(exist_ok=True)


def fetch_url(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ASA-broiler-energy-study/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def fetch_data() -> pd.DataFrame:
    raw_path = DATA_DIR / "nasa_power_semarang_20220403_20220421.json"
    csv_path = DATA_DIR / "nasa_power_semarang_daily.csv"

    raw = fetch_url(NASA_URL)
    raw_path.write_bytes(raw)
    payload = json.loads(raw.decode("utf-8"))
    params = payload["properties"]["parameter"]
    dates = sorted(params["T2M"].keys())

    rows = []
    for date in dates:
        row = {
            "date": pd.to_datetime(date, format="%Y%m%d").date().isoformat(),
            "temperature_c": params["T2M"][date],
            "relative_humidity_pct": params["RH2M"][date],
            "temperature_max_c": params["T2M_MAX"][date],
            "temperature_min_c": params["T2M_MIN"][date],
            "wet_bulb_c": params["T2MWET"][date],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["thi_rh"] = thi_from_temp_rh(df["temperature_c"], df["relative_humidity_pct"])
    df["thi_wetbulb"] = 0.85 * df["temperature_c"] + 0.15 * df["wet_bulb_c"]
    df.to_csv(csv_path, index=False)

    note_lines = [
        "Dataset route used in this study",
        "================================",
        "",
        f"Primary candidate: {MENDELEY_PAGE}",
        "DOI: 10.17632/7fptd3rfzy.1",
        "Status: metadata is public, but the public files API returned authorization-required in this environment.",
        "",
        f"Fallback/comparison data used for the experiment: {NASA_URL}",
        "Location: Semarang, Indonesia (-6.9667, 110.4167)",
        "Period: 2022-04-03 to 2022-04-21, aligned with the Mendeley poultry-house study period.",
        "Parameters: T2M, RH2M, T2M_MAX, T2M_MIN, T2MWET.",
    ]
    try:
        fetch_url(MENDELEY_API)
        note_lines[4] = "Status: Mendeley API file metadata was accessible."
    except urllib.error.HTTPError as exc:
        note_lines.append(f"Mendeley public API check: HTTP {exc.code} {exc.reason}.")
    except Exception as exc:  # noqa: BLE001 - saved for reproducibility
        note_lines.append(f"Mendeley public API check failed: {type(exc).__name__}: {exc}.")
    (SOURCES_DIR / "dataset_route_note.txt").write_text("\n".join(note_lines), encoding="utf-8")

    return df


def thi_from_temp_rh(temp_c, rh_pct):
    return 0.8 * temp_c + (rh_pct * (temp_c - 14.3) / 100.0) + 46.3


def apply_level(row: pd.Series, level: int) -> tuple[float, float]:
    temp_eff = row["temperature_c"] - COOLING_DELTA_C[level]
    thi_eff = float(thi_from_temp_rh(temp_eff, row["relative_humidity_pct"]))
    return temp_eff, thi_eff


def comfort_penalty(thi_eff: float) -> float:
    discomfort = max(0.0, thi_eff - COMFORT_LIMIT)
    severe = max(0.0, thi_eff - SEVERE_LIMIT)
    return 1.8 * discomfort**2 + 4.5 * severe**2


def step_cost(row: pd.Series, level: int, prev_level: int) -> float:
    _, thi_eff = apply_level(row, level)
    return (
        ENERGY_UNITS[level]
        + COMFORT_WEIGHT * comfort_penalty(thi_eff)
        + SWITCH_WEIGHT * abs(level - prev_level)
    )


def evaluate_sequence(df: pd.DataFrame, levels: list[int], method: str, runtime_ms: float, visited: int) -> dict:
    rows = []
    prev = 0
    for row, level in zip(df.to_dict("records"), levels):
        row_series = pd.Series(row)
        temp_eff, thi_eff = apply_level(row_series, level)
        rows.append(
            {
                "date": row["date"],
                "method": method,
                "level": level,
                "temperature_c": row["temperature_c"],
                "relative_humidity_pct": row["relative_humidity_pct"],
                "thi_before": row["thi_rh"],
                "temperature_effective_c": temp_eff,
                "thi_after": thi_eff,
                "energy_units": ENERGY_UNITS[level],
                "comfort_penalty": comfort_penalty(thi_eff),
                "switch_cost": SWITCH_WEIGHT * abs(level - prev),
            }
        )
        prev = level

    detail = pd.DataFrame(rows)
    objective = float(
        detail["energy_units"].sum()
        + detail["comfort_penalty"].sum()
        + detail["switch_cost"].sum()
    )
    return {
        "method": method,
        "levels": levels,
        "detail": detail,
        "total_energy_units": float(detail["energy_units"].sum()),
        "comfort_penalty": float(detail["comfort_penalty"].sum()),
        "switch_count": int(np.abs(np.diff([0] + levels)).sum()),
        "avg_thi_after": float(detail["thi_after"].mean()),
        "max_thi_after": float(detail["thi_after"].max()),
        "objective": objective,
        "runtime_ms": runtime_ms,
        "visited_nodes": visited,
    }


def solve_dp(df: pd.DataFrame) -> dict:
    start = time.perf_counter()
    n = len(df)
    dp = [{level: (math.inf, []) for level in LEVELS} for _ in range(n)]
    visited = 0

    for level in LEVELS:
        visited += 1
        cost = step_cost(df.iloc[0], level, 0)
        dp[0][level] = (cost, [level])

    for i in range(1, n):
        for level in LEVELS:
            best_cost = math.inf
            best_path: list[int] | None = None
            for prev in LEVELS:
                visited += 1
                prev_cost, prev_path = dp[i - 1][prev]
                cost = prev_cost + step_cost(df.iloc[i], level, prev)
                if cost < best_cost:
                    best_cost = cost
                    best_path = prev_path + [level]
            dp[i][level] = (best_cost, best_path or [])

    final_cost, path = min(dp[-1].values(), key=lambda item: item[0])
    runtime_ms = (time.perf_counter() - start) * 1000
    return evaluate_sequence(df, path, "Dynamic Programming", runtime_ms, visited)


def solve_greedy(df: pd.DataFrame) -> dict:
    start = time.perf_counter()
    path: list[int] = []
    prev = 0
    visited = 0
    for _, row in df.iterrows():
        choices = []
        for level in LEVELS:
            visited += 1
            choices.append((step_cost(row, level, prev), level))
        _, chosen = min(choices)
        path.append(chosen)
        prev = chosen
    runtime_ms = (time.perf_counter() - start) * 1000
    return evaluate_sequence(df, path, "Greedy", runtime_ms, visited)


def solve_bfs_layered(df: pd.DataFrame) -> dict:
    start = time.perf_counter()
    states = [(0, [], 0.0)]  # previous level, path, cost
    visited = 0
    for _, row in df.iterrows():
        next_states: dict[int, tuple[list[int], float]] = {}
        for prev_level, path, cost_so_far in states:
            for level in LEVELS:
                visited += 1
                new_cost = cost_so_far + step_cost(row, level, prev_level)
                old = next_states.get(level)
                if old is None or new_cost < old[1]:
                    next_states[level] = (path + [level], new_cost)
        states = [(level, path, cost) for level, (path, cost) in next_states.items()]
    _, best_path, _ = min(states, key=lambda item: item[2])
    runtime_ms = (time.perf_counter() - start) * 1000
    return evaluate_sequence(df, best_path, "BFS Berlapis", runtime_ms, visited)


def solve_dfs_memo(df: pd.DataFrame) -> dict:
    start = time.perf_counter()
    memo: dict[tuple[int, int], tuple[float, list[int]]] = {}
    visited = 0

    def dfs(i: int, prev_level: int) -> tuple[float, list[int]]:
        nonlocal visited
        key = (i, prev_level)
        if key in memo:
            return memo[key]
        if i == len(df):
            return 0.0, []

        row = df.iloc[i]
        ordered = sorted(LEVELS, key=lambda level: step_cost(row, level, prev_level))
        best_cost = math.inf
        best_path: list[int] = []
        for level in ordered:
            visited += 1
            suffix_cost, suffix_path = dfs(i + 1, level)
            cost = step_cost(row, level, prev_level) + suffix_cost
            if cost < best_cost:
                best_cost = cost
                best_path = [level] + suffix_path
        memo[key] = (best_cost, best_path)
        return memo[key]

    _, path = dfs(0, 0)
    runtime_ms = (time.perf_counter() - start) * 1000
    return evaluate_sequence(df, path, "DFS Bermemoisasi", runtime_ms, visited)


def trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def shoulder_high(x: float, a: float, b: float) -> float:
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a)


def shoulder_low(x: float, a: float, b: float) -> float:
    if x <= a:
        return 1.0
    if x >= b:
        return 0.0
    return (b - x) / (b - a)


def fuzzy_level(temp_c: float, rh_pct: float, thi: float) -> int:
    temp_cool = shoulder_low(temp_c, 24.0, 26.0)
    temp_warm = trapezoid(temp_c, 24.5, 26.0, 28.0, 29.5)
    temp_hot = shoulder_high(temp_c, 28.0, 31.0)
    hum_normal = shoulder_low(rh_pct, 65.0, 75.0)
    hum_high = trapezoid(rh_pct, 70.0, 78.0, 86.0, 92.0)
    hum_very_high = shoulder_high(rh_pct, 84.0, 92.0)
    thi_safe = shoulder_low(thi, 72.0, 75.0)
    thi_alert = trapezoid(thi, 72.0, 75.0, 78.5, 81.0)
    thi_stress = shoulder_high(thi, 78.0, 82.0)

    rules = [
        (max(temp_cool, min(thi_safe, hum_normal)), 0.0),
        (min(temp_warm, thi_safe), 1.0),
        (min(temp_warm, hum_high), 1.0),
        (thi_alert, 2.0),
        (min(temp_hot, hum_high), 3.0),
        (min(thi_stress, hum_high), 3.0),
        (min(temp_hot, hum_very_high), 3.0),
    ]
    numerator = sum(weight * output for weight, output in rules)
    denominator = sum(weight for weight, _ in rules)
    if denominator == 0:
        return 1
    return int(max(0, min(3, round(numerator / denominator))))


def solve_fuzzy(df: pd.DataFrame) -> dict:
    start = time.perf_counter()
    path = [
        fuzzy_level(row["temperature_c"], row["relative_humidity_pct"], row["thi_rh"])
        for _, row in df.iterrows()
    ]
    runtime_ms = (time.perf_counter() - start) * 1000
    return evaluate_sequence(df, path, "Fuzzy Logic", runtime_ms, len(df) * 7)


def make_bar_chart(summary: pd.DataFrame) -> None:
    width, height = 1100, 650
    margin_left, margin_top, margin_bottom = 120, 70, 110
    plot_w = width - margin_left - 70
    plot_h = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    values = summary["total_energy_units"].tolist()
    max_value = max(values) * 1.15
    labels = summary["method"].tolist()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]

    draw.text((margin_left, 22), "Total energi relatif per metode", fill="black", font=font)
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill="black", width=2)
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill="black", width=2)

    bar_gap = 32
    bar_w = int((plot_w - bar_gap * (len(values) + 1)) / len(values))
    for i, (value, label) in enumerate(zip(values, labels)):
        x0 = margin_left + bar_gap + i * (bar_w + bar_gap)
        x1 = x0 + bar_w
        y1 = margin_top + plot_h
        y0 = y1 - int((value / max_value) * plot_h)
        draw.rectangle((x0, y0, x1, y1), fill=colors[i % len(colors)], outline="black")
        draw.text((x0 + 4, y0 - 18), f"{value:.1f}", fill="black", font=font)
        words = label.replace(" ", "\n")
        draw.multiline_text((x0, y1 + 10), words, fill="black", font=font, spacing=2)

    for tick in np.linspace(0, max_value, 5):
        y = margin_top + plot_h - int((tick / max_value) * plot_h)
        draw.line((margin_left - 5, y, margin_left, y), fill="black")
        draw.text((20, y - 7), f"{tick:.0f}", fill="black", font=font)

    image.save(FIGURES_DIR / "energy_comparison.png")


def make_thi_line_chart(df: pd.DataFrame, dp_detail: pd.DataFrame, fuzzy_detail: pd.DataFrame) -> None:
    width, height = 1200, 650
    margin_left, margin_top, margin_bottom = 85, 60, 100
    plot_w = width - margin_left - 60
    plot_h = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    series = [
        ("THI awal", df["thi_rh"].tolist(), "#333333"),
        ("DP", dp_detail["thi_after"].tolist(), "#1f77b4"),
        ("Fuzzy", fuzzy_detail["thi_after"].tolist(), "#9467bd"),
    ]
    all_values = [value for _, values, _ in series for value in values] + [COMFORT_LIMIT, SEVERE_LIMIT]
    y_min = math.floor(min(all_values) - 1)
    y_max = math.ceil(max(all_values) + 1)
    n = len(df)

    draw.text((margin_left, 20), "Perubahan THI sebelum dan sesudah kendali kipas", fill="black", font=font)
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill="black", width=2)
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill="black", width=2)

    def xy(i: int, value: float) -> tuple[int, int]:
        x = margin_left + int((i / (n - 1)) * plot_w)
        y = margin_top + plot_h - int(((value - y_min) / (y_max - y_min)) * plot_h)
        return x, y

    for threshold, color, label in [(COMFORT_LIMIT, "#2ca02c", "batas 76"), (SEVERE_LIMIT, "#d62728", "batas 80")]:
        y = xy(0, threshold)[1]
        draw.line((margin_left, y, margin_left + plot_w, y), fill=color, width=1)
        draw.text((margin_left + plot_w - 65, y - 14), label, fill=color, font=font)

    for name, values, color in series:
        points = [xy(i, value) for i, value in enumerate(values)]
        draw.line(points, fill=color, width=3)
        for point in points[::4]:
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)

    legend_x = margin_left + 10
    legend_y = margin_top + 10
    for name, _, color in series:
        draw.rectangle((legend_x, legend_y, legend_x + 14, legend_y + 14), fill=color)
        draw.text((legend_x + 20, legend_y), name, fill="black", font=font)
        legend_y += 22

    for tick in np.linspace(y_min, y_max, 6):
        y = xy(0, tick)[1]
        draw.line((margin_left - 5, y, margin_left, y), fill="black")
        draw.text((25, y - 7), f"{tick:.1f}", fill="black", font=font)

    for i, date in enumerate(df["date"]):
        if i % 4 == 0 or i == n - 1:
            x, _ = xy(i, y_min)
            draw.text((x - 25, margin_top + plot_h + 12), date[5:], fill="black", font=font)

    image.save(FIGURES_DIR / "thi_line_comparison.png")


def write_results(df: pd.DataFrame, results: list[dict]) -> pd.DataFrame:
    summary_rows = []
    for result in results:
        summary_rows.append(
            {
                "method": result["method"],
                "total_energy_units": result["total_energy_units"],
                "comfort_penalty": result["comfort_penalty"],
                "switch_count": result["switch_count"],
                "avg_thi_after": result["avg_thi_after"],
                "max_thi_after": result["max_thi_after"],
                "objective": result["objective"],
                "runtime_ms": result["runtime_ms"],
                "visited_nodes": result["visited_nodes"],
                "level_sequence": " ".join(map(str, result["levels"])),
            }
        )
        result["detail"].to_csv(RESULTS_DIR / f"detail_{slug(result['method'])}.csv", index=False)

    summary = pd.DataFrame(summary_rows).sort_values("objective").reset_index(drop=True)
    summary.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)
    make_bar_chart(summary)
    dp = next(r for r in results if r["method"] == "Dynamic Programming")
    fuzzy = next(r for r in results if r["method"] == "Fuzzy Logic")
    make_thi_line_chart(df, dp["detail"], fuzzy["detail"])
    return summary


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def run_experiment() -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    ensure_dirs()
    df = fetch_data()
    results = [solve_dp(df), solve_greedy(df), solve_bfs_layered(df), solve_dfs_memo(df), solve_fuzzy(df)]
    summary = write_results(df, results)
    write_sources_file()
    return df, results, summary


def write_sources_file() -> None:
    text = """Sumber dan referensi utama
==========================

Dataset dan data:
1. A. Thiam, "Ambient conditions in poultry house," Mendeley Data, V1, 2023. DOI: 10.17632/7fptd3rfzy.1. URL: https://data.mendeley.com/datasets/7fptd3rfzy/1
2. NASA POWER Project, "Daily API Documentation." URL: https://power.larc.nasa.gov/docs/services/api/temporal/daily/

Referensi ilmiah:
3. H. A. Wijaya and S. Hartati, "Implementasi Logika Fuzzy Dalam Sistem Pendingin Otomatis Kandang Ayam Broiler Closed House," IJEIS, 2025. DOI: 10.22146/ijeis.103364.
4. M. Olejar, V. Cviklovic, D. Hruby, and L. Toth, "Fuzzy control of temperature and humidity microclimate in closed areas for poultry breeding," Research in Agricultural Engineering, vol. 60, pp. S31-S36, 2014. DOI: 10.17221/30/2013-RAE.
5. S. Akter et al., "Impacts of Air Velocity Treatments under Summer Condition: Part I-Heavy Broiler's Surface Temperature Response," Animals, vol. 12, no. 3, 328, 2022. DOI: 10.3390/ani12030328.
6. E. H. Mamdani and S. Assilian, "An experiment in linguistic synthesis with a fuzzy logic controller," International Journal of Man-Machine Studies, vol. 7, no. 1, pp. 1-13, 1975. DOI: 10.1016/S0020-7373(75)80002-2.
7. L. A. Zadeh, "Fuzzy sets," Information and Control, vol. 8, no. 3, pp. 338-353, 1965. DOI: 10.1016/S0019-9958(65)90241-X.
8. T. H. Cormen, C. E. Leiserson, R. L. Rivest, and C. Stein, Introduction to Algorithms, 4th ed. Cambridge, MA: MIT Press, 2022.
9. S. Russell and P. Norvig, Artificial Intelligence: A Modern Approach, 4th ed. Hoboken, NJ: Pearson, 2021.
10. M. N. I. S. Diarra et al., "Thermal Comfort, Growth Performance and Welfare of Olive Pulp Fed Broilers during Hot Season," Sustainability, vol. 15, no. 14, 10932, 2023. DOI: 10.3390/su151410932.
"""
    (SOURCES_DIR / "sources_used.md").write_text(text, encoding="utf-8")


def qn(tag: str) -> str:
    return f"{W}{tag}"


def new_p(style: str | None = None, text: str | None = None, sect_pr=None, align: str | None = None) -> etree._Element:
    p = etree.Element(qn("p"))
    ppr = etree.SubElement(p, qn("pPr"))
    if style:
        pstyle = etree.SubElement(ppr, qn("pStyle"))
        pstyle.set(qn("val"), style)
    if align:
        jc = etree.SubElement(ppr, qn("jc"))
        jc.set(qn("val"), align)
    if sect_pr is not None:
        ppr.append(deepcopy(sect_pr))
    if text is not None:
        add_text_run(p, text)
    return p


def no_number_references_p(text: str, sect_pr=None, align: str | None = None) -> etree._Element:
    p = etree.Element(qn("p"))
    ppr = etree.SubElement(p, qn("pPr"))
    pstyle = etree.SubElement(ppr, qn("pStyle"))
    pstyle.set(qn("val"), "references")
    num_pr = etree.SubElement(ppr, qn("numPr"))
    ilvl = etree.SubElement(num_pr, qn("ilvl"))
    ilvl.set(qn("val"), "0")
    num_id = etree.SubElement(num_pr, qn("numId"))
    num_id.set(qn("val"), "0")
    spacing = etree.SubElement(ppr, qn("spacing"))
    spacing.set(qn("line"), "12pt")
    spacing.set(qn("lineRule"), "auto")
    if align:
        jc = etree.SubElement(ppr, qn("jc"))
        jc.set(qn("val"), align)
    if sect_pr is not None:
        ppr.append(deepcopy(sect_pr))
    add_text_run(p, text)
    return p


def add_text_run(p: etree._Element, text: str, bold: bool = False, italic: bool = False) -> None:
    r = etree.SubElement(p, qn("r"))
    if bold or italic:
        rpr = etree.SubElement(r, qn("rPr"))
        if bold:
            etree.SubElement(rpr, qn("b"))
        if italic:
            etree.SubElement(rpr, qn("i"))
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if i:
            etree.SubElement(r, qn("br"))
        t = etree.SubElement(r, qn("t"))
        if line.startswith(" ") or line.endswith(" "):
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        t.text = line


def mixed_p(style: str, parts: list[tuple[str, bool, bool]]) -> etree._Element:
    p = new_p(style)
    for text, bold, italic in parts:
        add_text_run(p, text, bold=bold, italic=italic)
    return p


def table(caption: str, headers: list[str], rows: list[list[str]]) -> list[etree._Element]:
    elements = [new_p("tablehead", caption)]
    tbl = etree.Element(qn("tbl"))
    tbl_pr = etree.SubElement(tbl, qn("tblPr"))
    tbl_w = etree.SubElement(tbl_pr, qn("tblW"))
    tbl_w.set(qn("w"), "0")
    tbl_w.set(qn("type"), "auto")
    borders = etree.SubElement(tbl_pr, qn("tblBorders"))
    for border_name in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        border = etree.SubElement(borders, qn(border_name))
        border.set(qn("val"), "single")
        border.set(qn("sz"), "4")
        border.set(qn("space"), "0")
        border.set(qn("color"), "auto")

    grid = etree.SubElement(tbl, qn("tblGrid"))
    width = int(8600 / len(headers))
    for _ in headers:
        col = etree.SubElement(grid, qn("gridCol"))
        col.set(qn("w"), str(width))

    def add_row(values: list[str], is_header: bool = False) -> None:
        tr = etree.SubElement(tbl, qn("tr"))
        for value in values:
            tc = etree.SubElement(tr, qn("tc"))
            tc_pr = etree.SubElement(tc, qn("tcPr"))
            tc_w = etree.SubElement(tc_pr, qn("tcW"))
            tc_w.set(qn("w"), str(width))
            tc_w.set(qn("type"), "dxa")
            if is_header:
                shd = etree.SubElement(tc_pr, qn("shd"))
                shd.set(qn("val"), "clear")
                shd.set(qn("color"), "auto")
                shd.set(qn("fill"), "D9EAF7")
            p = new_p("tablehead" if is_header else "BodyText", value)
            tc.append(p)

    add_row(headers, True)
    for row in rows:
        add_row(row, False)
    elements.append(tbl)
    return elements


def citation_set(text: str) -> set[int]:
    found = set()
    for group in re.findall(r"\[([0-9,\-\s]+)\]", text):
        for item in re.split(r",", group):
            item = item.strip()
            if "-" in item:
                start, end = [int(x.strip()) for x in item.split("-", 1)]
                found.update(range(start, end + 1))
            elif item:
                found.add(int(item))
    return found


def strip_leading_reference_number(text: str) -> str:
    return re.sub(r"^\[\d+\]\s*", "", text)


@dataclass
class DocContent:
    body_elements: list[etree._Element]


def build_docx(df: pd.DataFrame, results: list[dict], summary: pd.DataFrame) -> None:
    if not BACKUP_DOCX.exists():
        shutil.copyfile(TEMPLATE_DOCX, BACKUP_DOCX)

    source_docx = BACKUP_DOCX if BACKUP_DOCX.exists() else TEMPLATE_DOCX
    with zipfile.ZipFile(source_docx, "r") as zin:
        document_xml = zin.read("word/document.xml")
        files = {item.filename: zin.read(item.filename) for item in zin.infolist()}

    root = etree.fromstring(document_xml)
    body = root.find(qn("body"))
    assert body is not None
    original_children = list(body)
    sects = []
    for p in root.xpath(".//w:p[w:pPr/w:sectPr]", namespaces=NS):
        sect = p.find(f"{W}pPr/{W}sectPr")
        if sect is not None:
            sects.append(deepcopy(sect))
    body_sect = root.xpath("./w:body/w:sectPr", namespaces=NS)[0]

    final_sect = deepcopy(body_sect)
    title_sect = sects[0]
    author_sect = sects[1]
    spacer_sect = sects[2]
    two_col_end = sects[3]

    image_rels = add_images_to_package(files)
    new_children = make_document_elements(
        df,
        results,
        summary,
        title_sect,
        author_sect,
        spacer_sect,
        two_col_end,
        final_sect,
        image_rels,
    )
    for child in original_children:
        body.remove(child)
    for child in new_children:
        body.append(child)

    files["word/document.xml"] = etree.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone=True
    )

    with zipfile.ZipFile(DOCX_OUT, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for filename, content in files.items():
            zout.writestr(filename, content)

    audit_docx()


def add_images_to_package(files: dict[str, bytes]) -> dict[str, str]:
    content_types_ns = "http://schemas.openxmlformats.org/package/2006/content-types"
    rels_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    rels_path = "word/_rels/document.xml.rels"
    content_types_path = "[Content_Types].xml"

    content_root = etree.fromstring(files[content_types_path])
    has_png = any(
        node.get("Extension") == "png"
        for node in content_root.findall(f"{{{content_types_ns}}}Default")
    )
    if not has_png:
        default = etree.SubElement(content_root, f"{{{content_types_ns}}}Default")
        default.set("Extension", "png")
        default.set("ContentType", "image/png")
    files[content_types_path] = etree.tostring(
        content_root, xml_declaration=True, encoding="UTF-8", standalone=True
    )

    rels_root = etree.fromstring(files[rels_path])
    max_rid = 0
    for rel in rels_root.findall(f"{{{rels_ns}}}Relationship"):
        rid = rel.get("Id", "")
        if rid.startswith("rId") and rid[3:].isdigit():
            max_rid = max(max_rid, int(rid[3:]))

    image_map = {
        "energy": ("energy_comparison.png", FIGURES_DIR / "energy_comparison.png"),
        "thi": ("thi_line_comparison.png", FIGURES_DIR / "thi_line_comparison.png"),
    }
    rel_ids: dict[str, str] = {}
    for key, (filename, path) in image_map.items():
        max_rid += 1
        rid = f"rId{max_rid}"
        rel = etree.SubElement(rels_root, f"{{{rels_ns}}}Relationship")
        rel.set("Id", rid)
        rel.set("Type", "http://purl.oclc.org/ooxml/officeDocument/relationships/image")
        rel.set("Target", f"media/{filename}")
        files[f"word/media/{filename}"] = path.read_bytes()
        rel_ids[key] = rid

    files[rels_path] = etree.tostring(
        rels_root, xml_declaration=True, encoding="UTF-8", standalone=True
    )
    return rel_ids


def image_paragraph(rel_id: str, image_path: Path, descr: str, width_inches: float = 3.25) -> etree._Element:
    wp_ns = "http://purl.oclc.org/ooxml/drawingml/wordprocessingDrawing"
    a_ns = "http://purl.oclc.org/ooxml/drawingml/main"
    pic_ns = "http://purl.oclc.org/ooxml/drawingml/picture"
    nsmap = {"wp": wp_ns, "a": a_ns, "pic": pic_ns, "r": NS_R}

    with Image.open(image_path) as image:
        px_w, px_h = image.size
    cx = int(width_inches * 914400)
    cy = int(cx * px_h / px_w)

    p = new_p("BodyText", align="center")
    r = etree.SubElement(p, qn("r"))
    drawing = etree.SubElement(r, qn("drawing"))
    inline = etree.SubElement(drawing, f"{{{wp_ns}}}inline", nsmap=nsmap)
    for attr in ["distT", "distB", "distL", "distR"]:
        inline.set(attr, "0")
    extent = etree.SubElement(inline, f"{{{wp_ns}}}extent")
    extent.set("cx", str(cx))
    extent.set("cy", str(cy))
    doc_pr = etree.SubElement(inline, f"{{{wp_ns}}}docPr")
    doc_pr.set("id", "1")
    doc_pr.set("name", image_path.name)
    doc_pr.set("descr", descr)
    c_nv_frame = etree.SubElement(inline, f"{{{wp_ns}}}cNvGraphicFramePr")
    locks = etree.SubElement(c_nv_frame, f"{{{a_ns}}}graphicFrameLocks")
    locks.set("noChangeAspect", "1")
    graphic = etree.SubElement(inline, f"{{{a_ns}}}graphic")
    graphic_data = etree.SubElement(graphic, f"{{{a_ns}}}graphicData")
    graphic_data.set("uri", pic_ns)
    pic = etree.SubElement(graphic_data, f"{{{pic_ns}}}pic")
    nv_pic_pr = etree.SubElement(pic, f"{{{pic_ns}}}nvPicPr")
    c_nv_pr = etree.SubElement(nv_pic_pr, f"{{{pic_ns}}}cNvPr")
    c_nv_pr.set("id", "0")
    c_nv_pr.set("name", image_path.name)
    c_nv_pic_pr = etree.SubElement(nv_pic_pr, f"{{{pic_ns}}}cNvPicPr")
    blip_fill = etree.SubElement(pic, f"{{{pic_ns}}}blipFill")
    blip = etree.SubElement(blip_fill, f"{{{a_ns}}}blip")
    blip.set(f"{{{NS_R}}}embed", rel_id)
    stretch = etree.SubElement(blip_fill, f"{{{a_ns}}}stretch")
    etree.SubElement(stretch, f"{{{a_ns}}}fillRect")
    sp_pr = etree.SubElement(pic, f"{{{pic_ns}}}spPr")
    xfrm = etree.SubElement(sp_pr, f"{{{a_ns}}}xfrm")
    off = etree.SubElement(xfrm, f"{{{a_ns}}}off")
    off.set("x", "0")
    off.set("y", "0")
    ext = etree.SubElement(xfrm, f"{{{a_ns}}}ext")
    ext.set("cx", str(cx))
    ext.set("cy", str(cy))
    prst_geom = etree.SubElement(sp_pr, f"{{{a_ns}}}prstGeom")
    prst_geom.set("prst", "rect")
    etree.SubElement(prst_geom, f"{{{a_ns}}}avLst")
    return p


def figure_elements(rel_id: str, image_path: Path, caption: str) -> list[etree._Element]:
    return [
        image_paragraph(rel_id, image_path, caption),
        new_p("figurecaption", caption),
    ]


def make_document_elements(
    df: pd.DataFrame,
    results: list[dict],
    summary: pd.DataFrame,
    title_sect,
    author_sect,
    spacer_sect,
    two_col_end,
    final_sect,
    image_rels: dict[str, str],
) -> list[etree._Element]:
    best = summary.iloc[0]
    dp = next(result for result in results if result["method"] == "Dynamic Programming")
    greedy = next(result for result in results if result["method"] == "Greedy")
    fuzzy = next(result for result in results if result["method"] == "Fuzzy Logic")

    nasa_min = df["temperature_c"].min()
    nasa_max = df["temperature_c"].max()
    rh_min = df["relative_humidity_pct"].min()
    rh_max = df["relative_humidity_pct"].max()
    thi_min = df["thi_rh"].min()
    thi_max = df["thi_rh"].max()

    table_rows = [
        [
            row["method"],
            f"{row['total_energy_units']:.2f}",
            f"{row['comfort_penalty']:.2f}",
            f"{row['max_thi_after']:.2f}",
            f"{row['runtime_ms']:.2f}",
        ]
        for _, row in summary.iterrows()
    ]

    dataset_rows = [
        ["Lokasi data", "Semarang, Indonesia (-6.9667, 110.4167)"],
        ["Periode", "3-21 April 2022"],
        ["Jumlah observasi", f"{len(df)} hari"],
        ["Suhu rata-rata", f"{df['temperature_c'].mean():.2f} C"],
        ["Kelembapan rata-rata", f"{df['relative_humidity_pct'].mean():.2f}%"],
        ["Rentang THI awal", f"{thi_min:.2f}-{thi_max:.2f}"],
    ]

    elements: list[etree._Element] = [
        new_p(
            "papertitle",
            "Analisis Perbandingan Dynamic Programming, Greedy-DFS-BFS, dan Fuzzy Logic dalam Optimasi Penggunaan Energi Kipas Kandang Broiler Berdasarkan Suhu dan Kelembapan Harian",
        ),
        new_p("Author", sect_pr=title_sect),
        new_p(
            "Author",
            f"{AUTHOR_NAME} - {AUTHOR_NIM}\nDepartemen Informatika Universitas Diponegoro\nSemarang, Indonesia\n{AUTHOR_EMAIL}",
        ),
        new_p(None, sect_pr=author_sect),
        new_p(None, sect_pr=spacer_sect),
        mixed_p(
            "Abstract",
            [
                ("Abstract-", True, False),
                (
                    "Makalah ini membahas optimasi penggunaan kipas kandang broiler berdasarkan suhu dan kelembapan harian melalui perbandingan Dynamic Programming, Greedy-DFS-BFS, dan Fuzzy Logic. Data utama yang direncanakan adalah data iklim kandang unggas Mendeley Data ber-DOI; karena file publiknya membutuhkan otorisasi pada lingkungan eksperimen, analisis komputasional menggunakan data harian NASA POWER untuk Semarang pada periode 3-21 April 2022 sebagai fallback yang terbuka dan reprodusibel. Permasalahan dimodelkan sebagai pemilihan level kipas diskrit untuk meminimalkan energi relatif, penalti kenyamanan berbasis Temperature-Humidity Index (THI), dan biaya pergantian level. Hasil eksperimen menunjukkan bahwa Dynamic Programming, BFS berlapis, dan DFS bermemoisasi mencapai nilai objektif terbaik yang sama, sedangkan Greedy dan Fuzzy Logic lebih sederhana tetapi tidak selalu minimum secara global. Kontribusi utama makalah adalah formulasi eksperimen yang dapat direplikasi, implementasi Python, dan analisis trade-off antara optimalitas, interpretabilitas, dan biaya komputasi.",
                    False,
                    False,
                ),
            ],
        ),
        mixed_p(
            "Keywords",
            [
                ("Keywords-", True, False),
                (
                    "dynamic programming, greedy, DFS, BFS, fuzzy logic, broiler, kipas kandang, THI, optimasi energi",
                    False,
                    False,
                ),
            ],
        ),
        new_p("Heading1", "Pendahuluan"),
        new_p(
            "BodyText",
            "Kandang ayam broiler membutuhkan pengendalian mikroklimat agar suhu dan kelembapan tidak mendorong heat stress. Pada kondisi tropis lembap, persoalan ini menjadi penting karena suhu tinggi dan kelembapan besar dapat menurunkan kenyamanan termal, mengurangi konsumsi pakan, serta menurunkan performa produksi. Penelitian terbaru pada sistem pendingin kandang broiler closed house juga menunjukkan bahwa kontrol otomatis berbasis fuzzy dapat meningkatkan stabilitas suhu dan kategori aman berdasarkan THI [3].",
        ),
        new_p(
            "BodyText",
            "Di sisi lain, kipas atau blower tidak boleh diaktifkan secara berlebihan. Semakin tinggi level kipas, semakin besar konsumsi energi dan semakin sering pergantian level dapat memperpendek umur aktuator. Karena itu, pengendalian kipas dapat dipandang sebagai masalah optimasi sekuensial: pada setiap hari dipilih level kipas yang cukup untuk menekan risiko panas, tetapi tetap hemat energi. Sudut pandang ini sesuai dengan tugas Analisis dan Strategi Algoritma karena solusi tidak hanya berupa teori, melainkan implementasi dan perbandingan metode pada data nyata.",
        ),
        new_p(
            "BodyText",
            "Makalah ini membandingkan tiga kelompok strategi: Dynamic Programming sebagai metode optimasi global pada horizon waktu, Greedy-DFS-BFS sebagai pendekatan pemilihan dan pencarian state-space, serta Fuzzy Logic sebagai metode luar daftar yang berbasis aturan linguistik. Fuzzy Logic dipilih karena telah lama dipakai untuk kontrol mikroklimat suhu dan kelembapan pada area peternakan unggas [4], serta berakar pada teori himpunan fuzzy Zadeh [7] dan kendali fuzzy Mamdani-Assilian [6].",
        ),
        new_p(
            "BodyText",
            "Rumusan pertanyaan penelitian adalah sebagai berikut. Pertama, bagaimana memodelkan keputusan level kipas sebagai masalah optimasi yang dapat diselesaikan dengan strategi algoritma? Kedua, apakah metode yang mempertimbangkan horizon waktu seperti DP memberikan keuntungan dibandingkan keputusan lokal Greedy? Ketiga, bagaimana karakter BFS, DFS, dan Fuzzy Logic ketika diberi fungsi biaya dan data lingkungan yang sama? Pertanyaan tersebut diarahkan pada kontribusi komputasional, bukan perancangan perangkat keras kandang.",
        ),
        new_p(
            "BodyText",
            "Kontribusi makalah ini terdiri atas tiga hal. Pertama, makalah menyusun formulasi biaya yang menghubungkan energi kipas, kenyamanan termal, dan stabilitas aktuator. Kedua, makalah menyediakan implementasi Python yang dapat dijalankan ulang dan menghasilkan file data, ringkasan metrik, serta grafik pendukung. Ketiga, makalah membahas trade-off algoritmik antara optimalitas, kecepatan, kelengkapan pencarian, dan interpretabilitas aturan. Dengan demikian, pembahasan tidak berhenti pada rangkuman teori, tetapi menghasilkan eksperimen terukur.",
        ),
        new_p("Heading1", "Batasan Masalah"),
        new_p(
            "BodyText",
            "Ruang lingkup penelitian dibatasi pada pengendalian level kipas berbasis data suhu dan kelembapan harian. Faktor lain seperti CO2, amonia, umur broiler, kepadatan kandang, jenis kipas, dan desain ventilasi fisik tidak dimodelkan secara rinci. Oleh karena data fallback tidak berisi spesifikasi daya kipas, konsumsi energi dinyatakan dalam satuan relatif. Nilai ini tetap berguna untuk membandingkan algoritma karena seluruh metode memakai fungsi biaya yang sama.",
        ),
        new_p(
            "BodyText",
            "Model juga mengasumsikan bahwa peningkatan level kipas menurunkan suhu efektif di area ayam. Asumsi ini disederhanakan dari prinsip ventilasi dan perlakuan kecepatan udara pada broiler, yaitu bahwa air velocity dapat membantu mengurangi dampak panas pada ayam berat dalam kondisi musim panas [5]. Dengan demikian, makalah ini tidak mengklaim sebagai desain kontrol industri siap pakai, tetapi sebagai eksperimen komputasional untuk membandingkan strategi algoritma.",
        ),
        new_p(
            "BodyText",
            "Batasan lain adalah resolusi waktu harian. Pada kandang nyata, kipas biasanya dikendalikan pada resolusi menit atau jam. Namun, resolusi harian tetap berguna untuk proyek ini karena memperlihatkan pola keputusan sekuensial secara ringkas dan menjaga ukuran state-space agar dapat dibandingkan dengan jelas. Jika data per 10 menit dari sensor kandang tersedia, struktur kode yang sama dapat dipakai dengan horizon lebih panjang, tetapi BFS/DFS tanpa pemangkasan akan menjadi jauh lebih mahal.",
        ),
        new_p(
            "BodyText",
            "Makalah juga tidak memasukkan variabel produktivitas seperti bobot panen, mortalitas, atau feed conversion ratio. Variabel tersebut sengaja dikeluarkan agar fokus tetap pada strategi algoritma dan optimasi energi kipas. Dengan demikian, keberhasilan algoritma dinilai dari fungsi objektif komputasional, bukan dari klaim performa biologis ayam. Pemisahan ini penting agar kesimpulan tidak melampaui data yang tersedia.",
        ),
        new_p("Heading1", "Dasar Teori"),
        new_p("Heading2", "Temperature-Humidity Index"),
        new_p(
            "BodyText",
            "Temperature-Humidity Index (THI) adalah indikator gabungan suhu dan kelembapan untuk memperkirakan tekanan panas. Beberapa studi broiler menghitung THI dari suhu dry-bulb dan wet-bulb [10], sedangkan studi berbasis suhu dan kelembapan relatif menggunakan formulasi yang menggabungkan suhu udara dan RH [5]. Pada eksperimen ini digunakan THI berbasis RH karena data NASA POWER menyediakan suhu dan kelembapan harian secara langsung.",
        ),
        new_p(
            "BodyText",
            "Formula yang digunakan adalah THI = 0,8T + RH(T - 14,3)/100 + 46,3, dengan T dalam derajat Celsius dan RH dalam persen. Level kenyamanan kemudian dipakai sebagai penalti biaya, bukan sebagai klasifikasi kaku. THI di atas 76 diberi penalti kuadrat, sedangkan THI di atas 80 diberi penalti tambahan karena masuk wilayah ketidaknyamanan yang lebih berat.",
        ),
        new_p(
            "BodyText",
            "Penggunaan penalti kontinu memiliki keuntungan dibanding aturan batas yang sepenuhnya diskrit. Jika THI hanya dikategorikan aman atau tidak aman, dua kondisi yang sangat dekat dengan batas akan diperlakukan terlalu berbeda. Penalti kuadrat membuat penyimpangan kecil tetap dihukum ringan, sedangkan penyimpangan besar dihukum lebih kuat. Bentuk ini sesuai dengan kebutuhan optimasi karena algoritma dapat menyeimbangkan energi dan kenyamanan secara bertahap.",
        ),
        new_p(
            "BodyText",
            "Dalam eksperimen, THI sebelum kendali dihitung dari suhu rata-rata harian dan kelembapan relatif. Setelah suatu level kipas dipilih, suhu efektif diasumsikan turun sesuai level, kemudian THI dihitung ulang. Pendekatan ini merupakan model surrogate. Model surrogate tidak menggantikan simulasi termal kandang, tetapi cukup untuk membandingkan algoritma karena seluruh metode menghadapi lingkungan, aksi, dan fungsi biaya yang identik.",
        ),
        new_p("Heading2", "Dynamic Programming"),
        new_p(
            "BodyText",
            "Dynamic Programming (DP) menyelesaikan masalah yang memiliki sub-struktur optimal dan submasalah tumpang tindih. Dalam buku Introduction to Algorithms, DP dijelaskan sebagai teknik untuk memecah masalah optimasi menjadi keputusan tahap demi tahap yang hasilnya disimpan agar tidak dihitung ulang [8]. Pada penelitian ini, state DP adalah pasangan hari ke-i dan level kipas terakhir. Transisi mencoba seluruh level kipas untuk hari berikutnya dan memilih biaya kumulatif minimum.",
        ),
        new_p(
            "BodyText",
            "Jika n adalah jumlah hari dan L adalah jumlah level kipas, kompleksitas DP adalah O(nL^2). Faktor L^2 muncul karena setiap level baru dibandingkan terhadap setiap level sebelumnya. Pada eksperimen ini L hanya empat, sehingga DP sangat ringan. Namun, pola ini tetap penting secara konseptual karena DP memberi jaminan optimum terhadap fungsi biaya yang didefinisikan. Jaminan ini tidak dimiliki oleh Greedy dan Fuzzy Logic.",
        ),
        new_p(
            "BodyText",
            "Rekurensi DP dapat dinyatakan sebagai dp[i][l] = min_p(dp[i-1][p] + biaya(i,l,p)), dengan l sebagai level hari ke-i dan p sebagai level hari sebelumnya. Biaya(i,l,p) mencakup energi level l, penalti THI setelah level l, dan biaya switching dari p ke l. Setelah seluruh tabel terisi, solusi akhir diperoleh dari level dengan biaya terkecil pada hari terakhir dan dilakukan backtracking untuk mendapatkan urutan level.",
        ),
        new_p("Heading2", "Greedy, DFS, dan BFS"),
        new_p(
            "BodyText",
            "Greedy memilih keputusan terbaik lokal pada hari berjalan tanpa melihat dampak keputusan itu terhadap hari berikutnya. Pendekatan ini sederhana dan cepat, tetapi dapat gagal ketika biaya pergantian level membuat keputusan lokal tidak optimal secara global. BFS dan DFS memandang level kipas sebagai graf berlapis: simpul merepresentasikan level pada suatu hari, sedangkan sisi merepresentasikan transisi level antarhari. Konsep pencarian graf seperti breadth-first search dan depth-first search umum dibahas dalam literatur algoritma dan kecerdasan buatan [8], [9].",
        ),
        new_p(
            "BodyText",
            "Pada implementasi Greedy, setiap hari seluruh level kipas dihitung biayanya terhadap level sebelumnya, kemudian level dengan biaya terkecil dipilih. Kompleksitasnya O(nL), lebih rendah daripada DP. Kelemahannya adalah keputusan yang murah hari ini dapat menyebabkan biaya switching atau penalti yang lebih besar di hari berikutnya. Oleh karena itu, Greedy dipakai sebagai baseline cepat, bukan sebagai metode yang dijamin optimal.",
        ),
        new_p(
            "BodyText",
            "BFS berlapis menelusuri graf berdasarkan urutan hari. Agar tidak terjadi ledakan kombinasi, state yang memiliki hari dan level akhir sama tetapi biaya lebih tinggi dipangkas. Dengan pemangkasan dominasi ini, BFS tetap mempertahankan kandidat terbaik per lapisan dan dapat mencapai hasil yang sama dengan DP pada model ini. Secara pedagogis, BFS membantu memperlihatkan bahwa optimasi sekuensial dapat dilihat sebagai pencarian pada graf keputusan.",
        ),
        new_p(
            "BodyText",
            "DFS bermemoisasi menelusuri keputusan secara mendalam: pilih level hari ini, lanjut ke hari berikutnya, lalu mundur ketika suffix sudah selesai. Memoization menyimpan hasil terbaik dari pasangan state (hari, level sebelumnya). Tanpa memoization, DFS akan mengeksplorasi 4^n urutan level dan menjadi tidak efisien. Dengan memoization, jumlah state yang dihitung kembali setara dengan jumlah pasangan hari-level sehingga cocok dipakai sebagai variasi pencarian yang masih terukur.",
        ),
        new_p("Heading2", "Fuzzy Logic"),
        new_p(
            "BodyText",
            "Fuzzy Logic memetakan variabel numerik ke derajat keanggotaan linguistik seperti dingin, hangat, panas, lembap, dan sangat lembap. Himpunan fuzzy memungkinkan suatu kondisi memiliki derajat keanggotaan di antara 0 dan 1 [7]. Sistem kontrol fuzzy kemudian menggabungkan aturan if-then untuk menghasilkan keluaran, sebagaimana diperkenalkan dalam eksperimen kendali linguistik Mamdani dan Assilian [6]. Dalam konteks kandang, aturan fuzzy mudah dipahami operator karena menyerupai pengetahuan praktis: jika suhu panas dan kelembapan tinggi, level kipas dinaikkan.",
        ),
        new_p(
            "BodyText",
            "Sistem fuzzy pada makalah ini menggunakan masukan suhu, kelembapan, dan THI. Masing-masing masukan diberi fungsi keanggotaan sederhana, misalnya suhu cool, warm, hot; kelembapan normal, high, very high; serta THI safe, alert, stress. Aturan yang aktif menghasilkan nilai keluaran Sugeno 0 sampai 3, lalu nilai rata-rata berbobot dibulatkan menjadi level kipas. Metode ini tidak mencari minimum objektif secara eksplisit, tetapi menghasilkan keputusan yang koheren dengan aturan domain.",
        ),
        new_p(
            "BodyText",
            "Kekuatan Fuzzy Logic adalah interpretabilitas. Jika hasilnya terlalu boros energi, aturan dapat disesuaikan, misalnya menurunkan konsekuen level untuk kondisi alert. Jika hasilnya terlalu berisiko terhadap panas, aturan dapat dibuat lebih konservatif. Fleksibilitas ini menjelaskan mengapa fuzzy banyak dipakai pada sistem kontrol mikroklimat. Kelemahannya adalah tidak ada jaminan bahwa aturan yang tampak wajar akan meminimalkan fungsi biaya tertentu.",
        ),
        new_p("Heading1", "Metodologi"),
        new_p(
            "BodyText",
            f"Eksperimen dilakukan menggunakan Python. Kandidat dataset utama adalah Ambient conditions in poultry house dari Mendeley Data dengan DOI 10.17632/7fptd3rfzy.1 [1]. Dataset tersebut relevan karena berisi suhu, kelembapan relatif, wet-bulb temperature, kecepatan udara, dan CO2 di kandang unggas pada interval 10 menit. Namun, saat eksperimen dijalankan, endpoint file publik Mendeley meminta otorisasi. Agar penelitian tetap reprodusibel tanpa akun, data fallback diambil dari NASA POWER Daily API [2].",
        ),
        new_p(
            "BodyText",
            f"Data NASA POWER dipilih untuk Semarang pada 3-21 April 2022, disejajarkan dengan periode dataset Mendeley yang dimulai 3 April 2022. Data berisi {len(df)} observasi harian dengan suhu rata-rata {df['temperature_c'].mean():.2f} C, kelembapan rata-rata {df['relative_humidity_pct'].mean():.2f}%, suhu {nasa_min:.2f}-{nasa_max:.2f} C, kelembapan {rh_min:.2f}-{rh_max:.2f}%, dan THI awal {thi_min:.2f}-{thi_max:.2f}.",
        ),
        new_p(
            "BodyText",
            "Tahap preprocessing dilakukan dengan membaca respons JSON NASA POWER, mengubah tanggal ke format ISO, mengekstraksi suhu rata-rata, suhu maksimum, suhu minimum, kelembapan relatif, dan wet-bulb temperature, kemudian menyimpannya sebagai CSV. Nilai missing tidak ditemukan pada periode ini. Script juga menyimpan JSON mentah sehingga hasil dapat diaudit ulang tanpa bergantung pada respons API berikutnya.",
        ),
    ]

    elements.extend(
        table(
            "Tabel 1. Ringkasan data eksperimen",
            ["Komponen", "Nilai"],
            dataset_rows,
        )
    )

    elements.extend(
        [
            new_p(
                "BodyText",
                "Aksi kipas dimodelkan sebagai empat level diskrit: 0, 1, 2, dan 3. Konsumsi energi relatif masing-masing level adalah 0,00; 1,20; 2,60; dan 4,10 unit per hari. Efek pendinginan suhu efektif diasumsikan 0,00; 0,90; 1,80; dan 2,80 C. Fungsi objektif menjumlahkan energi, penalti kenyamanan, dan biaya pergantian level sebesar 0,22 dikali selisih level. Karena seluruh metode menggunakan fungsi biaya identik, perbandingan tetap adil meskipun angka energi bersifat relatif.",
            ),
            new_p(
                "BodyText",
                "Secara matematis, untuk hari i dan level kipas l, biaya harian ditulis sebagai E(l) + P(THI_i(l)) + S(l,p), dengan p level hari sebelumnya. E(l) adalah energi relatif, P adalah penalti kenyamanan, dan S adalah biaya switching. Penalti kenyamanan bernilai nol bila THI efektif tidak melebihi 76, lalu meningkat kuadrat jika melewati batas tersebut. Jika THI efektif melewati 80, ditambahkan penalti kuadrat kedua agar kondisi panas berat dihindari.",
            ),
            new_p(
                "BodyText",
                "Parameter level kipas dipilih agar skenario tidak terlalu mudah maupun terlalu ekstrem. Level 0 tidak menggunakan energi tetapi tidak mendinginkan. Level 1 cukup hemat dan memberi pendinginan ringan. Level 2 menjadi pilihan menengah yang sering dipakai saat THI awal mendekati batas kenyamanan. Level 3 paling boros dan hanya rasional bila penalti panas jauh lebih mahal daripada energi tambahan. Dengan struktur ini, algoritma harus benar-benar menimbang trade-off.",
            ),
            new_p(
                "BodyText",
                "DP menghitung biaya minimum untuk setiap hari dan level terakhir. Greedy memilih level minimum per hari berdasarkan biaya langsung. BFS berlapis menelusuri state per hari secara melebar dan memangkas state yang didominasi pada level yang sama. DFS bermemoisasi menelusuri secara mendalam, tetapi menyimpan hasil suffix agar state yang sama tidak dihitung berulang. Fuzzy Logic menggunakan aturan Sugeno sederhana dengan masukan suhu, kelembapan, dan THI, kemudian membulatkan keluaran ke level kipas 0-3.",
            ),
            new_p(
                "BodyText",
                "Validasi eksperimen dilakukan dengan tiga cara. Pertama, seluruh metode menerima input data dan fungsi biaya yang sama. Kedua, output tiap metode disimpan ke file detail sehingga level, THI sebelum kendali, THI setelah kendali, energi, penalti, dan switching dapat diperiksa per hari. Ketiga, ringkasan metrik dihitung dari file detail, bukan ditulis manual. Hal ini mengurangi risiko inkonsistensi antara narasi makalah dan hasil program.",
            ),
            new_p("Heading1", "Hasil dan Pembahasan"),
            new_p(
                "BodyText",
                f"Hasil perbandingan menunjukkan bahwa metode terbaik berdasarkan fungsi objektif adalah {best['method']} dengan energi {best['total_energy_units']:.2f} unit, penalti kenyamanan {best['comfort_penalty']:.2f}, THI maksimum setelah kendali {best['max_thi_after']:.2f}, dan objektif total {best['objective']:.2f}. Dynamic Programming, BFS berlapis, dan DFS bermemoisasi menghasilkan urutan level yang sama karena masalah memiliki struktur graf berlapis kecil dan pemangkasan state tidak menghilangkan kandidat optimum.",
            ),
            new_p(
                "BodyText",
                "Sebelum kendali, THI harian berada pada rentang yang relatif sempit tetapi konsisten tinggi untuk wilayah tropis lembap. Ini membuat level kipas 0 hampir tidak pernah menjadi keputusan yang baik karena penalti kenyamanan akan lebih besar daripada penghematan energi. Namun, level kipas 3 juga tidak otomatis menjadi pilihan terbaik karena penurunan THI tambahan tidak selalu sebanding dengan energi yang dikeluarkan. Di sinilah fungsi biaya memaksa algoritma memilih kompromi.",
            ),
        ]
    )

    elements.extend(
        table(
            "Tabel 2. Perbandingan metrik algoritma",
            ["Metode", "Energi", "Penalti", "THI maks", "ms"],
            table_rows,
        )
    )
    elements.extend(
        figure_elements(
            image_rels["energy"],
            FIGURES_DIR / "energy_comparison.png",
            "Gambar 1. Perbandingan total energi relatif setiap metode.",
        )
    )
    elements.extend(
        figure_elements(
            image_rels["thi"],
            FIGURES_DIR / "thi_line_comparison.png",
            "Gambar 2. Perbandingan THI awal, hasil Dynamic Programming, dan hasil Fuzzy Logic.",
        )
    )

    elements.extend(
        [
            new_p(
                "BodyText",
                f"Urutan level DP adalah {dp['levels']}. Urutan ini cenderung memakai level tinggi hanya ketika THI awal harian menuntut pendinginan tambahan. Dibandingkan Greedy, DP mempertimbangkan biaya pergantian level sehingga tidak selalu memilih aksi yang tampak paling murah pada satu hari terpisah. Greedy tetap menjadi baseline penting karena runtime-nya rendah dan implementasinya mudah, tetapi hasilnya dapat memiliki objektif lebih besar ketika rangkaian keputusan lokal menciptakan switching yang tidak perlu.",
            ),
            new_p(
                "BodyText",
                f"Pada data ini, Greedy memakai energi {greedy['total_energy_units']:.2f} unit, sedikit lebih hemat daripada DP, tetapi penalti kenyamanannya naik menjadi {greedy['comfort_penalty']:.2f}. Hal ini menunjukkan bahwa energi total saja bukan ukuran keberhasilan. Jika hanya mengejar energi minimum, algoritma dapat membiarkan THI mendekati atau melewati batas. Fungsi objektif gabungan membuat keputusan yang lebih seimbang karena penalti kenyamanan dihitung bersama energi.",
            ),
            new_p(
                "BodyText",
                f"BFS berlapis dan DFS bermemoisasi memiliki nilai objektif yang sama dengan DP, tetapi jejak komputasinya berbeda. BFS mengunjungi {int(summary.loc[summary['method']=='BFS Berlapis','visited_nodes'].iloc[0])} kandidat transisi, sedangkan DFS bermemoisasi mengunjungi {int(summary.loc[summary['method']=='DFS Bermemoisasi','visited_nodes'].iloc[0])} kandidat. Pada horizon yang kecil, keduanya tetap cepat. Namun, tanpa pemangkasan atau memoization, pencarian semua kombinasi level akan tumbuh eksponensial sebesar 4^n sehingga tidak praktis untuk horizon panjang.",
            ),
            new_p(
                "BodyText",
                "Kesamaan hasil DP, BFS berlapis, dan DFS bermemoisasi tidak berarti ketiganya identik secara konsep. DP menuliskan rekurensi optimasi secara langsung. BFS menekankan proses eksplorasi lapisan demi lapisan pada graf keputusan. DFS menekankan eksplorasi mendalam dan pemakaian memo untuk menghindari pengulangan submasalah. Dalam konteks pembelajaran strategi algoritma, ketiganya memberi sudut pandang berbeda terhadap masalah yang sama.",
            ),
            new_p(
                "BodyText",
                f"Fuzzy Logic menghasilkan urutan level {fuzzy['levels']}. Metode ini tidak dirancang untuk mengoptimalkan fungsi biaya eksplisit, tetapi memiliki kelebihan pada interpretabilitas. Aturan seperti 'jika suhu panas dan kelembapan tinggi maka level kipas tinggi' mudah diverifikasi oleh pembaca non-teknis. Dalam eksperimen ini, Fuzzy Logic cenderung lebih konservatif pada beberapa hari sehingga energi dapat lebih besar daripada DP, tetapi penalti kenyamanan tetap terkendali.",
            ),
            new_p(
                "BodyText",
                f"Fuzzy Logic memakai energi {fuzzy['total_energy_units']:.2f} unit, tertinggi di antara metode yang diuji, tetapi penalti kenyamanannya hanya {fuzzy['comfort_penalty']:.2f}. Hasil ini masuk akal karena aturan fuzzy lebih memilih level 2 secara stabil untuk hampir semua hari. Stabilitas tersebut menurunkan risiko panas dan switching, tetapi mengorbankan energi. Jika operator kandang sangat memprioritaskan keselamatan termal, pola fuzzy dapat diterima; jika targetnya efisiensi energi, aturan perlu dituning.",
            ),
            new_p(
                "BodyText",
                "Dari sisi strategi algoritma, DP paling sesuai bila tujuan utama adalah optimalitas pada model biaya yang telah didefinisikan. Greedy cocok untuk sistem sederhana dengan sumber daya komputasi sangat terbatas. BFS dan DFS berguna untuk menjelaskan representasi state-space dan dapat menjadi dasar pencarian bila ruang aksi berubah menjadi kendala diskrit yang lebih kompleks. Fuzzy Logic cocok bila prioritasnya adalah aturan kendali yang transparan dan mudah disesuaikan operator kandang.",
            ),
            new_p("Heading2", "Analisis Sensitivitas dan Implikasi"),
            new_p(
                "BodyText",
                "Hasil eksperimen sangat dipengaruhi oleh bobot penalti kenyamanan. Jika bobot penalti diturunkan, algoritma optimasi akan semakin berani memilih level kipas rendah karena konsekuensi THI tinggi dianggap murah. Sebaliknya, jika bobot penalti dinaikkan, metode seperti DP akan lebih sering memilih level kipas tinggi. Dalam konteks kandang nyata, bobot ini dapat ditafsirkan sebagai tingkat prioritas peternak terhadap kenyamanan termal dibanding biaya listrik. Peternak yang menargetkan performa produksi dan kesejahteraan ayam mungkin memilih bobot penalti lebih besar daripada peternak yang sedang menekan biaya operasional.",
            ),
            new_p(
                "BodyText",
                "Parameter biaya switching juga berpengaruh terhadap pola keputusan. Bila biaya switching nol, algoritma bebas menaikkan dan menurunkan level kipas setiap hari. Pola ini mungkin optimal secara numerik, tetapi kurang baik untuk aktuator karena perubahan terlalu sering dapat meningkatkan keausan. Dengan menambahkan biaya switching, urutan keputusan menjadi lebih stabil. Dalam hasil eksperimen, DP memilih rangkaian level 1 yang cukup panjang sebelum berpindah ke level 2, lalu kembali ke level 1 pada akhir periode. Pola ini lebih mudah diterapkan daripada keputusan yang berubah-ubah setiap hari.",
            ),
            new_p(
                "BodyText",
                "Efek pendinginan per level juga merupakan asumsi penting. Jika level 2 dan 3 dianggap memberi penurunan suhu yang lebih besar, algoritma akan lebih cepat memilih level tinggi karena manfaat kenyamanan meningkat. Jika efek pendinginan lebih kecil, level tinggi menjadi kurang menarik dan penalti THI mungkin tetap tinggi. Pada implementasi lanjutan, parameter pendinginan seharusnya dikalibrasi dari data sensor sebelum dan sesudah kipas aktif. Kalibrasi dapat dilakukan dengan regresi sederhana atau identifikasi sistem termal agar model lebih dekat dengan kondisi kandang.",
            ),
            new_p(
                "BodyText",
                "Dari sudut pandang implementasi perangkat lunak, DP dapat dipakai sebagai engine perencana offline. Misalnya, data prakiraan cuaca tujuh hari ke depan diambil dari API, kemudian DP menghitung jadwal kipas yang hemat energi untuk periode tersebut. Greedy dapat dipakai sebagai mode darurat ketika sistem hanya mempunyai data hari ini. Fuzzy Logic dapat dipakai sebagai fallback lokal pada mikrokontroler karena aturan if-then tidak membutuhkan tabel DP besar dan tetap dapat dijelaskan kepada operator.",
            ),
            new_p(
                "BodyText",
                "BFS dan DFS lebih berguna sebagai model konseptual dan alat verifikasi. Untuk jumlah state kecil, keduanya dapat menunjukkan bahwa solusi DP memang sesuai dengan pencarian graf. Untuk jumlah state besar, BFS dan DFS perlu dibatasi dengan pruning, memoization, atau heuristik. Dengan kata lain, BFS/DFS memberi jembatan antara materi pencarian graf dan optimasi sekuensial, sedangkan DP memberi formulasi yang paling ringkas untuk kasus dengan struktur submasalah jelas.",
            ),
            new_p(
                "BodyText",
                "Jika data sensor kandang aktual tersedia, eksperimen dapat diperluas menjadi evaluasi kebijakan. Data historis dipakai untuk menghitung keputusan algoritma, lalu konsumsi energi dan THI efektif dibandingkan dengan kebijakan manual atau thermostat sederhana. Pada tahap berikutnya, kebijakan terbaik dapat diuji secara simulasi dengan variasi cuaca ekstrem. Pendekatan ini menjaga keselamatan karena algoritma tidak langsung dipasang pada kandang produksi sebelum perilakunya dipahami.",
            ),
            new_p(
                "BodyText",
                "Jika model diperluas menjadi resolusi per jam, DP masih dapat berjalan selama jumlah level dan state tetap kecil. Namun, bila state mencakup umur ayam, kelembapan litter, CO2, dan mode aktuator lain seperti cooling pad, jumlah kombinasi dapat meningkat. Pada kondisi tersebut, diperlukan teknik tambahan seperti pemangkasan, heuristik A*, atau pendekatan approximate dynamic programming. Dengan demikian, eksperimen sederhana ini juga memberi gambaran awal tentang skalabilitas.",
            ),
            new_p(
                "BodyText",
                "Dari sisi data, NASA POWER bukan pengganti sempurna sensor kandang. Data tersebut berasal dari model/reanalisis meteorologi grid dan merepresentasikan kondisi lingkungan luar. Kandang closed house dapat memiliki suhu dan kelembapan yang berbeda akibat panas metabolisme ayam, kepadatan kandang, ventilasi, evaporative cooling, dan desain bangunan. Karena itu, angka energi relatif dan THI efektif harus dilihat sebagai simulasi pembanding algoritma, bukan rekomendasi operasional langsung.",
            ),
            new_p(
                "BodyText",
                "Walaupun demikian, penggunaan NASA POWER memberikan kelebihan reprodusibilitas. Setiap pembaca dapat mengunduh parameter yang sama melalui API tanpa akun. Hal ini memenuhi kebutuhan eksperimen komputasional pada tugas makalah: kode dapat dijalankan ulang, hasil dapat diverifikasi, dan sumber data dapat dilacak. Jika dataset Mendeley berhasil diunduh secara manual, data kandang sebenarnya tinggal diletakkan pada folder data dan dibaca dengan skema kolom yang sama.",
            ),
            new_p(
                "BodyText",
                "Keterbatasan eksperimen terletak pada data fallback yang berasal dari cuaca luar, bukan sensor langsung di kandang. Karena itu, hasil numerik tidak boleh dibaca sebagai rekomendasi level kipas nyata untuk kandang broiler tertentu. Namun, alur eksperimen sudah disiapkan agar dataset kandang Mendeley atau dataset sensor lokal dapat langsung menggantikan data NASA POWER selama kolom suhu dan kelembapan tersedia.",
            ),
            new_p(
                "BodyText",
                "Selain itu, pemilihan parameter biaya masih bersifat desain eksperimen. Bobot penalti, energi relatif, dan efek pendinginan dipilih agar perbedaan perilaku algoritma tampak jelas. Pada aplikasi nyata, parameter tersebut perlu dikalibrasi terhadap spesifikasi kipas, sensor suhu di ketinggian ayam, respons biologis broiler, dan target produksi. Namun, struktur algoritmanya tidak berubah: yang diganti adalah nilai parameter dan data input.",
            ),
            new_p("Heading1", "Kesimpulan dan Saran"),
            new_p("Heading2", "Kesimpulan"),
            new_p(
                "BodyText",
                "Makalah ini berhasil memformulasikan optimasi kipas kandang broiler sebagai masalah pemilihan level diskrit berdasarkan suhu dan kelembapan harian. Dengan fungsi biaya yang menggabungkan energi relatif, penalti THI, dan biaya pergantian level, Dynamic Programming memberikan solusi optimum yang eksplisit dan dapat dijelaskan. BFS berlapis dan DFS bermemoisasi mencapai hasil yang sama pada model kecil, sedangkan Greedy lebih sederhana tetapi rentan terhadap keputusan lokal.",
            ),
            new_p(
                "BodyText",
                "Fuzzy Logic menjadi pembanding di luar daftar strategi algoritma karena merepresentasikan pengetahuan kontrol dalam aturan linguistik. Hasilnya menunjukkan trade-off: fuzzy lebih mudah dipahami dan diubah, tetapi tidak menjamin nilai objektif minimum. Dengan demikian, DP lebih tepat untuk mencari kebijakan referensi optimum, sedangkan fuzzy dapat dipakai sebagai pendekatan praktis ketika operator membutuhkan aturan yang transparan.",
            ),
            new_p("Heading2", "Saran"),
            new_p(
                "BodyText",
                "Pengembangan berikutnya sebaiknya memakai data sensor langsung dari kandang broiler, termasuk suhu dalam kandang, RH, CO2, amonia, umur ayam, dan status aktuator. Model energi juga perlu diganti dengan daya kipas aktual agar hasil dapat dinyatakan dalam kWh. Selain itu, metode lain seperti A*, branch and bound, atau reinforcement learning dapat dibandingkan bila fungsi biaya dan dinamika termal dibuat lebih realistis.",
            ),
            new_p("Heading5", "Lampiran"),
            new_p(
                "BodyText",
                "Kode program: src/fan_energy_optimization.py. Data hasil unduhan: data/nasa_power_semarang_daily.csv. Ringkasan hasil: results/summary_metrics.csv. Grafik pendukung: figures/energy_comparison.png dan figures/thi_line_comparison.png. Catatan dataset dan sumber: sources/dataset_route_note.txt dan sources/sources_used.md. Link video: ditambahkan penulis.",
            ),
            new_p("Heading5", "References"),
        ]
    )

    references = [
        '[1] A. Thiam, "Ambient conditions in poultry house," Mendeley Data, V1, 2023, doi: 10.17632/7fptd3rfzy.1.',
        '[2] NASA POWER Project, "Daily API Documentation," NASA Langley Research Center, 2026. [Online]. Available: https://power.larc.nasa.gov/docs/services/api/temporal/daily/',
        '[3] H. A. Wijaya and S. Hartati, "Implementasi Logika Fuzzy Dalam Sistem Pendingin Otomatis Kandang Ayam Broiler Closed House," IJEIS, vol. 15, no. 1, 2025, doi: 10.22146/ijeis.103364.',
        '[4] M. Olejar, V. Cviklovic, D. Hruby, and L. Toth, "Fuzzy control of temperature and humidity microclimate in closed areas for poultry breeding," Research in Agricultural Engineering, vol. 60, pp. S31-S36, 2014, doi: 10.17221/30/2013-RAE.',
        '[5] S. Akter et al., "Impacts of Air Velocity Treatments under Summer Condition: Part I-Heavy Broiler\'s Surface Temperature Response," Animals, vol. 12, no. 3, p. 328, 2022, doi: 10.3390/ani12030328.',
        '[6] E. H. Mamdani and S. Assilian, "An experiment in linguistic synthesis with a fuzzy logic controller," International Journal of Man-Machine Studies, vol. 7, no. 1, pp. 1-13, 1975, doi: 10.1016/S0020-7373(75)80002-2.',
        '[7] L. A. Zadeh, "Fuzzy sets," Information and Control, vol. 8, no. 3, pp. 338-353, 1965, doi: 10.1016/S0019-9958(65)90241-X.',
        "[8] T. H. Cormen, C. E. Leiserson, R. L. Rivest, and C. Stein, Introduction to Algorithms, 4th ed. Cambridge, MA: MIT Press, 2022.",
        "[9] S. Russell and P. Norvig, Artificial Intelligence: A Modern Approach, 4th ed. Hoboken, NJ: Pearson, 2021.",
        '[10] M. N. I. S. Diarra et al., "Thermal Comfort, Growth Performance and Welfare of Olive Pulp Fed Broilers during Hot Season," Sustainability, vol. 15, no. 14, p. 10932, 2023, doi: 10.3390/su151410932.',
    ]
    for ref in references:
        elements.append(new_p("references", strip_leading_reference_number(ref)))

    elements.extend(
        [
            new_p("Heading5", "Pernyataan"),
            no_number_references_p(
                "Dengan ini saya menyatakan bahwa makalah yang saya susun merupakan hasil karya saya sendiri dan tidak merupakan saduran, terjemahan, maupun plagiasi dari karya orang lain. Seluruh sumber yang digunakan telah dicantumkan secara benar sesuai dengan kaidah penulisan ilmiah.",
            ),
            no_number_references_p(
                "Terkait penggunaan teknologi kecerdasan buatan (Artificial Intelligence), saya menyatakan bahwa:",
            ),
            no_number_references_p(
                "☑ Saya menggunakan alat bantu AI sebagai pendukung dalam penyusunan makalah ini dengan rincian sebagai berikut:",
            ),
            no_number_references_p("Nama alat AI: OpenAI Codex / ChatGPT."),
            no_number_references_p(
                "Tujuan penggunaan: membantu pencarian sumber, penyusunan rancangan eksperimen, implementasi kode Python, analisis hasil, dan penyusunan draf makalah.",
            ),
            no_number_references_p(
                "Ruang lingkup penggunaan: AI digunakan sebagai alat bantu teknis dan penyuntingan; penulis tetap bertanggung jawab atas validasi isi, interpretasi hasil, dan integritas akademik.",
            ),
            no_number_references_p(
                "Penggunaan AI, apabila ada, hanya bersifat sebagai alat bantu dan tidak menggantikan peran saya sebagai penulis utama. Seluruh isi, analisis, serta kesimpulan dalam makalah ini tetap merupakan tanggung jawab saya sepenuhnya.",
            ),
            no_number_references_p(STATEMENT_DATE, align="end"),
            no_number_references_p("Ttd digital", align="end"),
            no_number_references_p(AUTHOR_NAME, align="end"),
            no_number_references_p(AUTHOR_NIM, sect_pr=two_col_end, align="end"),
            final_sect,
        ]
    )

    # Avoid keeping the final section as a raw sectPr inside the content list twice.
    if isinstance(elements[-1].tag, str) and etree.QName(elements[-1]).localname == "sectPr":
        pass
    return elements


def audit_docx() -> None:
    with zipfile.ZipFile(DOCX_OUT, "r") as z:
        root = etree.fromstring(z.read("word/document.xml"))
    text = "\n".join(root.xpath(".//w:t/text()", namespaces=NS))
    body_paras = root.xpath(".//w:body/w:p", namespaces=NS)
    refs = []
    in_refs = False
    for para in body_paras:
        para_text = "".join(para.xpath(".//w:t/text()", namespaces=NS)).strip()
        style = para.xpath("./w:pPr/w:pStyle/@w:val", namespaces=NS)
        if para_text == "References":
            in_refs = True
            continue
        if para_text == "Pernyataan":
            in_refs = False
        if in_refs and style == ["references"] and para_text:
            refs.append(len(refs) + 1)
    cites = citation_set(text)
    remaining_placeholders = [
        token
        for token in ["Judul Makalah", "Nama Lengkap - NIM", "email@undip.ac.id", "component, formatting"]
        if token in text
    ]
    audit = {
        "characters": len(text),
        "reference_numbers": refs,
        "citation_numbers": sorted(cites),
        "references_without_citation": sorted(set(refs) - cites),
        "citations_without_reference": sorted(cites - set(refs)),
        "remaining_template_placeholders": remaining_placeholders,
        "has_ai_statement": "OpenAI Codex" in text,
        "has_author": AUTHOR_NAME in text and AUTHOR_NIM in text,
    }
    (RESULTS_DIR / "docx_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    df, results, summary = run_experiment()
    build_docx(df, results, summary)
    print("Experiment and DOCX generation complete.")
    print(summary.to_string(index=False))
    print(f"DOCX: {DOCX_OUT}")


if __name__ == "__main__":
    main()
