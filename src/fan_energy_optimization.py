import csv
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Script ini sengaja dibuat sederhana mengikuti gaya praktikum ASA:
# - data disimpan sebagai list of dict
# - tabel DP dibuat eksplisit
# - BFS memakai queue/agenda
# - DFS memakai rekursi + memo
# - output eksperimen ditulis ke CSV dan gambar PNG

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
SOURCES_DIR = ROOT / "sources"

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


def ensure_dirs():
    for folder in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, SOURCES_DIR]:
        folder.mkdir(exist_ok=True)


def ambil_url(url):
    request = urllib.request.Request(url, headers={"User-Agent": "ASA-broiler-study/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def thi_from_temp_rh(suhu, rh):
    return 0.8 * suhu + (rh * (suhu - 14.3) / 100.0) + 46.3


def tulis_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def baca_data():
    raw_path = DATA_DIR / "nasa_power_semarang_20220403_20220421.json"
    csv_path = DATA_DIR / "nasa_power_semarang_daily.csv"

    try:
        raw = ambil_url(NASA_URL)
        raw_path.write_bytes(raw)
    except Exception:
        raw = raw_path.read_bytes()

    payload = json.loads(raw.decode("utf-8"))
    parameter = payload["properties"]["parameter"]
    tanggal = sorted(parameter["T2M"].keys())

    data = []
    for tgl in tanggal:
        tanggal_iso = f"{tgl[0:4]}-{tgl[4:6]}-{tgl[6:8]}"
        suhu = float(parameter["T2M"][tgl])
        rh = float(parameter["RH2M"][tgl])
        wet_bulb = float(parameter["T2MWET"][tgl])
        baris = {
            "date": tanggal_iso,
            "temperature_c": suhu,
            "relative_humidity_pct": rh,
            "temperature_max_c": float(parameter["T2M_MAX"][tgl]),
            "temperature_min_c": float(parameter["T2M_MIN"][tgl]),
            "wet_bulb_c": wet_bulb,
            "thi_rh": thi_from_temp_rh(suhu, rh),
            "thi_wetbulb": 0.85 * suhu + 0.15 * wet_bulb,
        }
        data.append(baris)

    fieldnames = [
        "date",
        "temperature_c",
        "relative_humidity_pct",
        "temperature_max_c",
        "temperature_min_c",
        "wet_bulb_c",
        "thi_rh",
        "thi_wetbulb",
    ]
    tulis_csv(csv_path, fieldnames, data)
    tulis_catatan_dataset()
    return data


def tulis_catatan_dataset():
    catatan = [
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
        ambil_url(MENDELEY_API)
        catatan[5] = "Status: Mendeley API file metadata was accessible."
    except urllib.error.HTTPError as e:
        catatan.append(f"Mendeley public API check: HTTP {e.code} {e.reason}.")
    except Exception as e:
        catatan.append(f"Mendeley public API check failed: {type(e).__name__}: {e}.")

    (SOURCES_DIR / "dataset_route_note.txt").write_text("\n".join(catatan), encoding="utf-8")


def suhu_dan_thi_setelah_kipas(baris, level):
    suhu_baru = baris["temperature_c"] - COOLING_DELTA_C[level]
    thi_baru = thi_from_temp_rh(suhu_baru, baris["relative_humidity_pct"])
    return suhu_baru, thi_baru


def penalti_kenyamanan(thi):
    lewat = max(0.0, thi - COMFORT_LIMIT)
    lewat_berat = max(0.0, thi - SEVERE_LIMIT)
    return 1.8 * lewat * lewat + 4.5 * lewat_berat * lewat_berat


def biaya_harian(baris, level, level_sebelum):
    _, thi_baru = suhu_dan_thi_setelah_kipas(baris, level)
    energi = ENERGY_UNITS[level]
    penalti = penalti_kenyamanan(thi_baru)
    pindah = SWITCH_WEIGHT * abs(level - level_sebelum)
    return energi + penalti + pindah


def evaluasi_urutan(data, urutan_level, nama_metode, waktu_ms, visited_nodes):
    detail = []
    level_sebelum = 0

    for i in range(len(data)):
        baris = data[i]
        level = urutan_level[i]
        suhu_baru, thi_baru = suhu_dan_thi_setelah_kipas(baris, level)
        energi = ENERGY_UNITS[level]
        penalti = penalti_kenyamanan(thi_baru)
        switch_cost = SWITCH_WEIGHT * abs(level - level_sebelum)

        detail.append(
            {
                "date": baris["date"],
                "method": nama_metode,
                "level": level,
                "temperature_c": baris["temperature_c"],
                "relative_humidity_pct": baris["relative_humidity_pct"],
                "thi_before": baris["thi_rh"],
                "temperature_effective_c": suhu_baru,
                "thi_after": thi_baru,
                "energy_units": energi,
                "comfort_penalty": penalti,
                "switch_cost": switch_cost,
            }
        )
        level_sebelum = level

    total_energi = 0.0
    total_penalti = 0.0
    total_switch_cost = 0.0
    total_thi = 0.0
    max_thi = -10**9
    switch_count = 0
    level_sebelum = 0

    for i in range(len(detail)):
        row = detail[i]
        total_energi += row["energy_units"]
        total_penalti += row["comfort_penalty"]
        total_switch_cost += row["switch_cost"]
        total_thi += row["thi_after"]
        if row["thi_after"] > max_thi:
            max_thi = row["thi_after"]
        switch_count += abs(row["level"] - level_sebelum)
        level_sebelum = row["level"]

    return {
        "method": nama_metode,
        "levels": urutan_level,
        "detail": detail,
        "total_energy_units": total_energi,
        "comfort_penalty": total_penalti,
        "switch_count": switch_count,
        "avg_thi_after": total_thi / len(detail),
        "max_thi_after": max_thi,
        "objective": total_energi + total_penalti + total_switch_cost,
        "runtime_ms": waktu_ms,
        "visited_nodes": visited_nodes,
    }


def solve_dp(data):
    # Program dinamis:
    # hari ke-i = tahap
    # level kipas = status
    # memilih level berikutnya = keputusan
    start = time.perf_counter()
    n = len(data)
    inf = 10**30
    visited = 0

    dp = []
    asal = []
    for _ in range(n):
        dp.append([inf] * len(LEVELS))
        asal.append([-1] * len(LEVELS))

    for level in LEVELS:
        visited += 1
        dp[0][level] = biaya_harian(data[0], level, 0)
        asal[0][level] = 0

    for hari in range(1, n):
        for level in LEVELS:
            terbaik = inf
            level_asal = 0
            for prev in LEVELS:
                visited += 1
                nilai = dp[hari - 1][prev] + biaya_harian(data[hari], level, prev)
                if nilai < terbaik:
                    terbaik = nilai
                    level_asal = prev
            dp[hari][level] = terbaik
            asal[hari][level] = level_asal

    level_akhir = 0
    for level in LEVELS:
        if dp[n - 1][level] < dp[n - 1][level_akhir]:
            level_akhir = level

    urutan = [0] * n
    sekarang = level_akhir
    for hari in range(n - 1, -1, -1):
        urutan[hari] = sekarang
        sekarang = asal[hari][sekarang]

    waktu_ms = (time.perf_counter() - start) * 1000
    return evaluasi_urutan(data, urutan, "Dynamic Programming", waktu_ms, visited)


def solve_greedy(data):
    # Greedy: setiap hari langsung ambil level dengan biaya lokal terkecil.
    start = time.perf_counter()
    urutan = []
    level_sebelum = 0
    visited = 0

    for baris in data:
        biaya_terbaik = 10**30
        level_terbaik = 0
        for level in LEVELS:
            visited += 1
            biaya = biaya_harian(baris, level, level_sebelum)
            if biaya < biaya_terbaik:
                biaya_terbaik = biaya
                level_terbaik = level
        urutan.append(level_terbaik)
        level_sebelum = level_terbaik

    waktu_ms = (time.perf_counter() - start) * 1000
    return evaluasi_urutan(data, urutan, "Greedy", waktu_ms, visited)


def solve_bfs(data):
    # BFS berlapis: agenda diperlakukan seperti queue.
    # State: (hari, level_terakhir, path, biaya).
    start = time.perf_counter()
    queue = [(0, 0, [], 0.0)]
    visited = 0

    for hari in range(len(data)):
        kandidat_level = {}
        while queue:
            _, level_sebelum, path, biaya_sebelum = queue.pop(0)
            for level in LEVELS:
                visited += 1
                biaya_baru = biaya_sebelum + biaya_harian(data[hari], level, level_sebelum)
                path_baru = path + [level]

                if level not in kandidat_level or biaya_baru < kandidat_level[level][1]:
                    kandidat_level[level] = (path_baru, biaya_baru)

        queue = []
        for level in LEVELS:
            path, biaya = kandidat_level[level]
            queue.append((hari + 1, level, path, biaya))

    path_terbaik = queue[0][2]
    biaya_terbaik = queue[0][3]
    for state in queue:
        if state[3] < biaya_terbaik:
            biaya_terbaik = state[3]
            path_terbaik = state[2]

    waktu_ms = (time.perf_counter() - start) * 1000
    return evaluasi_urutan(data, path_terbaik, "BFS Berlapis", waktu_ms, visited)


def solve_dfs(data):
    # DFS dengan memo:
    # rekursi menelusuri keputusan sedalam mungkin, lalu hasil state disimpan.
    start = time.perf_counter()
    memo = {}
    visited = 0

    def dfs(hari, level_sebelum):
        nonlocal visited
        key = (hari, level_sebelum)
        if key in memo:
            return memo[key]
        if hari == len(data):
            return 0.0, []

        biaya_terbaik = 10**30
        path_terbaik = []
        for level in LEVELS:
            visited += 1
            sisa_biaya, sisa_path = dfs(hari + 1, level)
            biaya = biaya_harian(data[hari], level, level_sebelum) + sisa_biaya
            if biaya < biaya_terbaik:
                biaya_terbaik = biaya
                path_terbaik = [level] + sisa_path

        memo[key] = (biaya_terbaik, path_terbaik)
        return memo[key]

    _, urutan = dfs(0, 0)
    waktu_ms = (time.perf_counter() - start) * 1000
    return evaluasi_urutan(data, urutan, "DFS Bermemoisasi", waktu_ms, visited)


def turun(x, a, b):
    if x <= a:
        return 1.0
    if x >= b:
        return 0.0
    return (b - x) / (b - a)


def naik(x, a, b):
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a)


def trapesium(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def fuzzy_level(suhu, rh, thi):
    suhu_dingin = turun(suhu, 24.0, 26.0)
    suhu_hangat = trapesium(suhu, 24.5, 26.0, 28.0, 29.5)
    suhu_panas = naik(suhu, 28.0, 31.0)

    rh_normal = turun(rh, 65.0, 75.0)
    rh_tinggi = trapesium(rh, 70.0, 78.0, 86.0, 92.0)
    rh_sangat_tinggi = naik(rh, 84.0, 92.0)

    thi_aman = turun(thi, 72.0, 75.0)
    thi_waspada = trapesium(thi, 72.0, 75.0, 78.5, 81.0)
    thi_stres = naik(thi, 78.0, 82.0)

    aturan = [
        (max(suhu_dingin, min(thi_aman, rh_normal)), 0.0),
        (min(suhu_hangat, thi_aman), 1.0),
        (min(suhu_hangat, rh_tinggi), 1.0),
        (thi_waspada, 2.0),
        (min(suhu_panas, rh_tinggi), 3.0),
        (min(thi_stres, rh_tinggi), 3.0),
        (min(suhu_panas, rh_sangat_tinggi), 3.0),
    ]

    pembilang = 0.0
    penyebut = 0.0
    for bobot, output in aturan:
        pembilang += bobot * output
        penyebut += bobot

    if penyebut == 0:
        return 1

    level = round(pembilang / penyebut)
    if level < 0:
        return 0
    if level > 3:
        return 3
    return int(level)


def solve_fuzzy(data):
    start = time.perf_counter()
    urutan = []
    for baris in data:
        level = fuzzy_level(
            baris["temperature_c"],
            baris["relative_humidity_pct"],
            baris["thi_rh"],
        )
        urutan.append(level)
    waktu_ms = (time.perf_counter() - start) * 1000
    return evaluasi_urutan(data, urutan, "Fuzzy Logic", waktu_ms, len(data) * 7)


def slug(teks):
    hasil = []
    terakhir_underscore = False
    for ch in teks.lower():
        if "a" <= ch <= "z" or "0" <= ch <= "9":
            hasil.append(ch)
            terakhir_underscore = False
        elif not terakhir_underscore:
            hasil.append("_")
            terakhir_underscore = True
    return "".join(hasil).strip("_")


def simpan_hasil(hasil):
    field_detail = [
        "date",
        "method",
        "level",
        "temperature_c",
        "relative_humidity_pct",
        "thi_before",
        "temperature_effective_c",
        "thi_after",
        "energy_units",
        "comfort_penalty",
        "switch_cost",
    ]

    ringkasan = []
    for item in hasil:
        nama_file = RESULTS_DIR / f"detail_{slug(item['method'])}.csv"
        tulis_csv(nama_file, field_detail, item["detail"])
        ringkasan.append(
            {
                "method": item["method"],
                "total_energy_units": item["total_energy_units"],
                "comfort_penalty": item["comfort_penalty"],
                "switch_count": item["switch_count"],
                "avg_thi_after": item["avg_thi_after"],
                "max_thi_after": item["max_thi_after"],
                "objective": item["objective"],
                "runtime_ms": item["runtime_ms"],
                "visited_nodes": item["visited_nodes"],
                "level_sequence": " ".join(str(x) for x in item["levels"]),
            }
        )

    ringkasan.sort(key=lambda row: row["objective"])
    field_summary = [
        "method",
        "total_energy_units",
        "comfort_penalty",
        "switch_count",
        "avg_thi_after",
        "max_thi_after",
        "objective",
        "runtime_ms",
        "visited_nodes",
        "level_sequence",
    ]
    tulis_csv(RESULTS_DIR / "summary_metrics.csv", field_summary, ringkasan)
    return ringkasan


def y_tick_values(max_value):
    hasil = []
    for i in range(5):
        hasil.append(max_value * i / 4)
    return hasil


def buat_grafik_energi(summary):
    width, height = 1100, 650
    kiri, atas, bawah = 120, 70, 110
    plot_w = width - kiri - 70
    plot_h = height - atas - bawah
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    nilai = [row["total_energy_units"] for row in summary]
    label = [row["method"] for row in summary]
    warna = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]
    nilai_maks = max(nilai) * 1.15

    draw.text((kiri, 22), "Total energi relatif per metode", fill="black", font=font)
    draw.line((kiri, atas, kiri, atas + plot_h), fill="black", width=2)
    draw.line((kiri, atas + plot_h, kiri + plot_w, atas + plot_h), fill="black", width=2)

    jarak = 32
    lebar_bar = int((plot_w - jarak * (len(nilai) + 1)) / len(nilai))
    for i in range(len(nilai)):
        x0 = kiri + jarak + i * (lebar_bar + jarak)
        x1 = x0 + lebar_bar
        y1 = atas + plot_h
        y0 = y1 - int((nilai[i] / nilai_maks) * plot_h)
        draw.rectangle((x0, y0, x1, y1), fill=warna[i % len(warna)], outline="black")
        draw.text((x0 + 4, y0 - 18), f"{nilai[i]:.1f}", fill="black", font=font)
        draw.multiline_text((x0, y1 + 10), label[i].replace(" ", "\n"), fill="black", font=font)

    for tick in y_tick_values(nilai_maks):
        y = atas + plot_h - int((tick / nilai_maks) * plot_h)
        draw.line((kiri - 5, y, kiri, y), fill="black")
        draw.text((20, y - 7), f"{tick:.0f}", fill="black", font=font)

    image.save(FIGURES_DIR / "energy_comparison.png")


def buat_grafik_thi(data, detail_dp, detail_fuzzy):
    width, height = 1200, 650
    kiri, atas, bawah = 85, 60, 100
    plot_w = width - kiri - 60
    plot_h = height - atas - bawah
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    seri = [
        ("THI awal", [row["thi_rh"] for row in data], "#333333"),
        ("DP", [row["thi_after"] for row in detail_dp], "#1f77b4"),
        ("Fuzzy", [row["thi_after"] for row in detail_fuzzy], "#9467bd"),
    ]

    semua_nilai = [COMFORT_LIMIT, SEVERE_LIMIT]
    for _, data_seri, _ in seri:
        for x in data_seri:
            semua_nilai.append(x)

    y_min = math.floor(min(semua_nilai) - 1)
    y_max = math.ceil(max(semua_nilai) + 1)
    n = len(data)

    def posisi(i, nilai):
        x = kiri + int((i / (n - 1)) * plot_w)
        y = atas + plot_h - int(((nilai - y_min) / (y_max - y_min)) * plot_h)
        return x, y

    draw.text((kiri, 20), "Perubahan THI sebelum dan sesudah kendali kipas", fill="black", font=font)
    draw.line((kiri, atas, kiri, atas + plot_h), fill="black", width=2)
    draw.line((kiri, atas + plot_h, kiri + plot_w, atas + plot_h), fill="black", width=2)

    for batas, warna, teks in [(COMFORT_LIMIT, "#2ca02c", "batas 76"), (SEVERE_LIMIT, "#d62728", "batas 80")]:
        _, y = posisi(0, batas)
        draw.line((kiri, y, kiri + plot_w, y), fill=warna, width=1)
        draw.text((kiri + plot_w - 65, y - 14), teks, fill=warna, font=font)

    for nama, data_seri, warna in seri:
        titik = []
        for i in range(n):
            titik.append(posisi(i, data_seri[i]))
        draw.line(titik, fill=warna, width=3)
        for p in titik[::4]:
            draw.ellipse((p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3), fill=warna)

    legend_y = atas + 10
    for nama, _, warna in seri:
        draw.rectangle((kiri + 10, legend_y, kiri + 24, legend_y + 14), fill=warna)
        draw.text((kiri + 30, legend_y), nama, fill="black", font=font)
        legend_y += 22

    for i in range(6):
        tick = y_min + (y_max - y_min) * i / 5
        _, y = posisi(0, tick)
        draw.line((kiri - 5, y, kiri, y), fill="black")
        draw.text((25, y - 7), f"{tick:.1f}", fill="black", font=font)

    for i in range(n):
        if i % 4 == 0 or i == n - 1:
            x, _ = posisi(i, y_min)
            draw.text((x - 25, atas + plot_h + 12), data[i]["date"][5:], fill="black", font=font)

    image.save(FIGURES_DIR / "thi_line_comparison.png")


def tulis_sumber():
    isi = """Sumber dan referensi utama
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
    (SOURCES_DIR / "sources_used.md").write_text(isi, encoding="utf-8")


def validasi(summary):
    target = {
        "Dynamic Programming": (32.6, 6.3536878677, 39.6136878677),
        "BFS Berlapis": (32.6, 6.3536878677, 39.6136878677),
        "DFS Bermemoisasi": (32.6, 6.3536878677, 39.6136878677),
        "Greedy": (31.2, 7.4272179235, 39.7272179235),
        "Fuzzy Logic": (49.4, 0.1972050028, 50.0372050028),
    }

    for row in summary:
        energi, penalti, objektif = target[row["method"]]
        if abs(row["total_energy_units"] - energi) > 1e-9:
            raise RuntimeError(f"Energi {row['method']} berubah")
        if abs(row["comfort_penalty"] - penalti) > 1e-8:
            raise RuntimeError(f"Penalti {row['method']} berubah")
        if abs(row["objective"] - objektif) > 1e-8:
            raise RuntimeError(f"Objektif {row['method']} berubah")

    for path in [
        DATA_DIR / "nasa_power_semarang_daily.csv",
        RESULTS_DIR / "summary_metrics.csv",
        FIGURES_DIR / "energy_comparison.png",
        FIGURES_DIR / "thi_line_comparison.png",
    ]:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Output tidak terbentuk: {path}")


def main():
    ensure_dirs()
    data = baca_data()

    hasil = [
        solve_dp(data),
        solve_bfs(data),
        solve_dfs(data),
        solve_greedy(data),
        solve_fuzzy(data),
    ]

    summary = simpan_hasil(hasil)

    hasil_dp = None
    hasil_fuzzy = None
    for item in hasil:
        if item["method"] == "Dynamic Programming":
            hasil_dp = item
        if item["method"] == "Fuzzy Logic":
            hasil_fuzzy = item

    buat_grafik_energi(summary)
    buat_grafik_thi(data, hasil_dp["detail"], hasil_fuzzy["detail"])
    tulis_sumber()
    validasi(summary)

    for row in summary:
        print(
            row["method"],
            "energi =", round(row["total_energy_units"], 4),
            "penalti =", round(row["comfort_penalty"], 4),
            "objektif =", round(row["objective"], 4),
            "level =", row["level_sequence"],
        )


if __name__ == "__main__":
    main()
