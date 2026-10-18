# -*- coding: utf-8 -*-
"""Cek Risiko Diabetes: sistem pendukung keputusan dengan K-Nearest Neighbors (KNN)."""
import io
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# DATA DAN KONSTANTA
# ============================================================
FITUR = ["Usia", "BMI", "Glukosa", "Tekanan Darah", "Riwayat Keluarga"]
KOLOM = ["Pasien"] + FITUR + ["Kelas"]
LABEL = {0: "Rendah", 1: "Tinggi"}

DATA_AWAL = pd.DataFrame(
    [
        ["P01", 25, 21.5, 90, 110, 0, 0],
        ["P02", 30, 23.1, 95, 115, 0, 0],
        ["P03", 35, 25.0, 105, 120, 0, 0],
        ["P04", 40, 27.5, 120, 125, 1, 1],
        ["P05", 45, 30.2, 140, 135, 1, 1],
        ["P06", 50, 31.5, 155, 140, 1, 1],
        ["P07", 55, 33.0, 165, 145, 1, 1],
        ["P08", 28, 22.5, 92, 112, 0, 0],
        ["P09", 38, 26.8, 115, 122, 0, 0],
        ["P10", 48, 29.5, 145, 130, 1, 1],
    ],
    columns=KOLOM,
)

CONTOH = {
    "rendah": {"in_usia": 26, "in_bmi": 22.0, "in_glu": 92, "in_tensi": 112, "in_riw": "Tidak ada"},
    "tinggi": {"in_usia": 52, "in_bmi": 31.0, "in_glu": 150, "in_tensi": 140, "in_riw": "Ada"},
}

C = {"ink": "#1D1B4C", "violet": "#5B4FD6", "low": "#13805F", "high": "#D63C50",
     "soft": "#5E5B82", "line": "#E1DEF3", "grid": "#ECEAF7", "amber": "#FFC857"}


# ============================================================
# INTI ALGORITMA KNN
# ============================================================
def normalisasi(data, pasien):
    """Min-Max: x' = (x - min) / (max - min)."""
    fmin = data[FITUR].min()
    fmax = data[FITUR].max()
    rentang = (fmax - fmin).replace(0, 1)
    norm_latih = (data[FITUR] - fmin) / rentang
    norm_baru = {f: float((pasien[f] - fmin[f]) / rentang[f]) for f in FITUR}
    return fmin, fmax, norm_latih, norm_baru


def voting(kelas, jarak, k, berbobot):
    """kelas dan jarak sudah terurut dari yang terdekat. Mengembalikan (pred, n0, n1, skor)."""
    kel = np.asarray(kelas[:k])
    jr = np.asarray(jarak[:k], dtype=float)
    n1 = int((kel == 1).sum())
    n0 = int(len(kel) - n1)
    if berbobot:
        w = 1.0 / (jr + 1e-6)
        s1 = float(w[kel == 1].sum())
        s0 = float(w[kel == 0].sum())
        skor = s1 / (s0 + s1)
        pred = 1 if s1 > s0 else 0
    else:
        skor = n1 / len(kel)
        if n1 > n0:
            pred = 1
        elif n0 > n1:
            pred = 0
        else:  # seri: ikuti tetangga paling dekat
            pred = int(kel[0])
    return pred, n0, n1, float(skor)


def hitung_knn(data, pasien, k, berbobot=False):
    data = data.reset_index(drop=True)
    k = max(1, min(int(k), len(data)))
    fmin, fmax, norm_latih, norm_baru = normalisasi(data, pasien)
    vektor = np.array([norm_baru[f] for f in FITUR])
    jarak = np.sqrt(((norm_latih.to_numpy(dtype=float) - vektor) ** 2).sum(axis=1))

    tabel = pd.DataFrame({
        "Idx": np.arange(len(data)),
        "Pasien": data["Pasien"].astype(str).to_numpy(),
        "Kelas": data["Kelas"].astype(int).to_numpy(),
        "Jarak": jarak,
    }).sort_values("Jarak", kind="stable").reset_index(drop=True)
    tabel.insert(0, "Peringkat", np.arange(1, len(tabel) + 1))
    tabel["Terpilih"] = tabel["Peringkat"] <= k

    pred, n0, n1, skor = voting(tabel["Kelas"].to_numpy(), tabel["Jarak"].to_numpy(), k, berbobot)

    kls = data["Kelas"].astype(int)
    prof = {}
    for c in (0, 1):
        m = norm_latih[kls == c].mean()
        prof[c] = {f: (0.0 if pd.isna(m[f]) else float(m[f])) for f in FITUR}

    return {
        "baru": dict(pasien), "k": k, "berbobot": bool(berbobot), "data": data.copy(),
        "tabel": tabel, "fmin": fmin, "fmax": fmax, "norm_latih": norm_latih,
        "norm_baru": norm_baru, "prof0": prof[0], "prof1": prof[1],
        "pred": pred, "n0": n0, "n1": n1, "skor": skor,
        "waktu": waktu_sekarang(), "tersimpan": None,
    }


@st.cache_data(show_spinner=False)
def evaluasi_loocv(data, kmaks, berbobot):
    """Leave-one-out: tiap pasien diprediksi memakai semua pasien lain.
    Mengembalikan {k: [tp, tn, fp, fn]} dengan kelas Tinggi sebagai positif."""
    X = data[FITUR].to_numpy(dtype=float)
    y = data["Kelas"].to_numpy(dtype=int)
    n = len(y)
    ks = list(range(1, min(kmaks, n - 1) + 1))
    hasil = {k: [0, 0, 0, 0] for k in ks}
    for i in range(n):
        mask = np.arange(n) != i
        Xt, yt = X[mask], y[mask]
        mn, mx = Xt.min(axis=0), Xt.max(axis=0)
        r = np.where(mx - mn == 0, 1.0, mx - mn)
        d = np.sqrt((((Xt - mn) / r - (X[i] - mn) / r) ** 2).sum(axis=1))
        urut = np.argsort(d, kind="stable")
        for k in ks:
            pred, _, _, _ = voting(yt[urut], d[urut], k, berbobot)
            if y[i] == 1:
                hasil[k][0 if pred == 1 else 3] += 1
            else:
                hasil[k][1 if pred == 0 else 2] += 1
    return hasil


def waktu_sekarang():
    return datetime.now(timezone(timedelta(hours=7))).strftime("%d-%m-%Y %H:%M")


def id_berikutnya(data):
    angka = [int(p[1:]) for p in data["Pasien"] if str(p)[1:].isdigit()]
    return f"P{(max(angka) if angka else 0) + 1:02d}"


def tambah_pasien(usia, bmi, glukosa, tensi, riwayat, kelas):
    pid = id_berikutnya(st.session_state.data)
    baris = {"Pasien": pid, "Usia": usia, "BMI": bmi, "Glukosa": glukosa,
             "Tekanan Darah": tensi, "Riwayat Keluarga": riwayat, "Kelas": kelas}
    st.session_state.data = pd.concat(
        [st.session_state.data, pd.DataFrame([baris])], ignore_index=True)
    return pid


def baca_csv(isi):
    try:
        df = pd.read_csv(io.BytesIO(isi))
    except Exception as ex:
        return None, f"File CSV tidak bisa dibaca: {ex}"
    kurang = [c for c in KOLOM if c not in df.columns]
    if kurang:
        return None, "Kolom yang belum ada: " + ", ".join(kurang)
    df = df[KOLOM].copy()
    peta = {"rendah": 0, "tinggi": 1, "0": 0, "1": 1, "0.0": 0, "1.0": 1}
    df["Kelas"] = df["Kelas"].map(lambda v: peta.get(str(v).strip().lower(), np.nan))
    for f in FITUR:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df["Kelas"] = df["Kelas"].astype(int)
    df["Pasien"] = df["Pasien"].astype(str)
    if not df["Riwayat Keluarga"].isin([0, 1]).all():
        return None, "Kolom Riwayat Keluarga hanya boleh berisi 0 atau 1."
    if len(df) < 3 or df["Kelas"].nunique() < 2:
        return None, "Dataset minimal berisi 3 pasien dan memuat kedua kelas (0 dan 1)."
    return df, None


# ============================================================
# KATEGORI KLINIS (informasi tambahan, tidak memengaruhi KNN)
# ============================================================
def kategori_bmi(v):
    if v < 18.5:
        return "Kurus", "info"
    if v < 25:
        return "Normal", "ok"
    if v < 30:
        return "Berat badan berlebih", "warn"
    return "Obesitas", "bad"


def kategori_glukosa(v):
    if v < 100:
        return "Normal", "ok"
    if v < 126:
        return "Prediabetes", "warn"
    return "Rentang diabetes", "bad"


def kategori_tensi(v):
    if v < 120:
        return "Normal", "ok"
    if v < 140:
        return "Meningkat", "warn"
    return "Tinggi", "bad"


# ============================================================
# GAYA (CSS)
# ============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Albert+Sans:wght@400;500;600&family=Sora:wght@600;700&display=swap');
:root{--ink:#1D1B4C;--ink2:#2E2B7A;--violet:#5B4FD6;--violet-d:#463BB5;--lav:#EEECFB;--bg:#F6F5FC;--line:#E1DEF3;--soft:#5E5B82;
--low:#13805F;--low-bg:#E2F5EE;--low-d:#0E5F46;--high:#D63C50;--high-bg:#FCEAED;--high-d:#A3283A;
--warn-bg:#FDF1D8;--warn-d:#7A5207;--info-bg:#E8EEFB;--info-d:#27468F;--amber:#FFC857;}
.stApp,[data-testid="stAppViewContainer"]{background:var(--bg);color:var(--ink);}
.stApp,.stApp p,.stApp label,.stApp li,.stApp input,.stApp textarea,.stApp button{font-family:'Albert Sans',system-ui,-apple-system,'Segoe UI',sans-serif;}
.stApp h1,.stApp h2,.stApp h3{font-family:'Sora','Albert Sans',sans-serif;color:var(--ink);letter-spacing:-.01em;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:820px;padding:1.2rem 1rem 4rem;}
#MainMenu,footer{visibility:hidden;}

.hero{display:flex;flex-wrap:wrap;align-items:center;gap:18px;background:linear-gradient(135deg,#1D1B4C 0%,#2E2B7A 70%,#3A35A0 100%);color:#fff;border-radius:22px;padding:26px 26px 24px;margin-bottom:14px;}
.hero-text{flex:1 1 300px;min-width:0;}
.hero-title{font-family:'Sora',sans-serif;font-size:2rem;line-height:1.12;font-weight:700;margin:0 0 10px;color:#fff;}
.hero-sub{margin:0;color:#D5D3F5;font-size:1rem;line-height:1.55;max-width:46ch;}
.hero-pills{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px;}
.hero-pill{background:rgba(255,255,255,.13);border-radius:999px;padding:5px 13px;font-size:.86rem;color:#fff;}
.hero-art{flex:0 0 190px;width:190px;max-width:100%;}
.hero-art svg{width:100%;height:auto;display:block;}

.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--line);overflow-x:auto;}
.stTabs [data-baseweb="tab"]{height:46px;padding:0 14px;color:var(--soft);font-weight:600;}
.stTabs [aria-selected="true"]{color:var(--violet-d);}
.stTabs [data-baseweb="tab-highlight"]{background-color:var(--violet);}

div[data-testid="stVerticalBlockBorderWrapper"]{border-color:var(--line) !important;border-radius:16px;}
.stNumberInput label,.stTextInput label,.stSlider label,.stRadio>label{color:var(--soft);font-weight:500;}

.stButton>button,.stDownloadButton>button{width:100%;border-radius:12px;min-height:48px;font-weight:600;border:1px solid var(--line);background:#fff;color:var(--violet-d);}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--violet);background:var(--lav);color:var(--violet-d);}
.stButton>button[kind="primary"],.stButton>button[data-testid="stBaseButton-primary"]{background:var(--violet);color:#fff;border-color:var(--violet);}
.stButton>button[kind="primary"]:hover,.stButton>button[data-testid="stBaseButton-primary"]:hover{background:var(--violet-d);border-color:var(--violet-d);color:#fff;}
.stButton>button:focus-visible,.stDownloadButton>button:focus-visible{outline:3px solid #A79FF0;outline-offset:2px;}

.step{display:flex;gap:14px;align-items:flex-start;margin:30px 0 12px;}
.step-n{flex:0 0 34px;height:34px;border-radius:50%;background:var(--violet);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;font-family:'Sora',sans-serif;}
.step-n.end{background:var(--ink);}
.step-t{font-family:'Sora',sans-serif;font-weight:700;font-size:1.15rem;color:var(--ink);line-height:1.25;}
.step-d{color:var(--soft);font-size:.93rem;margin-top:3px;line-height:1.5;}
.sec{font-family:'Sora',sans-serif;font-weight:700;font-size:1.05rem;color:var(--ink);margin:26px 0 8px;}
.note{font-size:.86rem;color:var(--soft);margin:8px 2px 0;line-height:1.5;}

.tbl-wrap{overflow:auto;background:#fff;border:1px solid var(--line);border-radius:14px;}
.tbl-wrap.scroll{max-height:420px;}
table.tbl{width:100%;border-collapse:collapse;font-size:.92rem;white-space:nowrap;}
table.tbl th{position:sticky;top:0;text-align:left;background:var(--lav);color:var(--ink2);font-weight:600;padding:10px 14px;border-bottom:1px solid var(--line);}
table.tbl td{padding:10px 14px;border-bottom:1px solid #EFEDF9;color:var(--ink);}
table.tbl tr:last-child td{border-bottom:none;}
table.tbl tr.pick td{background:#F1EFFE;}
table.tbl tr.pick td:first-child{box-shadow:inset 4px 0 0 var(--violet);}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.82rem;font-weight:600;}
.pill.low{background:var(--low-bg);color:var(--low-d);}
.pill.high{background:var(--high-bg);color:var(--high-d);}

.strip{display:flex;gap:10px;overflow-x:auto;padding:4px 2px 10px;}
.nb{flex:0 0 auto;text-align:center;min-width:64px;}
.nb .dot{width:52px;height:52px;border-radius:50%;margin:0 auto 6px;display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;font-family:'Sora',sans-serif;font-size:1.05rem;}
.nb .dot.low{background:var(--low);}
.nb .dot.high{background:var(--high);}
.nb .id{font-weight:600;font-size:.9rem;color:var(--ink);}
.nb .jr{font-size:.78rem;color:var(--soft);}

.votebar{display:flex;height:16px;border-radius:999px;overflow:hidden;background:var(--line);margin:8px 0;}
.votebar .low{background:var(--low);}
.votebar .high{background:var(--high);}
.votelabel{display:flex;justify-content:space-between;gap:10px;font-size:.92rem;font-weight:600;}
.votelabel .l{color:var(--low-d);}
.votelabel .h{color:var(--high-d);}

.chartbox{background:#fff;border:1px solid var(--line);border-radius:16px;padding:12px 10px 6px;}
.chartbox svg{width:100%;max-width:560px;height:auto;display:block;margin:0 auto;}
.ct{font-size:13px;fill:#5E5B82;font-family:'Albert Sans',sans-serif;}
.ctb{font-size:13px;font-weight:600;fill:#1D1B4C;font-family:'Albert Sans',sans-serif;}
.pulse{transform-box:fill-box;transform-origin:center;animation:pulse 2.6s ease-in-out infinite;}
@keyframes pulse{0%,100%{transform:scale(1);opacity:.30;}50%{transform:scale(1.3);opacity:.12;}}
@media (prefers-reduced-motion:reduce){.pulse{animation:none;}}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin:10px 4px 4px;font-size:.84rem;color:var(--soft);}
.legend span{display:inline-flex;align-items:center;gap:6px;}
.sw{display:inline-block;width:12px;height:12px;border-radius:50%;}
.sw.low{background:var(--low);}
.sw.high{background:var(--high);}
.sw.ring{background:#fff;border:2px solid var(--ink);}
.sw.pat{background:var(--ink);border-radius:2px;width:10px;height:10px;transform:rotate(45deg);}
.sw.me{background:var(--violet);border-radius:3px;width:14px;height:8px;}

.verdict{border-radius:18px;padding:24px 24px 22px;border-left:8px solid;}
.verdict.low{background:var(--low-bg);border-color:var(--low);}
.verdict.high{background:var(--high-bg);border-color:var(--high);}
.v-who{font-size:.95rem;color:var(--soft);}
.v-title{font-family:'Sora',sans-serif;font-size:2.1rem;font-weight:700;line-height:1.1;margin:4px 0 8px;}
.verdict.low .v-title{color:var(--low-d);}
.verdict.high .v-title{color:var(--high-d);}
.v-sub{color:var(--ink);font-size:1rem;line-height:1.5;}
.meter{margin-top:18px;}
.meter-track{position:relative;height:12px;border-radius:999px;background:linear-gradient(90deg,#BFE6D8 0%,#BFE6D8 50%,#F6C3CA 50%,#F6C3CA 100%);}
.meter-pin{position:absolute;top:50%;width:22px;height:22px;border-radius:50%;background:var(--ink);border:3px solid #fff;transform:translate(-50%,-50%);box-shadow:0 1px 4px rgba(29,27,76,.35);}
.meter-scale{display:flex;justify-content:space-between;gap:8px;margin-top:8px;font-size:.84rem;color:var(--soft);}
.meter-scale b{color:var(--ink);}
.disclaimer{font-size:.84rem;color:var(--soft);margin:10px 2px 0;}

.chips{display:flex;flex-wrap:wrap;gap:8px;}
.chip{display:inline-flex;align-items:center;gap:8px;padding:7px 13px;border-radius:999px;font-size:.9rem;font-weight:500;}
.chip b{font-weight:700;}
.chip.ok,.chip.low{background:var(--low-bg);color:var(--low-d);}
.chip.warn{background:var(--warn-bg);color:var(--warn-d);}
.chip.bad,.chip.high{background:var(--high-bg);color:var(--high-d);}
.chip.info{background:var(--info-bg);color:var(--info-d);}
.chip.cur{outline:2px solid var(--ink);outline-offset:1px;}

.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}
.kpi{background:#fff;border:1px solid var(--line);border-radius:14px;padding:14px 14px 12px;}
.kpi b{display:block;font-family:'Sora',sans-serif;font-size:1.6rem;color:var(--ink);line-height:1.15;}
.kpi span{display:block;font-size:.84rem;color:var(--soft);margin-top:4px;line-height:1.35;}

.guide{background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px 18px;margin-bottom:12px;line-height:1.6;}
.guide b{color:var(--ink);}
.guide ol,.guide ul{margin:8px 0 0 1.1rem;padding:0;}
.guide li{margin:4px 0;}

@media (max-width:640px){
 .hero{padding:22px 20px;}
 .hero-title{font-size:1.65rem;}
 .hero-art{flex-basis:150px;width:150px;margin:0 auto;}
 .v-title{font-size:1.7rem;}
 .kpi b{font-size:1.3rem;}
}
</style>
"""


# ============================================================
# PEMBANGUN HTML (semua satu baris agar aman di st.markdown)
# ============================================================
def e(x):
    import html
    return html.escape(str(x))


def fmt(v):
    try:
        return f"{float(v):g}"
    except Exception:
        return str(v)


def pill(kelas):
    return f'<span class="pill {"high" if int(kelas) == 1 else "low"}">{LABEL[int(kelas)]}</span>'


def tabel_html(headers, rows, pilih=None, scroll=False):
    pilih = pilih or [False] * len(rows)
    th = "".join(f"<th>{e(h)}</th>" for h in headers)
    trs = "".join(
        f'<tr class="{"pick" if p else ""}">' + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
        for row, p in zip(rows, pilih))
    return (f'<div class="tbl-wrap{" scroll" if scroll else ""}"><table class="tbl">'
            f'<thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>')


def langkah(n, judul, deskripsi, akhir=False):
    st.markdown(
        f'<div class="step"><div class="step-n{" end" if akhir else ""}">{n}</div><div>'
        f'<div class="step-t">{e(judul)}</div><div class="step-d">{e(deskripsi)}</div></div></div>',
        unsafe_allow_html=True)


def bagian(judul):
    st.markdown(f'<div class="sec">{e(judul)}</div>', unsafe_allow_html=True)


def catatan(teks):
    st.markdown(f'<div class="note">{e(teks)}</div>', unsafe_allow_html=True)


def svg_hero():
    rendah = [(34, 112), (62, 92), (44, 66), (80, 120), (96, 84)]
    tinggi = [(150, 48), (178, 30), (170, 76), (196, 56), (136, 26)]
    px, py = 120, 64
    dekat = [(150, 48), (136, 26), (170, 76)]
    o = ['<svg viewBox="0 0 220 150" role="img" aria-label="Ilustrasi: pasien baru dibandingkan dengan pasien terdekat">']
    for x, y in dekat:
        o.append(f'<line x1="{px}" y1="{py}" x2="{x}" y2="{y}" stroke="#FFFFFF" stroke-opacity=".55" stroke-width="1.5" stroke-dasharray="4 4"/>')
    for x, y in rendah:
        o.append(f'<circle cx="{x}" cy="{y}" r="7" fill="#3FC99B"/>')
    for x, y in tinggi:
        o.append(f'<circle cx="{x}" cy="{y}" r="7" fill="#FF7A8B"/>')
    o.append(f'<circle class="pulse" cx="{px}" cy="{py}" r="17" fill="#FFC857"/>')
    o.append(f'<polygon points="{px},{py-10} {px+10},{py} {px},{py+10} {px-10},{py}" fill="#FFC857" stroke="#1D1B4C" stroke-width="2"/>')
    o.append("</svg>")
    return "".join(o)


def html_hero(n, n_tinggi):
    return (
        '<div class="hero"><div class="hero-text">'
        '<div class="hero-title" role="heading" aria-level="1">Cek Risiko Diabetes</div>'
        '<div class="hero-sub">Masukkan data pasien. Sistem mencari pasien yang paling mirip di dataset, '
        'lalu menentukan risikonya dengan algoritma K-Nearest Neighbors.</div>'
        f'<div class="hero-pills"><span class="hero-pill">{n} pasien di dataset</span>'
        f'<span class="hero-pill">{n_tinggi} berisiko tinggi</span></div></div>'
        f'<div class="hero-art">{svg_hero()}</div></div>')


def tick_bagus(lo, hi, n=5):
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = mag * 10
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    v = math.ceil(lo / step) * step
    out = []
    while v <= hi + 1e-9:
        out.append(round(v, 6))
        v += step
    return out


def svg_peta(h):
    d, b, tab = h["data"], h["baru"], h["tabel"]
    W, H, L, R, T, B = 480, 340, 56, 14, 12, 48
    pw, ph = W - L - R, H - T - B

    def rng(vals):
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.08 or 1.0
        return lo - pad, hi + pad

    x0, x1 = rng(list(d["Glukosa"]) + [b["Glukosa"]])
    y0, y1 = rng(list(d["BMI"]) + [b["BMI"]])

    def sx(v):
        return L + (v - x0) / (x1 - x0) * pw

    def sy(v):
        return T + ph - (v - y0) / (y1 - y0) * ph

    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Peta posisi pasien baru terhadap pasien lain menurut glukosa dan BMI">']
    for v in tick_bagus(x0, x1):
        x = sx(v)
        o.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+ph}" stroke="{C["grid"]}"/>')
        o.append(f'<text class="ct" x="{x:.1f}" y="{T+ph+18}" text-anchor="middle">{v:g}</text>')
    for v in tick_bagus(y0, y1):
        y = sy(v)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}" stroke="{C["grid"]}"/>')
        o.append(f'<text class="ct" x="{L-8}" y="{y+4:.1f}" text-anchor="end">{v:g}</text>')
    o.append(f'<text class="ct" x="{L+pw/2:.0f}" y="{H-6}" text-anchor="middle">Glukosa (mg/dL)</text>')
    o.append(f'<text class="ct" x="13" y="{T+ph/2:.0f}" text-anchor="middle" transform="rotate(-90 13 {T+ph/2:.0f})">BMI</text>')

    px, py = sx(b["Glukosa"]), sy(b["BMI"])
    terpilih = tab[tab["Terpilih"]]
    for _, r in terpilih.iterrows():
        row = d.iloc[int(r["Idx"])]
        o.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{sx(row["Glukosa"]):.1f}" y2="{sy(row["BMI"]):.1f}" '
                 f'stroke="{C["ink"]}" stroke-opacity=".35" stroke-dasharray="4 4"/>')
    for i in range(len(d)):
        row = d.iloc[i]
        warna = C["high"] if int(row["Kelas"]) == 1 else C["low"]
        o.append(f'<circle cx="{sx(row["Glukosa"]):.1f}" cy="{sy(row["BMI"]):.1f}" r="6.5" fill="{warna}" stroke="#fff" stroke-width="2"/>')
    for _, r in terpilih.iterrows():
        row = d.iloc[int(r["Idx"])]
        cx, cy = sx(row["Glukosa"]), sy(row["BMI"])
        o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="11" fill="none" stroke="{C["ink"]}" stroke-width="2"/>')
        if h["k"] <= 7:
            o.append(f'<text class="ct" x="{cx+13:.1f}" y="{cy-9:.1f}">{e(r["Pasien"])}</text>')
    o.append(f'<circle class="pulse" cx="{px:.1f}" cy="{py:.1f}" r="19" fill="{C["violet"]}"/>')
    o.append(f'<polygon points="{px:.1f},{py-11:.1f} {px+11:.1f},{py:.1f} {px:.1f},{py+11:.1f} {px-11:.1f},{py:.1f}" '
             f'fill="{C["ink"]}" stroke="#fff" stroke-width="2"/>')
    lx = min(max(px, L + 34), W - R - 34)
    ly = max(py - 22, T + 12)
    o.append(f'<text class="ctb" x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle">Pasien baru</text>')
    o.append("</svg>")
    leg = ('<div class="legend"><span><i class="sw low"></i>Risiko rendah</span>'
           '<span><i class="sw high"></i>Risiko tinggi</span>'
           '<span><i class="sw ring"></i>Tetangga terpilih</span>'
           '<span><i class="sw pat"></i>Pasien baru</span></div>')
    return f'<div class="chartbox">{"".join(o)}</div>{leg}'


def svg_profil(h):
    nb, p0, p1, b = h["norm_baru"], h["prof0"], h["prof1"], h["baru"]
    W, X0, LEBAR = 480, 8, 380
    H = 34 + len(FITUR) * 70 + 4
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Perbandingan nilai pasien baru dengan rata-rata tiap kelas">']
    legenda = [(8, C["violet"], "Pasien"), (96, C["low"], "Rata-rata rendah"), (256, C["high"], "Rata-rata tinggi")]
    for x, warna, teks in legenda:
        o.append(f'<rect x="{x}" y="2" width="12" height="9" rx="2" fill="{warna}"/>')
        o.append(f'<text class="ct" x="{x+18}" y="11">{teks}</text>')
    for i, f in enumerate(FITUR):
        y0 = 34 + i * 70
        o.append(f'<text class="ctb" x="{X0}" y="{y0+12}">{e(f)} ({e(fmt(b[f]))})</text>')
        for j, (nilai, warna) in enumerate([(nb[f], C["violet"]), (p0[f], C["low"]), (p1[f], C["high"])]):
            y = y0 + 20 + j * 14
            w = max(0.0, min(1.0, nilai)) * LEBAR
            o.append(f'<rect x="{X0}" y="{y}" width="{LEBAR}" height="10" rx="3" fill="#F1EFFA"/>')
            if w > 0:
                o.append(f'<rect x="{X0}" y="{y}" width="{w:.1f}" height="10" rx="3" fill="{warna}"/>')
            o.append(f'<text class="ct" x="{X0+max(w,0)+6:.1f}" y="{y+9}">{nilai:.2f}</text>')
    o.append("</svg>")
    leg = ('<div class="legend"><span><i class="sw me"></i>Pasien baru</span>'
           '<span><i class="sw low"></i>Rata-rata kelas rendah</span>'
           '<span><i class="sw high"></i>Rata-rata kelas tinggi</span></div>')
    return f'<div class="chartbox">{"".join(o)}</div>{leg}'


def svg_kurva(ks, akurasi, kcur):
    W, H, L, R, T, B = 480, 240, 46, 16, 22, 40
    pw, ph = W - L - R, H - T - B
    n = len(ks)

    def xs(i):
        return L + (pw / 2 if n == 1 else i * pw / (n - 1))

    def ys(a):
        return T + ph - a * ph

    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Akurasi uji silang untuk setiap nilai K">']
    for a in (0, 0.25, 0.5, 0.75, 1):
        y = ys(a)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}" stroke="{C["grid"]}"/>')
        o.append(f'<text class="ct" x="{L-8}" y="{y+4:.1f}" text-anchor="end">{int(a*100)}%</text>')
    pts = " ".join(f"{xs(i):.1f},{ys(a):.1f}" for i, a in enumerate(akurasi))
    o.append(f'<polyline points="{pts}" fill="none" stroke="{C["violet"]}" stroke-width="2.5" stroke-linejoin="round"/>')
    for i, (k, a) in enumerate(zip(ks, akurasi)):
        x, y = xs(i), ys(a)
        o.append(f'<text class="ct" x="{x:.1f}" y="{T+ph+18}" text-anchor="middle">{k}</text>')
        if k == kcur:
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{C["violet"]}" stroke="#fff" stroke-width="2"/>')
            o.append(f'<text class="ctb" x="{min(max(x, L+30), W-R-30):.1f}" y="{max(y-12, 14):.1f}" text-anchor="middle">K={k}: {a*100:.0f}%</text>')
        else:
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#fff" stroke="{C["violet"]}" stroke-width="2"/>')
    o.append(f'<text class="ct" x="{L+pw/2:.0f}" y="{H-6}" text-anchor="middle">Nilai K</text>')
    o.append("</svg>")
    return f'<div class="chartbox">{"".join(o)}</div>'


def html_tetangga(h):
    chips = ""
    for _, r in h["tabel"][h["tabel"]["Terpilih"]].iterrows():
        c = "high" if int(r["Kelas"]) == 1 else "low"
        chips += (f'<div class="nb"><div class="dot {c}">{int(r["Peringkat"])}</div>'
                  f'<div class="id">{e(r["Pasien"])}</div><div class="jr">{r["Jarak"]:.3f}</div></div>')
    return f'<div class="strip">{chips}</div>'


def html_voting(h):
    p1 = h["skor"] * 100
    p0 = 100 - p1
    return (f'<div class="votebar"><div class="low" style="width:{p0:.1f}%"></div>'
            f'<div class="high" style="width:{p1:.1f}%"></div></div>'
            f'<div class="votelabel"><span class="l">Rendah: {h["n0"]} suara</span>'
            f'<span class="h">Tinggi: {h["n1"]} suara</span></div>')


def html_kesimpulan(h):
    pred = h["pred"]
    c = "high" if pred == 1 else "low"
    suara = h["n1"] if pred == 1 else h["n0"]
    skor = h["skor"] * 100
    ekstra = " Suara diberi bobot sesuai kedekatan." if h["berbobot"] else ""
    return (f'<div class="verdict {c}"><div class="v-who">Hasil untuk {e(h["nama"])}</div>'
            f'<div class="v-title">Risiko {LABEL[pred]}</div>'
            f'<div class="v-sub">{suara} dari {h["k"]} pasien terdekat termasuk kelas Risiko {LABEL[pred]}.{ekstra}</div>'
            f'<div class="meter"><div class="meter-track"><span class="meter-pin" style="left:{skor:.1f}%"></span></div>'
            f'<div class="meter-scale"><span>Rendah</span><span>Skor risiko tinggi: <b>{skor:.0f}%</b></span><span>Tinggi</span></div></div></div>'
            '<div class="disclaimer">Hasil ini alat bantu keputusan, bukan diagnosis medis.</div>')


def html_catatan(b):
    ada = int(b["Riwayat Keluarga"]) == 1
    items = []
    for nama, nilai, (teks, nada) in [
        ("BMI", fmt(b["BMI"]), kategori_bmi(b["BMI"])),
        ("Glukosa", fmt(b["Glukosa"]), kategori_glukosa(b["Glukosa"])),
        ("Tekanan darah", fmt(b["Tekanan Darah"]), kategori_tensi(b["Tekanan Darah"])),
        ("Riwayat keluarga", "", ("Faktor risiko", "warn") if ada else ("Tidak ada", "ok")),
    ]:
        judul = f"{nama} {nilai}".strip()
        items.append(f'<span class="chip {nada}"><b>{e(judul)}</b>{e(teks)}</span>')
    return f'<div class="chips">{"".join(items)}</div>'


def html_k_alternatif(h):
    t = h["tabel"]
    chips, hasil_set = "", set()
    for kk in (1, 3, 5, 7, 9):
        if kk > len(t):
            continue
        pred, _, _, _ = voting(t["Kelas"].to_numpy(), t["Jarak"].to_numpy(), kk, h["berbobot"])
        hasil_set.add(pred)
        cur = " cur" if kk == h["k"] else ""
        chips += f'<span class="chip {"high" if pred == 1 else "low"}{cur}"><b>K={kk}</b>{LABEL[pred]}</span>'
    if len(hasil_set) == 1:
        ket = "Hasil sama untuk semua nilai K yang dicoba, jadi hasil ini stabil."
    else:
        ket = "Hasil berubah pada beberapa nilai K. Anggap hasil ini kurang pasti dan konfirmasikan dengan pemeriksaan lain."
    return f'<div class="chips">{chips}</div><div class="note">{e(ket)}</div>'


def laporan_teks(h):
    b = h["baru"]
    baris = [
        "LAPORAN PREDIKSI RISIKO DIABETES (KNN)",
        f"Waktu   : {h['waktu']}",
        f"Pasien  : {h['nama']}",
        "",
        "Data input",
        f"- Usia            : {fmt(b['Usia'])} tahun",
        f"- BMI             : {fmt(b['BMI'])}",
        f"- Glukosa         : {fmt(b['Glukosa'])} mg/dL",
        f"- Tekanan darah   : {fmt(b['Tekanan Darah'])} mmHg",
        f"- Riwayat keluarga: {'Ada' if int(b['Riwayat Keluarga']) == 1 else 'Tidak ada'}",
        "",
        f"Parameter: K = {h['k']}, voting {'berbobot jarak' if h['berbobot'] else 'biasa'}",
        "Tetangga terdekat:",
    ]
    for _, r in h["tabel"][h["tabel"]["Terpilih"]].iterrows():
        baris.append(f"{int(r['Peringkat'])}. {r['Pasien']} (Risiko {LABEL[int(r['Kelas'])]}), jarak {r['Jarak']:.4f}")
    baris += [
        "",
        f"Voting: Rendah {h['n0']} suara, Tinggi {h['n1']} suara",
        f"Skor risiko tinggi: {h['skor'] * 100:.0f}%",
        f"KESIMPULAN: Risiko {LABEL[h['pred']]}",
        "",
        "Catatan: hasil ini alat bantu keputusan, bukan diagnosis medis.",
    ]
    return "\n".join(baris)


# ============================================================
# CALLBACK
# ============================================================
def cb_contoh(jenis):
    for kunci, nilai in CONTOH[jenis].items():
        st.session_state[kunci] = nilai


def cb_simpan():
    s = st.session_state
    h = s.hasil
    b = h["baru"]
    h["tersimpan"] = tambah_pasien(b["Usia"], b["BMI"], b["Glukosa"], b["Tekanan Darah"],
                                   b["Riwayat Keluarga"], int(s.simpan_kelas))


def cb_tambah():
    s = st.session_state
    pid = tambah_pasien(s.f_usia, s.f_bmi, s.f_glu, s.f_tensi,
                        1 if s.f_riw == "Ada" else 0, int(s.f_kelas))
    s.flash_data = f"Pasien {pid} ditambahkan. Dataset sekarang berisi {len(s.data)} pasien."


def cb_hapus():
    s = st.session_state
    if len(s.data) > 3:
        s.data = s.data.iloc[:-1].reset_index(drop=True)
        s.flash_data = "Pasien terakhir dihapus."


def cb_reset():
    s = st.session_state
    s.data = DATA_AWAL.copy()
    s.hasil = None
    s.flash_data = "Dataset dikembalikan ke 10 data awal."


def cb_muat():
    s = st.session_state
    up = s.get("csv_up")
    if up is None:
        s.flash_error = "Pilih file CSV dulu."
        return
    df, galat = baca_csv(up.getvalue())
    if galat:
        s.flash_error = galat
    else:
        s.data = df
        s.hasil = None
        s.flash_data = f"CSV dimuat. Dataset berisi {len(df)} pasien."


def cb_hapus_riwayat():
    st.session_state.riwayat = []


# ============================================================
# TAB: PREDIKSI
# ============================================================
def tampilkan_hasil(h):
    k = h["k"]
    t = h["tabel"]
    b = h["baru"]

    st.markdown('<div class="sec" style="margin-top:34px">Proses perhitungan</div>', unsafe_allow_html=True)

    langkah(3, "Samakan skala data", "Glukosa berskala ratusan, sedangkan BMI puluhan. Normalisasi Min-Max membuat semua fitur bernilai 0 sampai 1.")
    st.latex(r"x' = \dfrac{x - x_{min}}{x_{max} - x_{min}}")
    rows = [[e(f), fmt(h["fmin"][f]), fmt(h["fmax"][f]), fmt(b[f]), f'{h["norm_baru"][f]:.4f}'] for f in FITUR]
    st.markdown(tabel_html(["Fitur", "Min", "Maks", "Nilai pasien", "Hasil normalisasi"], rows), unsafe_allow_html=True)
    if any(v < 0 or v > 1 for v in h["norm_baru"].values()):
        catatan("Ada nilai yang berada di luar rentang data latih, sehingga hasil normalisasinya kurang dari 0 atau lebih dari 1. Perhitungan tetap berjalan.")

    langkah(4, "Hitung jarak ke tiap pasien", f"Makin kecil jarak, makin mirip. Baris berpenanda ungu adalah {k} pasien terdekat.")
    st.latex(r"d = \sqrt{\sum (x_i - y_i)^2}")
    rows = [[str(int(r["Peringkat"])), e(r["Pasien"]), f'{r["Jarak"]:.4f}', pill(r["Kelas"])] for _, r in t.iterrows()]
    st.markdown(tabel_html(["Peringkat", "Pasien", "Jarak", "Kelas"], rows, [bool(x) for x in t["Terpilih"]], scroll=len(t) > 12),
                unsafe_allow_html=True)
    with st.expander("Lihat contoh hitung jarak ke tetangga terdekat"):
        r0 = t.iloc[0]
        nl = h["norm_latih"].iloc[int(r0["Idx"])]
        sel = [(h["norm_baru"][f] - float(nl[f])) ** 2 for f in FITUR]
        rows = [[e(f), f'{h["norm_baru"][f]:.4f}', f'{float(nl[f]):.4f}', f'{v:.4f}'] for f, v in zip(FITUR, sel)]
        rows.append(["<b>Jumlah</b>", "", "", f"<b>{sum(sel):.4f}</b>"])
        st.markdown(tabel_html(["Fitur", "Pasien baru", str(r0["Pasien"]), "Selisih kuadrat"], rows), unsafe_allow_html=True)
        catatan(f"Jarak = akar dari {sum(sel):.4f} = {math.sqrt(sum(sel)):.4f}")

    langkah(5, f"Suara dari {k} tetangga terdekat", "Kelas yang paling banyak muncul menjadi hasil prediksi.")
    st.markdown(html_tetangga(h), unsafe_allow_html=True)
    catatan("Angka di lingkaran adalah peringkat kedekatan, angka di bawahnya jarak. Merah berarti risiko tinggi, hijau berarti risiko rendah.")
    st.markdown(html_voting(h), unsafe_allow_html=True)
    if h["berbobot"]:
        catatan("Suara diberi bobot 1 dibagi jarak, jadi tetangga yang lebih dekat berpengaruh lebih besar. Panjang batang mengikuti bobot tersebut.")
    elif h["n0"] == h["n1"]:
        catatan("Suara seri, sehingga kelas diambil dari pasien yang paling dekat.")

    langkah(6, "Lihat posisi pasien baru", "Peta ini menunjukkan di mana pasien baru berada di antara pasien lain.")
    st.markdown(svg_peta(h), unsafe_allow_html=True)
    catatan("Peta hanya memakai dua dari lima fitur (glukosa dan BMI), jadi jarak di peta tidak persis sama dengan jarak hitungan.")

    langkah(7, "Bandingkan profil nilai", "Nilai dibandingkan pada skala 0 sampai 1 agar adil untuk semua fitur.")
    st.markdown(svg_profil(h), unsafe_allow_html=True)
    catatan("Jika batang pasien lebih dekat ke rata-rata satu kelas, pasien lebih mirip dengan kelas itu pada fitur tersebut.")

    langkah("✓", "Kesimpulan", "Hasil akhir berdasarkan proses di atas.", akhir=True)
    st.markdown(html_kesimpulan(h), unsafe_allow_html=True)

    bagian("Catatan dari nilai yang dimasukkan")
    st.markdown(html_catatan(b), unsafe_allow_html=True)
    catatan("Kategori memakai acuan umum: BMI menurut WHO, glukosa puasa, dan tekanan sistolik. Ini informasi tambahan dan tidak memengaruhi hitungan KNN.")

    bagian("Bagaimana jika nilai K diganti?")
    st.markdown(html_k_alternatif(h), unsafe_allow_html=True)

    bagian("Simpan atau unduh")
    with st.container(border=True):
        if h["tersimpan"]:
            st.success(f"Pasien ini sudah disimpan ke dataset sebagai {h['tersimpan']}.")
        else:
            st.caption("Simpan pasien ini ke dataset latih agar ikut dihitung pada prediksi berikutnya. Ubah kelasnya jika kelas sebenarnya sudah diketahui.")
            st.radio("Kelas yang disimpan", [0, 1], index=int(h["pred"]), horizontal=True, key="simpan_kelas",
                     format_func=lambda x: f"Risiko {LABEL[x]}")
            st.button("Simpan ke dataset", key="btn_simpan", on_click=cb_simpan)
        st.download_button("Unduh laporan (TXT)", laporan_teks(h).encode("utf-8"),
                           file_name="laporan_risiko_diabetes.txt", mime="text/plain", key="dl_laporan")


def render_prediksi():
    s = st.session_state
    data = s.data

    langkah(1, "Isi data pasien", "Semua isian dipakai dalam perhitungan.")
    with st.container(border=True):
        st.caption("Isi cepat dengan contoh")
        b1, b2 = st.columns(2)
        b1.button("Contoh risiko rendah", on_click=cb_contoh, args=("rendah",), key="btn_c1")
        b2.button("Contoh risiko tinggi", on_click=cb_contoh, args=("tinggi",), key="btn_c2")
        st.text_input("Nama atau kode pasien (boleh dikosongkan)", key="in_nama", placeholder="Contoh: Budi")
        c1, c2 = st.columns(2)
        c1.number_input("Usia (tahun)", min_value=1, max_value=120, step=1, key="in_usia")
        c2.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1, format="%.1f", key="in_bmi")
        c1.number_input("Glukosa (mg/dL)", min_value=40, max_value=400, step=1, key="in_glu")
        c2.number_input("Tekanan darah sistolik (mmHg)", min_value=60, max_value=250, step=1, key="in_tensi")
        st.radio("Riwayat diabetes dalam keluarga", ["Ada", "Tidak ada"], horizontal=True, key="in_riw")

    langkah(2, "Tentukan nilai K", "K adalah jumlah pasien terdekat yang ikut memberi suara. Pilih angka ganjil agar suara tidak seri.")
    with st.container(border=True):
        kmax = min(15, len(data))
        s.in_k = int(min(max(int(s.in_k), 1), kmax))
        st.slider("Nilai K (jumlah pasien terdekat)", 1, kmax, key="in_k")
        st.toggle("Beri suara lebih besar pada pasien yang lebih dekat", key="in_bobot",
                  help="Voting berbobot: tiap tetangga bersuara sebesar 1 dibagi jaraknya.")

    if st.button("Hitung risiko", type="primary", key="btn_hitung"):
        baru = {"Usia": s.in_usia, "BMI": s.in_bmi, "Glukosa": s.in_glu,
                "Tekanan Darah": s.in_tensi, "Riwayat Keluarga": 1 if s.in_riw == "Ada" else 0}
        h = hitung_knn(s.data, baru, s.in_k, s.in_bobot)
        h["nama"] = (s.in_nama or "").strip() or "pasien ini"
        s.hasil = h
        s.riwayat.append({
            "Waktu": h["waktu"], "Pasien": h["nama"], "Usia": baru["Usia"], "BMI": baru["BMI"],
            "Glukosa": baru["Glukosa"], "Tekanan Darah": baru["Tekanan Darah"],
            "Riwayat Keluarga": baru["Riwayat Keluarga"], "K": h["k"],
            "Voting": "Berbobot" if h["berbobot"] else "Biasa",
            "Hasil": LABEL[h["pred"]], "Skor Risiko Tinggi (%)": round(h["skor"] * 100),
        })

    if s.hasil is not None:
        tampilkan_hasil(s.hasil)


# ============================================================
# TAB: ANALISIS
# ============================================================
def render_analisis():
    s = st.session_state
    data = s.data

    langkah("A", "Seberapa dapat dipercaya model ini?", "Uji silang leave-one-out: tiap pasien di dataset ditebak memakai semua pasien lain, lalu hasilnya dicocokkan dengan kelas aslinya.")
    hasil = evaluasi_loocv(data, 9, bool(s.in_bobot))
    ks = sorted(hasil)
    kcur = min(max(int(s.in_k), 1), ks[-1])
    tp, tn, fp, fn = hasil[kcur]
    total = tp + tn + fp + fn
    akurasi = (tp + tn) / total
    sens = f"{tp / (tp + fn) * 100:.0f}%" if (tp + fn) else "-"
    spes = f"{tn / (tn + fp) * 100:.0f}%" if (tn + fp) else "-"

    bagian(f"Hasil untuk K = {kcur}")
    st.markdown(
        '<div class="kpis">'
        f'<div class="kpi"><b>{akurasi*100:.0f}%</b><span>Akurasi: tebakan yang benar</span></div>'
        f'<div class="kpi"><b>{sens}</b><span>Sensitivitas: risiko tinggi yang terdeteksi</span></div>'
        f'<div class="kpi"><b>{spes}</b><span>Spesifisitas: risiko rendah yang terdeteksi</span></div></div>',
        unsafe_allow_html=True)

    bagian("Tabel kesalahan (confusion matrix)")
    st.markdown(tabel_html(
        ["", "Ditebak rendah", "Ditebak tinggi"],
        [["<b>Aslinya rendah</b>", str(tn), str(fp)], ["<b>Aslinya tinggi</b>", str(fn), str(tp)]]),
        unsafe_allow_html=True)

    bagian("Akurasi untuk setiap nilai K")
    st.markdown(svg_kurva(ks, [(hasil[k][0] + hasil[k][1]) / sum(hasil[k]) for k in ks], kcur), unsafe_allow_html=True)
    catatan("Titik besar adalah nilai K yang sedang dipilih di tab Prediksi. Dataset masih kecil, jadi akurasi yang tinggi belum tentu berlaku untuk pasien sungguhan.")

    bagian("Rata-rata nilai tiap kelas")
    rows = []
    for f in FITUR:
        m0 = data.loc[data["Kelas"] == 0, f].mean()
        m1 = data.loc[data["Kelas"] == 1, f].mean()
        if f == "Riwayat Keluarga":
            rows.append([e(f), f"{m0*100:.0f}% ada", f"{m1*100:.0f}% ada"])
        else:
            rows.append([e(f), f"{m0:.1f}", f"{m1:.1f}"])
    rows.append(["<b>Jumlah pasien</b>", f"<b>{int((data['Kelas'] == 0).sum())}</b>", f"<b>{int((data['Kelas'] == 1).sum())}</b>"])
    st.markdown(tabel_html(["Fitur", "Risiko rendah", "Risiko tinggi"], rows), unsafe_allow_html=True)


# ============================================================
# TAB: DATA DAN RIWAYAT
# ============================================================
def tabel_dataset(data):
    rows = []
    for _, r in data.iterrows():
        rows.append([e(r["Pasien"]), fmt(r["Usia"]), fmt(r["BMI"]), fmt(r["Glukosa"]), fmt(r["Tekanan Darah"]),
                     "Ada" if int(r["Riwayat Keluarga"]) == 1 else "Tidak ada", pill(r["Kelas"])])
    return tabel_html(["Pasien", "Usia", "BMI", "Glukosa", "Tekanan", "Riwayat", "Kelas"], rows, scroll=len(rows) > 12)


def render_data():
    s = st.session_state
    data = s.data

    if s.get("flash_data"):
        st.success(s.pop("flash_data"))
    if s.get("flash_error"):
        st.error(s.pop("flash_error"))

    langkah("+", "Tambah pasien ke dataset", "Pasien baru langsung ikut dihitung pada prediksi berikutnya.")
    with st.container(border=True):
        a1, a2 = st.columns(2)
        a1.number_input("Usia (tahun)", min_value=1, max_value=120, step=1, key="f_usia")
        a2.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1, format="%.1f", key="f_bmi")
        a1.number_input("Glukosa (mg/dL)", min_value=40, max_value=400, step=1, key="f_glu")
        a2.number_input("Tekanan darah sistolik (mmHg)", min_value=60, max_value=250, step=1, key="f_tensi")
        st.radio("Riwayat diabetes dalam keluarga", ["Ada", "Tidak ada"], horizontal=True, key="f_riw")
        st.radio("Kelas", [0, 1], horizontal=True, key="f_kelas", format_func=lambda x: f"Risiko {LABEL[x]}")
        st.button("Tambah ke dataset", type="primary", key="btn_tambah", on_click=cb_tambah)

    langkah("≡", f"Isi dataset ({len(data)} pasien)", "Data ini menjadi pembanding untuk setiap prediksi.")
    st.markdown(tabel_dataset(data), unsafe_allow_html=True)
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    b1.button("Hapus pasien terakhir", disabled=len(data) <= 3, key="btn_hapus", on_click=cb_hapus)
    b2.button("Kembali ke 10 data awal", key="btn_reset", on_click=cb_reset)

    langkah("↧", "Simpan dan muat dataset", "Data tambahan hilang saat halaman dimuat ulang. Unduh CSV untuk menyimpannya, lalu muat lagi saat dibutuhkan.")
    st.download_button("Unduh dataset (CSV)", data.to_csv(index=False).encode("utf-8"),
                       file_name="dataset_pasien.csv", mime="text/csv", key="dl_data")
    st.file_uploader("Muat dataset dari file CSV", type="csv", key="csv_up")
    st.button("Pakai CSV ini", key="btn_muat", on_click=cb_muat)
    catatan("Kolom CSV: " + ", ".join(KOLOM) + ". Kelas diisi 0 (rendah) atau 1 (tinggi). Riwayat Keluarga diisi 0 atau 1.")

    langkah("⟲", "Riwayat prediksi", "Daftar prediksi yang sudah dihitung selama halaman ini terbuka.")
    if not s.riwayat:
        catatan("Belum ada prediksi. Hitung satu pasien di tab Prediksi, nanti hasilnya muncul di sini.")
    else:
        rows = []
        for r in reversed(s.riwayat):
            rows.append([e(r["Waktu"]), e(r["Pasien"]), fmt(r["Usia"]), fmt(r["BMI"]), fmt(r["Glukosa"]),
                         fmt(r["Tekanan Darah"]), "Ada" if int(r["Riwayat Keluarga"]) == 1 else "Tidak ada",
                         str(r["K"]), e(r["Voting"]), pill(1 if r["Hasil"] == "Tinggi" else 0),
                         f'{r["Skor Risiko Tinggi (%)"]}%'])
        st.markdown(tabel_html(["Waktu", "Pasien", "Usia", "BMI", "Glukosa", "Tekanan", "Riwayat", "K", "Voting", "Hasil", "Skor"],
                               rows, scroll=len(rows) > 8), unsafe_allow_html=True)
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        d1.download_button("Unduh riwayat (CSV)", pd.DataFrame(s.riwayat).to_csv(index=False).encode("utf-8"),
                           file_name="riwayat_prediksi.csv", mime="text/csv", key="dl_riwayat")
        d2.button("Hapus riwayat", key="btn_hapus_riwayat", on_click=cb_hapus_riwayat)


# ============================================================
# TAB: PANDUAN
# ============================================================
def render_panduan():
    st.markdown(
        '<div class="guide"><b>Apa yang dilakukan sistem ini</b><br>'
        'Sistem membandingkan pasien baru dengan pasien yang sudah tercatat di dataset. '
        'Pasien yang paling mirip menjadi penentu: jika kebanyakan dari mereka berisiko tinggi, pasien baru diprediksi berisiko tinggi.</div>'
        '<div class="guide"><b>Cara kerja KNN</b><ol>'
        '<li>Samakan skala semua fitur dengan normalisasi Min-Max.</li>'
        '<li>Hitung jarak Euclidean pasien baru ke setiap pasien di dataset.</li>'
        '<li>Urutkan dari yang terdekat, lalu ambil K pasien teratas.</li>'
        '<li>Hitung suara tiap kelas. Pada voting berbobot, pasien yang lebih dekat bersuara lebih besar.</li>'
        '<li>Kelas dengan suara terbanyak menjadi kesimpulan.</li></ol></div>'
        '<div class="guide"><b>Arti data yang dimasukkan</b><ul>'
        '<li>Usia dalam tahun.</li>'
        '<li>BMI adalah indeks massa tubuh (kg per meter persegi).</li>'
        '<li>Glukosa dalam mg/dL.</li>'
        '<li>Tekanan darah memakai angka sistolik (angka atas) dalam mmHg.</li>'
        '<li>Riwayat keluarga: ada atau tidak ada anggota keluarga dengan diabetes.</li>'
        '<li>Kelas: 0 berarti risiko rendah, 1 berarti risiko tinggi.</li></ul></div>'
        '<div class="guide"><b>Batasan</b><br>'
        'Dataset awal hanya berisi 10 pasien dari soal, jadi hasilnya untuk latihan dan belum cocok dipakai sebagai keputusan medis. '
        'Hasil sebaiknya dikonfirmasi dengan pemeriksaan laboratorium dan tenaga medis.</div>',
        unsafe_allow_html=True)


# ============================================================
# HALAMAN UTAMA
# ============================================================
def inisialisasi():
    s = st.session_state
    awal = {
        "data": DATA_AWAL.copy(), "hasil": None, "riwayat": [],
        "in_nama": "", "in_usia": 42, "in_bmi": 28.5, "in_glu": 130, "in_tensi": 128,
        "in_riw": "Ada", "in_k": 3, "in_bobot": False,
        "f_usia": 40, "f_bmi": 25.0, "f_glu": 100, "f_tensi": 120, "f_riw": "Tidak ada", "f_kelas": 0,
    }
    for kunci, nilai in awal.items():
        if kunci not in s:
            s[kunci] = nilai


def main():
    st.set_page_config(page_title="Cek Risiko Diabetes KNN", page_icon="🩺", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    inisialisasi()
    data = st.session_state.data
    st.markdown(html_hero(len(data), int((data["Kelas"] == 1).sum())), unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["Prediksi", "Analisis", "Data", "Panduan"])
    with t1:
        render_prediksi()
    with t2:
        render_analisis()
    with t3:
        render_data()
    with t4:
        render_panduan()


main()
