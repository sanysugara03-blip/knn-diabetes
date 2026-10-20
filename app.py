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

C = {"ink": "#1D1B4C", "violet": "#5B4FD6", "low": "#13805F", "high": "#D63C50",
     "soft": "#5E5B82", "line": "#E1DEF3", "grid": "#ECEAF7", "amber": "#FFC857"}


# ============================================================
# INTI ALGORITMA KNN
# ============================================================
def normalisasi(data, pasien):
    fmin = data[FITUR].min()
    fmax = data[FITUR].max()
    rentang = (fmax - fmin).replace(0, 1)
    norm_latih = (data[FITUR] - fmin) / rentang
    norm_baru = {f: float((pasien[f] - fmin[f]) / rentang[f]) for f in FITUR}
    return fmin, fmax, norm_latih, norm_baru


def voting(kelas, jarak, k, berbobot):
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
        else:
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

    return {
        "baru": dict(pasien), "k": k, "berbobot": bool(berbobot), "data": data.copy(),
        "tabel": tabel, "fmin": fmin, "fmax": fmax, "norm_latih": norm_latih,
        "norm_baru": norm_baru, "pred": pred, "n0": n0, "n1": n1, "skor": skor,
        "waktu": waktu_sekarang(), "tersimpan": None,
    }


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
# GAYA (CSS) & GRAFIK SVG
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

.chartbox{background:#fff;border:1px solid var(--line);border-radius:16px;padding:12px 10px 6px;margin-top:10px;}
.chartbox svg{width:100%;max-width:560px;height:auto;display:block;margin:0 auto;}
.ct{font-size:13px;fill:#5E5B82;font-family:'Albert Sans',sans-serif;}
.ctb{font-size:13px;font-weight:600;fill:#1D1B4C;font-family:'Albert Sans',sans-serif;}
.pulse{transform-box:fill-box;transform-origin:center;animation:pulse 2.6s ease-in-out infinite;}
@keyframes pulse{0%,100%{transform:scale(1);opacity:.30;}50%{transform:scale(1.3);opacity:.12;}}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin:10px 4px 4px;font-size:.84rem;color:var(--soft);}
.legend span{display:inline-flex;align-items:center;gap:6px;}
.sw{display:inline-block;width:12px;height:12px;border-radius:50%;}
.sw.low{background:var(--low);}
.sw.high{background:var(--high);}
.sw.ring{background:#fff;border:2px solid var(--ink);}
.sw.pat{background:var(--ink);border-radius:2px;width:10px;height:10px;transform:rotate(45deg);}

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

.guide{background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px 18px;margin-bottom:12px;line-height:1.6;}
.guide b{color:var(--ink);}
.guide ol,.guide ul{margin:8px 0 0 1.1rem;padding:0;}
.guide li{margin:4px 0;}

.analisis-box{background:var(--lav);border:1px solid var(--line);border-radius:14px;padding:16px 18px;margin-top:16px;line-height:1.6;}
.analisis-box b{color:var(--ink);}
</style>
"""


def e(x):
    import html
    return html.escape(str(x))


def fmt(v):
    try:
        return f"{float(v):g}"
    except Exception:
        return str(v)


def svg_hero():
    rendah = [(34, 112), (62, 92), (44, 66), (80, 120), (96, 84)]
    tinggi = [(150, 48), (178, 30), (170, 76), (196, 56), (136, 26)]
    px, py = 120, 64
    o = ['<svg viewBox="0 0 220 150" role="img" aria-label="Ilustrasi">']
    for x, y in rendah:
        o.append(f'<circle cx="{x}" cy="{y}" r="7" fill="#3FC99B"/>')
    for x, y in tinggi:
        o.append(f'<circle cx="{x}" cy="{y}" r="7" fill="#FF7A8B"/>')
    o.append(f'<circle cx="{px}" cy="{py}" r="17" fill="#FFC857" opacity=".3"/>')
    o.append(f'<polygon points="{px},{py-10} {px+10},{py} {px},{py+10} {px-10},{py}" fill="#FFC857" stroke="#1D1B4C" stroke-width="2"/>')
    o.append("</svg>")
    return "".join(o)


def html_hero(n, n_tinggi):
    return (
        '<div class="hero"><div class="hero-text">'
        '<div class="hero-title" role="heading" aria-level="1">Cek Risiko Diabetes</div>'
        '<div class="hero-sub">Sistem Pendukung Keputusan medis berbasis algoritma K-Nearest Neighbors (KNN) '
        'untuk mengestimasi risiko diabetes secara cepat dan transparan.</div>'
        f'<div class="hero-pills"><span class="hero-pill">{n} pasien di dataset</span>'
        f'<span class="hero-pill">{n_tinggi} berisiko tinggi</span></div></div>'
        f'<div class="hero-art">{svg_hero()}</div></div>')


def tick_bagus(lo, hi, n=5):
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
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

    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Peta posisi pasien">']
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
    o.append("</svg>")
    leg = ('<div class="legend"><span><i class="sw low"></i>Risiko rendah</span>'
           '<span><i class="sw high"></i>Risiko tinggi</span>'
           '<span><i class="sw ring"></i>Tetangga terpilih</span>'
           '<span><i class="sw pat"></i>Pasien baru</span></div>')
    return f'<div class="chartbox">{"".join(o)}</div>{leg}'


def langkah(n, judul, deskripsi, akhir=False):
    st.markdown(
        f'<div class="step"><div class="step-n{" end" if akhir else ""}">{n}</div><div>'
        f'<div class="step-t">{e(judul)}</div><div class="step-d">{e(deskripsi)}</div></div></div>',
        unsafe_allow_html=True)


def bagian(judul):
    st.markdown(f'<div class="sec">{e(judul)}</div>', unsafe_allow_html=True)


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


def html_analisis(h):
    b = h["baru"]
    pred = h["pred"]
    skor = h["skor"] * 100
    
    faktor = []
    if b["Glukosa"] >= 140:
        faktor.append("kadar glukosa darah yang cukup tinggi (≥ 140 mg/dL)")
    elif b["Glukosa"] >= 120:
        faktor.append("kadar glukosa darah di atas batas rata-rata")
    
    if b["BMI"] >= 27:
        faktor.append("indeks massa tubuh (BMI) yang termasuk kategori berlebih")
    
    if b["Riwayat Keluarga"] == 1:
        faktor.append("memiliki riwayat keluarga dengan diabetes")

    penjelasan_faktor = ", ".join(faktor) if faktor else "kombinasi parameter klinis secara keseluruhan"
    
    narasi = (
        f"Berdasarkan penelusuran <b>{h['k']} tetangga terdekat</b> menggunakan algoritma KNN, "
        f"pasien <b>{e(h['nama'])}</b> dikategorikan ke dalam tingkat <b>Risiko {LABEL[pred]}</b> "
        f"dengan tingkat keyakinan skor sebesar <b>{skor:.0f}%</b>.<br><br>"
        f"Hal ini dipengaruhi oleh kemiripan karakteristik klinis dengan pasien lain di dataset, "
        f"terutama dari segi {penjelasan_faktor}. "
        f"{'Disarankan untuk melakukan pemeriksaan lanjutan berkala ke fasilitas kesehatan.' if pred == 1 else 'Pertahankan pola hidup sehat dan jaga kestabilan gula darah.'}"
    )
    return f'<div class="analisis-box"><b>Analisis & Interpretasi Hasil:</b><br>{narasi}</div>'


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
    baris.extend([
        "",
        f"Voting: Rendah {h['n0']} suara, Tinggi {h['n1']} suara",
        f"Skor risiko tinggi: {h['skor'] * 100:.0f}%",
        f"KESIMPULAN: Risiko {LABEL[h['pred']]}",
        "",
        "Catatan: hasil ini alat bantu keputusan, bukan diagnosis medis.",
    ])
    return "\n".join(baris)


# ============================================================
# INISIALISASI SESSION STATE
# ============================================================
if "data" not in st.session_state:
    st.session_state.data = DATA_AWAL.copy()
if "hasil" not in st.session_state:
    st.session_state.hasil = None
if "flash_data" not in st.session_state:
    st.session_state.flash_data = None
if "flash_error" not in st.session_state:
    st.session_state.flash_error = None


# ============================================================
# CALLBACK
# ============================================================
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


def cb_hapus_terakhir():
    s = st.session_state
    if len(s.data) > 3:
        s.data = s.data.iloc[:-1].reset_index(drop=True)
        s.flash_data = "Pasien terakhir dihapus."


def cb_hapus_pilih():
    s = st.session_state
    target_id = s.get("pilih_hapus_pasien")
    if len(s.data) <= 3:
        s.flash_error = "Dataset minimal harus memiliki 3 pasien."
        return
    if target_id:
        awal_len = len(s.data)
        s.data = s.data[s.data["Pasien"] != target_id].reset_index(drop=True)
        if len(s.data) < awal_len:
            s.flash_data = f"Pasien {target_id} berhasil dihapus dari dataset."
        else:
            s.flash_error = f"Pasien {target_id} tidak ditemukan."


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


# ============================================================
# TAMPILAN UTAMA
# ============================================================
st.set_page_config(page_title="Cek Risiko Diabetes (KNN)", page_icon="🩺", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

data_kini = st.session_state.data
n_tinggi = int((data_kini["Kelas"] == 1).sum())
st.markdown(html_hero(len(data_kini), n_tinggi), unsafe_allow_html=True)

if st.session_state.flash_data:
    st.success(st.session_state.flash_data)
    st.session_state.flash_data = None
if st.session_state.flash_error:
    st.error(st.session_state.flash_error)
    st.session_state.flash_error = None

tab_pred, tab_data = st.tabs(["🔮 Prediksi Pasien", "📊 Kelola Dataset"])

with tab_pred:
    st.markdown(
        '<div class="guide"><b>Panduan Penggunaan Sistem:</b><ol>'
        '<li>Masukkan parameter klinis pasien baru (Usia, BMI, Glukosa, Tekanan Darah, Riwayat Keluarga) pada formulir.</li>'
        '<li>Atur nilai <b>K</b> untuk menentukan berapa banyak tetangga terdekat yang dianalisis.</li>'
        '<li>Klik tombol <b>Hitung Prediksi Risiko</b> untuk melihat proses normalisasi, perhitungan jarak, grafik peta, serta analisis mendalam.</li>'
        '</ol></div>',
        unsafe_allow_html=True
    )

    bagian("1. Masukkan Data Pasien Baru")
    with st.form("form_pred"):
        nama = st.text_input("Nama atau Kode Pasien", value="Pasien 1")
        c1, c2 = st.columns(2)
        usia = c1.number_input("Usia (tahun)", 1, 120, 42, 1)
        bmi = c2.number_input("BMI", 10.0, 60.0, 28.5, 0.1, format="%.1f")
        glu = c1.number_input("Glukosa (mg/dL)", 40, 400, 130, 1)
        tensi = c2.number_input("Tekanan Darah (mmHg)", 60, 250, 128, 1)
        riw = st.radio("Riwayat Keluarga dengan Diabetes", ["Tidak ada", "Ada"], index=1, horizontal=True)
        k = st.slider("Nilai K (Jumlah Tetangga Terdekat)", 1, min(15, len(data_kini)), 3, 1)
        berbobot = st.checkbox("Gunakan pembobotan jarak (Distance Weighting)", value=False)
        hitung = st.form_submit_button("Hitung Prediksi Risiko", type="primary")

    if hitung:
        pasien = {
            "Usia": usia, "BMI": bmi, "Glukosa": glu,
            "Tekanan Darah": tensi, "Riwayat Keluarga": 1 if riw == "Ada" else 0
        }
        st.session_state.hasil = hitung_knn(data_kini, pasien, k, berbobot)
        st.session_state.hasil["nama"] = nama.strip() or "Pasien"

    h = st.session_state.hasil
    if h is not None:
        bagian("2. Proses Perhitungan K-Nearest Neighbors")
        
        langkah(1, "Normalisasi Fitur (Min-Max)", "Setiap nilai fitur diubah ke rentang [0, 1] agar porsinya setara.")
        st.markdown("Rumus Normalisasi Min-Max:")
        st.latex(r"x' = \frac{x - x_{min}}{x_{max} - x_{min}}")

        df_norm = pd.DataFrame({
            "Fitur": FITUR,
            "Min": [h["fmin"][f] for f in FITUR],
            "Max": [h["fmax"][f] for f in FITUR],
            "Nilai Pasien": [h["baru"][f] for f in FITUR],
            "Normalisasi": [round(h["norm_baru"][f], 4) for f in FITUR],
        })
        st.markdown(
            f'<div class="tbl-wrap"><table class="tbl"><thead><tr>'
            f'<th>Fitur</th><th>Min</th><th>Max</th><th>Nilai Pasien</th><th>Normalisasi</th>'
            f'</thead><tbody>' + "".join(
                f'<tr><td>{r.Fitur}</td><td>{fmt(r.Min)}</td><td>{fmt(r.Max)}</td><td>{fmt(r["Nilai Pasien"])}</td><td>{r.Normalisasi}</td></tr>'
                for _, r in df_norm.iterrows()) + '</tbody></table></div>',
            unsafe_allow_html=True)

        langkah(2, "Hitung Jarak Euclidean", "Menghitung jarak kuadrat antara pasien baru dan seluruh pasien di dataset latih.")
        st.markdown("Rumus Jarak Euclidean:")
        st.latex(r"d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}")

        t = h["tabel"].copy()
        t_headers = ["Peringkat", "ID", "Kelas Asli", "Jarak Euclidean", "Terpilih"]
        t_rows = []
        for _, r in t.iterrows():
            kls_str = f'<span class="pill {"high" if int(r["Kelas"])==1 else "low"}">{LABEL[int(r["Kelas"])]}</span>'
            pilih_str = "✔" if r["Terpilih"] else ""
            t_rows.append([int(r["Peringkat"]), e(r["Pasien"]), kls_str, f'{r["Jarak"]:.4f}', pilih_str])
        st.markdown(
            f'<div class="tbl-wrap scroll">'
            f'<table class="tbl"><thead><tr>' + "".join(f"<th>{th}</th>" for th in t_headers) + '</tr></thead><tbody>' +
            "".join(f'<tr class="{"pick" if r[4] else ""}">' + "".join(f"<td>{cell}</td>" for cell in r) + '</tr>' for r in t_rows) +
            '</tbody></table></div>',
            unsafe_allow_html=True)
        st.markdown(html_tetangga(h), unsafe_allow_html=True)

        # Grafik Peta Sebaran Spasial (Scatter Plot)
        st.markdown(
            '<div class="sec" style="margin-top:24px">Peta Sebaran Pasien (Glukosa vs BMI)</div>',
            unsafe_allow_html=True
        )
        st.markdown(svg_peta(h), unsafe_allow_html=True)

        langkah(3, f"Voting dari {h['k']} Tetangga Terdekat", "Menentukan mayoritas kelas dari tetangga terdekat yang terpilih.")
        st.markdown(html_voting(h), unsafe_allow_html=True)

        langkah(4, "Kesimpulan & Analisis Hasil", "Hasil akhir estimasi risiko beserta penjelasan naratif.", akhir=True)
        st.markdown(html_kesimpulan(h), unsafe_allow_html=True)
        st.markdown(html_analisis(h), unsafe_allow_html=True)

        bagian("Simpan Hasil ke Dataset")
        st.radio("Pilih kelas untuk disimpan ke dataset:", [0, 1], index=h["pred"], horizontal=True,
                 format_func=lambda x: f"Risiko {LABEL[x]}", key="simpan_kelas")
        st.button("Simpan Pasien Ini ke Dataset Latih", on_click=cb_simpan)
        if h["tersimpan"]:
            st.success(f"Berhasil disimpan sebagai {h['tersimpan']}.")

        bagian("Unduh Laporan")
        st.download_button("Unduh Laporan Teks (.txt)", data=laporan_teks(h),
                           file_name=f"laporan_knn_{h['nama'].replace(' ', '_')}.txt", mime="text/plain")

with tab_data:
    bagian("Manajemen Dataset Latih")
    st.write(f"Total pasien aktif di dataset: **{len(data_kini)}**")
    
    # Keterangan Informasi Kode Kolom
    st.markdown(
        '<div class="guide" style="font-size:0.9rem; padding: 12px 16px; margin-bottom: 12px;">'
        '<b>Keterangan Kolom Dataset:</b><br>'
        '• <b>Riwayat Keluarga:</b> <code>0</code> = Tidak Ada Risiko, <code>1</code> = Ada Risiko.<br>'
        '• <b>Kelas Risiko:</b> Rendah (0) atau Tinggi (1).'
        '</div>',
        unsafe_allow_html=True
    )

    # Render Tabel Dataset dengan HTML agar rapi dan sejajar
    t_tampil = data_kini.copy()
    t_tampil["Riwayat Keluarga"] = t_tampil["Riwayat Keluarga"].map(lambda x: f"{x} ({'Ada' if int(x)==1 else 'Tidak Ada'})")
    t_tampil["Kelas"] = t_tampil["Kelas"].map(lambda x: f'<span class="pill {"high" if int(x)==1 else "low"}">Risiko {LABEL[int(x)]}</span>')
    
    html_rows = []
    for _, row in t_tampil.iterrows():
        html_rows.append(
            f"<tr><td><b>{e(row['Pasien'])}</b></td>"
            f"<td>{fmt(row['Usia'])}</td>"
            f"<td>{fmt(row['BMI'])}</td>"
            f"<td>{fmt(row['Glukosa'])}</td>"
            f"<td>{fmt(row['Tekanan Darah'])}</td>"
            f"<td>{row['Riwayat Keluarga']}</td>"
            f"<td>{row['Kelas']}</td></tr>"
        )
    
    st.markdown(
        f'<div class="tbl-wrap scroll"><table class="tbl"><thead><tr>'
        f'<th>Pasien</th><th>Usia</th><th>BMI</th><th>Glukosa</th><th>Tekanan Darah</th><th>Riwayat Keluarga</th><th>Kelas Risiko</th>'
        f'</tr></thead><tbody>' + "".join(html_rows) + '</tbody></table></div>',
        unsafe_allow_html=True
    )

    bagian("Tambah Pasien Baru")
    with st.form("f_tambah"):
        c1, c2 = st.columns(2)
        st.session_state.f_usia = c1.number_input("Usia", 1, 120, 35, key="inp_u")
        st.session_state.f_bmi = c2.number_input("BMI", 10.0, 60.0, 24.0, key="inp_b")
        st.session_state.f_glu = c1.number_input("Glukosa", 40, 400, 100, key="inp_g")
        st.session_state.f_tensi = c2.number_input("Tekanan Darah", 60, 250, 120, key="inp_t")
        st.session_state.f_riw = st.radio("Riwayat Keluarga", ["Tidak ada", "Ada"], horizontal=True, key="inp_r")
        st.session_state.f_kelas = st.radio("Kelas Risiko", [0, 1], horizontal=True,
                                            format_func=lambda x: LABEL[x], key="inp_k")
        st.form_submit_button("Tambah ke Dataset", on_click=cb_tambah)

    bagian("Hapus Pasien dari Dataset")
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("**Hapus Berdasarkan ID Pasien**")
        daftar_id = list(data_kini["Pasien"].astype(str))
        st.selectbox("Pilih ID Pasien yang ingin dihapus:", daftar_id, key="pilih_hapus_pasien")
        st.button("Hapus Pasien Terpilih", on_click=cb_hapus_pilih, disabled=len(data_kini) <= 3)

    with col_d2:
        st.markdown("**Aksi Cepat Lainnya**")
        st.write("Hapus pasien yang paling terakhir ditambahkan:")
        st.button("Hapus Pasien Terakhir", on_click=cb_hapus_terakhir, disabled=len(data_kini) <= 3)
        st.write("Kembalikan seluruh dataset ke pengaturan awal:")
        st.button("Reset ke Dataset Awal (10 Pasien)", on_click=cb_reset)

    bagian("Impor & Ekspor CSV")
    st.download_button("Unduh Dataset sebagai CSV", data=data_kini.to_csv(index=False).encode("utf-8"),
                       file_name="dataset_diabetes_knn.csv", mime="text/csv")
    st.file_uploader("Unggah Dataset CSV", type="csv", key="csv_up", on_change=cb_muat)
