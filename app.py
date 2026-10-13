import html
import io
import math
import pandas as pd
import streamlit as st

# ============================================================
# DATA AWAL (10 pasien dari soal)
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


# ============================================================
# FUNGSI KNN
# ============================================================
def hitung_knn(data, pasien_baru, k):
    fmin = data[FITUR].min()
    fmax = data[FITUR].max()
    rentang = (fmax - fmin).replace(0, 1)  # hindari pembagian nol

    norm_latih = (data[FITUR] - fmin) / rentang
    norm_baru = {f: (pasien_baru[f] - fmin[f]) / rentang[f] for f in FITUR}

    jarak = []
    for i in range(len(data)):
        s = sum((norm_latih.iloc[i][f] - norm_baru[f]) ** 2 for f in FITUR)
        jarak.append(math.sqrt(s))

    tabel = data[["Pasien", "Kelas"]].copy()
    tabel["Jarak"] = jarak
    tabel = tabel.sort_values("Jarak").reset_index(drop=True)
    tabel.insert(0, "Peringkat", range(1, len(tabel) + 1))
    tabel["Terpilih"] = tabel["Peringkat"] <= k

    tetangga = tabel[tabel["Terpilih"]]
    vote = tetangga["Kelas"].value_counts().to_dict()
    n0, n1 = vote.get(0, 0), vote.get(1, 0)
    if n1 > n0:
        pred = 1
    elif n0 > n1:
        pred = 0
    else:
        pred = int(tetangga.iloc[0]["Kelas"])
    return tabel, fmin, fmax, norm_baru, (n0, n1), pred


def id_berikutnya(data):
    angka = [int(p[1:]) for p in data["Pasien"] if str(p)[1:].isdigit()]
    return f"P{(max(angka) if angka else 0) + 1:02d}"


def tambah_pasien(usia, bmi, glukosa, tensi, riwayat, kelas):
    baris = {"Pasien": id_berikutnya(st.session_state.data), "Usia": usia, "BMI": bmi,
             "Glukosa": glukosa, "Tekanan Darah": tensi,
             "Riwayat Keluarga": riwayat, "Kelas": kelas}
    st.session_state.data = pd.concat(
        [st.session_state.data, pd.DataFrame([baris])], ignore_index=True)
    return baris["Pasien"]


# ============================================================
# TAMPILAN: GAYA
# ============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700&family=Figtree:wght@400;500;600&display=swap');

:root {
  --ink: #0E2F3A;
  --ink-soft: #4A6670;
  --teal: #0B5563;
  --teal-deep: #08404B;
  --mint: #E6F1F0;
  --paper: #F4F8F8;
  --line: #D3E2E1;
  --low: #2E9E6B;
  --low-bg: #E4F4EC;
  --high: #D1495B;
  --high-bg: #FBE9EC;
}

.stApp { background: var(--paper); color: var(--ink); }
.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea,
.stApp button, .stApp [data-baseweb="tab"] { font-family: 'Figtree', system-ui, sans-serif; }
.stApp h1, .stApp h2, .stApp h3 { font-family: 'Bricolage Grotesque', 'Figtree', sans-serif; color: var(--ink); letter-spacing: -0.01em; }

header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 760px; padding-top: 1.6rem; padding-bottom: 4rem; }
#MainMenu, footer { visibility: hidden; }

/* Hero */
.hero { background: linear-gradient(135deg, var(--teal-deep) 0%, var(--teal) 62%, #137A83 100%);
  color: #fff; border-radius: 20px; padding: 28px 28px 26px; margin-bottom: 18px; }
.hero h1 { font-family: 'Bricolage Grotesque', sans-serif; font-size: 2rem; line-height: 1.12; margin: 0 0 8px;
  color: #fff !important; font-weight: 700; padding: 0; }
.hero p { margin: 0; color: #CDE5E6; font-size: 1rem; max-width: 52ch; line-height: 1.5; }
.hero .stat { display: inline-block; margin-top: 16px; background: rgba(255,255,255,.12);
  border-radius: 999px; padding: 6px 14px; font-size: .86rem; color: #fff; }

/* Tab */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { height: 44px; padding: 0 16px; color: var(--ink-soft); font-weight: 600; }
.stTabs [aria-selected="true"] { color: var(--teal); }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--teal); }

/* Kotak input */
div[data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border-color: var(--line) !important; border-radius: 16px; }
.stNumberInput input, .stTextInput input { background: var(--paper); border-radius: 10px; color: var(--ink); }
.stNumberInput label, .stTextInput label, .stSlider label, .stRadio > label { color: var(--ink-soft); font-weight: 500; }

/* Tombol */
.stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {
  width: 100%; border-radius: 12px; min-height: 48px; font-weight: 600; border: 1px solid var(--line);
  background: #fff; color: var(--teal); }
.stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
  background: var(--teal); color: #fff; border-color: var(--teal); }
.stButton > button[kind="primary"]:hover { background: var(--teal-deep); border-color: var(--teal-deep); color: #fff; }
.stButton > button:focus-visible { outline: 3px solid #7CC4C4; outline-offset: 2px; }

/* Langkah proses */
.step { display: flex; gap: 14px; align-items: flex-start; margin: 30px 0 12px; }
.step-n { flex: 0 0 34px; height: 34px; border-radius: 50%; background: var(--teal); color: #fff;
  display: flex; align-items: center; justify-content: center; font-weight: 700;
  font-family: 'Bricolage Grotesque', sans-serif; }
.step-t { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.15rem; color: var(--ink); line-height: 1.25; }
.step-d { color: var(--ink-soft); font-size: .93rem; margin-top: 2px; }

/* Tabel */
.tbl-wrap { overflow-x: auto; background: #fff; border: 1px solid var(--line); border-radius: 14px; }
table.tbl { width: 100%; border-collapse: collapse; font-size: .92rem; white-space: nowrap; }
table.tbl th { text-align: left; background: var(--mint); color: var(--teal-deep); font-weight: 600;
  padding: 10px 14px; border-bottom: 1px solid var(--line); }
table.tbl td { padding: 10px 14px; border-bottom: 1px solid #EAF1F1; color: var(--ink); }
table.tbl tr:last-child td { border-bottom: none; }
table.tbl tr.pick td { background: #EEF8F4; }
table.tbl tr.pick td:first-child { box-shadow: inset 4px 0 0 var(--teal); }
.pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .82rem; font-weight: 600; }
.pill.low { background: var(--low-bg); color: #1D6E49; }
.pill.high { background: var(--high-bg); color: #A12C3C; }

/* Jajaran tetangga: elemen utama */
.strip { display: flex; gap: 10px; overflow-x: auto; padding: 4px 2px 10px; }
.nb { flex: 0 0 auto; text-align: center; min-width: 64px; }
.nb .dot { width: 52px; height: 52px; border-radius: 50%; margin: 0 auto 6px; display: flex; align-items: center;
  justify-content: center; color: #fff; font-weight: 700; font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.05rem; }
.nb .dot.low { background: var(--low); }
.nb .dot.high { background: var(--high); }
.nb .id { font-weight: 600; font-size: .9rem; color: var(--ink); }
.nb .jr { font-size: .78rem; color: var(--ink-soft); }
.strip-note { font-size: .85rem; color: var(--ink-soft); margin-top: 2px; }

/* Voting */
.votebar { display: flex; height: 16px; border-radius: 999px; overflow: hidden; background: var(--line); margin: 8px 0 8px; }
.votebar .low { background: var(--low); }
.votebar .high { background: var(--high); }
.votelabel { display: flex; justify-content: space-between; font-size: .92rem; font-weight: 600; }
.votelabel .l { color: #1D6E49; }
.votelabel .h { color: #A12C3C; }

/* Kesimpulan */
.verdict { border-radius: 18px; padding: 24px 24px 22px; margin-top: 8px; border-left: 8px solid; }
.verdict.low { background: var(--low-bg); border-color: var(--low); }
.verdict.high { background: var(--high-bg); border-color: var(--high); }
.verdict .v-who { font-size: .95rem; color: var(--ink-soft); }
.verdict .v-title { font-family: 'Bricolage Grotesque', sans-serif; font-size: 2.1rem; font-weight: 700; line-height: 1.1; margin: 4px 0 8px; }
.verdict.low .v-title { color: #1D6E49; }
.verdict.high .v-title { color: #A12C3C; }
.verdict .v-sub { color: var(--ink); font-size: 1rem; }
.disclaimer { font-size: .84rem; color: var(--ink-soft); margin-top: 10px; }

@media (max-width: 640px) {
  .hero { padding: 22px 20px; }
  .hero h1 { font-size: 1.6rem; }
  .verdict .v-title { font-size: 1.7rem; }
}
</style>
"""


def e(x):
    return html.escape(str(x))


def fmt(v):
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def tabel_html(headers, rows, pilih=None):
    pilih = pilih or [False] * len(rows)
    th = "".join(f"<th>{e(h)}</th>" for h in headers)
    trs = ""
    for row, p in zip(rows, pilih):
        tds = "".join(f"<td>{c}</td>" for c in row)  # sel sudah aman / sudah di-escape
        trs += f'<tr class="{"pick" if p else ""}">{tds}</tr>'
    return f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def pill(kelas):
    return f'<span class="pill {"high" if kelas == 1 else "low"}">{LABEL[kelas]}</span>'


def langkah(n, judul, deskripsi):
    st.markdown(
        f'<div class="step"><div class="step-n">{n}</div><div>'
        f'<div class="step-t">{e(judul)}</div><div class="step-d">{e(deskripsi)}</div></div></div>',
        unsafe_allow_html=True)


def tabel_dataset(data):
    rows = []
    for _, r in data.iterrows():
        rows.append([e(r["Pasien"]), fmt(r["Usia"]), fmt(r["BMI"]), fmt(r["Glukosa"]),
                     fmt(r["Tekanan Darah"]), "Ada" if r["Riwayat Keluarga"] == 1 else "Tidak ada",
                     pill(int(r["Kelas"]))])
    return tabel_html(["Pasien", "Usia", "BMI", "Glukosa", "Tekanan", "Riwayat", "Kelas"], rows)


def tampilkan_hasil(h):
    k, baru, tabel = h["k"], h["baru"], h["tabel"]
    fmin, fmax, norm_baru = h["fmin"], h["fmax"], h["norm_baru"]
    n0, n1 = h["votes"]
    pred = h["pred"]
    total = n0 + n1

    # 3. Normalisasi
    langkah(3, "Samakan skala data", "Glukosa berskala puluhan sampai ratusan, sedangkan BMI puluhan. "
            "Normalisasi Min-Max membuat semua fitur bernilai 0 sampai 1.")
    st.latex(r"x' = \dfrac{x - x_{min}}{x_{max} - x_{min}}")
    rows = [[e(f), fmt(fmin[f]), fmt(fmax[f]), fmt(baru[f]), f"{norm_baru[f]:.4f}"] for f in FITUR]
    st.markdown(tabel_html(["Fitur", "Min", "Maks", "Nilai pasien", "Hasil normalisasi"], rows),
                unsafe_allow_html=True)

    # 4. Jarak
    langkah(4, "Hitung jarak ke tiap pasien", f"Makin kecil jarak, makin mirip. Baris bertanda hijau adalah {k} pasien terdekat.")
    st.latex(r"d = \sqrt{\sum (x_i - y_i)^2}")
    rows, pilih = [], []
    for _, r in tabel.iterrows():
        rows.append([str(int(r["Peringkat"])), e(r["Pasien"]), f'{r["Jarak"]:.4f}', pill(int(r["Kelas"]))])
        pilih.append(bool(r["Terpilih"]))
    st.markdown(tabel_html(["Peringkat", "Pasien", "Jarak", "Kelas"], rows, pilih), unsafe_allow_html=True)

    # 5. Tetangga + voting
    langkah(5, f"Suara dari {k} tetangga terdekat", "Kelas yang paling banyak muncul menjadi hasil prediksi.")
    chips = ""
    for _, r in tabel[tabel["Terpilih"]].iterrows():
        c = "high" if int(r["Kelas"]) == 1 else "low"
        chips += (f'<div class="nb"><div class="dot {c}">{int(r["Peringkat"])}</div>'
                  f'<div class="id">{e(r["Pasien"])}</div><div class="jr">{r["Jarak"]:.3f}</div></div>')
    st.markdown(f'<div class="strip">{chips}</div>'
                '<div class="strip-note">Angka di lingkaran adalah peringkat kedekatan. Merah berarti risiko tinggi, hijau berarti risiko rendah.</div>',
                unsafe_allow_html=True)
    p0 = n0 / total * 100 if total else 0
    p1 = 100 - p0 if total else 0
    st.markdown(
        f'<div class="votebar"><div class="low" style="width:{p0:.1f}%"></div><div class="high" style="width:{p1:.1f}%"></div></div>'
        f'<div class="votelabel"><span class="l">Rendah: {n0} suara</span><span class="h">Tinggi: {n1} suara</span></div>',
        unsafe_allow_html=True)
    if n0 == n1:
        st.info("Suara seri, sehingga kelas diambil dari pasien yang paling dekat.")

    # Kesimpulan
    langkah("✓", "Kesimpulan", "Hasil akhir berdasarkan proses di atas.")
    c = "high" if pred == 1 else "low"
    suara = n1 if pred == 1 else n0
    st.markdown(
        f'<div class="verdict {c}"><div class="v-who">Hasil untuk {e(h["nama"])}</div>'
        f'<div class="v-title">Risiko {LABEL[pred]}</div>'
        f'<div class="v-sub">{suara} dari {total} pasien terdekat termasuk kelas Risiko {LABEL[pred]}.</div></div>'
        '<div class="disclaimer">Hasil ini alat bantu keputusan, bukan diagnosis medis.</div>',
        unsafe_allow_html=True)

    # Simpan ke dataset
    st.markdown("&nbsp;", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Simpan pasien ini ke dataset latih**")
        st.caption("Kelas bawaan mengikuti hasil prediksi. Ubah jika kelas sebenarnya sudah diketahui.")
        st.radio("Kelas yang disimpan", [0, 1], index=pred, horizontal=True, key="simpan_kelas",
                 format_func=lambda x: f"Risiko {LABEL[x]}")
        st.button("Simpan ke dataset", key="btn_simpan", on_click=cb_simpan)
        if st.session_state.get("flash_simpan"):
            st.success(st.session_state.pop("flash_simpan"))


# ============================================================
# CALLBACK
# ============================================================
def cb_simpan():
    s = st.session_state
    b = s.hasil["baru"]
    pid = tambah_pasien(b["Usia"], b["BMI"], b["Glukosa"], b["Tekanan Darah"],
                        b["Riwayat Keluarga"], s.simpan_kelas)
    s.flash_simpan = f"Pasien disimpan sebagai {pid}. Dataset sekarang berisi {len(s.data)} pasien."


def cb_tambah():
    s = st.session_state
    pid = tambah_pasien(s.f_usia, s.f_bmi, s.f_glu, s.f_tensi,
                        1 if s.f_riw == "Ada" else 0, s.f_kelas)
    s.flash_data = f"Pasien {pid} ditambahkan ke dataset."


def cb_hapus():
    s = st.session_state
    if len(s.data) > 3:
        s.data = s.data.iloc[:-1].reset_index(drop=True)
        s.flash_data = "Pasien terakhir dihapus."


def cb_reset():
    st.session_state.data = DATA_AWAL.copy()
    st.session_state.hasil = None
    st.session_state.flash_data = "Dataset dikembalikan ke 10 data awal."


def cb_muat():
    s = st.session_state
    up = s.get("csv_up")
    if up is None:
        return
    try:
        df = pd.read_csv(io.BytesIO(up.getvalue()))
        if not set(KOLOM).issubset(df.columns):
            s.flash_error = "Kolom CSV harus: " + ", ".join(KOLOM)
        else:
            s.data = df[KOLOM].reset_index(drop=True)
            s.hasil = None
            s.flash_data = f"CSV dimuat. Dataset berisi {len(s.data)} pasien."
    except Exception as ex:
        s.flash_error = f"CSV tidak bisa dibaca: {ex}"


# ============================================================
# HALAMAN
# ============================================================
st.set_page_config(page_title="Cek Risiko Diabetes", page_icon="🩺", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

if "data" not in st.session_state:
    st.session_state.data = DATA_AWAL.copy()
if "hasil" not in st.session_state:
    st.session_state.hasil = None

data = st.session_state.data
n_tinggi = int((data["Kelas"] == 1).sum())

st.markdown(
    '<div class="hero"><h1>Cek Risiko Diabetes</h1>'
    '<p>Masukkan data pasien. Sistem membandingkannya dengan pasien yang sudah tercatat memakai algoritma K-Nearest Neighbors.</p>'
    f'<span class="stat">{len(data)} pasien di dataset, {n_tinggi} berisiko tinggi</span></div>',
    unsafe_allow_html=True)

tab_prediksi, tab_data = st.tabs(["Prediksi", "Dataset latih"])

# ---------------- TAB PREDIKSI ----------------
with tab_prediksi:
    langkah(1, "Isi data pasien", "Semua isian dipakai dalam perhitungan.")
    with st.container(border=True):
        nama = st.text_input("Nama atau kode pasien (boleh dikosongkan)", placeholder="Contoh: Budi")
        c1, c2 = st.columns(2)
        usia = c1.number_input("Usia (tahun)", min_value=1, max_value=120, value=42, step=1)
        bmi = c2.number_input("BMI", min_value=10.0, max_value=60.0, value=28.5, step=0.1, format="%.1f")
        glukosa = c1.number_input("Glukosa (mg/dL)", min_value=40, max_value=400, value=130, step=1)
        tensi = c2.number_input("Tekanan darah sistolik (mmHg)", min_value=60, max_value=250, value=128, step=1)
        riwayat_txt = st.radio("Riwayat diabetes dalam keluarga", ["Ada", "Tidak ada"], horizontal=True)
        riwayat = 1 if riwayat_txt == "Ada" else 0

    langkah(2, "Tentukan nilai K", "K adalah jumlah pasien terdekat yang ikut memberi suara. Pilih angka ganjil agar suara tidak seri.")
    kmax = len(data)
    k = st.slider("Nilai K", min_value=1, max_value=min(15, kmax), value=min(3, kmax), step=1)

    if st.button("Hitung risiko", type="primary"):
        baru = {"Usia": usia, "BMI": bmi, "Glukosa": glukosa,
                "Tekanan Darah": tensi, "Riwayat Keluarga": riwayat}
        tabel, fmin, fmax, norm_baru, votes, pred = hitung_knn(st.session_state.data, baru, k)
        st.session_state.hasil = {
            "baru": baru, "k": k, "tabel": tabel, "fmin": fmin, "fmax": fmax,
            "norm_baru": norm_baru, "votes": votes, "pred": pred,
            "nama": nama.strip() or "pasien ini",
        }

    if st.session_state.hasil is not None:
        tampilkan_hasil(st.session_state.hasil)

# ---------------- TAB DATASET ----------------
with tab_data:
    if st.session_state.get("flash_data"):
        st.success(st.session_state.pop("flash_data"))
    if st.session_state.get("flash_error"):
        st.error(st.session_state.pop("flash_error"))

    langkah("+", "Tambah pasien ke dataset", "Pasien baru langsung ikut dihitung pada prediksi berikutnya.")
    with st.container(border=True):
        with st.form("form_tambah", border=False):
            a1, a2 = st.columns(2)
            a1.number_input("Usia (tahun)", 1, 120, 40, 1, key="f_usia")
            a2.number_input("BMI", 10.0, 60.0, 25.0, 0.1, format="%.1f", key="f_bmi")
            a1.number_input("Glukosa (mg/dL)", 40, 400, 100, 1, key="f_glu")
            a2.number_input("Tekanan darah sistolik (mmHg)", 60, 250, 120, 1, key="f_tensi")
            st.radio("Riwayat diabetes dalam keluarga", ["Ada", "Tidak ada"], horizontal=True, key="f_riw")
            st.radio("Kelas", [0, 1], horizontal=True, key="f_kelas",
                     format_func=lambda x: f"Risiko {LABEL[x]}")
            st.form_submit_button("Tambah ke dataset", type="primary", on_click=cb_tambah)

    langkah("≡", f"Isi dataset ({len(data)} pasien)", "Data ini menjadi pembanding untuk setiap prediksi.")
    st.markdown(tabel_dataset(data), unsafe_allow_html=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    b1.button("Hapus pasien terakhir", disabled=len(data) <= 3, on_click=cb_hapus)
    b2.button("Kembali ke 10 data awal", on_click=cb_reset)

    langkah("↧", "Simpan dan muat dataset", "Data tambahan hilang saat halaman dimuat ulang. Unduh CSV untuk menyimpannya.")
    st.download_button("Unduh dataset (CSV)", data.to_csv(index=False).encode("utf-8"),
                       file_name="dataset_pasien.csv", mime="text/csv")
    st.file_uploader("Muat dataset dari file CSV", type="csv", key="csv_up")
    st.button("Pakai CSV ini", on_click=cb_muat)
