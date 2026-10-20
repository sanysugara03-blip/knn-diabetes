# -*- coding: utf-8 -*-
"""Cek Risiko Diabetes: sistem pendukung keputusan dengan K-Nearest Neighbors (KNN)."""
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


def tampilkan_hasil(h):
    k, baru, tabel = h["k"], h["baru"], h["tabel"]
    fmin, fmax, norm_baru = h["fmin"], h["fmax"], h["norm_baru"]
    n0, n1 = h["votes"]
    pred = h["pred"]

    st.divider()
    st.header("Proses Perhitungan")

    st.subheader("3. Normalisasi Min-Max")
    st.markdown("Rumus: **x' = (x − min) / (max − min)**")
    df_norm = pd.DataFrame({
        "Fitur": FITUR,
        "Min": [fmin[f] for f in FITUR],
        "Max": [fmax[f] for f in FITUR],
        "Nilai Pasien Baru": [baru[f] for f in FITUR],
        "Hasil Normalisasi": [round(norm_baru[f], 4) for f in FITUR],
    })
    st.dataframe(df_norm, hide_index=True, use_container_width=True)

    st.subheader("4. Jarak Euclidean ke Setiap Pasien")
    st.markdown("Rumus: **d = √( Σ (xᵢ − yᵢ)² )** pada data yang sudah dinormalisasi. "
                f"Baris berwarna hijau adalah **{k} tetangga terdekat**.")
    t = tabel.copy()
    t["Kelas"] = t["Kelas"].map(LABEL)
    t["Jarak"] = t["Jarak"].round(4)
    t["Terpilih"] = t["Terpilih"].map({True: "✔ Ya", False: ""})

    def warna(row):
        return ["background-color: #d4edda; color: #155724" if row["Terpilih"] else ""] * len(row)

    st.dataframe(t.style.apply(warna, axis=1), hide_index=True, use_container_width=True)

    st.subheader(f"5. Voting dari {k} Tetangga Terdekat")
    v1, v2 = st.columns(2)
    v1.metric("Risiko Rendah (0)", f"{n0} suara")
    v2.metric("Risiko Tinggi (1)", f"{n1} suara")
    if n0 == n1:
        st.info("Suara seri, sehingga kelas diambil dari tetangga yang paling dekat.")

    st.divider()
    st.header("Kesimpulan")
    total = n0 + n1
    nama = h["nama"]
    if pred == 1:
        st.error(f"### {nama} berisiko **TINGGI** diabetes\n"
                 f"{n1} dari {total} tetangga terdekat termasuk kelas Risiko Tinggi.")
    else:
        st.success(f"### {nama} berisiko **RENDAH** diabetes\n"
                   f"{n0} dari {total} tetangga terdekat termasuk kelas Risiko Rendah.")
    st.caption("Hasil ini hanya alat bantu keputusan, bukan diagnosis medis.")

    # Simpan pasien ini ke dataset
    st.subheader("Simpan pasien ini ke dataset?")
    kelas_simpan = st.radio(
        "Kelas yang disimpan", [0, 1], index=pred, horizontal=True,
        format_func=lambda x: f"{x} = Risiko {LABEL[x]}",
        help="Bawaan mengikuti hasil prediksi. Ubah jika dokter sudah mengetahui kelas sebenarnya.")
    if st.button("Simpan ke dataset latih"):
        pid = tambah_pasien(baru["Usia"], baru["BMI"], baru["Glukosa"],
                            baru["Tekanan Darah"], baru["Riwayat Keluarga"], kelas_simpan)
        st.success(f"Pasien disimpan sebagai {pid}. Dataset sekarang berisi "
                   f"{len(st.session_state.data)} pasien.")


# ============================================================
# TAMPILAN
# ============================================================
st.set_page_config(page_title="Prediksi Risiko Diabetes (KNN)", page_icon="🩺", layout="centered")

if "data" not in st.session_state:
    st.session_state.data = DATA_AWAL.copy()
if "hasil" not in st.session_state:
    st.session_state.hasil = None

st.title("🩺 Prediksi Risiko Diabetes")
st.caption("Sistem pendukung keputusan klinik dengan algoritma K-Nearest Neighbors (KNN)")

# ---------------- Dataset ----------------
st.subheader("1. Dataset Latih")
data = st.session_state.data
st.write(f"Jumlah pasien di dataset: **{len(data)}**")
with st.expander("Lihat / kelola dataset"):
    tampil = data.copy()
    tampil["Kelas"] = tampil["Kelas"].map(LABEL)
    st.dataframe(tampil, hide_index=True, use_container_width=True)

    st.markdown("**Tambah pasien ke dataset**")
    with st.form("form_tambah", clear_on_submit=False):
        a1, a2 = st.columns(2)
        t_usia = a1.number_input("Usia", 1, 120, 40, 1)
        t_bmi = a2.number_input("BMI", 10.0, 60.0, 25.0, 0.1, format="%.1f")
        t_glu = a1.number_input("Glukosa", 40, 400, 100, 1)
        t_tensi = a2.number_input("Tekanan Darah", 60, 250, 120, 1)
        t_riw = st.radio("Riwayat keluarga", ["Ada", "Tidak ada"], horizontal=True)
        t_kelas = st.radio("Kelas", [0, 1], horizontal=True,
                           format_func=lambda x: f"{x} = Risiko {LABEL[x]}")
        kirim = st.form_submit_button("Tambah ke dataset")
    if kirim:
        pid = tambah_pasien(t_usia, t_bmi, t_glu, t_tensi, 1 if t_riw == "Ada" else 0, t_kelas)
        st.success(f"Pasien {pid} ditambahkan.")
        st.rerun()

    b1, b2 = st.columns(2)
    if b1.button("Hapus pasien terakhir", disabled=len(data) <= 3):
        st.session_state.data = data.iloc[:-1].reset_index(drop=True)
        st.rerun()
    if b2.button("Reset ke 10 data awal"):
        st.session_state.data = DATA_AWAL.copy()
        st.session_state.hasil = None
        st.rerun()

    st.markdown("**Simpan / muat dataset (CSV)**")
    st.download_button("Unduh dataset (CSV)", data.to_csv(index=False).encode("utf-8"),
                       file_name="dataset_pasien.csv", mime="text/csv")
    up = st.file_uploader("Muat dataset dari CSV", type="csv")
    if up is not None and st.button("Gunakan CSV ini"):
        try:
            df = pd.read_csv(io.BytesIO(up.getvalue()))
            if not set(KOLOM).issubset(df.columns):
                st.error("Kolom CSV harus: " + ", ".join(KOLOM))
            else:
                st.session_state.data = df[KOLOM].reset_index(drop=True)
                st.session_state.hasil = None
                st.rerun()
        except Exception as e:
            st.error(f"CSV tidak bisa dibaca: {e}")
    st.caption("Catatan: data tambahan tersimpan selama halaman terbuka. "
               "Unduh CSV agar tidak hilang saat halaman dimuat ulang.")

# ---------------- Input pasien baru ----------------
st.subheader("2. Input Data Pasien Baru")
nama = st.text_input("Nama / kode pasien (opsional)", value="", placeholder="contoh: Budi")
c1, c2 = st.columns(2)
usia = c1.number_input("Usia (tahun)", min_value=1, max_value=120, value=42, step=1)
bmi = c2.number_input("BMI", min_value=10.0, max_value=60.0, value=28.5, step=0.1, format="%.1f")
glukosa = c1.number_input("Glukosa (mg/dL)", min_value=40, max_value=400, value=130, step=1)
tensi = c2.number_input("Tekanan Darah (sistolik, mmHg)", min_value=60, max_value=250, value=128, step=1)
riwayat_txt = st.radio("Riwayat diabetes dalam keluarga", ["Ada", "Tidak ada"], horizontal=True)
riwayat = 1 if riwayat_txt == "Ada" else 0
k = st.slider("Nilai K (jumlah tetangga)", min_value=1, max_value=min(15, len(st.session_state.data)),
              value=min(3, len(st.session_state.data)), step=1,
              help="Disarankan ganjil agar voting tidak seri.")

if st.button("Hitung Prediksi", type="primary", use_container_width=True):
    baru = {"Usia": usia, "BMI": bmi, "Glukosa": glukosa,
            "Tekanan Darah": tensi, "Riwayat Keluarga": riwayat}
    tabel, fmin, fmax, norm_baru, votes, pred = hitung_knn(st.session_state.data, baru, k)
    st.session_state.hasil = {
        "baru": baru, "k": k, "tabel": tabel, "fmin": fmin, "fmax": fmax,
        "norm_baru": norm_baru, "votes": votes, "pred": pred,
        "nama": nama.strip() or "Pasien",
    }

if st.session_state.hasil is not None:
    tampilkan_hasil(st.session_state.hasil)
