import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import geopandas as gpd
import contextily as ctx
import streamlit as st

df = pd.read_csv('df_final.csv')

df = df.rename(columns={
    'Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)': 'Persentase Penduduk Miskin',
    'Rata-rata Lama Sekolah Penduduk 15+ (Tahun)': 'Rata-rata Lama Sekolah',
    'Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)': 'Pengeluaran perKapita (Ribu rupiah)',
    'Persentase rumah tangga yang memiliki akses terhadap sanitasi layak': 'Persentase Kelayakan Sanitasi',
    'Persentase rumah tangga yang memiliki akses terhadap air minum layak': 'Persentase Kelayakan Air Minum',
    'PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)': 'PDRB (Rupiah)'
    })

kabKot_lat_long = pd.read_csv('kota_kab_lat_long_final.csv')

# Gabungkan berdasarkan kolom 'Kabupaten/Kota'
df = df.merge(kabKot_lat_long, on='Kabupaten/Kota', how='left')
kolom = df.columns.tolist()
idx = kolom.index('Kabupaten/Kota')
urutan_baru = kolom[:idx+1] + ['lat', 'long'] + kolom[idx+1:-2]
df = df[urutan_baru]

df['Persentase Penduduk Tidak Miskin'] = 100 - df['Persentase Penduduk Miskin']

cols = df.columns.tolist()
pos_miskin = cols.index("Persentase Penduduk Miskin")
cols.remove("Persentase Penduduk Tidak Miskin")
cols.insert(pos_miskin + 1, "Persentase Penduduk Tidak Miskin")
df = df[cols]

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

reading_scaled = scaler.fit_transform(df[["Reading Frequency per week",
                                          "Number of Readings per Quarter",
                                          "Daily Reading Duration (in minutes)",
                                          "Tingkat Kegemaran Membaca (Reading Interest)"
                                          ]])

df["Skor Membaca"] = reading_scaled.mean(axis=1)

df = df.drop(columns=[
    "Reading Frequency per week",
    "Number of Readings per Quarter",
    "Internet Access Frequency per Week",
    "Daily Internet Duration (in minutes)",
    "Daily Reading Duration (in minutes)",
    "Tingkat Kegemaran Membaca (Reading Interest)",
    "Category"
])

import geopandas as gpd
gdf = gpd.read_file('indonesia-prov.geojson')
gdf = gdf.rename(columns={'Propinsi':'Provinsi'})
provinsi_mapping = {
    'NUSATENGGARA BARAT': 'NUSA TENGGARA BARAT',
    'NUSA TENGGARA BARAT': 'NUSA TENGGARA BARAT',
    'NUSATENGGARA TIMUR': 'NUSA TENGGARA TIMUR',
    'NUSA TENGGARA TIMUR': 'NUSA TENGGARA TIMUR',
    'DAERAH ISTIMEWA YOGYAKARTA': 'DI YOGYAKARTA',
    'DI. ACEH': 'ACEH',
    'KEPULAUAN RIAU': 'KEP. RIAU',
    'BANGKA BELITUNG': 'KEP. BANGKA BELITUNG',
}

gdf['Provinsi'] = gdf['Provinsi'].replace(provinsi_mapping)

gdf = gdf.rename(columns={'ID':'Id_provinsi',
                          'kode': 'Kode_provinsi',
                          'SUMBER': 'Sumber_provinsi',
                          'geometry': 'Geometry_provinsi'
                          })

merged_df = pd.merge(df,
                     gdf[['Provinsi', 'Id_provinsi', 'Kode_provinsi', 'Sumber_provinsi', 'Geometry_provinsi']],
                     on='Provinsi', how='left')

def insert_after(df, insert_cols, after_col):
    cols = df.columns.tolist()
    insert_cols = [col for col in insert_cols if col in cols and col != 'Provinsi']
    insert_pos = cols.index(after_col) + 1
    for col in insert_cols:
        cols.remove(col)
    for i, col in enumerate(insert_cols):
        cols.insert(insert_pos + i, col)
    return df[cols]

df = insert_after(merged_df,
                  ['Id_provinsi', 'Kode_provinsi', 'Sumber_provinsi', 'Geometry_provinsi'],
                  'long')

normalityTest_df = df.drop(columns=['Provinsi', 'Kabupaten/Kota', 'Klasifikasi Kemiskinan', 'YEAR', 'lat', 'long',
                                    'Id_provinsi', 'Kode_provinsi', 'Sumber_provinsi', 'Geometry_provinsi'])

normalisasi_df = df.drop(columns=['Provinsi', 'Kabupaten/Kota', 'YEAR', 'lat', 'long', 'Id_provinsi', 'Kode_provinsi', 
                                  'Sumber_provinsi', 'Geometry_provinsi'])

column_names = normalisasi_df.columns
normalisasi_df = scaler.fit_transform(normalisasi_df)
normalisasi_df = pd.DataFrame(normalisasi_df, columns=column_names)

from sklearn.cluster import OPTICS
sum_cluster_df = normalisasi_df.copy()
optics_model =OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05)
labels = optics_model.fit_predict(sum_cluster_df)
df['Cluster_OPTICS'] = labels

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(sum_cluster_df)

df['Klasifikasi'] = df['Cluster_OPTICS'].apply(lambda x: 'Baik' if x == 0 else None)

df.at[155, 'Klasifikasi'] = "Lebih baik"
df.at[156, 'Klasifikasi'] = "Lebih baik"
df.at[157, 'Klasifikasi'] = "Lebih baik"
df.at[158, 'Klasifikasi'] = "Lebih baik"
df.at[159, 'Klasifikasi'] = "Lebih baik"
df.at[160, 'Klasifikasi'] = "Lebih baik"
df.at[174, 'Klasifikasi'] = "Lebih baik"
df.at[175, 'Klasifikasi'] = "Lebih baik"
df.at[180, 'Klasifikasi'] = "Lebih baik"
df.at[263, 'Klasifikasi'] = "Lebih baik"

df.at[474, 'Klasifikasi'] = "Kurang"
df.at[480, 'Klasifikasi'] = "Kurang"
df.at[481, 'Klasifikasi'] = "Kurang"
df.at[482, 'Klasifikasi'] = "Kurang"
df.at[483, 'Klasifikasi'] = "Kurang"
df.at[486, 'Klasifikasi'] = "Kurang"
df.at[491, 'Klasifikasi'] = "Kurang"
df.at[492, 'Klasifikasi'] = "Kurang"
df.at[496, 'Klasifikasi'] = "Kurang"
df.at[497, 'Klasifikasi'] = "Kurang"
df.at[498, 'Klasifikasi'] = "Kurang"
df.at[499, 'Klasifikasi'] = "Kurang"
df.at[503, 'Klasifikasi'] = "Kurang"
df.at[504, 'Klasifikasi'] = "Kurang"
df.at[505, 'Klasifikasi'] = "Kurang"
df.at[506, 'Klasifikasi'] = "Kurang"
df.at[507, 'Klasifikasi'] = "Kurang"
df.at[508, 'Klasifikasi'] = "Kurang"
df.at[509, 'Klasifikasi'] = "Kurang"
df.at[510, 'Klasifikasi'] = "Kurang"
df.at[511, 'Klasifikasi'] = "Kurang"
df.at[512, 'Klasifikasi'] = "Kurang"

cluster_summary = df.drop(columns=['Provinsi', 'Kabupaten/Kota','Id_provinsi', 'Kode_provinsi', 'Sumber_provinsi', 
                                   'Geometry_provinsi']).groupby('Klasifikasi').mean()

df_pca = pd.DataFrame(reduced, columns=["PCA 1", "PCA 2"])
df_pca["Klasifikasi"] = df["Klasifikasi"].values

df_pca = pd.DataFrame(reduced, columns=["PCA 1", "PCA 2"])
df_pca["Klasifikasi"] = df["Klasifikasi"].values
df_pca["Kabupaten/Kota"] = df["Kabupaten/Kota"].values

# Deploy
import visualisasi

st.set_page_config(page_title="Dashboard Visualisasi", layout="wide")

st.title("Dashboard Data Kabupaten/Kota Indonesia")
st.markdown("Silakan pilih visualisasi dari sidebar di kiri.")

st.sidebar.title("Menu Visualisasi")
visual_option = st.sidebar.radio("Pilih Jenis Visualisasi", [
    "Persentase Penduduk Miskin",
    "Rata-rata Lama Sekolah",
    "Tingkat Pengangguran Terbuka",
    "Umur Harapan Hidup",
    "Rata-rata Salary per Provinsi",
    "Korelasi IPM dan Umur Harapan Hidup",
    "Spearman Correlation Heatmap",
    "Visualisasi PCA Interaktif"  # ini Plotly
])

# Pemanggilan visualisasi
if visual_option == "Persentase Penduduk Miskin":
    fig = visualisasi.vis_penduduk_miskin(df)
    st.pyplot(fig)

elif visual_option == "Rata-rata Lama Sekolah":
    fig = visualisasi.rata_lama_sekolah(df)
    st.pyplot(fig)

elif visual_option == "Tingkat Pengangguran Terbuka":
    fig = visualisasi.vis_tngkt_nganggur(df)
    st.pyplot(fig)

elif visual_option == "Umur Harapan Hidup":
    fig = visualisasi.visualisasi_umur_harapan_hidup(df)
    st.pyplot(fig)

elif visual_option == "Rata-rata Salary per Provinsi":
    fig = visualisasi.visualisasi_rata_rata_salary(df)
    st.pyplot(fig)

elif visual_option == "Korelasi IPM dan Umur Harapan Hidup":
    fig = visualisasi.visualisasi_korelasi_ipm_umur(df)
    st.pyplot(fig)

elif visual_option == "Spearman Correlation Heatmap":
    fig = visualisasi.visualisasi_spearman_heatmap(df)
    st.pyplot(fig)

elif visual_option == "Visualisasi PCA Interaktif":
    fig = visualisasi.visualisasi_cluster_hover(df)
    st.plotly_chart(fig, use_container_width=True)

