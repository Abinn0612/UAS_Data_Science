import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import geopandas as gpd
import contextily as ctx

df = pd.read_csv(r'df_final.csv')

df = df.rename(columns={
    'Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)': 'Persentase Penduduk Miskin',
    'Rata-rata Lama Sekolah Penduduk 15+ (Tahun)': 'Rata-rata Lama Sekolah',
    'Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)': 'Pengeluaran perKapita (Ribu rupiah)',
    'Persentase rumah tangga yang memiliki akses terhadap sanitasi layak': 'Persentase Kelayakan Sanitasi',
    'Persentase rumah tangga yang memiliki akses terhadap air minum layak': 'Persentase Kelayakan Air Minum',
    'PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)': 'PDRB (Rupiah)'
    })

kabKot_lat_long = pd.read_csv(r'kota_kab_lat_long_final.csv')

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
gdf = gpd.read_file(r'indonesia-prov.geojson')
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

from scipy.stats import shapiro

normalityTest_df = df.drop(columns=['Provinsi', 'Kabupaten/Kota', 'Klasifikasi Kemiskinan', 'YEAR', 'lat', 'long','Id_provinsi', 'Kode_provinsi', 'Sumber_provinsi', 'Geometry_provinsi'])

normalityTest_df.info()

# Visualisasi
def vis_penduduk_miskin (df):
        # Jitter koordinat agar tidak tumpang tindih
    np.random.seed(42)
    long_peta = df['long'] + np.random.uniform(-0.05, 0.05, size=len(df))
    lat_peta = df['lat'] + np.random.uniform(-0.05, 0.05, size=len(df))

    # Konversi ke GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(long_peta, lat_peta),
        crs='EPSG:4326'
    )

    # Proyeksi ke Web Mercator
    gdf = gdf.to_crs(epsg=3857)

    persentasePendudukMiskin = df.groupby("Kabupaten/Kota")["Persentase Penduduk Miskin"].mean()
    top_5 = persentasePendudukMiskin.sort_values(ascending=False).head(5)
    bottom_5 = persentasePendudukMiskin.sort_values().head(5)

    # Buat layout dengan GridSpec
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 2])

    # --- Row 1: Peta sebaran (colspan=2) ---
    ax_map = plt.subplot(gs[0, :])
    divider = make_axes_locatable(ax_map)
    cax = divider.append_axes("right", size="3%", pad=0.1)

    gdf.plot(
        ax=ax_map,
        column='Persentase Penduduk Miskin',
        cmap='OrRd',
        legend=True,
        legend_kwds={'label': "Persentase Penduduk Miskin", 'orientation': "vertical"},
        cax=cax,
        markersize=30,
        edgecolor='black',
        linewidth=0.3,
        alpha=0.6
    )

    ctx.add_basemap(ax_map)
    ax_map.set_title("Sebaran Persentase Penduduk Miskin dengan Basemap", fontsize=14)
    ax_map.axis('off')

    # --- Row 2: Bar chart top & bottom (stacked vertically di satu kolom) ---
    ax_top = plt.subplot(gs[1, 0])
    top_5.plot(kind='bar', ax=ax_top, color='red')
    ax_top.set_title("5 Kota dengan Persentase Penduduk Miskin Tertinggi (2021)")
    ax_top.set_ylabel("Persentase Penduduk Miskin")
    ax_top.set_xlabel("Kabupaten/Kota")
    ax_top.tick_params(axis='x', rotation=45)

    ax_bottom = plt.subplot(gs[1, 1])
    bottom_5.plot(kind='bar', ax=ax_bottom, color='green')
    ax_bottom.set_title("5 Kota dengan Persentase Penduduk Miskin Terendah (2021)")
    ax_bottom.set_ylabel("Persentase Penduduk Miskin")
    ax_bottom.set_xlabel("Kabupaten/Kota")
    ax_bottom.tick_params(axis='x', rotation=45)

    # Sinkronisasi skala Y kedua chart
    max_val = max(top_5.max(), bottom_5.max())
    ax_top.set_ylim(0, max_val + 1)
    ax_bottom.set_ylim(0, max_val + 1)

    plt.tight_layout()
    return fig

def rata_lama_sekolah(df):
    averange_school = df.groupby("Kabupaten/Kota")["Rata-rata Lama Sekolah"].mean()
    top_5 = averange_school.sort_values(ascending=False).head(5)
    bottom_5 = averange_school.sort_values().head(5)

    # Buat layout dengan GridSpec
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 2])

    # Buat GeoDataFrame dari lat-long
    np.random.seed(42)
    long_peta = df['long'] + np.random.uniform(-0.05, 0.05, size=len(df))
    lat_peta = df['lat'] + np.random.uniform(-0.05, 0.05, size=len(df))

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(long_peta, lat_peta),
        crs='EPSG:4326'
    )

    gdf = gdf.to_crs(epsg=3857)

    # Plot peta
    ax_map = plt.subplot(gs[0, :])
    divider = make_axes_locatable(ax_map)
    cax = divider.append_axes("right", size="3%", pad=0.1)

    gdf.plot(
        ax=ax_map,
        column='Rata-rata Lama Sekolah',
        cmap='Greens',
        legend=True,
        legend_kwds={'label': "Rata-rata Lama Sekolah", 'orientation': "vertical"},
        cax=cax,
        markersize=30,
        edgecolor='black',
        linewidth=0.3,
        alpha=0.6
    )

    ctx.add_basemap(ax_map)
    ax_map.set_title("Sebaran Rata-rata Lama Sekolah dengan Basemap", fontsize=14)
    ax_map.axis('off')

    # Bar chart
    ax_top = plt.subplot(gs[1, 0])
    top_5.plot(kind='bar', ax=ax_top, color='green')
    ax_top.set_title("5 Kota dengan Rata-rata Lama Sekolah Tertinggi (2021)")
    ax_top.set_ylabel("Rata-rata Lama Sekolah")
    ax_top.set_xlabel("Kabupaten/Kota")
    ax_top.tick_params(axis='x', rotation=45)

    ax_bottom = plt.subplot(gs[1, 1])
    bottom_5.plot(kind='bar', ax=ax_bottom, color='red')
    ax_bottom.set_title("5 Kota dengan Rata-rata Lama Sekolah Terendah (2021)")
    ax_bottom.set_ylabel("Rata-rata Lama Sekolah")
    ax_bottom.set_xlabel("Kabupaten/Kota")
    ax_bottom.tick_params(axis='x', rotation=45)

    max_val = max(top_5.max(), bottom_5.max())
    ax_top.set_ylim(0, max_val + 1)
    ax_bottom.set_ylim(0, max_val + 1)

    plt.tight_layout()
    return fig

def vis_tngkt_nganggur(df):
    pengangguranTerbuka = df.groupby("Kabupaten/Kota")["Tingkat Pengangguran Terbuka"].mean()
    top_5 = pengangguranTerbuka.sort_values(ascending=False).head(5)
    bottom_5 = pengangguranTerbuka.sort_values().head(5)

    # Buat layout dengan GridSpec
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 2])

    # --- Row 1: Peta sebaran (colspan=2) ---
    ax_map = plt.subplot(gs[0, :])
    divider = make_axes_locatable(ax_map)
    cax = divider.append_axes("right", size="3%", pad=0.1)

    # Buat GeoDataFrame dari lat-long
    np.random.seed(42)
    long_peta = df['long'] + np.random.uniform(-0.05, 0.05, size=len(df))
    lat_peta = df['lat'] + np.random.uniform(-0.05, 0.05, size=len(df))

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(long_peta, lat_peta),
        crs='EPSG:4326'
    )

    gdf = gdf.to_crs(epsg=3857)

    gdf.plot(
        ax=ax_map,
        column='Tingkat Pengangguran Terbuka',
        cmap='OrRd',
        legend=True,
        legend_kwds={'label': "Tingkat Pengangguran Terbuka", 'orientation': "vertical"},
        cax=cax,
        markersize=30,
        edgecolor='black',
        linewidth=0.3,
        alpha=0.6
    )

    ctx.add_basemap(ax_map)
    ax_map.set_title("Sebaran Tingkat Pengangguran Terbuka dengan Basemap", fontsize=14)
    ax_map.axis('off')

    # --- Row 2: Bar chart top & bottom ---
    ax_top = plt.subplot(gs[1, 0])
    top_5.plot(kind='bar', ax=ax_top, color='red')
    ax_top.set_title("5 Kota dengan Tingkat Pengangguran Terbuka Tertinggi (2021)")
    ax_top.set_ylabel("Tingkat Pengangguran Terbuka")
    ax_top.set_xlabel("Kabupaten/Kota")
    ax_top.tick_params(axis='x', rotation=45)

    ax_bottom = plt.subplot(gs[1, 1])
    bottom_5.plot(kind='bar', ax=ax_bottom, color='green')
    ax_bottom.set_title("5 Kota dengan Tingkat Pengangguran Terbuka Terendah (2021)")
    ax_bottom.set_ylabel("Tingkat Pengangguran Terbuka")
    ax_bottom.set_xlabel("Kabupaten/Kota")
    ax_bottom.tick_params(axis='x', rotation=45)

    # Sinkronisasi skala Y
    max_val = max(top_5.max(), bottom_5.max())
    ax_top.set_ylim(0, max_val + 1)
    ax_bottom.set_ylim(0, max_val + 1)

    plt.tight_layout()
    return fig

def visualisasi_umur_harapan_hidup(df):
    umurHarapanHidup = df.groupby("Kabupaten/Kota")["Umur Harapan Hidup (Tahun)"].mean()
    top_5 = umurHarapanHidup.sort_values(ascending=False).head(5)
    bottom_5 = umurHarapanHidup.sort_values().head(5)

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 2])

    # Buat GeoDataFrame dari lat-long
    np.random.seed(42)
    long_peta = df['long'] + np.random.uniform(-0.05, 0.05, size=len(df))
    lat_peta = df['lat'] + np.random.uniform(-0.05, 0.05, size=len(df))

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(long_peta, lat_peta),
        crs='EPSG:4326'
    )

    gdf = gdf.to_crs(epsg=3857)

    # Plot peta
    ax_map = plt.subplot(gs[0, :])
    divider = make_axes_locatable(ax_map)
    cax = divider.append_axes("right", size="3%", pad=0.1)

    gdf.plot(
        ax=ax_map,
        column='Umur Harapan Hidup (Tahun)',
        cmap='Greens',
        legend=True,
        legend_kwds={'label': "Umur Harapan Hidup (Tahun)", 'orientation': "vertical"},
        cax=cax,
        markersize=30,
        edgecolor='black',
        linewidth=0.3,
        alpha=0.6
    )

    ctx.add_basemap(ax_map)
    ax_map.set_title("Sebaran Umur Harapan Hidup (Tahun) dengan Basemap", fontsize=14)
    ax_map.axis('off')

    # Bar chart
    ax_top = plt.subplot(gs[1, 0])
    top_5.plot(kind='bar', ax=ax_top, color='green')
    ax_top.set_title("5 Kota dengan Umur Harapan Hidup Tertinggi (2021)")
    ax_top.set_ylabel("Umur Harapan Hidup (Tahun)")
    ax_top.set_xlabel("Kabupaten/Kota")
    ax_top.tick_params(axis='x', rotation=45)

    ax_bottom = plt.subplot(gs[1, 1])
    bottom_5.plot(kind='bar', ax=ax_bottom, color='red')
    ax_bottom.set_title("5 Kota dengan Umur Harapan Hidup Terendah (2021)")
    ax_bottom.set_ylabel("Umur Harapan Hidup (Tahun)")
    ax_bottom.set_xlabel("Kabupaten/Kota")
    ax_bottom.tick_params(axis='x', rotation=45)

    max_val = max(top_5.max(), bottom_5.max())
    ax_top.set_ylim(0, max_val + 5)
    ax_bottom.set_ylim(0, max_val + 5)

    plt.tight_layout()
    return fig

def visualisasi_rata_rata_salary(df):
    # Hitung rata-rata salary per provinsi
    rata_rata_salary = df.groupby('Provinsi')['SALARY'].mean().reset_index()

    # Ambil geometri unik untuk tiap provinsi
    unique_provinces = df[['Provinsi', 'Geometry_provinsi']].drop_duplicates(subset='Provinsi').reset_index(drop=True)

    # Gabungkan geometri dengan rata-rata salary
    gdf_skor = unique_provinces.merge(rata_rata_salary, on='Provinsi', how='left')
    gdf_skor = gpd.GeoDataFrame(gdf_skor, geometry='Geometry_provinsi')

    # Buat plot
    fig, ax = plt.subplots(figsize=(24, 14))
    gdf_skor.plot(
        column='SALARY',
        cmap='Greens',
        linewidth=0.8,
        edgecolor='0.8',
        legend=True,
        ax=ax,
        legend_kwds={
            'label': "Rata-rata SALARY",
            'orientation': "vertical",
            'shrink': 0.6,
            'aspect': 20
        }
    )

    # Tambahkan label provinsi di tengah-tengah
    for idx, row in gdf_skor.iterrows():
        if row['Geometry_provinsi'].centroid.is_empty:
            continue
        plt.annotate(
            text=row['Provinsi'],
            xy=(row['Geometry_provinsi'].centroid.x, row['Geometry_provinsi'].centroid.y),
            ha='center',
            fontsize=8,
            color='black'
        )

    ax.set_title("Sebaran Rata-rata SALARY per Provinsi di Indonesia", fontsize=22)
    ax.axis('off')
    plt.tight_layout()
    return fig

def visualisasi_korelasi_ipm_umur(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(
        x="Indeks Pembangunan Manusia",
        y="Umur Harapan Hidup (Tahun)",
        data=df,
        scatter_kws={'alpha': 0.6},
        line_kws={"color": "red"},
        ax=ax
    )
    ax.set_title("Korelasi antara IPM dan Umur Harapan Hidup", fontsize=14)
    ax.set_xlabel("Indeks Pembangunan Manusia")
    ax.set_ylabel("Umur Harapan Hidup (Tahun)")
    ax.grid(True)
    plt.tight_layout()
    return fig

def visualisasi_spearman_heatmap(df):

    # Siapkan data korelasi (hanya kolom numerik)
    correlation_df = df.select_dtypes(include=['number']).copy()

    # Hitung korelasi Spearman
    correlation_matrix = correlation_df.corr(method='spearman')

    # Buat plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='viridis',
        fmt=".2f",
        linewidths=0.5,
        ax=ax
    )
    ax.set_title("Spearman Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    return fig

def visualisasi_cluster_hover(df):
    import plotly.express as px
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler

    # Drop kolom non-numerik yang tidak diperlukan untuk PCA
    exclude_cols = ['Provinsi', 'Kabupaten/Kota', 'YEAR', 'lat', 'long',
                    'Id_provinsi', 'Kode_provinsi', 'Sumber_provinsi',
                    'Geometry_provinsi', 'Klasifikasi', 'Cluster_OPTICS']

    fitur_numerik = df.drop(columns=[col for col in exclude_cols if col in df.columns])

    # Normalisasi
    scaler = RobustScaler()
    fitur_scaled = scaler.fit_transform(fitur_numerik)

    # PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(fitur_scaled)

    # Buat DataFrame hasil PCA
    df_pca = pd.DataFrame(reduced, columns=["PCA 1", "PCA 2"])
    df_pca["Klasifikasi"] = df["Klasifikasi"].values
    df_pca["Kabupaten/Kota"] = df["Kabupaten/Kota"].values

    # Plot dengan Plotly
    fig = px.scatter(
        df_pca,
        x="PCA 1",
        y="PCA 2",
        color="Klasifikasi",
        hover_name="Kabupaten/Kota",
        title="Persebaran Data Berdasarkan Klasifikasi (Interaktif)",
        labels={"color": "Klasifikasi"},
        width=1500,
        height=900
    )

    return fig
