import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard Analisis Forecast",
    page_icon="💰",
    layout="wide"
)

# --- FUNGSI LOADING DATA ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_excel(
            file_path,
            index_col='BSM_CREATED_ON', 
            parse_dates=True         
        )
        df.index = df.index.normalize()
        return df
    except FileNotFoundError:
        st.warning(f"File tidak ditemukan: {file_path}")
        return None
    except KeyError:
        st.error(f"GAGAL: Pastikan kolom 'BSM_CREATED_ON' ada di file {file_path}.")
        return None
    except Exception as e:
        st.error(f"Error saat membaca {file_path}: {e}")
        return None

# --- MEMUAT SEMUA DATA ---
df_aktual = load_data('stl_decomp.xlsx')
df_train = load_data('train_pred.xlsx')
df_test = load_data('test_final_evaluation.xlsx')
df_forecast = load_data('forecast.xlsx')

# --- SIDEBAR UNTUK KONTROL TAMPILAN ---
st.sidebar.header("Tampilkan Lapisan Data")
show_aktual = st.sidebar.checkbox("1. Data Aktual (Observed)", value=True)
show_train = st.sidebar.checkbox("2. Data Hasil Training (Fit)", value=True)
show_test = st.sidebar.checkbox("3. Prediksi Data Test", value=True)
show_forecast = st.sidebar.checkbox("4. Forecast Masa Depan", value=True)

# --- JUDUL DASHBOARD ---
st.title("💰 Visualisasi Hasil Forecast BSM Alat Berat")
st.write("Gunakan checkbox di sidebar untuk menampilkan atau menyembunyikan subset data.")

# --- ANALISIS & METRIK PERFORMA TEST ---
st.markdown("---")
st.subheader("Ringkasan Performa pada Data Test")

# Gabungkan data aktual dan test untuk menghitung error
if df_test is not None and df_aktual is not None:
    # Gabungkan berdasarkan tanggal
    df_error_test = pd.concat([df_aktual['observed'], df_test['prediksi_inti']], axis=1).dropna()
    
    # Hitung selisih absolut (absolute error)
    df_error_test['error'] = (df_error_test['observed'] - df_error_test['prediksi_inti']).abs()
    
    # Hitung metrik
    mae = df_error_test['error'].mean()
    max_error_date = df_error_test['error'].idxmax()
    max_error_value = df_error_test['error'].max()

    # Tampilkan metrik dalam kolom
    col1, col2 = st.columns(2)
    col1.metric(
        label="Rata-rata Error (MAE)", 
        value=f"Rp {mae:,.0f}"
    )
    col2.metric(
        label="Error Tertinggi Terjadi Pada", 
        value=max_error_date.strftime('%d %B %Y'),
        help=f"Selisih sebesar Rp {max_error_value:,.0f}" # Info tambahan saat hover
    )
else:
    st.info("Data test atau data aktual tidak ditemukan untuk menghitung metrik performa.")


# --- MEMBUAT GRAFIK PLOTLY ---
st.markdown("---")
fig = go.Figure()

if show_aktual and df_aktual is not None:
    fig.add_trace(go.Scatter(x=df_aktual.index, y=df_aktual['observed'], mode='lines', name='Data Aktual (Observed)', line=dict(color='#0068C9', width=2)))
if show_train and df_train is not None:
    fig.add_trace(go.Scatter(x=df_train.index, y=df_train['Train Pred'], mode='lines', name='Hasil Training (Fit)', line=dict(color='#FFAA00', width=2, dash='dash')))
if show_test and df_test is not None:
    fig.add_trace(go.Scatter(x=df_test.index, y=df_test['batas_atas_95'], mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=df_test.index, y=df_test['batas_bawah_95'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 75, 75, 0.2)', hoverinfo='skip', name='Batas Keyakinan 95% (Test)'))
    fig.add_trace(go.Scatter(x=df_test.index, y=df_test['prediksi_inti'], mode='lines', name='Prediksi Test', line=dict(color='#FF4B4B', width=2), customdata=df_test[['batas_bawah_95', 'batas_atas_95']], hovertemplate="<b>Prediksi Test: Rp %{y:,.0f}</b><br>Batas Bawah: Rp %{customdata[0]:,.0f}<br>Batas Atas: Rp %{customdata[1]:,.0f}<extra></extra>"))
if show_forecast and df_forecast is not None:
    fig.add_trace(go.Scatter(x=df_forecast.index, y=df_forecast['batas_atas_95'], mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=df_forecast.index, y=df_forecast['batas_bawah_95'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(45, 201, 55, 0.2)', hoverinfo='skip', name='Batas Keyakinan 95% (Forecast)'))
    fig.add_trace(go.Scatter(x=df_forecast.index, y=df_forecast['prediksi_inti'], mode='lines', name='Forecast Masa Depan', line=dict(color='#2DC937', width=3), customdata=df_forecast[['batas_bawah_95', 'batas_atas_95']], hovertemplate="<b>Forecast: Rp %{y:,.0f}</b><br>Batas Bawah: Rp %{customdata[0]:,.0f}<br>Batas Atas: Rp %{customdata[1]:,.0f}<extra></extra>"))

fig.update_layout(xaxis_title='Tanggal', yaxis_title='Nilai (Rp)', yaxis_tickprefix='Rp ', yaxis_tickformat=',.0f', hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True)

# --- FUNGSI UNTUK FORMAT TABEL ---
def format_dataframe_for_display(df):
    df_display = df.copy()
    numeric_cols = df_display.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df_display[col] = df_display[col].apply(lambda x: f"Rp {x:,.0f}")
    return df_display

# --- MENAMPILKAN DATA MENTAH DENGAN FORMAT BARU ---
st.markdown("---")
st.subheader("Tinjau Data Mentah")
col1, col2 = st.columns(2)

with col1:
    with st.expander("Data Aktual & Training", expanded=True):
        if df_aktual is not None:
            st.dataframe(format_dataframe_for_display(df_aktual))
        if df_train is not None:
            st.dataframe(format_dataframe_for_display(df_train))
with col2:
    with st.expander("Data Test & Forecast", expanded=True):
        if df_test is not None:
            st.dataframe(format_dataframe_for_display(df_test))
        if df_forecast is not None:
            st.dataframe(format_dataframe_for_display(df_forecast))