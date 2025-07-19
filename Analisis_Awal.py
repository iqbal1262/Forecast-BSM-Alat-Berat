import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Analisis Utama",
    page_icon="📊",
    layout="wide"
)

# --- FUNGSI LOADING DATA (DARI GOOGLE SHEETS) ---
@st.cache_data
def load_data_from_gsheets(spreadsheet_id, tab_id):
    url = f'https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={tab_id}'
    try:
        df = pd.read_csv(
            url,
            index_col='BSM_CREATED_ON', 
            parse_dates=True         
        )
        df.index = df.index.normalize()
        return df
    except Exception as e:
        st.error(f"Gagal memuat data dari Google Sheets. Cek kembali ID & hak akses. Error: {e}")
        return None

# --- FUNGSI UNTUK FORMAT TABEL ---
def format_dataframe_for_display(df):
    if df is None:
        return pd.DataFrame()
    df_display = df.copy()
    numeric_cols = df_display.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df_display[col] = df_display[col].apply(lambda x: f"Rp {x:,.0f}")
    return df_display

# --- DATA SOURCE UNTUK HALAMAN INI ---
SPREADSHEET_ID = "1x8QdY6Y4VkoYsqE5ojaFsyq5JntEaNeKRFY1eluOQ80" 
TAB_ID_AKTUAL = "542042387"
TAB_ID_TRAIN = "1217230565"
TAB_ID_TEST = "1677129148"
TAB_ID_FORECAST = "0"

# --- MEMUAT DATA ---
df_aktual = load_data_from_gsheets(SPREADSHEET_ID, TAB_ID_AKTUAL)
df_train = load_data_from_gsheets(SPREADSHEET_ID, TAB_ID_TRAIN)
df_test = load_data_from_gsheets(SPREADSHEET_ID, TAB_ID_TEST)
df_forecast = load_data_from_gsheets(SPREADSHEET_ID, TAB_ID_FORECAST)

# --- KONTEN HALAMAN ---
st.title("Hasil Forecast Total Pengeluaran Mingguan")

# --- SIDEBAR KONTROL UNTUK HALAMAN INI ---
st.sidebar.header("Filter Lapisan Data")
show_aktual = st.sidebar.checkbox("1. Data Aktual (Observed)", value=True, key="aktual_utama")
show_train = st.sidebar.checkbox("2. Data Hasil Training (Fit)", value=True, key="train_utama")
show_test = st.sidebar.checkbox("3. Prediksi Data Test", value=True, key="test_utama")
show_forecast = st.sidebar.checkbox("4. Forecast Masa Depan", value=True, key="forecast_utama")

# --- ANALISIS & METRIK PERFORMA TEST ---
st.subheader("Ringkasan Performa pada Data Test")
if df_test is not None and df_aktual is not None:
    df_error_test = pd.concat([df_aktual['observed'], df_test['prediksi_inti']], axis=1).dropna()
    df_error_test['error'] = (df_error_test['observed'] - df_error_test['prediksi_inti']).abs()
    mae = df_error_test['error'].mean()
    max_error_date = df_error_test['error'].idxmax()
    max_error_value = df_error_test['error'].max()
    col1, col2 = st.columns(2)
    col1.metric(label="Rata-rata Error (MAE)", value=f"Rp {mae:,.0f}")
    col2.metric(label="Error Tertinggi Terjadi Pada", value=max_error_date.strftime('%d %B %Y'), help=f"Selisih sebesar Rp {max_error_value:,.0f}")
else:
    st.info("Data test atau data aktual tidak ditemukan untuk menghitung metrik performa.")

# --- GRAFIK PLOTLY ---
st.markdown("---")
fig = go.Figure()
if show_aktual and df_aktual is not None: fig.add_trace(go.Scatter(x=df_aktual.index, y=df_aktual['observed'], mode='lines', name='Data Aktual', line=dict(color='#0068C9', width=2)))
if show_train and df_train is not None: fig.add_trace(go.Scatter(x=df_train.index, y=df_train['Train Pred'], mode='lines', name='Hasil Training', line=dict(color='#FFAA00', width=2, dash='dash')))
if show_test and df_test is not None:
    fig.add_trace(go.Scatter(x=df_test.index, y=df_test['batas_atas_95'], mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=df_test.index, y=df_test['batas_bawah_95'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 75, 75, 0.2)', hoverinfo='skip', name='Batas Keyakinan 95% (Test)'))
    fig.add_trace(go.Scatter(x=df_test.index, y=df_test['prediksi_inti'], mode='lines', name='Prediksi Test', line=dict(color='#FF4B4B', width=2), customdata=df_test[['batas_bawah_95', 'batas_atas_95']], hovertemplate="<b>Prediksi Test: Rp %{y:,.0f}</b><br>Batas Bawah: Rp %{customdata[0]:,.0f}<br>Batas Atas: Rp %{customdata[1]:,.0f}<extra></extra>"))
if show_forecast and df_forecast is not None:
    fig.add_trace(go.Scatter(x=df_forecast.index, y=df_forecast['batas_atas_95'], mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=df_forecast.index, y=df_forecast['batas_bawah_95'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(45, 201, 55, 0.2)', hoverinfo='skip', name='Batas Keyakinan 95% (Forecast)'))
    fig.add_trace(go.Scatter(x=df_forecast.index, y=df_forecast['prediksi_inti'], mode='lines', name='Forecast Masa Depan', line=dict(color='#2DC937', width=3), customdata=df_forecast[['batas_bawah_95', 'batas_atas_95']], hovertemplate="<b>Forecast: Rp %{y:,.0f}</b><br>Batas Bawah: Rp %{customdata[0]:,.0f}<br>Batas Atas: Rp %{customdata[1]:,.0f}<extra></extra>"))
fig.update_layout(title="Grafik Perbandingan Data", xaxis_title='Tanggal', yaxis_title='Nilai (Rp)', yaxis_tickprefix='Rp ', yaxis_tickformat=',.0f', hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True)

# --- TABEL DATA MENTAH ---
st.markdown("---")
st.subheader("Tinjau Data Mentah")
col1, col2 = st.columns(2)
with col1:
    with st.expander("Data Aktual & Training", expanded=True):
        st.dataframe(format_dataframe_for_display(df_aktual))
        st.dataframe(format_dataframe_for_display(df_train))
with col2:
    with st.expander("Data Test & Forecast", expanded=True):
        st.dataframe(format_dataframe_for_display(df_test))
        st.dataframe(format_dataframe_for_display(df_forecast))