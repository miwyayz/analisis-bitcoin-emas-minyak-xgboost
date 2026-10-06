import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

# ============================================================
# KONFIGURASI
# ============================================================
st.set_page_config(
    page_title="Dashboard Prediksi Harga Bitcoin",
    page_icon="📈",
    layout="wide",
)

DATA_PATH = "merged_data.csv"

BEST_PARAMS = {
    "subsample": 0.6,
    "reg_lambda": 1,
    "reg_alpha": 1,
    "n_estimators": 100,
    "min_child_weight": 5,
    "max_depth": 2,
    "learning_rate": 0.01,
    "colsample_bytree": 1.0,
}

ASET = {"Bitcoin": "btc_price", "Emas": "gold_price", "Minyak": "oil_price"}


# ============================================================
# FUNGSI BANTU
# ============================================================
@st.cache_data
def muat_data(path):
    df = pd.read_csv(path, parse_dates=["recorded_date"])
    df = df.sort_values("recorded_date").drop_duplicates("recorded_date")
    return df.set_index("recorded_date")[["btc_price", "gold_price", "oil_price"]]


def buat_fitur(df):
    d = df[["btc_price", "gold_price", "oil_price"]].copy()

    d["btc_ret"] = np.log(d["btc_price"]).diff()
    d["gold_ret"] = d["gold_price"].pct_change()
    d["oil_ret"] = d["oil_price"].pct_change()
    d[["gold_ret", "oil_ret"]] = (
        d[["gold_ret", "oil_ret"]].replace([np.inf, -np.inf], np.nan).fillna(0)
    )

    fitur = pd.DataFrame(index=d.index)

    for k in [1, 2, 3, 7, 14]:
        fitur[f"btc_ret_lag{k}"] = d["btc_ret"].shift(k)

    for k in [1, 2, 3, 7]:
        fitur[f"gold_ret_lag{k}"] = d["gold_ret"].shift(k)
        fitur[f"oil_ret_lag{k}"] = d["oil_ret"].shift(k)

    for w in [7, 30]:
        fitur[f"btc_mom_{w}"] = d["btc_ret"].shift(1).rolling(w).mean()
        fitur[f"btc_vol_{w}"] = d["btc_ret"].shift(1).rolling(w).std()

    fitur["gold_mom_7"] = d["gold_ret"].shift(1).rolling(7).mean()
    fitur["oil_mom_7"] = d["oil_ret"].shift(1).rolling(7).mean()
    fitur["gold_vol_7"] = d["gold_ret"].shift(1).rolling(7).std()
    fitur["oil_vol_7"] = d["oil_ret"].shift(1).rolling(7).std()

    btc_lag = d["btc_price"].shift(1)
    for w in [7, 30]:
        fitur[f"btc_ma_ratio_{w}"] = np.log(btc_lag / btc_lag.rolling(w).mean())

    fitur["day_of_week"] = d.index.dayofweek
    fitur["month"] = d.index.month

    fitur["target"] = d["btc_ret"]
    fitur["btc_price"] = d["btc_price"]
    fitur["btc_price_prev"] = d["btc_price"].shift(1)
    return fitur


def hitung_metrik(actual, pred):
    return {
        "MAE": mean_absolute_error(actual, pred),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "MAPE (%)": float(np.mean(np.abs((actual - pred) / actual)) * 100),
        "R²": r2_score(actual, pred),
    }


def kekuatan_korelasi(r):
    a = abs(r)
    if a < 0.20:
        return "sangat lemah"
    if a < 0.40:
        return "lemah"
    if a < 0.60:
        return "sedang"
    if a < 0.80:
        return "kuat"
    return "sangat kuat"


def arah_korelasi(r):
    if r > 0:
        return "positif"
    if r < 0:
        return "negatif"
    return "tidak ada hubungan"


def rapikan(fig, y_title=None, height=430):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    if y_title:
        fig.update_yaxes(title_text=y_title)
    fig.update_xaxes(title_text="Tanggal")
    return fig


def tambah_interval(fig, x, lower, upper, nama="Interval prediksi 95%"):
    fig.add_trace(go.Scatter(
        x=x, y=upper, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=lower, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(255,127,14,0.20)", name=nama,
    ))


# ============================================================
# PEMODELAN
# ============================================================
@st.cache_resource(show_spinner="Melatih model XGBoost...")
def latih_model(base_data):
    feature_data = buat_fitur(base_data).replace([np.inf, -np.inf], np.nan).dropna()
    fitur = [c for c in feature_data.columns
             if c not in ["target", "btc_price", "btc_price_prev"]]

    split_index = int(len(feature_data) * 0.8)
    train = feature_data.iloc[:split_index]
    test = feature_data.iloc[split_index:]

    X_train, y_train = train[fitur], train["target"]
    X_test = test[fitur]

    def buat_model():
        return XGBRegressor(
            objective="reg:squarederror", random_state=42, n_jobs=-1, **BEST_PARAMS
        )

    model = buat_model()
    model.fit(X_train, y_train)

    # Interval prediksi dari residual out-of-fold pada data train
    tscv = TimeSeriesSplit(n_splits=5)
    oof = []
    for tr_idx, va_idx in tscv.split(X_train):
        m_cv = buat_model()
        m_cv.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof.append(y_train.iloc[va_idx].values - m_cv.predict(X_train.iloc[va_idx]))
    oof = np.concatenate(oof)
    q_low, q_high = np.quantile(oof, [0.025, 0.975])

    # Prediksi one-step-ahead pada data test
    pred_ret = model.predict(X_test)
    prev = test["btc_price_prev"]
    forecast = pd.Series(prev.values * np.exp(pred_ret), index=test.index)
    lower = pd.Series(prev.values * np.exp(pred_ret + q_low), index=test.index)
    upper = pd.Series(prev.values * np.exp(pred_ret + q_high), index=test.index)

    # Evaluasi
    y_true = test["btc_price"]
    m_xgb = hitung_metrik(y_true, forecast)
    m_naive = hitung_metrik(y_true, prev)

    arah_aktual = np.sign(y_true - prev)
    arah_pred = np.sign(forecast - prev)
    mask = arah_pred != 0
    akurasi_arah = (
        float((arah_aktual[mask] == arah_pred[mask]).mean() * 100)
        if mask.any() else np.nan
    )
    coverage = float(((y_true >= lower) & (y_true <= upper)).mean() * 100)

    importance = pd.Series(model.feature_importances_, index=fitur).sort_values()

    # Model final (seluruh data) untuk prediksi masa depan
    final_model = buat_model()
    final_model.fit(feature_data[fitur], feature_data["target"])

    return {
        "fitur": fitur,
        "train": train,
        "test": test,
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
        "m_xgb": m_xgb,
        "m_naive": m_naive,
        "akurasi_arah": akurasi_arah,
        "coverage": coverage,
        "importance": importance,
        "q_low": float(q_low),
        "q_high": float(q_high),
        "final_model": final_model,
    }


@st.cache_data(show_spinner="Menghitung prediksi ke depan...")
def prediksi_rekursif(base_data, horizon, _model, fitur, q_low, q_high):
    sim = base_data.copy()
    last_price = sim["btc_price"].iloc[-1]
    cum_ret = 0.0
    rows = []

    for h in range(1, horizon + 1):
        next_date = sim.index[-1] + pd.Timedelta(days=1)
        row = pd.DataFrame(
            {
                "btc_price": [sim["btc_price"].iloc[-1]],
                "gold_price": [sim["gold_price"].iloc[-1]],
                "oil_price": [sim["oil_price"].iloc[-1]],
            },
            index=[next_date],
        )
        sim = pd.concat([sim, row])

        fitur_next = buat_fitur(sim.tail(120))[fitur].iloc[[-1]]
        ret_next = _model.predict(fitur_next)[0]

        harga_next = sim["btc_price"].iloc[-2] * np.exp(ret_next)
        sim.loc[next_date, "btc_price"] = harga_next

        cum_ret += ret_next
        rows.append({
            "Tanggal": next_date,
            "Prediksi (USD)": harga_next,
            "Batas bawah 95% (USD)": last_price * np.exp(cum_ret + q_low * np.sqrt(h)),
            "Batas atas 95% (USD)": last_price * np.exp(cum_ret + q_high * np.sqrt(h)),
        })

    return pd.DataFrame(rows)


# ============================================================
# MUAT DATA
# ============================================================
try:
    data = muat_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"File '{DATA_PATH}' tidak ditemukan. Letakkan di folder yang sama dengan app.py.")
    st.stop()

base_data = data.asfreq("D").ffill()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Pengaturan")

tgl_min, tgl_max = data.index.min().date(), data.index.max().date()
rentang = st.sidebar.date_input(
    "Rentang tanggal analisis",
    value=(tgl_min, tgl_max),
    min_value=tgl_min,
    max_value=tgl_max,
)
if isinstance(rentang, (tuple, list)) and len(rentang) == 2:
    tgl_awal, tgl_akhir = rentang
else:
    tgl_awal, tgl_akhir = tgl_min, tgl_max

horizon = st.sidebar.slider("Horizon prediksi (hari)", 7, 90, 30)

st.sidebar.markdown("---")
st.sidebar.caption("Rahmi · E1E14048")

df_f = data.loc[str(tgl_awal):str(tgl_akhir)]
if len(df_f) < 30:
    st.warning("Rentang tanggal terlalu pendek. Pilih minimal 30 hari.")
    st.stop()

# ============================================================
# HEADER
# ============================================================
st.title("📈 Analisis Korelasi dan Prediksi Harga Bitcoin")
st.caption(
    "Hubungan harga Bitcoin dengan emas dan minyak serta prediksi harga Bitcoin "
    "menggunakan XGBoost."
)

k1, k2, k3, k4 = st.columns(4)
for kolom, (nama, col) in zip((k1, k2, k3), ASET.items()):
    terakhir, sebelum = df_f[col].iloc[-1], df_f[col].iloc[-2]
    kolom.metric(f"{nama} (USD)", f"{terakhir:,.2f}", f"{(terakhir / sebelum - 1) * 100:+.2f}%")
k4.metric("Jumlah hari data", f"{len(df_f):,}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Tren Harga", "Korelasi", "Pola Perubahan", "Evaluasi Model", "Prediksi"]
)

# ============================================================
# TAB 1 - TREN HARGA
# ============================================================
with tab1:
    pilihan = st.radio("Aset", list(ASET.keys()), horizontal=True)
    fig = px.line(df_f.reset_index(), x="recorded_date", y=ASET[pilihan])
    fig.update_layout(title=f"Perkembangan Harga {pilihan}")
    st.plotly_chart(rapikan(fig, "Harga (USD)"))

    norm = df_f / df_f.iloc[0] * 100
    norm.columns = list(ASET.keys())
    fig = px.line(norm.reset_index(), x="recorded_date", y=list(ASET.keys()))
    fig.update_layout(title="Perbandingan Harga Relatif (Awal Periode = 100)", legend_title_text="")
    st.plotly_chart(rapikan(fig, "Indeks harga"))

    with st.expander("Statistik deskriptif"):
        desc = df_f.describe().T
        desc.index = ["Bitcoin", "Emas", "Minyak"]
        st.dataframe(desc.style.format("{:,.2f}"))

# ============================================================
# TAB 2 - KORELASI
# ============================================================
with tab2:
    r_gold = df_f["btc_price"].corr(df_f["gold_price"])
    r_oil = df_f["btc_price"].corr(df_f["oil_price"])

    c1, c2 = st.columns(2)
    c1.metric(
        "Korelasi Bitcoin – Emas", f"{r_gold:.4f}",
        f"{kekuatan_korelasi(r_gold)}, {arah_korelasi(r_gold)}", delta_color="off",
    )
    c2.metric(
        "Korelasi Bitcoin – Minyak", f"{r_oil:.4f}",
        f"{kekuatan_korelasi(r_oil)}, {arah_korelasi(r_oil)}", delta_color="off",
    )

    corr = df_f.corr()
    corr.index = corr.columns = list(ASET.keys())
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
    )
    fig.update_layout(title="Matriks Korelasi", height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig)

    s1, s2 = st.columns(2)
    with s1:
        fig = px.scatter(
            df_f.reset_index(), x="gold_price", y="btc_price", opacity=0.5,
            labels={"gold_price": "Harga Emas (USD)", "btc_price": "Harga Bitcoin (USD)"},
            title="Bitcoin vs Emas",
        )
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig)
    with s2:
        fig = px.scatter(
            df_f.reset_index(), x="oil_price", y="btc_price", opacity=0.5,
            labels={"oil_price": "Harga Minyak (USD)", "btc_price": "Harga Bitcoin (USD)"},
            title="Bitcoin vs Minyak",
        )
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig)

    st.caption(
        "Korelasi dihitung pada level harga. Korelasi level harga dapat menyesatkan "
        "pada deret waktu yang bertren, sehingga tidak menunjukkan hubungan sebab-akibat."
    )

# ============================================================
# TAB 3 - POLA PERUBAHAN
# ============================================================
with tab3:
    chg = df_f.pct_change().mul(100).dropna()
    chg.columns = list(ASET.keys())
    chg["Kondisi"] = np.where(
        chg["Bitcoin"] > 0, "Naik", np.where(chg["Bitcoin"] < 0, "Turun", "Tetap")
    )
    pola = chg.groupby("Kondisi")[list(ASET.keys())].mean()
    pola = pola.loc[[k for k in ["Naik", "Turun"] if k in pola.index]]

    long = pola.reset_index().melt(
        id_vars="Kondisi", var_name="Aset", value_name="Rata-rata perubahan harian (%)"
    )
    fig = px.bar(
        long, x="Kondisi", y="Rata-rata perubahan harian (%)",
        color="Aset", barmode="group",
        title="Rata-rata Perubahan Harga Saat Bitcoin Naik atau Turun",
    )
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig)

    st.dataframe(pola.style.format("{:.2f}"))

    ret = chg[list(ASET.keys())] / 100
    fig = go.Figure()
    for nama in ASET:
        fig.add_trace(go.Scatter(x=ret.index, y=ret[nama], mode="lines", name=nama))
    fig.update_layout(title="Daily Return Bitcoin, Emas, dan Minyak")
    st.plotly_chart(rapikan(fig, "Daily return"))

# ============================================================
# TAB 4 - EVALUASI MODEL
# ============================================================
hasil = latih_model(base_data)

with tab4:
    m_xgb, m_naive = hasil["m_xgb"], hasil["m_naive"]
    test = hasil["test"]

    st.caption(
        f"Data training: {hasil['train'].index.min().date()} s.d. {hasil['train'].index.max().date()} "
        f"({len(hasil['train']):,} hari) · Data testing: {test.index.min().date()} s.d. "
        f"{test.index.max().date()} ({len(test):,} hari)"
    )

    tabel = pd.DataFrame({"XGBoost": m_xgb, "Naive (harga kemarin)": m_naive}).T
    st.dataframe(tabel.style.format({
        "MAE": "{:,.2f}", "RMSE": "{:,.2f}", "MAPE (%)": "{:.2f}", "R²": "{:.4f}",
    }))

    e1, e2 = st.columns(2)
    e1.metric(
        "Akurasi arah naik/turun",
        "n/a" if np.isnan(hasil["akurasi_arah"]) else f"{hasil['akurasi_arah']:.2f}%",
    )
    e2.metric("Cakupan interval 95%", f"{hasil['coverage']:.2f}%")

    selisih = (m_naive["RMSE"] - m_xgb["RMSE"]) / m_naive["RMSE"] * 100
    if abs(selisih) < 1:
        st.info(
            f"Perbaikan RMSE terhadap naive: {selisih:.2f}%. XGBoost setara dengan naive "
            "(random walk), sehingga tidak ada keunggulan prediktif yang berarti."
        )
    elif selisih > 0:
        st.success(f"Perbaikan RMSE terhadap naive: {selisih:.2f}%. XGBoost lebih akurat daripada naive.")
    else:
        st.warning(f"Perbaikan RMSE terhadap naive: {selisih:.2f}%. XGBoost kurang akurat daripada naive.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test.index, y=test["btc_price"], mode="lines", name="Aktual"))
    fig.add_trace(go.Scatter(x=test.index, y=hasil["forecast"], mode="lines", name="Prediksi XGBoost"))
    tambah_interval(fig, test.index, hasil["lower"], hasil["upper"])
    fig.update_layout(title="Prediksi One-Step-Ahead pada Data Test")
    st.plotly_chart(rapikan(fig, "Harga Bitcoin (USD)"))

    imp = hasil["importance"].tail(15)
    fig = px.bar(
        x=imp.values, y=imp.index, orientation="h",
        labels={"x": "Importance", "y": "Fitur"}, title="15 Fitur Terpenting",
    )
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig)

# ============================================================
# TAB 5 - PREDIKSI
# ============================================================
with tab5:
    future = prediksi_rekursif(
        base_data, horizon, hasil["final_model"],
        hasil["fitur"], hasil["q_low"], hasil["q_high"],
    )

    riwayat = base_data.iloc[-180:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=riwayat.index, y=riwayat["btc_price"], mode="lines", name="Historis (180 hari)"))
    fig.add_trace(go.Scatter(x=future["Tanggal"], y=future["Prediksi (USD)"], mode="lines", name=f"Prediksi {horizon} hari"))
    tambah_interval(fig, future["Tanggal"], future["Batas bawah 95% (USD)"], future["Batas atas 95% (USD)"])
    fig.update_layout(title=f"Prediksi Harga Bitcoin {horizon} Hari ke Depan")
    st.plotly_chart(rapikan(fig, "Harga Bitcoin (USD)"))

    p1, p2, p3 = st.columns(3)
    p1.metric("Harga terakhir (USD)", f"{base_data['btc_price'].iloc[-1]:,.2f}")
    p2.metric(
        f"Prediksi hari ke-{horizon} (USD)",
        f"{future['Prediksi (USD)'].iloc[-1]:,.2f}",
        f"{(future['Prediksi (USD)'].iloc[-1] / base_data['btc_price'].iloc[-1] - 1) * 100:+.2f}%",
    )
    p3.metric(
        "Rentang interval 95% (USD)",
        f"{future['Batas bawah 95% (USD)'].iloc[-1]:,.0f} – {future['Batas atas 95% (USD)'].iloc[-1]:,.0f}",
    )

    tampil = future.copy()
    tampil["Tanggal"] = tampil["Tanggal"].dt.strftime("%Y-%m-%d")
    st.dataframe(tampil.style.format({
        "Prediksi (USD)": "{:,.2f}",
        "Batas bawah 95% (USD)": "{:,.2f}",
        "Batas atas 95% (USD)": "{:,.2f}",
    }), hide_index=True)

    st.download_button(
        "Unduh prediksi (CSV)",
        tampil.to_csv(index=False).encode("utf-8"),
        file_name="prediksi_bitcoin.csv",
        mime="text/csv",
    )

    st.caption(
        "Prediksi dihitung secara rekursif dengan asumsi harga emas dan minyak tetap pada nilai "
        "terakhir. Interval melebar seiring bertambahnya horizon. Hasil ini bersifat akademik "
        "dan bukan saran investasi."
    )
