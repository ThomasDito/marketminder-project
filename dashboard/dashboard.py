import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# ==============================
# Load Model & Dataset
# ==============================
model_path = "final_gradient_boosting_model.pkl"
gb_model = joblib.load(model_path)

data_path = "main_data.csv"
df = pd.read_csv(data_path)

# Mendapatkan nilai min & max dari dataset asli sebelum normalisasi
price_min, price_max = df["Price"].min(), df["Price"].max()
demand_min, demand_max = df["Demand Forecast"].min(), df["Demand Forecast"].max()

# Memastikan format data sesuai dengan yang dipakai saat training
selected_features = ['Category', 'Price', 'Discount', 'Demand Forecast', 'Holiday/Promotion', 'Competitor Pricing', 'Seasonality']

# Fungsi untuk menormalisasi input agar sesuai dengan skala yang digunakan saat preprocessing
def normalize_value(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# Fungsi untuk melakukan prediksi
def predict_units_sold(input_data):
    prediction = gb_model.predict(input_data)
    return prediction

# ==============================
# UI Streamlit
# ==============================
st.title("🛍️ MarketMinder: Optimizing Store Operations Through Sales Forecasting 📊")
st.write("Menggunakan **Gradient Boosting Regressor** untuk memprediksi penjualan unit berdasarkan berbagai faktor.")

# Sidebar input untuk prediksi manual
st.sidebar.header("🛍️ Masukkan Data untuk Prediksi")

# Pilihan kategori produk
category = st.sidebar.selectbox("Kategori Produk:", ["Clothing", "Electronics", "Furniture", "Groceries", "Toys"])
category_mapping = {"Clothing": 0, "Electronics": 1, "Furniture": 2, "Groceries": 3, "Toys": 4}
category_encoded = category_mapping[category]

# Input nilai asli (sebelum normalisasi)
price_input = st.sidebar.number_input(f"Harga Produk (Rentang: {price_min} - {price_max})", min_value=price_min, max_value=price_max, step=1.0)
demand_forecast_input = st.sidebar.number_input(f"Perkiraan Permintaan (Rentang: {demand_min} - {demand_max})", min_value=demand_min, max_value=demand_max, step=1.0)

# Normalisasi input agar sesuai dengan model yang telah dilatih
price = normalize_value(price_input, price_min, price_max)
demand_forecast = normalize_value(demand_forecast_input, demand_min, demand_max)

# Input lainnya
discount = st.sidebar.number_input("Diskon (%)", min_value=0.0, max_value=100.0, step=1.0)
holiday_promotion = st.sidebar.selectbox("Ada Promo/Hari Libur?", ["Tidak", "Ya"])
holiday_encoded = 1 if holiday_promotion == "Ya" else 0

competitor_pricing = st.sidebar.number_input("Harga Kompetitor", min_value=0.0, step=1.0)
seasonality = st.sidebar.selectbox("Seasonality:", ["Autumn", "Spring", "Summer", "Winter"])
seasonality_mapping = {"Autumn": 0, "Spring": 1, "Summer": 2, "Winter": 3}
seasonality_encoded = seasonality_mapping[seasonality]

# Konversi input ke dalam bentuk DataFrame yang sesuai dengan model
input_features = np.array([[category_encoded, price, discount, demand_forecast, holiday_encoded, competitor_pricing, seasonality_encoded]])
input_df = pd.DataFrame(input_features, columns=selected_features)

# ==============================
# Prediksi
# ==============================
if st.sidebar.button("Prediksi"):
    prediction = predict_units_sold(input_df)
    st.sidebar.subheader("📊 Hasil Prediksi")
    st.sidebar.write(f"**Prediksi Units Sold: {int(prediction[0])}**")

# ==============================
# Visualisasi Tren Penjualan
# ==============================

if 'Date' in df.columns:
    st.subheader("📈 Tren Penjualan dari Waktu ke Waktu")

    # Memastikan format date dalam datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Mengelompokkan total penjualan per tanggal
    df_grouped = df.groupby("Date")["Units Sold"].sum().reset_index()
    
    # Menambahkan label pada visualisasi menggunakan Markdown
    st.markdown("""
    **📌 Keterangan:**
    - **Sumbu X (Tanggal):** Menunjukkan rentang waktu dari data historis.
    - **Sumbu Y (Total Units Sold):** Jumlah unit yang terjual pada hari tertentu.
    - **Interaktivitas:** Hover pada titik data untuk melihat jumlah penjualan pada tanggal tertentu.
    """)

    # Menggunakan Streamlit Line Chart untuk interaktifitas lebih baik
    st.line_chart(df_grouped.set_index("Date"))

# ==============================
# Visualisasi Distribusi Data
# ==============================
st.subheader("📊 Distribusi Fitur dalam Dataset")
feature_choice = st.selectbox("Pilih fitur untuk visualisasi", selected_features)

plt.figure(figsize=(8, 4))
sns.histplot(df[feature_choice], kde=True, bins=20, color="skyblue", edgecolor="black")
plt.xlabel(feature_choice, fontsize=12)
plt.ylabel("Frekuensi", fontsize=12)
plt.title(f"Distribusi {feature_choice}", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)

st.pyplot(plt)

# ==============================
# Visualisasi
# ==============================

# Average Units Sold by Day of Week (Per Region)
st.subheader("📆 Rata-rata Units Sold Berdasarkan Hari dalam Seminggu (Per Region)")
avg_daily_sales = df.groupby(["DayOfWeek", "Region"])["Units Sold"].mean().unstack()
# mengurutkan berdasarkan urutan hari dalam seminggu
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
avg_daily_sales = avg_daily_sales.reindex(order)

plt.figure(figsize=(12, 6))
avg_daily_sales.plot(kind="bar", figsize=(12, 6))
plt.title("Average Units Sold by Day of Week (Per Region)")
plt.xlabel("Day of Week")
plt.ylabel("Average Units Sold")
plt.legend(title="Region")
st.pyplot(plt)

# Average Units Sold by Month (Per Region)
st.subheader("📅 Rata-rata Units Sold Berdasarkan Bulan (Per Region)")
avg_monthly_sales = df.groupby(["Month", "Region"])["Units Sold"].mean().unstack()

plt.figure(figsize=(12, 6))
avg_monthly_sales.plot(kind="bar", figsize=(12, 6))
plt.title("Average Units Sold by Month (Per Region)")
plt.xlabel("Month")
plt.ylabel("Average Units Sold")
plt.legend(title="Region")
st.pyplot(plt)

# Average Units Sold by Season (Per Region)
st.subheader("🍂 Rata-rata Units Sold Berdasarkan Musim (Per Region)")
avg_seasonal_sales = df.groupby(["Seasonality", "Region"])["Units Sold"].mean().unstack()

plt.figure(figsize=(12, 6))
avg_seasonal_sales.plot(kind="bar", figsize=(12, 6))
plt.title("Average Units Sold by Season (Per Region)")
plt.xlabel("Season")
plt.ylabel("Average Units Sold")
plt.legend(title="Region")
st.pyplot(plt)

# Average Units Sold with and without Promotion (Per Region)
st.subheader("🎁 Rata-rata Units Sold dengan dan tanpa Promo (Per Region)")
avg_promo_sales = df.groupby(["Holiday/Promotion", "Region"])["Units Sold"].mean().unstack()

plt.figure(figsize=(12, 6))
avg_promo_sales.plot(kind="bar", figsize=(12, 6))
plt.title("Average Units Sold with and without Promotion (Per Region)")
plt.xlabel("Promotion Status (0 = No, 1 = Yes)")
plt.ylabel("Average Units Sold")
plt.legend(title="Region")
st.pyplot(plt)
