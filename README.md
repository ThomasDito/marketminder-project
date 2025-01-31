# MarketMinder: Optimizing Store Operations Through Sales Forecasting
Proyek ini bertujuan untuk memprediksi penjualan toko dengan integrasi machine learning.

## Struktur Proyek
1. **data/**: Berisi dataset.
2. **notebooks/**: 
- Berisi Jupyter Notebook untuk eksplorasi dan analisis
- insight.txt yang berisi seluruh insight yang didapat dari proses EDA
- dokumentasi.txt yang berisi seluruh insight dan dokumentasi lengkap dari setiap model
3. **dashboard/**:
- Berisi final model machine learning yang telah disimpan menggunakan joblib
- main_data.csv yang digunakan untuk dashboard.py
- dashboard.py berisi source code untuk membangun tampilan dashboard yang berisi I/O prediksi dan visualisasi data

## Langkah-Langkah
1. Eksplorasi Data.
2. Preprocessing.
3. Pengembangan Model.
4. Integrasi dan Visualisasi.

-----------------------------------------------------------------------------------------------------------------------------------

# Cara Menjalankan Project Submission dan Menjalankan Dashboard

## Setup Environment - Anaconda
```
conda create --name main-ds python=3.9
conda activate main-ds
pip install -r requirements.txt
```

## Setup Environment - Shell/Terminal
```
mkdir proyek_analisis_data
cd proyek_analisis_data
pipenv install
pipenv shell
pip install -r requirements.txt
```

## Run steamlit app
```
streamlit run dashboard.py
```

## Dashboard Link
[MarketMinder](https://marketminder-project-dashboard.streamlit.app/)