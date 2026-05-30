# Fingo AI — API Integration Guide

**Base URL:** `https://mes1205-fingo.hf.space`

---

## Endpoints

### 1. `POST /predict/income`
Prediksi pendapatan 4 minggu ke depan berdasarkan histori 12 minggu terakhir.

**Request Body:**
```json
{
  "income_history": [800000, 750000, 820000, 790000, 810000, 770000, 800000, 830000, 760000, 800000, 790000, 810000],
  "usia": 27,
  "hari_kerja_per_minggu": 5,
  "jam_kerja_per_hari": 8,
  "gig_ojek_online": 1,
  "gig_kurir": 0,
  "gig_jualan_online": 0,
  "gig_freelance_desain": 0,
  "gig_freelance_it": 0,
  "gig_content_creator": 0,
  "gig_tutor": 0,
  "gig_pekerja_harian": 0
}
```

> `income_history` wajib berisi tepat **12 nilai** (Rupiah). Semua field lainnya opsional — kalau tidak dikirim, pakai default profil ojol.

**Response:**
```json
{
  "status": "success",
  "prediction_next_week": 812500.00,
  "prediction_4_weeks_ahead": [812500.00, 798125.00, 828750.00, 771875.00],
  "total_projected_income": 3211250.00,
  "income_direction": "Up",
  "direction_proba": { "Down": 0.12, "Stable": 0.31, "Up": 0.57 },
  "avg_income_last_4w": 800000.00
}
```

| Field | Tipe | Keterangan |
|---|---|---|
| `prediction_next_week` | float | Prediksi pendapatan minggu depan (IDR) |
| `prediction_4_weeks_ahead` | float[] | Proyeksi 4 minggu ke depan (IDR) |
| `total_projected_income` | float | Total proyeksi 4 minggu (IDR) |
| `income_direction` | string | `"Up"` / `"Down"` / `"Stable"` |
| `direction_proba` | object | Probabilitas tiap arah tren |
| `avg_income_last_4w` | float | Rata-rata pendapatan 4 minggu terakhir (IDR) |

---

### 2. `POST /chat`
Konsultasi keuangan berbasis Generative AI (Gemini).

**Request Body:**
```json
{
  "user_message": "Gimana cara aku nabung dari pendapatan bulan ini?",
  "financial_context": {
    "income": 5000000,
    "expense": 3500000,
    "budget_remaining": 1500000,
    "impulsive_count": 3
  }
}
```

**Response:**
```json
{
  "reply": "Dengan sisa budget 1.5jt, kamu udah di posisi yang oke banget! ..."
}
```

---

### 3. `GET /`
Health check — cek apakah server aktif dan model sudah loaded.

```json
{
  "status": "active",
  "model_loaded": true
}
```

---

## Error Handling

| HTTP Code | Kondisi |
|---|---|
| `400` | `income_history` kurang dari 12 data |
| `500` | Model belum siap / inferensi gagal |
| `429` | Gemini API rate limit (endpoint `/chat`) |

Semua error punya format:
```json
{ "detail": "Pesan error di sini" }
```

---

## Catatan Penting

- **Model income prediction** hanya akurat di rentang **Rp 28 — Rp 1.940.300 per minggu** (range data training). Input di luar range ini akan ter-clip otomatis.
- **Endpoint `/chat`** membutuhkan `GEMINI_API_KEY` yang sudah dikonfigurasi di environment HuggingFace Space. Kalau belum ada, response akan berisi pesan error.
- Timeout yang disarankan: **30 detik** per request.
