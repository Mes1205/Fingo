import os
import pickle
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(
    title="Fingo AI Inference Service",
    description="Income Prediction & GenAI Financial Assistant for Gig Workers",
    version="7.0"
)

# ── Load model package ────────────────────────────────────────────────────────
try:
    with open("fingo_deploy.pkl", "rb") as f:
        model_package = pickle.load(f)
    print("✅ fingo_deploy.pkl loaded.")
    _fc = model_package.get("feature_columns", [])
    print(f"   {len(_fc)} features | model: {type(model_package.get('sk_reg_model')).__name__}")
except Exception as e:
    model_package = None
    print(f"⚠️  Gagal load pkl: {e}")


@app.get("/")
def home():
    return {
        "status"      : "active",
        "message"     : "Fingo AI Inference Service is running!",
        "engineer"    : "Martha Meslina Florencia - AI Engineer 2",
        "model_loaded": model_package is not None,
    }


@app.get("/debug/model_info")
def debug_model_info():
    if model_package is None:
        raise HTTPException(status_code=500, detail="Model tidak loaded.")
    feature_cols       = model_package.get("feature_columns", [])
    income_cols_scaled = model_package.get("income_cols_scaled", [])
    income_scaler      = model_package.get("income_scaler", None)
    local_ts           = model_package.get("local_target_scaler", None)
    sk_reg             = model_package.get("sk_reg_model", None)
    income_scaler_range = {}
    if income_scaler is not None and hasattr(income_scaler, "data_min_"):
        for col, mn, mx in zip(income_cols_scaled,
                               income_scaler.data_min_, income_scaler.data_max_):
            income_scaler_range[col] = {"min": float(mn), "max": float(mx)}
    target_range = {}
    if local_ts is not None and hasattr(local_ts, "data_min_"):
        target_range = {
            "income_min_idr": float(local_ts.data_min_[0]),
            "income_max_idr": float(local_ts.data_max_[0]),
        }
    return {
        "n_feature_columns"  : len(feature_cols),
        "feature_columns"    : feature_cols,
        "income_cols_scaled" : income_cols_scaled,
        "income_scaler_range": income_scaler_range,
        "target_scaler_range": target_range,
        "sk_reg_model_type"  : type(sk_reg).__name__ if sk_reg else None,
    }


# ── Schema ────────────────────────────────────────────────────────────────────
class SimpleIncomeInput(BaseModel):
    income_history       : List[float]   # ← sekarang cukup 4 minggu
    usia                 : float = 27.0
    hari_kerja_per_minggu: float = 5.0
    jam_kerja_per_hari   : float = 8.0
    gig_ojek_online      : float = 1.0
    gig_kurir            : float = 0.0
    gig_jualan_online    : float = 0.0
    gig_freelance_desain : float = 0.0
    gig_freelance_it     : float = 0.0
    gig_content_creator  : float = 0.0
    gig_tutor            : float = 0.0
    gig_pekerja_harian   : float = 0.0

class FeatureInput(BaseModel):
    features: Dict[str, float]

class FinancialContext(BaseModel):
    income          : float
    expense         : float
    budget_remaining: float
    impulsive_count : int

class ChatRequest(BaseModel):
    user_message     : str
    financial_context: FinancialContext


def _mm_scale(val: float, mn: float, mx: float) -> float:
    return float(np.clip((val - mn) / (mx - mn + 1e-8), 0.0, 1.0))


def build_X(data: SimpleIncomeInput) -> np.ndarray:
    """
    Build 59-feature vector dari INPUT 4 MINGGU.
    Adjustment untuk 4 minggu (vs 12 minggu sebelumnya):
      - rolling_mean_8w / rolling_std_8w → proxy pakai rolling_mean_4w / rolling_std_4w
        karena hanya 4 titik tersedia (8w info tidak ada)
      - income_trend_4w_abs = h[-1] - h[-4]  (bukan h[-1] - h[-5])
      - income_trend_4w_pct = trend_4w_abs / h[-4]
      - lag_4_income = h[-4] = h[0] (minggu paling lama dari 4 input)
      - lag_3/2/current: h[-3], h[-2], h[-1] → tetap valid
      - Semua fitur yang tidak bergantung window > 4w → tidak berubah
    """
    pkg = model_package
    feature_cols       = pkg.get("feature_columns", [])
    income_cols_scaled = pkg.get("income_cols_scaled", [])
    income_scaler      = pkg.get("income_scaler", None)

    if income_scaler is not None and hasattr(income_scaler, "data_min_"):
        sc_min = {col: float(mn) for col, mn in zip(income_cols_scaled, income_scaler.data_min_)}
        sc_max = {col: float(mx) for col, mx in zip(income_cols_scaled, income_scaler.data_max_)}
    else:
        sc_min = {"current_income": 28, "lag_2_income": 28,
                  "lag_3_income": 28, "bps_jasa_weekly": 390946}
        sc_max = {"current_income": 1940300, "lag_2_income": 1940300,
                  "lag_3_income": 1940300, "bps_jasa_weekly": 692983}

    # Ambil 4 titik terakhir — kalau ada lebih tetap jalan
    h     = np.array(data.income_history[-4:], dtype=np.float64)
    roll4 = h                     # semua 4 titik = roll4
    roll2 = h[-2:]

    # ── Trend & change ────────────────────────────────────────────────────────
    last_change_abs = float(h[-1] - h[-2])
    last_change_pct = float(last_change_abs / (abs(h[-2]) + 1e-8))

    # FIX 1: trend_4w pakai h[-1] - h[-4] = h[-1] - h[0] (range 4 titik)
    trend_4w_abs    = float(h[-1] - h[0])
    trend_4w_pct    = float(trend_4w_abs / (abs(h[0]) + 1e-8))

    x4              = np.arange(4, dtype=np.float64)
    slope_4w        = float(np.polyfit(x4, roll4, 1)[0]) if len(set(h.tolist())) > 1 else 0.0
    prev_change     = float(h[-2] - h[-3])

    # ── Profil user ───────────────────────────────────────────────────────────
    experience_months = data.usia * 12 * 0.3
    exp_log           = float(np.log1p(experience_months))
    total_jam         = data.hari_kerja_per_minggu * data.jam_kerja_per_hari
    pref_payday       = 0.4 if data.gig_ojek_online or data.gig_kurir else 0.2
    pref_weekend      = 0.5 if data.gig_ojek_online or data.gig_kurir else 0.3

    feature_map = {
        # Temporal
        "target_idx"               : 4.0,    # 4 minggu observed
        "target_month"             : 6.0,
        "target_week_of_month"     : 2.0,
        "target_quarter"           : 2.0,
        "target_is_month_start"    : 0.0,
        "target_is_month_end"      : 0.0,
        "target_is_payday_period"  : 1.0,
        "target_is_weekend"        : 0.0,
        "target_is_ramadan_lebaran": 0.0,
        "target_is_harbolnas"      : 0.0,
        "target_is_christmas_year_end": 0.0,
        "target_is_new_year"       : 0.0,
        # Income — MinMaxScaled
        "current_income"           : _mm_scale(h[-1], sc_min["current_income"], sc_max["current_income"]),
        "lag_2_income"             : _mm_scale(h[-2], sc_min["lag_2_income"],   sc_max["lag_2_income"]),
        "lag_3_income"             : _mm_scale(h[-3], sc_min["lag_3_income"],   sc_max["lag_3_income"]),
        # FIX 2: lag_4 = h[0] (minggu pertama dari 4 input) — RAW
        "lag_4_income"             : float(h[0]),
        # Rolling 4w — dari 4 titik input
        "rolling_mean_4w"          : float(np.mean(roll4)),
        "rolling_std_4w"           : float(np.std(roll4) + 1e-8),
        "rolling_min_4w"           : float(np.min(roll4)),
        "rolling_max_4w"           : float(np.max(roll4)),
        "rolling_range_4w"         : float(np.max(roll4) - np.min(roll4)),
        "rolling_median_4w"        : float(np.median(roll4)),
        "rolling_cv_4w"            : float(np.std(roll4) / (np.mean(roll4) + 1e-8)),
        "rolling_last_vs_median_pct": float((h[-1] - np.median(roll4)) / (np.median(roll4) + 1e-8)),
        "rolling_mean_2w"          : float(np.mean(roll2)),
        # FIX 3: 8w features → proxy dengan 4w (tidak ada data 8w)
        "rolling_mean_8w"          : float(np.mean(roll4)),   # proxy
        "rolling_std_8w"           : float(np.std(roll4) + 1e-8),  # proxy
        # Trend
        "income_trend_4w_abs"      : trend_4w_abs,
        "income_trend_4w_pct"      : trend_4w_pct,
        "last_income_change_abs"   : last_change_abs,
        "last_income_change_pct"   : last_change_pct,
        "income_growth_1w"         : last_change_pct,
        "income_volatility"        : float(np.std(h) / (np.mean(h) + 1e-8)),
        "trend_slope_4w"           : slope_4w,
        "is_previous_week_up"      : 1.0 if prev_change > 0 else 0.0,
        "is_previous_week_down"    : 1.0 if prev_change < 0 else 0.0,
        "is_previous_week_stable"  : 1.0 if prev_change == 0 else 0.0,
        "lag_ratio_1_to_mean"      : float(h[-1] / (np.mean(h) + 1e-8)),
        # Profil user
        "usia"                     : data.usia,
        "experience_months_log"    : exp_log,
        "hari_kerja_per_minggu"    : data.hari_kerja_per_minggu,
        "jam_kerja_per_hari"       : data.jam_kerja_per_hari,
        "total_jam_seminggu"       : total_jam,
        "bps_jasa_weekly"          : _mm_scale(0.0, sc_min["bps_jasa_weekly"], sc_max["bps_jasa_weekly"]),
        # Preferences
        "pref_awal_bulan"          : 0.3,
        "pref_payday"              : pref_payday,
        "pref_weekend"             : pref_weekend,
        "pref_ramadan_lebaran"     : 0.0,
        "pref_natal_tahun_baru"    : 0.0,
        "pref_harbolnas"           : 0.0,
        "pref_promo_aplikasi"      : 0.3,
        # Tipe gig
        "gig_ojek_online"          : data.gig_ojek_online,
        "gig_kurir"                : data.gig_kurir,
        "gig_jualan_online"        : data.gig_jualan_online,
        "gig_freelance_desain"     : data.gig_freelance_desain,
        "gig_freelance_it"         : data.gig_freelance_it,
        "gig_content_creator"      : data.gig_content_creator,
        "gig_tutor"                : data.gig_tutor,
        "gig_pekerja_harian"       : data.gig_pekerja_harian,
    }

    X = np.array(
        [feature_map.get(col, 0.0) for col in feature_cols],
        dtype=np.float64
    ).reshape(1, -1)

    return X


@app.post("/predict/income")
def predict_income(data: SimpleIncomeInput):
    if model_package is None:
        raise HTTPException(status_code=500, detail="Model pkl belum siap.")

    # ← minimum sekarang 4, bukan 12
    if len(data.income_history) < 4:
        raise HTTPException(status_code=400,
            detail=f"Butuh minimal 4 minggu. Diterima: {len(data.income_history)}.")

    try:
        sk_reg = model_package.get("sk_reg_model")
        sk_cls = model_package.get("sk_cls_model")

        if sk_reg is None:
            raise HTTPException(status_code=500, detail="sk_reg_model tidak ada di pkl.")

        X = build_X(data)

        pred_log = float(sk_reg.predict(X)[0])
        pred_idr = float(max(np.expm1(pred_log), 0))

        direction       = "Stable"
        direction_proba = {"Down": 0.33, "Stable": 0.34, "Up": 0.33}
        if sk_cls is not None:
            try:
                proba     = sk_cls.predict_proba(X)[0]
                direction = {0: "Down", 1: "Stable", 2: "Up"}[int(np.argmax(proba))]
                direction_proba = {
                    "Down"  : round(float(proba[0]), 4),
                    "Stable": round(float(proba[1]), 4),
                    "Up"    : round(float(proba[2]), 4),
                }
            except Exception:
                pass

        h          = np.array(data.income_history[-4:])
        avg_recent = float(np.mean(h))

        # Proyeksi 4 minggu: extend window dengan slope historis
        future_4_weeks = []
        h_extended = list(h)
        slope = float(np.polyfit(np.arange(len(h)), h, 1)[0])
        for _ in range(4):
            h_extended.append(h_extended[-1] + slope)
            data_ext = SimpleIncomeInput(
                income_history       = h_extended[-4:],
                usia                 = data.usia,
                hari_kerja_per_minggu= data.hari_kerja_per_minggu,
                jam_kerja_per_hari   = data.jam_kerja_per_hari,
                gig_ojek_online      = data.gig_ojek_online,
                gig_kurir            = data.gig_kurir,
                gig_jualan_online    = data.gig_jualan_online,
                gig_freelance_desain = data.gig_freelance_desain,
                gig_freelance_it     = data.gig_freelance_it,
                gig_content_creator  = data.gig_content_creator,
                gig_tutor            = data.gig_tutor,
                gig_pekerja_harian   = data.gig_pekerja_harian,
            )
            X_r   = build_X(data_ext)
            p_r   = float(max(np.expm1(sk_reg.predict(X_r)[0]), 0))
            future_4_weeks.append(round(p_r, 2))

        return {
            "status"                  : "success",
            "input_weeks_received"    : len(data.income_history),
            "prediction_next_week"    : round(pred_idr, 2),
            "prediction_4_weeks_ahead": future_4_weeks,
            "total_projected_income"  : round(sum(future_4_weeks), 2),
            "income_direction"        : direction,
            "direction_proba"         : direction_proba,
            "avg_income_last_4w"      : round(avg_recent, 2),
            "_debug_pred_log"         : round(pred_log, 4),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inferensi gagal: {str(e)}")


@app.post("/predict/full")
def predict_income_full(data: FeatureInput):
    if model_package is None:
        raise HTTPException(status_code=500, detail="Model pkl belum siap.")
    try:
        feature_cols = model_package.get("feature_columns", [])
        sk_reg = model_package.get("sk_reg_model")
        sk_cls = model_package.get("sk_cls_model")
        X = np.array([data.features.get(col, 0.0) for col in feature_cols],
                     dtype=np.float64).reshape(1, -1)
        pred_idr = float(max(np.expm1(sk_reg.predict(X)[0]), 0))
        direction, direction_proba = "Stable", {"Down": 0.33, "Stable": 0.34, "Up": 0.33}
        if sk_cls:
            proba = sk_cls.predict_proba(X)[0]
            direction = {0: "Down", 1: "Stable", 2: "Up"}[int(np.argmax(proba))]
            direction_proba = {"Down": round(float(proba[0]), 4),
                               "Stable": round(float(proba[1]), 4),
                               "Up": round(float(proba[2]), 4)}
        return {"status": "success", "next_week_income_idr": round(pred_idr, 2),
                "income_direction": direction, "direction_proba": direction_proba}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inferensi gagal: {str(e)}")


@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        return {"reply": "Error: GEMINI_API_KEY belum dikonfigurasi di HuggingFace Space settings."}

    system_prompt = f"""Kamu adalah Fingo, AI Financial Consultant cerdas dan ramah yang membantu Gig Workers Indonesia mengatur keuangan mereka.
Gunakan bahasa Indonesia yang santai, kasual, dan informatif. Pakai istilah seperti 'hype', 'budgeting', 'saving', 'aman cuy', 'cuan', 'boncos' bila relevan, tapi tetap solutif dan taktis.
Konteks Finansial Pengguna:
- Total Pendapatan Bulan Ini : Rp {request.financial_context.income:,.0f}
- Total Pengeluaran Bulan Ini: Rp {request.financial_context.expense:,.0f}
- Sisa Anggaran (Budget)    : Rp {request.financial_context.budget_remaining:,.0f}
- Belanja Impulsif Terdeteksi: {request.financial_context.impulsive_count} kali
INSTRUKSI:
- Jawab dalam bahasa Indonesia yang santai dan ramah
- Maksimal 3 paragraf pendek
- WAJIB selesaikan kalimat dengan sempurna, jangan putus di tengah kalimat
- Akhiri dengan 1 tips actionable konkret
- Langsung ke poin, jangan sapaan panjang"""

    payload = {
        "contents": [{"role": "user",
                      "parts": [{"text": f"{system_prompt}\n\nPertanyaan: {request.user_message}"}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 10024}
    }
    try:
        resp = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": GEMINI_API_KEY},
            json=payload, timeout=20
        )
        if resp.status_code == 200:
            candidate = resp.json()["candidates"][0]
            reply = candidate["content"]["parts"][0]["text"]
            if candidate.get("finishReason") == "MAX_TOKENS":
                reply += "..."
            return {"reply": reply.strip()}
        elif resp.status_code == 429:
            return {"reply": "Fingo lagi overloaded (rate limit). Tunggu sebentar ya!"}
        else:
            return {"reply": f"Gemini API error {resp.status_code}. Coba lagi ya!"}
    except requests.Timeout:
        return {"reply": "Timeout. Coba lagi ya!"}
    except Exception as e:
        return {"reply": f"Error: {str(e)}"}
