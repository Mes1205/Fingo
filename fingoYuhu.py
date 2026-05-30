import streamlit as st
import requests

API_BASE = "https://mes1205-fingo.hf.space"

st.set_page_config(
    page_title="Fingo AI",
    page_icon="💸",
    layout="centered"
)

st.title("💸 Fingo AI")
st.caption("Financial Assistant untuk Gig Workers Indonesia")

tab1, tab2 = st.tabs(["📈 Prediksi Pendapatan", "🤖 Chat dengan Fingo"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Income Prediction
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Prediksi Pendapatan 4 Minggu ke Depan")
    st.write("Masukkan pendapatan kamu 12 minggu terakhir (dalam Rupiah):")

    with st.form("income_form"):
        cols = st.columns(3)
        history = []
        for i in range(12):
            col = cols[i % 3]
            val = col.number_input(
                f"Minggu {i+1}",
                min_value=0.0,
                value=400_000.0,
                step=100_000.0,
                format="%.0f",
                key=f"week_{i}"
            )
            history.append(val)

        submitted = st.form_submit_button("🔮 Prediksi Sekarang", use_container_width=True)

    if submitted:
        with st.spinner("Fingo lagi ngitung pendapatan kamu..."):
            try:
                res = requests.post(
                    f"{API_BASE}/predict/income",
                    json={"income_history": history},
                    timeout=30
                )

                if res.status_code == 200:
                    data = res.json()
                    pred_4w    = data.get("prediction_4_weeks_ahead", [])
                    total_proj = data.get("total_projected_income", 0)
                    direction  = data.get("income_direction", "Stable")
                    avg_4w     = data.get("avg_income_last_4w", 0)

                    # Minggu +1 diambil dari array (index 0) — lebih valid & konsisten
                    pred_minggu1 = pred_4w[0] if pred_4w else 0

                    dir_icon = {"Up": "📈", "Down": "📉", "Stable": "➡️"}.get(direction, "➡️")

                    st.success("✅ Prediksi berhasil!")

                    col1, col2, col3 = st.columns(3)
                    col1.metric(
                        "Minggu +1",
                        f"Rp {pred_minggu1:,.0f}",
                        delta=f"Rp {pred_minggu1 - avg_4w:,.0f} vs rata-rata"
                    )
                    col2.metric("Total Proyeksi 4 Minggu", f"Rp {total_proj:,.0f}")
                    col3.metric("Tren Pendapatan", f"{dir_icon} {direction}")

                    if pred_4w:
                        st.write("**Proyeksi per Minggu:**")

                        import pandas as pd
                        week_labels = [f"Minggu +{i+1}" for i in range(len(pred_4w))]
                        chart_data  = pd.DataFrame(
                            {"Proyeksi (Rp)": pred_4w},
                            index=week_labels
                        )
                        st.bar_chart(chart_data)

                        week_cols = st.columns(len(pred_4w))
                        for i, (col, val) in enumerate(zip(week_cols, pred_4w)):
                            col.metric(f"Minggu +{i+1}", f"Rp {val:,.0f}")

                    st.divider()
                    if direction == "Up":
                        st.info("📈 **Tren naik!** Bagus banget! Ini saat yang tepat buat nambah dana darurat atau investasi. Jangan lupa sisihkan minimal 20% dari pendapatan ya!")
                    elif direction == "Down":
                        st.warning("📉 **Hati-hati, tren turun.** Coba review pengeluaran dan fokus ke gigs yang paling cuan. Kurangi pengeluaran non-essential dulu ya!")
                    else:
                        st.info("➡️ **Pendapatan stabil.** Mantap! Konsisten itu kunci. Coba explore gigs baru untuk ningkatin pendapatan!")

                else:
                    st.error(f"❌ Gagal prediksi! Status: {res.status_code}")
                    st.code(res.text, language="json")

            except requests.Timeout:
                st.error("⏱️ Request timeout. Server Fingo lagi sibuk, coba lagi ya!")
            except requests.ConnectionError:
                st.error("🔌 Gagal konek ke server Fingo. Pastikan API server sedang aktif.")
            except Exception as e:
                st.error(f"❌ Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Chat with Fingo
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🤖 Konsultasi Keuangan dengan Fingo")
    st.caption("Tanya apa aja soal keuangan kamu ke Fingo!")

    with st.expander("⚙️ Atur Konteks Keuangan Kamu", expanded=True):
        c1, c2 = st.columns(2)
        income = c1.number_input(
            "💰 Pendapatan Bulan Ini (Rp)",
            min_value=0.0,
            value=5_000_000.0,
            step=100_000.0,
            format="%.0f"
        )
        expense = c2.number_input(
            "💸 Pengeluaran Bulan Ini (Rp)",
            min_value=0.0,
            value=3_500_000.0,
            step=100_000.0,
            format="%.0f"
        )
        budget_remaining = income - expense
        impulsive_count  = st.slider("🛒 Jumlah Belanja Impulsif (kali)", 0, 20, 3)

        st.metric("Sisa Budget", f"Rp {budget_remaining:,.0f}",
                  delta="surplus" if budget_remaining >= 0 else "defisit")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Tanya Fingo sesuatu... (contoh: 'Gimana cara manage gaji aku bulan ini?')")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Fingo lagi mikir..."):
                try:
                    res = requests.post(
                        f"{API_BASE}/chat",
                        json={
                            "user_message": user_input,
                            "financial_context": {
                                "income"           : income,
                                "expense"          : expense,
                                "budget_remaining" : budget_remaining,
                                "impulsive_count"  : impulsive_count,
                            }
                        },
                        timeout=30
                    )

                    if res.status_code == 200:
                        reply = res.json().get("reply", "Fingo nggak bisa jawab sekarang.")
                    else:
                        reply = f"❌ Error {res.status_code}: {res.text}"

                except requests.Timeout:
                    reply = "⏱️ Fingo timeout, coba lagi ya!"
                except requests.ConnectionError:
                    reply = "🔌 Gagal konek ke Fingo AI. Server mungkin lagi istirahat."
                except Exception as e:
                    reply = f"❌ Error: {e}"

                st.write(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
