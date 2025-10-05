import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="MVT Analyzer", layout="centered")

st.title("📈 MVT Analyzer – Hiểu toán qua dữ liệu kinh doanh")
st.markdown("""
Ứng dụng giúp bạn *hiểu Định lý Giá trị Trung bình (MVT)* thông qua *báo cáo kinh doanh thực tế*.

Chỉ cần nhập doanh thu/lợi nhuận theo thời gian, chương trình sẽ hiển thị:
- Tốc độ thay đổi trung bình giữa các kỳ (slope)
- Đánh giá định tính (tăng trưởng, giảm, biến động)
- Giải thích ý nghĩa của MVT trong dữ liệu thực tế
""")

st.header("1️⃣ Nhập dữ liệu kinh doanh")

st.markdown("Bạn có thể nhập dữ liệu thủ công hoặc tải file CSV (có cột Kỳ và Giá trị).")

uploaded_file = st.file_uploader("📂 Tải lên file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.write("Hoặc dùng dữ liệu mẫu:")
    data = {
        "Kỳ": ["Q1 2025", "Q2 2025", "Q3 2025"],
        "Giá trị": [90.1, 95.4, 94.0]
    }
    df = pd.DataFrame(data)

st.dataframe(df)

st.header("2️⃣ Tính tốc độ thay đổi trung bình (Mean Value)")

if len(df) < 2:
    st.warning("⚠️ Cần ít nhất 2 kỳ để tính toán.")
else:
    slopes, comments, periods = [], [], []

    for i in range(len(df) - 1):
        a, b = df.iloc[i, 1], df.iloc[i + 1, 1]
        slope = b - a
        slopes.append(slope)
        periods.append(f"{df.iloc[i, 0]} → {df.iloc[i + 1, 0]}")

        if slope > 0:
            comments.append("🔼 Tăng trưởng")
        elif slope < 0:
            comments.append("🔻 Suy giảm")
        else:
            comments.append("⏸ Ổn định")

    results = pd.DataFrame({
        "Khoảng thời gian": periods,
        "Tốc độ thay đổi (Δ)": slopes,
        "Nhận xét": comments
    })

    st.dataframe(results)

    st.header("3️⃣ Trực quan hóa dữ liệu và MVT")
    fig, ax = plt.subplots()
    ax.plot(df["Kỳ"], df["Giá trị"], marker="o", color="dodgerblue", label="Giá trị thực tế")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("Giá trị (tỷ USD)")
    ax.set_title("Biểu đồ doanh thu / giá trị theo thời gian")
    for i in range(len(slopes)):
        ax.text(i + 0.4, (df.iloc[i, 1] + df.iloc[i + 1, 1]) / 2, f"{slopes[i]:+.2f}", color="red")
    st.pyplot(fig)

    st.header("4️⃣ Phân tích định tính")
    avg_slope = np.mean(slopes)
    if avg_slope > 0:
        overall = "✅ Doanh nghiệp đang *tăng trưởng trung bình ổn định*."
    elif avg_slope < 0:
        overall = "⚠️ Doanh nghiệp có xu hướng *suy giảm nhẹ* trong giai đoạn này."
    else:
        overall = "ℹ️ Doanh nghiệp *ổn định, không thay đổi đáng kể*."

    st.success(overall)

    st.markdown("""
    *Giải thích theo Định lý Giá trị Trung bình (MVT):*  
    Giữa hai kỳ báo cáo liên tiếp, tồn tại ít nhất một thời điểm mà *tốc độ thay đổi tức thời* của chỉ tiêu kinh doanh  
    (ví dụ doanh thu hoặc lợi nhuận) *bằng đúng tốc độ thay đổi trung bình* đã tính ở trên.  
    Điều này có nghĩa là, trong khoảng giữa hai quý, có một giai đoạn thực tế mà công ty đang hoạt động  
    với đúng mức "động lượng" trung bình đó – phản ánh xu hướng tăng trưởng hoặc suy giảm bền vững.
    """)
