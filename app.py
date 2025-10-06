import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="MVT Analyzer - Detailed MVT Steps", layout="wide")

st.title("📈 MVT Analyzer – Giải thích chi tiết theo Định lý Giá trị Trung bình (MVT)")
st.markdown("""
Ứng dụng này sẽ:
- Tính *tốc độ thay đổi trung bình* giữa các kỳ (slope),
- Ước lượng *tốc độ thay đổi tức thời* (đạo hàm xấp xỉ) tại từng điểm,
- Tìm *điểm MVT ước lượng* cho mỗi đoạn bằng cách nội suy giữa các đạo hàm,
- Hiển thị *bảng số* và *giải thích từng bước* cho mỗi đoạn, cùng biểu đồ minh họa.
""")

# --- Input dữ liệu ---
st.header("1. Nhập dữ liệu")
st.markdown("Upload CSV (cột Kỳ, Giá trị) hoặc dùng dữ liệu mẫu để thử.")

uploaded_file = st.file_uploader("Tải lên file CSV (Kỳ, Giá trị)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.write("Dữ liệu mẫu (công ty giả định):")
    sample = {
        "Kỳ": ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024", "Q1 2025", "Q2 2025"],
        "Giá trị": [120.5, 138.2, 142.8, 135.6, 150.3, 155.9]
    }
    df = pd.DataFrame(sample)

# sanitize
if "Giá trị" not in df.columns or "Kỳ" not in df.columns:
    st.error("CSV phải có cột 'Kỳ' và 'Giá trị'.")
    st.stop()

df["Kỳ"] = df["Kỳ"].astype(str)
df["Giá trị"] = pd.to_numeric(df["Giá trị"], errors="coerce")
if df["Giá trị"].isna().any():
    st.warning("Có giá trị không phải số trong cột 'Giá trị' — những hàng đó sẽ bị bỏ.")
    df = df.dropna(subset=["Giá trị"]).reset_index(drop=True)

st.dataframe(df)

n = len(df)
if n < 2:
    st.warning("Cần ít nhất 2 kỳ để phân tích.")
    st.stop()

# --- Tính slope giữa từng cặp ---
st.header("2. Tính toán cơ bản")
t = np.arange(n)  # đơn vị thời gian giả định đều (mỗi kỳ = 1)
y = df["Giá trị"].to_numpy()

slopes = np.diff(y) / np.diff(t)  # dt = 1 -> chỉ là diff
periods = [f"{df.loc[i,'Kỳ']} → {df.loc[i+1,'Kỳ']}" for i in range(n-1)]

# hiển thị bảng slopes
slopes_df = pd.DataFrame({
    "Khoảng thời gian": periods,
    "Giá trị tại a": [y[i] for i in range(n-1)],
    "Giá trị tại b": [y[i+1] for i in range(n-1)],
    "Slope (Δ = b - a)": slopes
})
st.subheader("Slope (tốc độ thay đổi trung bình) giữa các kỳ")
st.dataframe(slopes_df.style.format({"Slope (Δ = b - a)": "{:+.3f}"}))

# --- Ước lượng đạo hàm tại mỗi điểm (forward/backward/central) ---
deriv = np.zeros(n)
if n == 2:
    # trivial: forward/backward same
    deriv[0] = slopes[0]
    deriv[1] = slopes[0]
else:
    deriv[0] = (y[1] - y[0]) / (t[1] - t[0])  # forward diff
    deriv[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])  # backward diff
    for i in range(1, n-1):
        deriv[i] = (y[i+1] - y[i-1]) / (t[i+1] - t[i-1])  # central difference

deriv_df = pd.DataFrame({
    "Kỳ": df["Kỳ"],
    "Giá trị": df["Giá trị"],
    "Đạo hàm xấp xỉ f'(t) (tốc độ tức thời)": deriv
})
st.subheader("Đạo hàm xấp xỉ tại từng điểm (tốc độ tức thời)")
st.dataframe(deriv_df.style.format({"Đạo hàm xấp xỉ f'(t) (tốc độ tức thời)": "{:+.3f}"}))

# --- Phân tích MVT cho từng đoạn ---
st.header("2. Phân tích MVT – từng bước cho mỗi đoạn")

records = []
plot_mvt_points = []  # (c, y_c, slope, segment_index)
for i in range(n - 1):
    a_idx, b_idx = i, i + 1
    a_label, b_label = df.loc[a_idx, "Kỳ"], df.loc[b_idx, "Kỳ"]
    a_val, b_val = float(y[a_idx]), float(y[b_idx])
    slope = float(slopes[i])

    # đạo hàm tại hai đầu điểm (ước lượng)
    deriv_a = float(deriv[a_idx])
    deriv_b = float(deriv[b_idx])

    # B1: Tính slope (đã có)
    # B2: Tính đạo hàm tại đầu và cuối (deriv_a, deriv_b)
    # B3: Kiểm tra điều kiện trung gian: slope nằm giữa deriv_a và deriv_b?
    bracket = (min(deriv_a, deriv_b) <= slope <= max(deriv_a, deriv_b))

    # B4: ước lượng vị trí c trong (i, i+1)
    if deriv_b != deriv_a and bracket:
        # nội suy tuyến tính để tìm c fractional trong (i, i+1)
        frac = (slope - deriv_a) / (deriv_b - deriv_a)  # t position fraction within interval
        c = i + frac
        method = "internal linear interpolation between derivative estimates"
    elif deriv_b == deriv_a and abs(deriv_a - slope) < 1e-9 and bracket:
        c = i + 0.5
        method = "derivatives equal to slope (any point works) -> choose midpoint"
    else:
        # không bracket: chọn điểm có derivative gần nhất (i hoặc i+1)
        if abs(deriv_a - slope) <= abs(deriv_b - slope):
            c = float(i)
            method = "no sign change -> choose left endpoint (closest derivative)"
        else:
            c = float(i + 1)
            method = "no sign change -> choose right endpoint (closest derivative)"

    # nội suy giá trị hàm tại c bằng nội suy tuyến tính giữa a và b
    y_c = a_val + (c - i) * (b_val - a_val)  # since t spacing = 1

    # đạo hàm ước lượng tại c (linear interp between deriv_a and deriv_b)
    deriv_c = deriv_a + ( (c - i) * (deriv_b - deriv_a) )

    residual = abs(deriv_c - slope)

    records.append({
        "Segment": f"{a_label} → {b_label}",
        "a_label": a_label,
        "b_label": b_label,
        "a_val": a_val,
        "b_val": b_val,
        "slope": slope,
        "deriv_a": deriv_a,
        "deriv_b": deriv_b,
        "bracket": bracket,
        "c_pos": c,
        "c_loc_text": f"{(a_label)} -- {(b_label)} (c ≈ {c:.3f})" if (i != int(c)) else f"{df.loc[int(c),'Kỳ']} (exact index)",
        "y_c": y_c,
        "deriv_c": deriv_c,
        "residual": residual,
        "method": method
    })

    plot_mvt_points.append((c, y_c, slope, i))

mvt_table = pd.DataFrame(records)
# hiển thị bảng tóm tắt
st.subheader("Bảng tóm tắt MVT ước lượng cho từng đoạn")
display_cols = ["Segment", "a_val", "b_val", "slope", "deriv_a", "deriv_b", "y_c", "deriv_c"]
st.dataframe(mvt_table[display_cols].rename(columns={
    "a_val": "Giá trị a",
    "b_val": "Giá trị b",
    "slope": "Slope (Δ)",
    "deriv_a": "Đạo hàm tại a",
    "deriv_b": "Đạo hàm tại b",
    "y_c": "Giá trị f(c) ước lượng",
    "deriv_c": "Đạo hàm ước lượng tại c",

}).style.format({
    "Giá trị a": "{:.3f}",
    "Giá trị b": "{:.3f}",
    "Slope (Δ)": "{:+.3f}",
    "Đạo hàm tại a": "{:+.3f}",
    "Đạo hàm tại b": "{:+.3f}",
    "Giá trị f(c) ước lượng": "{:.3f}",
    "Đạo hàm ước lượng tại c": "{:+.3f}",

}))

# --- Chi tiết từng bước: dùng expander cho mỗi đoạn ---
st.header("2. Giải thích chi tiết theo từng bước (cho mỗi đoạn)")
#tinh slope giua tung cap#
t = np.arange(n)  # đơn vị thời gian giả định đều (mỗi kỳ = 1)
y = df["Giá trị"].to_numpy()

slopes = np.diff(y) / np.diff(t)  # dt = 1 -> chỉ là diff
periods = [f"{df.loc[i,'Kỳ']} → {df.loc[i+1,'Kỳ']}" for i in range(n-1)]

# --- Ước lượng đạo hàm tại mỗi điểm (forward/backward/central) ---
deriv = np.zeros(n)
if n == 2:
    # trivial: forward/backward same
    deriv[0] = slopes[0]
    deriv[1] = slopes[0]
else:
    deriv[0] = (y[1] - y[0]) / (t[1] - t[0])  # forward diff
    deriv[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])  # backward diff
    for i in range(1, n-1):
        deriv[i] = (y[i+1] - y[i-1]) / (t[i+1] - t[i-1])  # central difference


for rec in records:
    seg = rec["Segment"]
    with st.expander(f"Giải thích: {seg}", expanded=False):
        st.markdown(f"*Bước 1 – Dữ liệu đầu vào:* a = {rec['a_label']} = {rec['a_val']:.3f}, b = {rec['b_label']} = {rec['b_val']:.3f}")
        st.markdown(f"*Bước 2 – Tính tốc độ thay đổi trung bình (slope):*")
        st.markdown(f"> slope = (f(b) - f(a)) / (t_b - t_a) = ({rec['b_val']:.3f} - {rec['a_val']:.3f}) / 1 = *{rec['slope']:+.3f}*")
        st.markdown(f"*Bước 3 – Ước lượng đạo hàm tại hai đầu đoạn:*")
        st.markdown(f"> f'({rec['a_label']}) ≈ {rec['deriv_a']:+.3f},  f'({rec['b_label']}) ≈ {rec['deriv_b']:+.3f}")
        st.markdown(f"*Bước 4 – Kiểm tra tính trung gian (Intermediate Value):*")
        if rec["bracket"]:
            st.markdown(f"> Vì slope = *{rec['slope']:+.3f}* nằm giữa f'({rec['a_label']}) và f'({rec['b_label']}), theo tính chất trung gian có tồn tại c ∈ ({rec['a_label']}, {rec['b_label']}).")
            st.markdown(f"*Bước 5 – Tìm c ước lượng:* phương pháp: {rec['method']}")
            st.markdown(f"> Vị trí ước lượng c ≈ *{rec['c_pos']:.3f}* (tức nằm cách {rec['a_label']} khoảng {(rec['c_pos']-int(rec['c_pos'])):.3f} phần của khoảng đến {rec['b_label']}).")
            st.markdown(f"> Ước lượng f(c) bằng nội suy tuyến tính ≈ *{rec['y_c']:.3f}*.")
            st.markdown(f"> Ước lượng f'(c) bằng nội suy giữa f'({rec['a_label']}) và f'({rec['b_label']}) ≈ *{rec['deriv_c']:+.3f}*.")
            st.markdown(f"> Sai khác |f'(c) - slope| = *{rec['residual']:.3e}* (mong muốn nhỏ).")
        else:
            st.markdown(f"> slope = *{rec['slope']:+.3f}* không nằm giữa f'({rec['a_label']}) và f'({rec['b_label']}).")
            st.markdown(f"> Chúng tôi chọn điểm có f' gần nhất theo phương pháp: {rec['method']}, ước lượng c = *{rec['c_pos']:.3f}*.")
            st.markdown(f"> Giá trị f(c) ước lượng = *{rec['y_c']:.3f}, f'(c) ≈ *{rec['deriv_c']:+.3f}*, sai khác = *{rec['residual']:.3e}**.")
        # Interpretation
        st.markdown("*Diễn giải ý nghĩa kinh doanh (gợi ý):*")
        if rec["slope"] > 0:
            st.markdown("- slope > 0: doanh thu (hoặc chỉ tiêu) tăng trung bình trong khoảng này.")
            st.markdown(f"- Điểm c ước lượng cho thấy giai đoạn mà doanh nghiệp có tốc độ tăng đúng bằng tốc độ trung bình (thời điểm có 'momentum' nhất).")
        elif rec["slope"] < 0:
            st.markdown("- slope < 0: doanh thu giảm trung bình trong khoảng này.")
            st.markdown(f"- Điểm c ước lượng có thể là thời điểm cảnh báo quản trị (nên xem chi tiết nguyên nhân tại khoảng này).")
        else:
            st.markdown("- slope = 0: không thay đổi tổng thể trong khoảng này.")
        st.markdown("---")

# --- Vẽ biểu đồ với điểm c và tiếp tuyến ước lượng ---
st.header("4. Biểu đồ minh họa (các điểm MVT & tiếp tuyến ước lượng)")
fig, ax = plt.subplots(figsize=(10, 4))
x = t
ax.plot(x, y, marker="o", linestyle="-", label="Giá trị thực tế")
# vẽ các secant line
for i in range(n-1):
    ax.plot([i, i+1], [y[i], y[i+1]], color="gray", linestyle="--", alpha=0.6)
# đánh dấu các điểm c và vẽ tiếp tuyến (tangent) ước lượng
colors = ["orange", "red", "green", "purple", "brown", "cyan"]
for idx, (c, y_c, slope, seg_i) in enumerate(plot_mvt_points):
    ax.scatter(c, y_c, color=colors[idx % len(colors)], s=80, zorder=5)
    # vẽ tiếp tuyến khoảng nhỏ quanh c
    x_line = np.linspace(max(0, c-0.8), min(n-1, c+0.8), 50)
    y_line = y_c + slope * (x_line - c)
    ax.plot(x_line, y_line, color=colors[idx % len(colors)], linestyle='-', linewidth=1.5, alpha=0.8,
            label=f"Tangent approx seg {seg_i} ({slope:+.2f})")
    # chú thích
    ax.text(c, y_c, f" c≈{c:.2f}", fontsize=8, verticalalignment="bottom")

ax.set_xticks(x)
ax.set_xticklabels(df["Kỳ"], rotation=30)
ax.set_xlabel("Kỳ")
ax.set_ylabel("Giá trị")
ax.set_title("Dữ liệu & các điểm MVT ước lượng (với các tiếp tuyến ước lượng)")
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
st.pyplot(fig)

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

# 🔹 Tổng kết định tính toàn giai đoạn dựa trên slope trung bình
avg_slope = np.mean(slopes)
if avg_slope > 0:
    overall = "✅ Doanh nghiệp đang tăng trưởng trung bình ổn định."
elif avg_slope < 0:
    overall = "⚠️ Doanh nghiệp có xu hướng suy giảm nhẹ trong giai đoạn này."
else:
    overall = "ℹ️ Doanh nghiệp ổn định, không thay đổi đáng kể."

st.success(overall)

# --- Tính slope giữa từng cặp ---
st.header("2. Tính toán cơ bản")
t = np.arange(n)  # đơn vị thời gian giả định đều (mỗi kỳ = 1)
y = df["Giá trị"].to_numpy()

slopes = np.diff(y) / np.diff(t)  # dt = 1 -> chỉ là diff
periods = [f"{df.loc[i,'Kỳ']} → {df.loc[i+1,'Kỳ']}" for i in range(n-1)]

# hiển thị bảng slopes
slopes_df = pd.DataFrame({
    "Khoảng thời gian": periods,
    "Giá trị tại a": [y[i] for i in range(n-1)],
    "Giá trị tại b": [y[i+1] for i in range(n-1)],
    "Slope (Δ = b - a)": slopes
})
st.subheader("Slope (tốc độ thay đổi trung bình) giữa các kỳ")
st.dataframe(slopes_df.style.format({"Slope (Δ = b - a)": "{:+.3f}"}))

# --- Ước lượng đạo hàm tại mỗi điểm (forward/backward/central) ---
deriv = np.zeros(n)
if n == 2:
    # trivial: forward/backward same
    deriv[0] = slopes[0]
    deriv[1] = slopes[0]
else:
    deriv[0] = (y[1] - y[0]) / (t[1] - t[0])  # forward diff
    deriv[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])  # backward diff
    for i in range(1, n-1):
        deriv[i] = (y[i+1] - y[i-1]) / (t[i+1] - t[i-1])  # central difference

deriv_df = pd.DataFrame({
    "Kỳ": df["Kỳ"],
    "Giá trị": df["Giá trị"],
    "Đạo hàm xấp xỉ f'(t) (tốc độ tức thời)": deriv
})
st.subheader("Đạo hàm xấp xỉ tại từng điểm (tốc độ tức thời)")
st.dataframe(deriv_df.style.format({"Đạo hàm xấp xỉ f'(t) (tốc độ tức thời)": "{:+.3f}"}))

# --- Phân tích MVT cho từng đoạn ---
st.header("2. Phân tích MVT – từng bước cho mỗi đoạn")

records = []
plot_mvt_points = []  # (c, y_c, slope, segment_index)
for i in range(n - 1):
    a_idx, b_idx = i, i + 1
    a_label, b_label = df.loc[a_idx, "Kỳ"], df.loc[b_idx, "Kỳ"]
    a_val, b_val = float(y[a_idx]), float(y[b_idx])
    slope = float(slopes[i])

    # đạo hàm tại hai đầu điểm (ước lượng)
    deriv_a = float(deriv[a_idx])
    deriv_b = float(deriv[b_idx])

    # B1: Tính slope (đã có)
    # B2: Tính đạo hàm tại đầu và cuối (deriv_a, deriv_b)
    # B3: Kiểm tra điều kiện trung gian: slope nằm giữa deriv_a và deriv_b?
    bracket = (min(deriv_a, deriv_b) <= slope <= max(deriv_a, deriv_b))

    # B4: ước lượng vị trí c trong (i, i+1)
    if deriv_b != deriv_a and bracket:
        # nội suy tuyến tính để tìm c fractional trong (i, i+1)
        frac = (slope - deriv_a) / (deriv_b - deriv_a)  # t position fraction within interval
        c = i + frac
        method = "internal linear interpolation between derivative estimates"
    elif deriv_b == deriv_a and abs(deriv_a - slope) < 1e-9 and bracket:
        c = i + 0.5
        method = "derivatives equal to slope (any point works) -> choose midpoint"
    else:
        # không bracket: chọn điểm có derivative gần nhất (i hoặc i+1)
        if abs(deriv_a - slope) <= abs(deriv_b - slope):
            c = float(i)
            method = "no sign change -> choose left endpoint (closest derivative)"
        else:
            c = float(i + 1)
            method = "no sign change -> choose right endpoint (closest derivative)"

    # nội suy giá trị hàm tại c bằng nội suy tuyến tính giữa a và b
    y_c = a_val + (c - i) * (b_val - a_val)  # since t spacing = 1

    # đạo hàm ước lượng tại c (linear interp between deriv_a and deriv_b)
    deriv_c = deriv_a + ( (c - i) * (deriv_b - deriv_a) )

    residual = abs(deriv_c - slope)

    records.append({
        "Segment": f"{a_label} → {b_label}",
        "a_label": a_label,
        "b_label": b_label,
        "a_val": a_val,
        "b_val": b_val,
        "slope": slope,
        "deriv_a": deriv_a,
        "deriv_b": deriv_b,
        "bracket": bracket,
        "c_pos": c,
        "c_loc_text": f"{(a_label)} -- {(b_label)} (c ≈ {c:.3f})" if (i != int(c)) else f"{df.loc[int(c),'Kỳ']} (exact index)",
        "y_c": y_c,
        "deriv_c": deriv_c,
        "residual": residual,
        "method": method
    })

    plot_mvt_points.append((c, y_c, slope, i))

mvt_table = pd.DataFrame(records)
# hiển thị bảng tóm tắt
st.subheader("Bảng tóm tắt MVT ước lượng cho từng đoạn")
display_cols = ["Segment", "a_val", "b_val", "slope", "deriv_a", "deriv_b", "y_c", "deriv_c"]
st.dataframe(mvt_table[display_cols].rename(columns={
    "a_val": "Giá trị a",
    "b_val": "Giá trị b",
    "slope": "Slope (Δ)",
    "deriv_a": "Đạo hàm tại a",
    "deriv_b": "Đạo hàm tại b",
    "y_c": "Giá trị f(c) ước lượng",
    "deriv_c": "Đạo hàm ước lượng tại c",

}).style.format({
    "Giá trị a": "{:.3f}",
    "Giá trị b": "{:.3f}",
    "Slope (Δ)": "{:+.3f}",
    "Đạo hàm tại a": "{:+.3f}",
    "Đạo hàm tại b": "{:+.3f}",
    "Giá trị f(c) ước lượng": "{:.3f}",
    "Đạo hàm ước lượng tại c": "{:+.3f}",

}))

# --- Chi tiết từng bước: dùng expander cho mỗi đoạn ---
st.header("2. Giải thích chi tiết theo từng bước (cho mỗi đoạn)")
#tinh slope giua tung cap#
t = np.arange(n)  # đơn vị thời gian giả định đều (mỗi kỳ = 1)
y = df["Giá trị"].to_numpy()

slopes = np.diff(y) / np.diff(t)  # dt = 1 -> chỉ là diff
periods = [f"{df.loc[i,'Kỳ']} → {df.loc[i+1,'Kỳ']}" for i in range(n-1)]

# --- Ước lượng đạo hàm tại mỗi điểm (forward/backward/central) ---
deriv = np.zeros(n)
if n == 2:
    # trivial: forward/backward same
    deriv[0] = slopes[0]
    deriv[1] = slopes[0]
else:
    deriv[0] = (y[1] - y[0]) / (t[1] - t[0])  # forward diff
    deriv[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])  # backward diff
    for i in range(1, n-1):
        deriv[i] = (y[i+1] - y[i-1]) / (t[i+1] - t[i-1])  # central difference


for rec in records:
    seg = rec["Segment"]
    with st.expander(f"Giải thích: {seg}", expanded=False):
        st.markdown(f"*Bước 1 – Dữ liệu đầu vào:* a = {rec['a_label']} = {rec['a_val']:.3f}, b = {rec['b_label']} = {rec['b_val']:.3f}")
        st.markdown(f"*Bước 2 – Tính tốc độ thay đổi trung bình (slope):*")
        st.markdown(f"> slope = (f(b) - f(a)) / (t_b - t_a) = ({rec['b_val']:.3f} - {rec['a_val']:.3f}) / 1 = *{rec['slope']:+.3f}*")
        st.markdown(f"*Bước 3 – Ước lượng đạo hàm tại hai đầu đoạn:*")
        st.markdown(f"> f'({rec['a_label']}) ≈ {rec['deriv_a']:+.3f},  f'({rec['b_label']}) ≈ {rec['deriv_b']:+.3f}")
        st.markdown(f"*Bước 4 – Kiểm tra tính trung gian (Intermediate Value):*")
        if rec["bracket"]:
            st.markdown(f"> Vì slope = *{rec['slope']:+.3f}* nằm giữa f'({rec['a_label']}) và f'({rec['b_label']}), theo tính chất trung gian có tồn tại c ∈ ({rec['a_label']}, {rec['b_label']}).")
            st.markdown(f"*Bước 5 – Tìm c ước lượng:* phương pháp: {rec['method']}")
            st.markdown(f"> Vị trí ước lượng c ≈ *{rec['c_pos']:.3f}* (tức nằm cách {rec['a_label']} khoảng {(rec['c_pos']-int(rec['c_pos'])):.3f} phần của khoảng đến {rec['b_label']}).")
            st.markdown(f"> Ước lượng f(c) bằng nội suy tuyến tính ≈ *{rec['y_c']:.3f}*.")
            st.markdown(f"> Ước lượng f'(c) bằng nội suy giữa f'({rec['a_label']}) và f'({rec['b_label']}) ≈ *{rec['deriv_c']:+.3f}*.")
            st.markdown(f"> Sai khác |f'(c) - slope| = *{rec['residual']:.3e}* (mong muốn nhỏ).")
        else:
            st.markdown(f"> slope = *{rec['slope']:+.3f}* không nằm giữa f'({rec['a_label']}) và f'({rec['b_label']}).")
            st.markdown(f"> Chúng tôi chọn điểm có f' gần nhất theo phương pháp: {rec['method']}, ước lượng c = *{rec['c_pos']:.3f}*.")
            st.markdown(f"> Giá trị f(c) ước lượng = *{rec['y_c']:.3f}, f'(c) ≈ *{rec['deriv_c']:+.3f}*, sai khác = *{rec['residual']:.3e}**.")
        # Interpretation
        st.markdown("*Diễn giải ý nghĩa kinh doanh (gợi ý):*")
        if rec["slope"] > 0:
            st.markdown("- slope > 0: doanh thu (hoặc chỉ tiêu) tăng trung bình trong khoảng này.")
            st.markdown(f"- Điểm c ước lượng cho thấy giai đoạn mà doanh nghiệp có tốc độ tăng đúng bằng tốc độ trung bình (thời điểm có 'momentum' nhất).")
        elif rec["slope"] < 0:
            st.markdown("- slope < 0: doanh thu giảm trung bình trong khoảng này.")
            st.markdown(f"- Điểm c ước lượng có thể là thời điểm cảnh báo quản trị (nên xem chi tiết nguyên nhân tại khoảng này).")
        else:
            st.markdown("- slope = 0: không thay đổi tổng thể trong khoảng này.")
            
st.markdown("""
*Ghi chú về phương pháp:*  
- Bởi dữ liệu thực là rời rạc (theo quý/năm), ta dùng các xấp xỉ đạo hàm (forward/backward/central) để mô phỏng f'(t).  
- Khi f' tại hai đầu bao lấy slope trung bình, ta nội suy tuyến tính giữa hai giá trị đạo hàm để tìm vị trí c ước lượng (giả thiết đạo hàm biến đổi đều trong khoảng nhỏ).  
- Phương pháp này gần đúng; nếu cần chính xác hơn có thể:
  - Dùng nội suy spline để có hàm mượt hơn rồi giải f'(t)=slope trong khoảng
  - Dùng dữ liệu có phân giải cao hơn (theo ngày/tuần).
""")

st.markdown("""
    *Giải thích theo Định lý Giá trị Trung bình (MVT):*  
    Giữa hai kỳ báo cáo liên tiếp, tồn tại ít nhất một thời điểm mà *tốc độ thay đổi tức thời* của chỉ tiêu kinh doanh  
    (ví dụ doanh thu hoặc lợi nhuận) *bằng đúng tốc độ thay đổi trung bình* đã tính ở trên.  
    Điều này có nghĩa là, trong khoảng giữa hai quý, có một giai đoạn thực tế mà công ty đang hoạt động  
    với đúng mức "động lượng" trung bình đó – phản ánh xu hướng tăng trưởng hoặc suy giảm bền vững.
    """)
