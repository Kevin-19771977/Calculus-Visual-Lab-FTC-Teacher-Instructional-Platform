import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="FTC 學生互動學習平台",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Style
# -----------------------------
st.markdown(
    """
    <style>
    .main {
        padding-top: 1.2rem;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    h1, h2, h3 {
        letter-spacing: 0.2px;
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #eef5ff 0%, #f7fbff 100%);
        border: 1px solid #d8e6ff;
        margin-bottom: 1rem;
    }
    .panel {
        background: #fafcff;
        border: 1px solid #e4ecf7;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
    }
    .focus-box {
        background: #fffaf0;
        border-left: 6px solid #f5b041;
        padding: 0.9rem 1rem;
        border-radius: 10px;
        margin: 0.8rem 0 1rem 0;
    }
    .formula-box {
        background: #f8f9fb;
        border: 1px solid #e7ebf0;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        margin: 0.7rem 0;
    }
    .small-note {
        color: #4f6478;
        font-size: 0.95rem;
    }
div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6eef7;
        padding: 0.7rem 0.8rem;
        border-radius: 14px;
    }
    div[data-testid="stButton"] > button {
        border-radius: 999px;
        border: 1px solid #cfe0ff;
        background: linear-gradient(180deg, #ffffff 0%, #f3f8ff 100%);
        color: #184a90;
        font-weight: 700;
        padding: 0.5rem 1.1rem;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: #8fb6ff;
        color: #0f3d82;
    }
    .module-toolbar {
        background: linear-gradient(135deg, #f8fbff 0%, #edf4ff 100%);
        border: 1px solid #dbe8fb;
        border-radius: 18px;
        padding: 1rem 1.1rem 0.5rem 1.1rem;
        margin: 0.4rem 0 1rem 0;
    }
    .module-chip {
        display: inline-block;
        margin-right: 0.45rem;
        margin-bottom: 0.45rem;
        padding: 0.36rem 0.8rem;
        border-radius: 999px;
        background: #ffffff;
        border: 1px solid #d9e7fb;
        color: #355070;
        font-size: 0.95rem;
        font-weight: 600;
    }
    .big-note {
        font-size: 1.02rem;
        color: #38506a;
        line-height: 1.8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Math utilities
# -----------------------------
def safe_gradient(y, x):
    return np.gradient(y, x)


def cumulative_integral(func, a, xs):
    ys = func(xs)
    area = np.zeros_like(xs)
    dx = np.diff(xs)
    trap = 0.5 * (ys[:-1] + ys[1:]) * dx
    area[1:] = np.cumsum(trap)
    return area


def antiderivative_factory(name: str):
    if name == "x":
        return lambda x: 0.5 * x**2
    if name == "x^2":
        return lambda x: x**3 / 3
    if name == "sin(x)":
        return lambda x: -np.cos(x)
    if name == "cos(x)":
        return lambda x: np.sin(x)
    if name == "x^2 - 1":
        return lambda x: x**3 / 3 - x
    if name == "0.5x^3 - x":
        return lambda x: 0.125 * x**4 - 0.5 * x**2
    raise ValueError("Unknown function")


def function_factory(name: str):
    if name == "x":
        return lambda x: x
    if name == "x^2":
        return lambda x: x**2
    if name == "sin(x)":
        return lambda x: np.sin(x)
    if name == "cos(x)":
        return lambda x: np.cos(x)
    if name == "x^2 - 1":
        return lambda x: x**2 - 1
    if name == "0.5x^3 - x":
        return lambda x: 0.5 * x**3 - x
    raise ValueError("Unknown function")


def g_factory(name: str):
    if name == "x":
        return lambda x: x
    if name == "x^2":
        return lambda x: x**2
    if name == "sin(x)+1":
        return lambda x: np.sin(x) + 1
    if name == "0.5x+1":
        return lambda x: 0.5 * x + 1
    if name == "2-x":
        return lambda x: 2 - x
    raise ValueError("Unknown g(x)")


def gprime_factory(name: str):
    if name == "x":
        return lambda x: np.ones_like(x)
    if name == "x^2":
        return lambda x: 2 * x
    if name == "sin(x)+1":
        return lambda x: np.cos(x)
    if name == "0.5x+1":
        return lambda x: np.full_like(x, 0.5)
    if name == "2-x":
        return lambda x: np.full_like(x, -1.0)
    raise ValueError("Unknown g(x)")


def add_common_style(ax):
    ax.grid(alpha=0.22)
    ax.axhline(0, linewidth=1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom:0.4rem;">📘 微積分基本定理互動學習平台</h1>
        <div style="font-size:1.08rem; line-height:1.7;">
            這個版本專門設計給學生在網頁上操作。<br>
            你可以直接拖曳滑桿、切換函數、觀察圖形與公式之間的關係。
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

colA, colB, colC = st.columns([1.2, 1.2, 1.6])
with colA:
    st.markdown('<div class="panel"><b>建議使用方式</b><br>先看左邊設定，再依序操作 4 個模組。</div>', unsafe_allow_html=True)
with colB:
    st.markdown('<div class="panel"><b>學習目標</b><br>看懂「面積累積 → 導數 → 定積分」的完整連結。</div>', unsafe_allow_html=True)
with colC:
    st.markdown('<div class="panel"><b>適合情境</b><br>課堂操作、翻轉學習、研究試教、學生自主練習。</div>', unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("操作設定")
    fname = st.selectbox(
        "選擇原函數 f(x)",
        ["x", "x^2", "sin(x)", "cos(x)", "x^2 - 1", "0.5x^3 - x"],
    )
    f = function_factory(fname)
    F = antiderivative_factory(fname)

    st.markdown("---")
    a = st.number_input("固定點 a", min_value=-3.0, max_value=2.0, value=0.0, step=0.5, format="%.2f")
    left_col, right_col = st.columns(2)
    with left_col:
        domain_left = st.number_input("左端點", min_value=-5.0, max_value=0.0, value=-3.0, step=0.5, format="%.2f")
    with right_col:
        domain_right = st.number_input("右端點", min_value=1.0, max_value=5.0, value=3.0, step=0.5, format="%.2f")
    if domain_right <= domain_left + 0.5:
        domain_right = domain_left + 0.5

    st.markdown("---")
    st.subheader("模組 1 圖形樣式")
    curve_color_m1 = st.color_picker("函數曲線顏色", "#1f77b4")
    fill_color_m1 = st.color_picker("面積塗色顏色", "#ff7f0e")

    st.markdown("---")
    show_help = st.checkbox("顯示操作提醒", value=True)
    show_formula = st.checkbox("顯示公式區", value=True)

xs = np.linspace(domain_left, domain_right, 800)
ys = f(xs)
Axs = cumulative_integral(f, a, xs)
Aprime = safe_gradient(Axs, xs)

if show_help:
    st.markdown(
        """
        <div class="focus-box">
        <b>操作提醒</b><br>
        先固定一個函數，然後只改一個滑桿，慢慢觀察圖形怎麼變。<br>
        每個模組下方都有「你現在應該看到什麼」可以直接拿來當學生提示語。
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Tabs
# -----------------------------
module1, module2, module3, module4 = st.tabs([
    "模組 1｜累積函數動態生成",
    "模組 2｜導數與累積同步",
    "模組 3｜變上限積分符號辨識",
    "模組 4｜FTC Part 2 幾何意義",
])

# -----------------------------
# Module 1
# -----------------------------
with module1:
    st.subheader("模組 1：累積函數動態生成")
    st.caption("看懂 A(x) 不是固定數字，而是會跟著 x 改變的累積函數。")

    st.markdown(
        """
        <div class="module-toolbar">
            <div class="module-chip">步驟 1：選函數</div>
            <div class="module-chip">步驟 2：拖動 x</div>
            <div class="module-chip">步驟 3：看從固定點 a 到 x 的面積變化</div>
            <div class="module-chip">步驟 4：對照 A(x) 上的點</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_left, top_right = st.columns([1.55, 1.0])
    with top_left:
        if show_formula:
            st.markdown('<div class="formula-box">$$A(x)=\int_a^x f(t)\,dt$$</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="big-note">
            觀察重點：當你把 x 往右拖時，從固定點 a 到 x 的面積會持續累積，
            而下方的 <b>A(x)</b> 也會跟著改變。
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_right:
        reset_default = float((domain_left + domain_right) / 2)
        if st.button("把 x 回到中間位置", key="m1_reset_button", use_container_width=True):
            st.session_state["m1x"] = reset_default
        st.markdown(
            f"""
            <div class="panel" style="margin-top:0.55rem;">
            <b>目前設定</b><br>
            函數：<b>{fname}</b><br>
            固定點：<b>a = {a:.2f}</b><br>
            顏色：曲線 <span style="color:{curve_color_m1};font-weight:700;">■</span>　
            面積 <span style="color:{fill_color_m1};font-weight:700;">■</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    x1 = st.number_input(
        "輸入 x，觀察從固定點 a 到 x 的面積如何形成 A(x)",
        min_value=float(domain_left),
        max_value=float(domain_right),
        value=float(st.session_state.get("m1x", (domain_left + domain_right) / 2)),
        step=0.05,
        format="%.2f",
        key="m1x",
    )

    current_A = np.interp(x1, xs, Axs)
    current_f = f(np.array([x1]))[0]
    mask = (xs >= min(a, x1)) & (xs <= max(a, x1))

    m1c1, m1c2, m1c3 = st.columns(3)
    m1c1.metric("目前 x", f"{x1:.3f}")
    m1c2.metric("目前 f(x)", f"{current_f:.4f}")
    m1c3.metric("目前 A(x)", f"{current_A:.4f}")

    chart_col_left, chart_col_right = st.columns([1.35, 1.35], gap="large")

    # 累積函數只顯示到目前滑桿位置，形成「逐漸長出來」的效果
    if x1 >= domain_left:
        mask_A = xs <= x1
    else:
        mask_A = xs >= x1

    with chart_col_left:
        fig12, ax12 = plt.subplots(figsize=(9.8, 6.6), constrained_layout=True)
        ax12.plot(xs[mask_A], Axs[mask_A], linewidth=3.0, color=curve_color_m1)
        ax12.axvline(x1, linestyle="--", linewidth=1.4, color="#666666")
        ax12.scatter([x1], [current_A], s=95, color=curve_color_m1, zorder=5)
        ax12.set_title("累積函數 A(x)（會隨著滑桿逐步生成）", fontsize=16, pad=14)
        ax12.set_xlabel("x", fontsize=12)
        ax12.set_ylabel("A(x)", fontsize=12)
        ax12.tick_params(labelsize=11)
        add_common_style(ax12)
        st.pyplot(fig12, use_container_width=True)

    with chart_col_right:
        fig11, ax11 = plt.subplots(figsize=(9.8, 6.6), constrained_layout=True)
        ax11.plot(xs, ys, linewidth=3.0, color=curve_color_m1)
        ax11.axvline(a, linestyle="--", linewidth=1.4, color="#666666")
        ax11.axvline(x1, linestyle="--", linewidth=1.4, color="#666666")
        ax11.fill_between(xs[mask], ys[mask], 0, alpha=0.40, color=fill_color_m1)
        ax11.scatter([x1], [current_f], s=95, color=curve_color_m1, zorder=5)
        ax11.set_title("原函數 f(x) 與從固定點 a 到 x 的累積面積", fontsize=16, pad=14)
        ax11.set_xlabel("x", fontsize=12)
        ax11.set_ylabel("f(x)", fontsize=12)
        ax11.tick_params(labelsize=11)
        add_common_style(ax11)
        st.pyplot(fig11, use_container_width=True)

    st.markdown(
        f"""
        <div class="panel">
        <b>你現在應該看到什麼</b><br>
        1. 當你拖動 x 時，左圖的 A(x) 曲線會逐步長出來。<br>
        2. 右圖的陰影面積會跟著改變，代表新的累積量來源。<br>
        3. 這表示積分可以看成「從 a 開始，一路累積到 x」。<br><br>
        目前：當 x = <b>{x1:.2f}</b> 時，f(x) ≈ <b>{current_f:.4f}</b>，A(x) ≈ <b>{current_A:.4f}</b>。
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Module 2
# -----------------------------
with module2:
    st.subheader("模組 2：導數與累積同步")
    st.caption("觀察 A'(x) 為什麼會接近 f(x)，這就是 FTC 第一部分的核心。")

    if show_formula:
        st.markdown('<div class="formula-box">\n$$A(x)=\\int_a^x f(t)\,dt \quad \Rightarrow \quad A\'(x)=f(x)$$\n</div>', unsafe_allow_html=True)

    x2 = st.slider(
        "拖動 x，觀察 f(x)、A(x)、A'(x) 的同步變化",
        min_value=float(domain_left),
        max_value=float(domain_right),
        value=float((domain_left + domain_right) / 3),
        step=0.05,
        key="m2x",
    )

    current_A2 = np.interp(x2, xs, Axs)
    current_f2 = f(np.array([x2]))[0]
    current_Ap2 = np.interp(x2, xs, Aprime)

    if current_f2 > 1e-3:
        trend = "A(x) 正在上升"
    elif current_f2 < -1e-3:
        trend = "A(x) 正在下降"
    else:
        trend = "A(x) 在這附近斜率接近 0"

    m2c1, m2c2, m2c3, m2c4 = st.columns(4)
    m2c1.metric("x", f"{x2:.3f}")
    m2c2.metric("f(x)", f"{current_f2:.4f}")
    m2c3.metric("A(x)", f"{current_A2:.4f}")
    m2c4.metric("A'(x) 近似", f"{current_Ap2:.4f}")

    left, right = st.columns([1.25, 1])
    with left:
        fig2, ax2 = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
        ax2.plot(xs, ys, linewidth=2, label="f(x)")
        ax2.plot(xs, Aprime, linestyle="--", linewidth=2, label="A'(x) 的數值近似")
        ax2.axvline(x2, linestyle=":", linewidth=1.2)
        ax2.scatter([x2], [current_f2], s=55)
        ax2.scatter([x2], [current_Ap2], s=55)
        ax2.set_title("比較 f(x) 與 A'(x)", fontsize=14)
        ax2.set_xlabel("x")
        ax2.set_ylabel("數值")
        ax2.legend()
        add_common_style(ax2)
        st.pyplot(fig2, use_container_width=True)

    with right:
        fig22, ax22 = plt.subplots(figsize=(8, 5.4), constrained_layout=True)
        ax22.plot(xs, Axs, linewidth=2)
        ax22.axvline(x2, linestyle=":", linewidth=1.2)
        ax22.scatter([x2], [current_A2], s=55)
        ax22.set_title("累積函數 A(x)", fontsize=14)
        ax22.set_xlabel("x")
        ax22.set_ylabel("A(x)")
        add_common_style(ax22)
        st.pyplot(fig22, use_container_width=True)

    st.markdown(
        f"""
        <div class="panel">
        <b>你現在應該看到什麼</b><br>
        - 當 f(x) &gt; 0，A(x) 會往上走。<br>
        - 當 f(x) &lt; 0，A(x) 會往下走。<br>
        - 當 f(x) 接近 0，A(x) 的斜率也會接近 0。<br><br>
        目前這個位置的判讀：<b>{trend}</b>。
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Module 3
# -----------------------------
with module3:
    st.subheader("模組 3：變上限積分符號辨識")
    st.caption("分清楚誰是虛變數、誰才是真正控制上限變化的變數。")

    gname = st.selectbox("選擇上限函數 g(x)", ["x", "x^2", "sin(x)+1", "0.5x+1", "2-x"], key="gselect")
    g = g_factory(gname)
    gp = gprime_factory(gname)

    if show_formula:
        st.markdown(
            '<div class="formula-box">\n$$\\frac{d}{dx}\\int_a^x f(t)\,dt = f(x)$$\n$$\\frac{d}{dx}\\int_a^{g(x)} f(t)\,dt = f(g(x))g\'(x)$$\n</div>',
            unsafe_allow_html=True,
        )

    x3_left, x3_right = -1.5, 1.5
    xs3 = np.linspace(x3_left, x3_right, 600)
    gx = g(xs3)
    Hxs = F(gx) - F(np.full_like(gx, a))
    Hprime_numeric = safe_gradient(Hxs, xs3)
    Hprime_formula = f(gx) * gp(xs3)

    x3 = st.slider(
        "拖動 x，觀察 H(x)=∫_a^{g(x)} f(t)dt 的變化",
        min_value=float(x3_left),
        max_value=float(x3_right),
        value=0.5,
        step=0.05,
        key="m3x",
    )

    g_now = g(np.array([x3]))[0]
    gp_now = gp(np.array([x3]))[0]
    H_now = (F(np.array([g_now])) - F(np.array([a])))[0]
    Hf_now = f(np.array([g_now]))[0] * gp_now

    m3c1, m3c2, m3c3, m3c4 = st.columns(4)
    m3c1.metric("x", f"{x3:.3f}")
    m3c2.metric("g(x)", f"{g_now:.4f}")
    m3c3.metric("g'(x)", f"{gp_now:.4f}")
    m3c4.metric("f(g(x))g'(x)", f"{Hf_now:.4f}")

    left, right = st.columns([1.15, 1.15])
    with left:
        fig3, ax3 = plt.subplots(figsize=(8.8, 5.3), constrained_layout=True)
        dense_t = np.linspace(min(a, np.min(gx)) - 0.5, max(a, np.max(gx)) + 0.5, 800)
        ax3.plot(dense_t, f(dense_t), linewidth=2)
        ax3.axvline(a, linestyle="--", linewidth=1.2)
        ax3.axvline(g_now, linestyle="--", linewidth=1.2)
        lo, hi = min(a, g_now), max(a, g_now)
        mask3 = (dense_t >= lo) & (dense_t <= hi)
        ax3.fill_between(dense_t[mask3], f(dense_t[mask3]), 0, alpha=0.3)
        ax3.scatter([g_now], [f(np.array([g_now]))[0]], s=55)
        ax3.set_title("以 t 為積分變數的面積", fontsize=14)
        ax3.set_xlabel("t")
        ax3.set_ylabel("f(t)")
        add_common_style(ax3)
        st.pyplot(fig3, use_container_width=True)

    with right:
        fig32, ax32 = plt.subplots(figsize=(8.8, 5.3), constrained_layout=True)
        ax32.plot(xs3, Hprime_numeric, linewidth=2, label="數值近似 H'(x)")
        ax32.plot(xs3, Hprime_formula, linestyle="--", linewidth=2, label="f(g(x))g'(x)")
        ax32.axvline(x3, linestyle=":", linewidth=1.2)
        ax32.scatter([x3], [np.interp(x3, xs3, Hprime_numeric)], s=55)
        ax32.scatter([x3], [Hf_now], s=55)
        ax32.set_title("比較導數結果", fontsize=14)
        ax32.set_xlabel("x")
        ax32.set_ylabel("數值")
        ax32.legend()
        add_common_style(ax32)
        st.pyplot(fig32, use_container_width=True)

    st.markdown(
        f"""
        <div class="panel">
        <b>你現在應該看到什麼</b><br>
        - 在積分號裡的 <b>t</b> 只是內部記號。<br>
        - 真正控制面積怎麼變的是上限 <b>g(x)</b>。<br>
        - 所以導數會變成 <b>f(g(x))g'(x)</b>。<br><br>
        目前選擇 g(x)=<b>{gname}</b>，當 x = <b>{x3:.2f}</b> 時：<br>
        g(x) ≈ <b>{g_now:.4f}</b>，g'(x) ≈ <b>{gp_now:.4f}</b>，H(x) ≈ <b>{H_now:.4f}</b>。
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Module 4
# -----------------------------
with module4:
    st.subheader("模組 4：FTC Part 2 幾何意義")
    st.caption("把定積分看成原函數的總改變量，而不是一條要背的公式。")

    if show_formula:
        st.markdown('<div class="formula-box">\n$$F\'(x)=f(x) \quad \Rightarrow \quad \\int_a^b f(x)\,dx = F(b)-F(a)$$\n</div>', unsafe_allow_html=True)

    b4 = st.slider(
        "選擇右端點 b",
        min_value=float(domain_left + 0.2),
        max_value=float(domain_right),
        value=float(min(domain_right, 2.0)),
        step=0.05,
        key="m4b",
    )

    if b4 <= a:
        st.warning("目前 b 小於等於 a，請把 b 調整到大於 a。")
    else:
        exact_area = (F(np.array([b4])) - F(np.array([a])))[0]
        Fa = F(np.array([a]))[0]
        Fb = F(np.array([b4]))[0]

        m4c1, m4c2, m4c3 = st.columns(3)
        m4c1.metric("F(a)", f"{Fa:.4f}")
        m4c2.metric("F(b)", f"{Fb:.4f}")
        m4c3.metric("F(b)-F(a)", f"{exact_area:.4f}")

        left, right = st.columns([1.15, 1])
        with left:
            fig4, ax4 = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
            ax4.plot(xs, ys, linewidth=2)
            ax4.axvline(a, linestyle="--", linewidth=1.2)
            ax4.axvline(b4, linestyle="--", linewidth=1.2)
            mask4 = (xs >= a) & (xs <= b4)
            ax4.fill_between(xs[mask4], ys[mask4], 0, alpha=0.3)
            ax4.set_title("陰影面積：定積分", fontsize=14)
            ax4.set_xlabel("x")
            ax4.set_ylabel("f(x)")
            add_common_style(ax4)
            st.pyplot(fig4, use_container_width=True)

        with right:
            Fx = F(xs)
            fig42, ax42 = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
            ax42.plot(xs, Fx, linewidth=2)
            ax42.axvline(a, linestyle="--", linewidth=1.2)
            ax42.axvline(b4, linestyle="--", linewidth=1.2)
            ax42.scatter([a, b4], [Fa, Fb], s=55)
            ax42.set_title("原函數的總改變量", fontsize=14)
            ax42.set_xlabel("x")
            ax42.set_ylabel("F(x)")
            add_common_style(ax42)
            st.pyplot(fig42, use_container_width=True)

        st.markdown(
            f"""
            <div class="panel">
            <b>你現在應該看到什麼</b><br>
            左邊的陰影面積，對應到右邊原函數從 F(a) 走到 F(b) 的總改變量。<br>
            這就是：<b>定積分 = 總改變量</b>。<br><br>
            現在 a = <b>{a:.2f}</b>，b = <b>{b4:.2f}</b>，所以<br>
            F(a) ≈ <b>{Fa:.4f}</b>，F(b) ≈ <b>{Fb:.4f}</b>，
            因此定積分 ≈ <b>{exact_area:.4f}</b>。
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
st.subheader("教師／研究者下一步可加的功能")
st.markdown(
    """
    - 加入學生姓名、學號、班級欄位。  
    - 每個模組後面放 2–3 題概念題。  
    - 把答案、自評、操作時間存成 CSV。  
    - 分成學生版與教師版。  
    - 接上前測、後測、問卷頁面。  
    """
)

st.info("執行方式：python -m streamlit run ftc_teaching_tool_web_clear.py")
