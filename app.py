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
    .soft-control-box {
        background: linear-gradient(180deg, #fcfdff 0%, #f4f8ff 100%);
        border: 1px solid #dbe7f7;
        border-radius: 16px;
        padding: 0.8rem 0.9rem 0.4rem 0.9rem;
        margin-bottom: 0.75rem;
    }
    .center-soft-control-box {
        background: linear-gradient(180deg, #fcfdff 0%, #f4f8ff 100%);
        border: 1px solid #dbe7f7;
        border-radius: 16px;
        padding: 0.8rem 0.9rem 0.4rem 0.9rem;
        margin: 0 auto 0.75rem auto;
        width: 100%;
    }

    div[data-testid="stTabs"] [role="tablist"] {
        justify-content: space-between;
        gap: 1.2rem;
    }
    div[data-testid="stTabs"] [role="tab"] {
        flex: 1 1 0;
        min-width: 0;
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
    dx = np.diff(xs)
    trap = 0.5 * (ys[:-1] + ys[1:]) * dx

    # cumulative_from_left[i] ≈ ∫_{xs[0]}^{xs[i]} f(t) dt
    cumulative_from_left = np.concatenate(([0.0], np.cumsum(trap)))

    # 重新以固定點 a 為基準：
    # A(x) = ∫_a^x f(t) dt = ∫_{xs[0]}^x f(t) dt - ∫_{xs[0]}^a f(t) dt
    area_at_a = np.interp(a, xs, cumulative_from_left)
    area = cumulative_from_left - area_at_a
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



def build_custom_function(expr_text: str):
    x = sp.symbols("x")
    allowed = {
        "x": x,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "Abs": sp.Abs,
        "pi": sp.pi,
        "E": sp.E,
    }
    expr = sp.sympify(expr_text, locals=allowed)
    func = sp.lambdify(x, expr, modules=["numpy"])
    return expr, func


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
    ax.axhline(0, linewidth=1.6, color="#b0b0b0", zorder=0)
    ax.axvline(0, linewidth=1.6, color="#b0b0b0", zorder=0)
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
    function_mode = st.radio("函數來源", ["使用內建函數", "自行輸入函數"], index=0)

    if function_mode == "使用內建函數":
        fname = st.selectbox(
            "選擇原函數 f(x)",
            ["x", "x^2", "sin(x)", "cos(x)", "x^2 - 1", "0.5x^3 - x"],
        )
        f = function_factory(fname)
    else:
        if st.button("顯示常用函數輸入示範", use_container_width=True):
            st.session_state["show_function_examples"] = not st.session_state.get("show_function_examples", False)

        if st.session_state.get("show_function_examples", False):
            st.info(
                "可參考這些輸入格式：\n"
                "• x\n"
                "• x**2\n"
                "• x**3 - x\n"
                "• sin(x)\n"
                "• cos(x) + x\n"
                "• exp(x)\n"
                "• log(x+2)\n"
                "• sqrt(x+3)\n"
                "• Abs(x)\n"
                "• sin(x) + x/2"
            )

        custom_expr_text = st.text_input(
            "自行輸入 f(x)",
            value="sin(x)+x/2",
            help="可輸入：x**2、sin(x)、cos(x)、exp(x)、log(x+2)、sqrt(x+3) 等",
        )
        try:
            custom_expr, custom_func = build_custom_function(custom_expr_text)
            fname = str(custom_expr)
            f = lambda x: np.array(custom_func(x), dtype=float)
            st.success("自訂函數已載入")
        except Exception:
            fname = "x"
            f = function_factory(fname)
            st.error("自訂函數格式有誤，已暫時改用 f(x)=x")

    st.markdown("---")
    left_input_col, right_input_col = st.columns(2)
    with left_input_col:
        domain_left = st.number_input("顯示區間左端點", value=-3.0, step=0.5, format="%.2f")
    with right_input_col:
        domain_right = st.number_input("顯示區間右端點", value=3.0, step=0.5, format="%.2f")
    if domain_right <= domain_left:
        st.warning("右端點必須大於左端點，已暫時使用預設區間 [-3, 3]。")
        domain_left, domain_right = -3.0, 3.0


    y_input_col1, y_input_col2 = st.columns(2)
    with y_input_col1:
        y_min_input = st.number_input("圖形下界", value=-5.0, step=0.5, format="%.2f")
    with y_input_col2:
        y_max_input = st.number_input("圖形上界", value=5.0, step=0.5, format="%.2f")

    if y_max_input <= y_min_input:
        st.warning("圖形上界必須大於圖形下界，已暫時使用預設範圍 [-5, 5]。")
        y_min_input, y_max_input = -5.0, 5.0

    st.markdown("---")
    st.subheader("模組 1 圖形樣式")
    fill_color_m1 = st.color_picker("面積塗色顏色", "#ff7f0e")

    st.markdown("---")
    show_help = st.checkbox("顯示操作提醒", value=True)
    show_formula = st.checkbox("顯示公式區", value=True)


if "m1a" not in st.session_state:
    st.session_state["m1a"] = float(min(max(0.0, domain_left), domain_right))
if "m2a" not in st.session_state:
    st.session_state["m2a"] = float(min(max(0.0, domain_left), domain_right))
if "m4a" not in st.session_state:
    st.session_state["m4a"] = float(min(max(0.0, domain_left), domain_right))

xs = np.linspace(domain_left, domain_right, 800)

try:
    ys = np.array(f(xs), dtype=float)
    if ys.shape == ():
        ys = np.full_like(xs, float(ys))
except Exception:
    st.error("目前函數無法在這個區間正常計算，請修改函數或調整顯示區間。")
    st.stop()

a_default = float(st.session_state.get("m1a", min(max(0.0, domain_left), domain_right)))
Axs = cumulative_integral(f, a_default, xs)
Aprime = safe_gradient(Axs, xs)

# 以數值方式建立一個原函數 F，使得 F'(x)=f(x) 且 F(a)=0
def F(x):
    arr = np.array(x, dtype=float)
    return np.interp(arr, xs, Axs)

Fxs = F(xs)

# 固定共同座標範圍：不因固定點 a 改變而跳動
x_min_common = float(domain_left)
x_max_common = float(domain_right)
y_min_common = float(y_min_input)
y_max_common = float(y_max_input)

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
module1, module2, module4 = st.tabs([
    "模組 1｜累積函數動態生成",
    "模組 2｜導數與累積同步",
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
            <div class="module-chip">步驟 3：看面積變化</div>
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
            觀察重點：圖形只顯示固定點 a 右邊的部分。當你把 x 往右拖時，從固定點 a 到 x 的面積會持續累積，
            而下方的 <b>A(x)</b> 也會跟著改變。
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_right:
        a = float(st.session_state.get("m1a", min(max(0.0, domain_left), domain_right)))
        x1 = float(st.session_state.get("m1x", (domain_left + domain_right) / 2))
        st.markdown(
            f"""
            <div class="panel" style="margin-top:0.55rem;">
            <b>目前設定</b><br>
            函數：<b>{fname}</b><br>
            固定點：<b>a = {a:.2f}</b><br>
            顯示區間：<b>[{domain_left:.2f}, {domain_right:.2f}]</b><br>
            顏色：左圖曲線 <span style="color:#8fc9a8;font-weight:700;">■</span>　右圖曲線 <span style="color:#8bbce9;font-weight:700;">■</span>　
            面積 <span style="color:{fill_color_m1};font-weight:700;">■</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="center-soft-control-box">', unsafe_allow_html=True)
    a = st.slider(
        "固定點 a",
        min_value=float(domain_left),
        max_value=float(domain_right),
        value=float(st.session_state.get("m1a", min(max(0.0, domain_left), domain_right))),
        step=0.05,
        key="m1a",
    )
    x1 = st.slider(
        "拖動 x",
        min_value=float(domain_left),
        max_value=float(domain_right),
        value=float(st.session_state.get("m1x", (domain_left + domain_right) / 2)),
        step=0.05,
        key="m1x",
    )
    reset_default = float((domain_left + domain_right) / 2)
    if st.button("把 x 回到中間位置", key="m1_reset_button", use_container_width=True):
        st.session_state["m1x"] = reset_default
    st.markdown('</div>', unsafe_allow_html=True)

    Axs = cumulative_integral(f, a, xs)
    current_A = np.interp(x1, xs, Axs)
    current_f = f(np.array([x1]))[0]
    mask = (xs >= min(a, x1)) & (xs <= max(a, x1))

    m1c1, m1c2, m1c3 = st.columns(3)
    m1c1.metric("目前 x", f"{x1:.3f}")
    m1c2.metric("目前 f(x)", f"{current_f:.4f}")
    m1c3.metric("目前 A(x)", f"{current_A:.4f}")

    chart_col_left, chart_col_right = st.columns(2, gap="large")

    # 累積函數只顯示到目前滑桿位置，形成「逐漸長出來」的效果
    if x1 >= domain_left:
        mask_A = xs <= x1
    else:
        mask_A = xs >= x1

    with chart_col_left:
        fig12, ax12 = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
        mask_A_display = (xs >= a) & mask_A
        ax12.plot(xs[mask_A_display], Axs[mask_A_display], linewidth=4.2, color="#8fc9a8")
        ax12.axvline(a, linestyle="--", linewidth=1.6, color="#f2a3c7")
        ax12.axvline(x1, linestyle="--", linewidth=1.6, color="#9bd18b")
        ax12.scatter([x1], [current_A], s=95, color="#8fc9a8", zorder=5)
        ax12.set_title("累積函數 A(x)（會隨著滑桿逐步生成）", fontsize=16, pad=14)
        ax12.set_xlabel("x", fontsize=12)
        ax12.set_ylabel("A(x)", fontsize=12)
        ax12.set_xlim(x_min_common, x_max_common)
        ax12.set_ylim(y_min_common, y_max_common)
        ax12.tick_params(labelsize=11)
        add_common_style(ax12)
        st.pyplot(fig12, use_container_width=True)

    with chart_col_right:
        fig11, ax11 = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
        mask_f_display = xs >= a
        ax11.plot(xs[mask_f_display], ys[mask_f_display], linewidth=4.2, color="#8bbce9")
        ax11.axvline(a, linestyle="--", linewidth=1.6, color="#f2a3c7")
        ax11.axvline(x1, linestyle="--", linewidth=1.6, color="#9bd18b")
        ax11.fill_between(xs[mask], ys[mask], 0, alpha=0.40, color=fill_color_m1)
        ax11.scatter([x1], [current_f], s=95, color="#8bbce9", zorder=5)
        ax11.set_title("原函數 f(x) 與從固定點 a 到 x 的累積面積", fontsize=16, pad=14)
        ax11.set_xlabel("x", fontsize=12)
        ax11.set_ylabel("f(x)", fontsize=12)
        ax11.set_xlim(x_min_common, x_max_common)
        ax11.set_ylim(y_min_common, y_max_common)
        ax11.tick_params(labelsize=11)
        add_common_style(ax11)
        st.pyplot(fig11, use_container_width=True)

    st.markdown(
        f"""
        <div class="panel">
        <b>你現在應該看到什麼</b><br>
        1. 當你拖動 x 時，左圖的 A(x) 曲線會逐步長出來。<br>
        2. 右圖的陰影面積會跟著改變，代表新的累積量來源。<br>
        3. 這表示積分可以看成「從固定點 a 開始，一路累積到 x」。<br><br>
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
        st.markdown('<div class="formula-box">\n$$A(x)=\\int_a^x f(t)\\,dt \quad \Rightarrow \quad A\'(x)=f(x)$$\n</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="big-note">
        觀察重點：當你拖動 x 時，左圖的 A(x) 會顯示目前位置與切線，
        右圖的 f(x) 會同步顯示對應的函數值，幫助你理解 A'(x)=f(x)。
        </div>
        """,
        unsafe_allow_html=True,
    )

    full_width_col = st.container()
    with full_width_col:
        st.markdown('<div class="soft-control-box">', unsafe_allow_html=True)
        a2 = st.slider(
            "固定點 a",
            min_value=float(domain_left),
            max_value=float(domain_right),
            value=float(st.session_state.get("m2a", min(max(0.0, domain_left), domain_right))),
            step=0.05,
            key="m2a",
        )
        x2 = st.slider(
            "拖動 x",
            min_value=float(domain_left),
            max_value=float(domain_right),
            value=float(st.session_state.get("m2x", (domain_left + domain_right) / 3)),
            step=0.05,
            key="m2x",
        )
        reset_default_m2 = float((domain_left + domain_right) / 3)
        if st.button("把 x 回到中間位置", key="m2_reset_button", use_container_width=True):
            st.session_state["m2x"] = reset_default_m2
            x2 = reset_default_m2
        st.markdown('</div>', unsafe_allow_html=True)

        Axs_m2 = cumulative_integral(f, a2, xs)
        Aprime_m2 = safe_gradient(Axs_m2, xs)
        current_A2 = np.interp(x2, xs, Axs_m2)
        current_f2 = f(np.array([x2]))[0]
        current_Ap2 = np.interp(x2, xs, Aprime_m2)

        if current_f2 > 1e-3:
            trend = "A(x) 正在上升"
        elif current_f2 < -1e-3:
            trend = "A(x) 正在下降"
        else:
            trend = "A(x) 在這附近斜率接近 0"

    m2_info_left, m2_info_center, m2_info_right = st.columns([0.20, 0.60, 0.20])
    with m2_info_center:
        st.markdown(
            f"""
            <div class="panel" style="margin-top:0.55rem;">
            <b>目前設定</b><br>
            固定點：<b>a = {a2:.2f}</b><br>
            x 範圍：<b>[{domain_left:.2f}, {domain_right:.2f}]</b><br>
            y 範圍：<b>[{y_min_common:.2f}, {y_max_common:.2f}]</b><br>
            左圖曲線 <span style="color:#8fc9a8;font-weight:700;">■</span>　
            右圖曲線 <span style="color:#8bbce9;font-weight:700;">■</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    m2c1, m2c2, m2c3, m2c4 = st.columns(4)
    m2c1.metric("x", f"{x2:.3f}")
    m2c2.metric("f(x)", f"{current_f2:.4f}")
    m2c3.metric("A(x)", f"{current_A2:.4f}")
    m2c4.metric("A'(x) 近似", f"{current_Ap2:.4f}")

    left, right = st.columns(2, gap="large")
    with left:
        fig22, ax22 = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
        ax22.plot(xs, Axs_m2, linewidth=3.4, color="#8fc9a8")
        ax22.axvline(a2, linestyle="--", linewidth=1.6, color="#f2a3c7")
        ax22.axvline(x2, linestyle="--", linewidth=1.6, color="#9bd18b")
        ax22.scatter([x2], [current_A2], s=55, color="#8fc9a8")

        tangent_half_width = 0.60
        tangent_x = np.linspace(
            max(x_min_common, x2 - tangent_half_width),
            min(x_max_common, x2 + tangent_half_width),
            40,
        )
        tangent_y = current_A2 + current_Ap2 * (tangent_x - x2)
        ax22.plot(tangent_x, tangent_y, linewidth=3.0, color="#ffb347")

        ax22.set_title("累積函數 A(x)", fontsize=14)
        ax22.set_xlabel("x")
        ax22.set_ylabel("A(x)")
        ax22.set_xlim(x_min_common, x_max_common)
        ax22.set_ylim(y_min_common, y_max_common)
        add_common_style(ax22)
        st.pyplot(fig22, use_container_width=True)

    with right:
        fig2, ax2 = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
        ax2.plot(xs, ys, linewidth=3.4, label="f(x)", color="#8bbce9")
        ax2.axvline(a2, linestyle="--", linewidth=1.6, color="#f2a3c7")
        ax2.axvline(x2, linestyle="--", linewidth=1.6, color="#9bd18b")
        ax2.scatter([x2], [current_f2], s=55, color="#8bbce9")
        ax2.set_title("函數 f(x)", fontsize=14)
        ax2.set_xlabel("x")
        ax2.set_ylabel("f(x)")
        ax2.set_xlim(x_min_common, x_max_common)
        ax2.set_ylim(y_min_common, y_max_common)
        ax2.legend()
        add_common_style(ax2)
        st.pyplot(fig2, use_container_width=True)

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
# Module 4
# -----------------------------
with module4:
    st.subheader("模組 4：FTC Part 2 幾何意義")
    st.caption("把定積分看成原函數的總改變量，而不是一條要背的公式。")

    if show_formula:
        st.markdown('<div class="formula-box">\n$$F\'(x)=f(x) \quad \Rightarrow \quad \\int_a^b f(x)\,dx = F(b)-F(a)$$\n</div>', unsafe_allow_html=True)

    st.markdown('<div class="soft-control-box">', unsafe_allow_html=True)
    a = st.slider(
        "固定點 a",
        min_value=float(domain_left),
        max_value=float(domain_right),
        value=float(st.session_state.get("m4a", min(max(0.0, domain_left), domain_right))),
        step=0.05,
        key="m4a",
    )
    b4 = st.slider(
        "選擇右端點 b",
        min_value=float(domain_left),
        max_value=float(domain_right),
        value=float(st.session_state.get("m4b", min(domain_right, 2.0))),
        step=0.05,
        key="m4b",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    Axs_m4 = cumulative_integral(f, a, xs)
    def F_m4(x):
        arr = np.array(x, dtype=float)
        return np.interp(arr, xs, Axs_m4)

    if b4 <= a:
        st.warning("目前 b 小於等於 a，請把 b 調整到大於 a。")
    else:
        exact_area = (F_m4(np.array([b4])) - F_m4(np.array([a])))[0]
        Fa = F_m4(np.array([a]))[0]
        Fb = F_m4(np.array([b4]))[0]

        m4c1, m4c2, m4c3 = st.columns(3)
        m4c1.metric("F(a)", f"{Fa:.4f}")
        m4c2.metric("F(b)", f"{Fb:.4f}")
        m4c3.metric("F(b)-F(a)", f"{exact_area:.4f}")

        left, right = st.columns(2)
        with left:
            Fx = Axs_m4
            fig42, ax42 = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
            ax42.plot(xs, Fx, linewidth=3.4, color="#8bbce9")
            ax42.axvline(a, linestyle="--", linewidth=1.6, color="#f2a3c7")
            ax42.axvline(b4, linestyle="--", linewidth=1.2)
            ax42.scatter([a, b4], [Fa, Fb], s=55, color="#8bbce9")
            ax42.set_title("原函數的總改變量", fontsize=14)
            ax42.set_xlabel("x")
            ax42.set_ylabel("F(x)")
            ax42.set_xlim(x_min_common, x_max_common)
            ax42.set_ylim(y_min_common, y_max_common)
            add_common_style(ax42)
            st.pyplot(fig42, use_container_width=True)

        with right:
            fig4, ax4 = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
            ax4.plot(xs, ys, linewidth=3.4, color="#8fc9a8")
            ax4.axvline(a, linestyle="--", linewidth=1.6, color="#f2a3c7")
            ax4.axvline(b4, linestyle="--", linewidth=1.2)
            mask4 = (xs >= a) & (xs <= b4)
            ax4.fill_between(xs[mask4], ys[mask4], 0, alpha=0.3, color=fill_color_m1)
            ax4.set_title("陰影面積：定積分", fontsize=14)
            ax4.set_xlabel("x")
            ax4.set_ylabel("f(x)")
            ax4.set_xlim(x_min_common, x_max_common)
            ax4.set_ylim(y_min_common, y_max_common)
            add_common_style(ax4)
            st.pyplot(fig4, use_container_width=True)

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
