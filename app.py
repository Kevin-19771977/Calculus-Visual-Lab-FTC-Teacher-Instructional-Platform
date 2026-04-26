import re
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

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
    code {
        background: #f3f6fb;
        padding: 0.12rem 0.35rem;
        border-radius: 6px;
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
    .goal-panel {
        padding: 1.2rem 1.4rem;
        font-size: 1.08rem;
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


def fill_area_by_sign(ax, xs_segment, ys_segment, pos_color, neg_color, alpha=0.4):
    xs_segment = np.array(xs_segment, dtype=float)
    ys_segment = np.array(ys_segment, dtype=float)
    if len(xs_segment) == 0:
        return
    ax.fill_between(xs_segment, ys_segment, 0, where=(ys_segment >= 0), interpolate=True, alpha=alpha, color=pos_color)
    ax.fill_between(xs_segment, ys_segment, 0, where=(ys_segment < 0), interpolate=True, alpha=alpha, color=neg_color)


def draw_to_x_axis(ax, x0, y0, color, linewidth=1.6, marker_size=55):
    ax.plot([x0, x0], [0, y0], linestyle="--", linewidth=linewidth, color=color)
    ax.scatter([x0], [y0], s=marker_size, color=color, zorder=6)


def fill_area_by_sign(ax, xs_segment, ys_segment, pos_color, neg_color, alpha=0.4):
    xs_segment = np.array(xs_segment, dtype=float)
    ys_segment = np.array(ys_segment, dtype=float)
    if len(xs_segment) == 0:
        return
    ax.fill_between(xs_segment, ys_segment, 0, where=(ys_segment >= 0), interpolate=True, alpha=alpha, color=pos_color)
    ax.fill_between(xs_segment, ys_segment, 0, where=(ys_segment < 0), interpolate=True, alpha=alpha, color=neg_color)


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
    if name == "3":
        return lambda x: 3 * np.array(x, dtype=float)
    if name == "x":
        return lambda x: 0.5 * np.array(x, dtype=float)**2
    if name == "x**2":
        return lambda x: np.array(x, dtype=float)**3 / 3
    if name == "x**3":
        return lambda x: np.array(x, dtype=float)**4 / 4
    raise ValueError("Unknown function")


def function_factory(name: str):
    if name == "1":
        return lambda x: np.full_like(np.array(x, dtype=float), 1.0)
    if name == "3":
        return lambda x: np.full_like(np.array(x, dtype=float), 3.0)
    if name == "x":
        return lambda x: x
    if name == "x**2":
        return lambda x: x**2
    if name == "x**3":
        return lambda x: x**3
    raise ValueError("Unknown function")



def convert_absolute_bars(expr: str) -> str:
    """Convert paired |...| into abs(...)."""
    result = []
    open_bar = True
    for ch in expr:
        if ch == '|':
            if open_bar:
                result.append('abs(')
            else:
                result.append(')')
            open_bar = not open_bar
        else:
            result.append(ch)
    if not open_bar:
        raise ValueError('絕對值符號 | | 未成對出現')
    return ''.join(result)


def normalize_function_input(func_str: str) -> str:
    s = func_str.strip()
    s = convert_absolute_bars(s)
    s = s.replace('^', '**')
    s = re.sub(r"\s+", "", s)

    s = re.sub(r"np\.log10(?=\()", "__NPLOGTEN__", s)
    s = re.sub(r"np\.log2(?=\()", "__NPLOGTWO__", s)
    s = re.sub(r"log10(?=\()", "__LOGTEN__", s)
    s = re.sub(r"log2(?=\()", "__LOGTWO__", s)
    s = re.sub(r"ln(?=\()", "__LOG__", s)

    func_pattern = r"(?:__NPLOGTEN__|__NPLOGTWO__|__LOGTEN__|__LOGTWO__|__LOG__|np\.(?:sin|cos|tan|exp|log|sqrt|abs)|sin|cos|tan|exp|log|sqrt|abs|pi|e)"

    s = re.sub(rf"(?<=\d)(?=(?:x|\(|{func_pattern}))", "*", s)
    s = re.sub(rf"(?<=x)(?=(?:\(|{func_pattern}|\d))", "*", s)
    s = re.sub(rf"(?<=\))(?=(?:x|\(|{func_pattern}|\d))", "*", s)

    s = s.replace("__NPLOGTEN__", "np.log10")
    s = s.replace("__NPLOGTWO__", "np.log2")
    s = s.replace("__LOGTEN__", "log10")
    s = s.replace("__LOGTWO__", "log2")
    s = s.replace("__LOG__", "log")

    return s


def parse_function(func_str: str):
    allowed_names = {
        'np': np,
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'exp': np.exp,
        'log': np.log,
        'ln': np.log,
        'log10': np.log10,
        'log2': np.log2,
        'sqrt': np.sqrt,
        'abs': np.abs,
        'pi': np.pi,
        'e': np.e,
    }

    normalized_func_str = normalize_function_input(func_str)

    def f(x):
        return eval(normalized_func_str, {'__builtins__': {}}, {'x': x, **allowed_names})

    return f


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


RAINBOW_COLORS = [
    "#e74c3c",
    "#f39c12",
    "#f1c40f",
    "#2ecc71",
    "#3498db",
    "#5b6ee1",
    "#9b59b6",
]


def get_axis_label_levels(values, threshold=0.42):
    values = [float(v) for v in values]
    order = np.argsort(values)
    levels = [0] * len(values)
    active = []
    for idx in order:
        x = values[idx]
        active = [(px, lvl) for px, lvl in active if abs(x - px) < threshold]
        used = {lvl for px, lvl in active if abs(x - px) < threshold}
        lvl = 0
        while lvl in used:
            lvl += 1
        levels[idx] = lvl
        active.append((x, lvl))
    return levels


def pretty_a_label_kwargs():
    return dict(
        ha="center",
        va="top",
        fontsize=13.5,
        fontweight="bold",
        color="#d84a4a",
        bbox=dict(
            boxstyle="round,pad=0.16,rounding_size=0.12",
            fc="#fff6bf",
            ec="#f2d97c",
            lw=0.9,
            alpha=0.98,
        ),
    )


def smart_point_xytext(x, y, x_min, x_max, y_min, y_max, other_points=None):
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    candidates = [
        (14 if x <= x_mid else -98, 14 if y <= y_mid else -26),
        (14 if x <= x_mid else -98, -30 if y <= y_mid else 18),
        (-100 if x <= x_mid else 18, 14 if y <= y_mid else -26),
        (-100 if x <= x_mid else 18, -30 if y <= y_mid else 18),
        (18, 26),
        (-104, -34),
    ]
    other_points = other_points or []
    x_thr = 0.10 * max(x_max - x_min, 1e-6)
    y_thr = 0.10 * max(y_max - y_min, 1e-6)
    close_count = sum(abs(x - ox) < x_thr and abs(y - oy) < y_thr for ox, oy in other_points)
    choice = min(close_count, len(candidates) - 1)
    return candidates[choice]


def smart_value_bbox():
    return dict(boxstyle="round,pad=0.12", fc="white", ec="#d9e7d9", lw=0.6, alpha=0.92)


def smart_area_bbox():
    return dict(
        boxstyle="round,pad=0.28,rounding_size=0.16",
        fc="white",
        ec="#c9d2de",
        lw=1.0,
        alpha=0.94,
    )


if "func_str" not in st.session_state:
    st.session_state["func_str"] = "x"

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom:0.4rem;">📘 微積分第一基本定理教學平台</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("操作設定")

    st.markdown("### 函數設定")
    func_str = st.text_input("輸入原函數 f(x)", key="func_str")
    with st.expander("輸入語法說明"):
        st.markdown("""
        <style>
        .syntax-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.93rem;
            margin-top: 0.2rem;
        }
        .syntax-table th, .syntax-table td {
            border: 1px solid #e5e7eb;
            padding: 8px 10px;
            vertical-align: top;
        }
        .syntax-table th {
            background: #f8fafc;
            font-weight: 700;
            text-align: left;
        }
        .syntax-group {
            background: #fefce8;
            font-weight: 700;
            white-space: nowrap;
            width: 28%;
        }
        </style>

        <table class="syntax-table">
            <thead>
                <tr>
                    <th>類別</th>
                    <th>可輸入語法與說明</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="syntax-group">基本代數</td>
                    <td>
                        <code>x^2</code>：x 的平方<br>
                        <code>x^3</code>：x 的立方<br>
                        <code>|x|</code>：x 的絕對值<br>
                        <code>2x</code>：<code>2*x</code><br>
                        <code>(2x+1)(x-3)</code>：<code>(2x+1)*(x-3)</code>
                    </td>
                </tr>
                <tr>
                    <td class="syntax-group">三角函數</td>
                    <td>
                        <code>sin(x)</code><br>
                        <code>cos(x)</code><br>
                        <code>tan(x)</code><br>
                    </td>
                </tr>
                <tr>
                    <td class="syntax-group">指數與根號</td>
                    <td>
                        <code>e^x</code>：e 的 x 次方<br>
                        <code>sqrt(x)</code>：x 的平方根
                    </td>
                </tr>
                <tr>
                    <td class="syntax-group">對數函數</td>
                    <td>
                        <code>ln(x)</code>：以 e 為底的自然對數<br>
                        <code>log10(x)</code>：以 10 為底的常用對數<br>
                        <code>log2(x)</code>：以 2 為底的對數
                    </td>
                </tr>
                <tr>
                    <td class="syntax-group">常數</td>
                    <td>
                        <code>pi</code>：圓周率<br>
                        <code>e</code>：自然常數
                    </td>
                </tr>
                <tr>
                    <td class="syntax-group">混合範例</td>
                    <td>
                        <code>x^2-3x+5</code><br>
                        <code>2x^2+3x-1</code><br>
                        <code>sin(x)+x^2</code><br>
                        <code>e^(-x)+2sin(x)</code><br>
                        <code>log(x)+x^2</code><br>
                        <code>sqrt(x+1)+x</code><br>
                        <code>|x-3|+2</code>
                    </td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

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
    st.subheader("圖形樣式")
    fill_pos_color = st.color_picker("面積塗色（x軸上方）", "#f4b183")
    fill_neg_color = st.color_picker("面積塗色（x軸下方）", "#9cc2e5")

    st.markdown("---")


try:
    f = parse_function(func_str)
    fname = normalize_function_input(func_str)
    test_x = np.linspace(domain_left, domain_right, 50)
    test_y = np.asarray(f(test_x), dtype=float)

    if not np.all(np.isfinite(test_y)):
        st.error("函數在此區間內出現無效值，請調整函數或區間。若使用 ln(x)、log10(x)、log2(x)，請設定區間滿足 x > 0，例如 a = 1、b = 5。")
        st.stop()

except Exception:
    st.error("函數輸入錯誤。可輸入例如：x^2、2x、3(x+1)、2sin(x)、|x|、sqrt(x+1)、ln(x)、log10(x)、log2(x)")
    st.stop()

if "m1a" not in st.session_state:
    st.session_state["m1a"] = float(min(max(0.0, domain_left), domain_right))
if "m2a" not in st.session_state:
    st.session_state["m2a"] = float(min(max(0.0, domain_left), domain_right))
if "m4a" not in st.session_state:
    st.session_state["m4a"] = float(min(max(0.0, domain_left), domain_right))
if "m1x_raw" not in st.session_state:
    st.session_state["m1x_raw"] = float((domain_left + domain_right) / 2)
if "m1z_raw" not in st.session_state:
    st.session_state["m1z_raw"] = float((domain_left + domain_right) / 2)
if "m4b_raw" not in st.session_state:
    st.session_state["m4b_raw"] = float(min(domain_right, 2.0))
if "m1_saved_a_curves" not in st.session_state:
    st.session_state["m1_saved_a_curves"] = []
if "m1_saved_curve_color_idx" not in st.session_state:
    st.session_state["m1_saved_curve_color_idx"] = 0
if "m2_saved_a_curves" not in st.session_state:
    st.session_state["m2_saved_a_curves"] = []
if "m2_saved_curve_color_idx" not in st.session_state:
    st.session_state["m2_saved_curve_color_idx"] = 0

show_help = False
show_formula = True

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


def enforce_m1x_not_below_a():
    a_val = float(st.session_state.get("m1a", domain_left))
    raw_val = float(st.session_state.get("m1x_raw", a_val))
    if raw_val < a_val:
        st.session_state["m1x_raw"] = a_val

def enforce_m1z_not_above_a():
    a_val = float(st.session_state.get("m1a", domain_left))
    raw_val = float(st.session_state.get("m1z_raw", a_val))
    if raw_val > a_val:
        st.session_state["m1z_raw"] = a_val

def enforce_m4b_not_below_a():
    a_val = float(st.session_state.get("m4a", domain_left))
    raw_val = float(st.session_state.get("m4b_raw", a_val))
    if raw_val < a_val:
        st.session_state["m4b_raw"] = a_val

# -----------------------------
# Tabs
# -----------------------------
module1, module2, module4 = st.tabs([
    "模組 1｜原函數f(x)動態生成累積函數A(x)",
    "模組 2｜累積函數 A(x) 的導函數等於原函數 f(x)",
    "模組 3｜FTC Part 2 幾何意義",
])

# -----------------------------
# Module 1
# -----------------------------
with module1:
    st.subheader("模組 1：原函數f(x)動態生成累積函數A(x)")
    st.markdown(
        """
        <div class="module-toolbar">
            <div class="module-chip">步驟 1：輸入原函數f(x)</div>
            <div class="module-chip">步驟 2：設定固定點 a</div>
            <div class="module-chip">步驟 3：拖動 x</div>
            <div class="module-chip">步驟 4：觀察面積累積值</div>
            <div class="module-chip">步驟 5：對照 A(x) 上的點坐標</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if show_formula:
        st.markdown(
            '<div style="text-align:center; padding: 0.6rem 0 0.9rem 0;">',
            unsafe_allow_html=True
        )
        st.latex(r"\Huge A(x)=\int_a^x f(t)\,dt")
    
    top_formula_col, top_control_col = st.columns([1.05, 0.95], gap="large")

    with top_control_col:
        st.markdown(
            '<div style="font-size:0.98rem; color:#60758c; margin:0.15rem 0 0.35rem 0;"><b>控制區</b></div>',
            unsafe_allow_html=True
        )
        a = st.slider(
            "固定點 a",
            min_value=float(domain_left),
            max_value=float(domain_right),
            value=float(st.session_state.get("m1a", min(max(0.0, domain_left), domain_right))),
            step=0.05,
            key="m1a",
        )
        if st.session_state.get("m1x_raw", (domain_left + domain_right) / 2) < a:
            st.session_state["m1x_raw"] = float(a)
        x1 = st.slider(
            "向右拖動x",
            min_value=float(domain_left),
            max_value=float(domain_right),
            value=float(st.session_state.get("m1x_raw", max((domain_left + domain_right) / 2, float(a)))),
            step=0.05,
            key="m1x_raw",
            on_change=enforce_m1x_not_below_a,
        )
        x1 = float(max(x1, a))
        if st.session_state.get("m1z_raw", (domain_left + domain_right) / 2) > a:
            st.session_state["m1z_raw"] = float(a)
        z1 = st.slider(
            "向左拖動x",
            min_value=float(domain_left),
            max_value=float(domain_right),
            value=float(st.session_state.get("m1z_raw", min((domain_left + domain_right) / 2, float(a)))),
            step=0.05,
            key="m1z_raw",
            on_change=enforce_m1z_not_above_a,
        )
        z1 = float(min(z1, a))
        button_col_left, button_col_right = st.columns(2, gap="small")
        with button_col_left:
            if st.button("留下固定點a的累積函數圖形", key="m1_save_a_curve", use_container_width=True):
                saved_curve = cumulative_integral(f, a, xs)
                color_idx = int(st.session_state.get("m1_saved_curve_color_idx", 0))
                curve_color = RAINBOW_COLORS[color_idx % len(RAINBOW_COLORS)]
                st.session_state["m1_saved_curve_color_idx"] = color_idx + 1
                st.session_state["m1_saved_a_curves"].append(
                    {
                        "a": float(a),
                        "curve": np.array(saved_curve, dtype=float),
                        "color": curve_color,
                    }
                )
        with button_col_right:
            if st.button("清除留下的圖形", key="m1_clear_saved_curves", use_container_width=True):
                st.session_state["m1_saved_a_curves"] = []
                st.session_state["m1_saved_curve_color_idx"] = 0
        show_full_A_curve = st.checkbox("顯示累積函數全部圖形", value=False, key="m1_show_full_curve")

    components.html(
        """
        <script>
        const repaintAllModuleSliders = () => {
            const doc = window.parent.document;
            const sliders = doc.querySelectorAll('div[data-testid="stSlider"]');
            sliders.forEach((slider) => {
                const trackBits = slider.querySelectorAll('div[data-baseweb="slider"] div');
                trackBits.forEach((el) => {
                    const style = window.parent.getComputedStyle(el);
                    const h = parseFloat(style.height || "0");
                    const w = parseFloat(style.width || "0");
                    const radius = parseFloat(style.borderTopLeftRadius || "0");

                    const isThumbLike = h >= 12 && w >= 12 && Math.abs(h - w) <= 6 && radius >= 8;
                    const isTrackLike = h > 0 && h <= 8 && w > 20;

                    if (isTrackLike) {
                        el.style.background = "#d9dee7";
                        el.style.backgroundColor = "#d9dee7";
                        el.style.borderColor = "#d9dee7";
                        el.style.boxShadow = "none";
                    }

                    if (isThumbLike) {
                        el.style.background = "#ff4b4b";
                        el.style.backgroundColor = "#ff4b4b";
                        el.style.borderColor = "#ff4b4b";
                    }
                });
            });
        };

        repaintAllModuleSliders();
        const intervalId = setInterval(repaintAllModuleSliders, 500);

        window.addEventListener("beforeunload", () => clearInterval(intervalId));
        </script>
        """,
        height=0,
    )

    Axs = cumulative_integral(f, a, xs)
    current_A = np.interp(x1, xs, Axs)
    current_f = f(np.array([x1]))[0]
    current_Z = np.interp(z1, xs, Axs)
    current_fz = f(np.array([z1]))[0]
    mask = (xs >= min(a, x1)) & (xs <= max(a, x1))
    mask_z = (xs >= min(z1, a)) & (xs <= max(z1, a))
    m1_axis_positions = [a, x1, z1]
    m1_axis_levels = get_axis_label_levels(m1_axis_positions, threshold=0.45)
    m1_axis_y_offsets = [-0.15 - 0.32 * lvl for lvl in m1_axis_levels]

    with top_formula_col:
        st.markdown('<div style="padding: 1.2rem 0 0.3rem 0;">', unsafe_allow_html=True)

        show_left_formula = st.checkbox("顯示向左累積算式", value=False, key="m1_show_left_formula")
        if show_left_formula:
            st.latex(
                rf"\Large A({{\color{{green}}{{{z1:.2f}}}}})=\int_{{\color{{red}}{{{a:.2f}}}}}^{{\color{{green}}{{{z1:.2f}}}}} f(t)\,dt"
                rf"={current_Z:.4f}"
            )
        else:
            st.markdown('<div style="height: 2.2rem;"></div>', unsafe_allow_html=True)

        st.markdown('<div style="height: 0.9rem;"></div>', unsafe_allow_html=True)

        show_right_formula = st.checkbox("顯示向右累積算式", value=False, key="m1_show_right_formula")
        if show_right_formula:
            st.latex(
                rf"\Large A({{\color{{green}}{{{x1:.2f}}}}})=\int_{{\color{{red}}{{{a:.2f}}}}}^{{\color{{green}}{{{x1:.2f}}}}} f(t)\,dt"
                rf"={current_A:.4f}"
            )
        else:
            st.markdown('<div style="height: 2.2rem;"></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    chart_col_left, chart_col_right = st.columns(2, gap="large")

    # 累積函數只顯示到目前滑桿位置，形成「逐漸長出來」的效果
    if x1 >= domain_left:
        mask_A = xs <= x1
    else:
        mask_A = xs >= x1

    with chart_col_left:
        fig12, ax12 = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)

        for saved_item in st.session_state.get("m1_saved_a_curves", []):
            saved_a = float(saved_item["a"])
            saved_curve = np.array(saved_item["curve"], dtype=float)
            saved_color = saved_item.get("color", "#9fd8b3")
            ax12.plot(xs, saved_curve, linewidth=2.2, color=saved_color, alpha=0.70)
            saved_y_at_a = np.interp(saved_a, xs, saved_curve)
            ax12.scatter([saved_a], [saved_y_at_a], s=34, color="#d84a4a", zorder=7)

        if show_full_A_curve:
            mask_A_display = np.full_like(xs, True, dtype=bool)
            ax12.plot(xs[mask_A_display], Axs[mask_A_display], linewidth=4.2, color="#8fc9a8")
        else:
            mask_A_display = (xs >= a) & mask_A
            mask_Z_display = (xs >= z1) & (xs <= a)
            ax12.plot(xs[mask_A_display], Axs[mask_A_display], linewidth=4.2, color="#8fc9a8")
            ax12.plot(xs[mask_Z_display], Axs[mask_Z_display], linewidth=4.2, color="#8fc9a8")
        draw_to_x_axis(ax12, a, np.interp(a, xs, Axs), "#f2a3c7", linewidth=1.6, marker_size=45)
        ax12.text(
            a,
            0 + m1_axis_y_offsets[0] - 0.02,
            f"{a:.2f}",
            **pretty_a_label_kwargs(),
        )
        draw_to_x_axis(ax12, x1, current_A, "#9bd18b", linewidth=1.6, marker_size=55)
        draw_to_x_axis(ax12, z1, current_Z, "#9bd18b", linewidth=1.6, marker_size=55)
        # 顯示 x 的數值（左圖綠色線與 x 軸交點）
        ax12.text(
            x1,
            0 + m1_axis_y_offsets[1],
            f"{x1:.2f}",
            ha="center",
            va="top",
            fontsize=13,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#d9e7d9", lw=0.6, alpha=0.92),
        )
        ax12.text(
            z1,
            0 + m1_axis_y_offsets[2],
            f"{z1:.2f}",
            ha="center",
            va="top",
            fontsize=13,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#d9e7d9", lw=0.6, alpha=0.92),
        )

        x_xytext = smart_point_xytext(
            x1, current_A, x_min_common, x_max_common, y_min_common, y_max_common, other_points=[(z1, current_Z)]
        )
        ax12.annotate(
            f"({x1:.2f}, {current_A:.2f})",
            xy=(x1, current_A),
            xytext=x_xytext,
            textcoords="offset points",
            color="#2f6f4f",
            fontsize=13.5,
            fontweight="semibold",
            bbox=dict(
                boxstyle="round,pad=0.24,rounding_size=0.18",
                fc="white",
                ec="#86c79d",
                lw=1.0,
                alpha=0.96,
            ),
            arrowprops=dict(arrowstyle="-", color="#86c79d", lw=1.0, alpha=0.9),
        )
        z_xytext = smart_point_xytext(
            z1, current_Z, x_min_common, x_max_common, y_min_common, y_max_common, other_points=[(x1, current_A)]
        )
        ax12.annotate(
            f"({z1:.2f}, {current_Z:.2f})",
            xy=(z1, current_Z),
            xytext=z_xytext,
            textcoords="offset points",
            color="#2f6f4f",
            fontsize=13.5,
            fontweight="semibold",
            bbox=dict(
                boxstyle="round,pad=0.24,rounding_size=0.18",
                fc="white",
                ec="#86c79d",
                lw=1.0,
                alpha=0.96,
            ),
            arrowprops=dict(arrowstyle="-", color="#86c79d", lw=1.0, alpha=0.9),
        )
        ax12.set_title("y=A(x)", fontsize=16, pad=14)
        ax12.set_xlabel("x", fontsize=12)
        ax12.set_ylabel("A(x)", fontsize=12)
        ax12.set_xlim(x_min_common, x_max_common)
        ax12.set_ylim(y_min_common, y_max_common)
        ax12.tick_params(labelsize=11)
        add_common_style(ax12)
        st.pyplot(fig12, use_container_width=True)

    with chart_col_right:
        fig11, ax11 = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
        ax11.plot(xs, ys, linewidth=4.2, color="#8bbce9")
        draw_to_x_axis(ax11, a, f(np.array([a]))[0], "#f2a3c7", linewidth=1.6, marker_size=45)
        ax11.text(
            a,
            0 + m1_axis_y_offsets[0] - 0.02,
            f"{a:.2f}",
            **pretty_a_label_kwargs(),
        )
        draw_to_x_axis(ax11, x1, current_f, "#9bd18b", linewidth=1.6, marker_size=55)
        draw_to_x_axis(ax11, z1, current_fz, "#9bd18b", linewidth=1.6, marker_size=55)
        # 顯示 x 的數值（綠色線與 x 軸交點）
        ax11.text(
            x1,
            0 + m1_axis_y_offsets[1],
            f"{x1:.2f}",
            ha="center",
            va="top",
            fontsize=13,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#d9e7d9", lw=0.6, alpha=0.92),
        )
        ax11.text(
            z1,
            0 + m1_axis_y_offsets[2],
            f"{z1:.2f}",
            ha="center",
            va="top",
            fontsize=13,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#d9e7d9", lw=0.6, alpha=0.92),
        )
        if x1 >= a:
            fill_area_by_sign(ax11, xs[mask], ys[mask], fill_pos_color, fill_neg_color, alpha=0.40)
            x_mid = (a + x1) / 2
            if abs(x1 - z1) < 0.90:
                x_mid = a + 0.72 * (x1 - a)
            ys_mask = ys[mask]
            positive_part = ys_mask[ys_mask >= 0]
            negative_part = ys_mask[ys_mask < 0]

            if current_A >= 0:
                if len(positive_part) > 0:
                    y_mid = 0.52 * np.max(positive_part)
                else:
                    y_mid = 0.38 * max(y_max_common, 1.0)
            else:
                if len(negative_part) > 0:
                    y_mid = 0.52 * np.min(negative_part)
                else:
                    y_mid = 0.38 * min(y_min_common, -1.0)

            ax11.text(
                x_mid,
                y_mid,
                f"{current_A:.2f}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="semibold",
                color="#2f2f2f",
                bbox=dict(
                    boxstyle="round,pad=0.28,rounding_size=0.16",
                    fc="white",
                    ec="#c9d2de",
                    lw=1.0,
                    alpha=0.94,
                ),
            )
        if z1 <= a:
            fill_area_by_sign(ax11, xs[mask_z], ys[mask_z], fill_pos_color, fill_neg_color, alpha=0.40)
            z_mid = (z1 + a) / 2
            ys_mask_z = ys[mask_z]
            positive_part_z = ys_mask_z[ys_mask_z >= 0]
            negative_part_z = ys_mask_z[ys_mask_z < 0]

            if current_Z >= 0:
                if len(positive_part_z) > 0:
                    y_mid_z = 0.52 * np.max(positive_part_z)
                else:
                    y_mid_z = 0.38 * max(y_max_common, 1.0)
            else:
                if len(negative_part_z) > 0:
                    y_mid_z = 0.52 * np.min(negative_part_z)
                else:
                    y_mid_z = 0.38 * min(y_min_common, -1.0)

            z_mid = z1 + 0.22 * (a - z1)
            if abs(x1 - z1) < 0.90:
                z_mid = z1 + 0.12 * (a - z1)
            ax11.text(
                z_mid,
                y_mid_z,
                f"{current_Z:.2f}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="semibold",
                color="#2f2f2f",
                bbox=dict(
                    boxstyle="round,pad=0.28,rounding_size=0.16",
                    fc="white",
                    ec="#c9d2de",
                    lw=1.0,
                    alpha=0.94,
                ),
            )
        ax11.set_title("y=f(x)", fontsize=16, pad=14)
        ax11.set_xlabel("x", fontsize=12)
        ax11.set_ylabel("f(x)", fontsize=12)
        ax11.set_xlim(x_min_common, x_max_common)
        ax11.set_ylim(y_min_common, y_max_common)
        ax11.tick_params(labelsize=11)
        add_common_style(ax11)
        st.pyplot(fig11, use_container_width=True)

    st.markdown(
        """
        <div style="margin-top:0.65rem; font-size:1rem; line-height:1.85; color:#42586f;">
        <b>觀察重點</b><br>
        1. 當 x 改變時，A(x) 也跟著改變。<br>
        2. A(x) 不是一個單一數值，而是一個會隨著 x 改變的新函數。
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Module 2
# -----------------------------
with module2:
    st.subheader("模組 2：累積函數 A(x) 的導函數等於原函數 f(x)")

    if show_formula:
        st.markdown(
            '<div style="text-align:center; padding: 0.6rem 0 0.9rem 0;">',
            unsafe_allow_html=True
        )
        st.latex(r"\Huge A(x)=\int_a^x f(t)\,dt \quad \Rightarrow \quad A'(x)=f(x)")
        st.markdown('</div>', unsafe_allow_html=True)

    full_width_col = st.container()
    with full_width_col:
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

        Axs_m2 = cumulative_integral(f, a2, xs)
        Aprime_m2 = safe_gradient(Axs_m2, xs)
        current_A2 = np.interp(x2, xs, Axs_m2)
        current_f2 = f(np.array([x2]))[0]
        current_Ap2 = np.interp(x2, xs, Aprime_m2)

        m2_button_col_left, m2_button_col_right = st.columns(2, gap="small")
        with m2_button_col_left:
            if st.button("留下固定點a的累積函數圖形", key="m2_save_a_curve", use_container_width=True):
                color_idx = int(st.session_state.get("m2_saved_curve_color_idx", 0))
                curve_color = RAINBOW_COLORS[color_idx % len(RAINBOW_COLORS)]
                st.session_state["m2_saved_curve_color_idx"] = color_idx + 1
                st.session_state["m2_saved_a_curves"].append(
                    {
                        "a": float(a2),
                        "curve": np.array(Axs_m2, dtype=float),
                        "color": curve_color,
                        "x": float(x2),
                        "A": float(current_A2),
                        "Aprime": float(current_Ap2),
                    }
                )
        with m2_button_col_right:
            if st.button("清除留下的圖形", key="m2_clear_saved_curves", use_container_width=True):
                st.session_state["m2_saved_a_curves"] = []
                st.session_state["m2_saved_curve_color_idx"] = 0

        m2_axis_positions = [a2, x2]
        m2_axis_levels = get_axis_label_levels(m2_axis_positions, threshold=0.45)
        m2_axis_y_offsets = [-0.17 - 0.32 * lvl for lvl in m2_axis_levels]

        if current_f2 > 1e-3:
            trend = "A(x) 正在上升"
        elif current_f2 < -1e-3:
            trend = "A(x) 正在下降"
        else:
            trend = "A(x) 在這附近斜率接近 0"

    left, right = st.columns(2, gap="large")
    with left:
        fig22, ax22 = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)

        for saved_item in st.session_state.get("m2_saved_a_curves", []):
            saved_curve = np.array(saved_item["curve"], dtype=float)
            saved_color = saved_item.get("color", "#9fd8b3")
            saved_x = float(saved_item["x"])
            saved_A = float(saved_item["A"])
            saved_Aprime = float(saved_item["Aprime"])

            ax22.plot(xs, saved_curve, linewidth=2.2, color=saved_color, alpha=0.70)
            ax22.scatter([saved_x], [saved_A], s=48, color="#9bd18b", zorder=7)

            saved_tangent_half_width = 1.05
            saved_tangent_x = np.linspace(
                max(x_min_common, saved_x - saved_tangent_half_width),
                min(x_max_common, saved_x + saved_tangent_half_width),
                40,
            )
            saved_tangent_y = saved_A + saved_Aprime * (saved_tangent_x - saved_x)
            ax22.plot(saved_tangent_x, saved_tangent_y, linewidth=2.6, color="#ffb347", alpha=0.72)

        ax22.plot(xs, Axs_m2, linewidth=3.4, color="#8fc9a8")
        draw_to_x_axis(ax22, a2, np.interp(a2, xs, Axs_m2), "#f2a3c7", linewidth=1.6, marker_size=45)
        ax22.text(
            a2,
            0 + m2_axis_y_offsets[0],
            f"{a2:.2f}",
            **pretty_a_label_kwargs(),
        )
        draw_to_x_axis(ax22, x2, current_A2, "#9bd18b", linewidth=1.6, marker_size=55)
        ax22.text(
            x2,
            0 + m2_axis_y_offsets[1],
            f"{x2:.2f}",
            ha="center",
            va="top",
            fontsize=13,
            bbox=smart_value_bbox(),
        )
        tangent_half_width = 1.05
        tangent_x = np.linspace(
            max(x_min_common, x2 - tangent_half_width),
            min(x_max_common, x2 + tangent_half_width),
            40,
        )
        tangent_y = current_A2 + current_Ap2 * (tangent_x - x2)
        ax22.plot(tangent_x, tangent_y, linewidth=3.0, color="#ffb347")

        tangent_label_x = min(max(x2 + 0.35, x_min_common + 0.35), x_max_common - 0.35)
        tangent_label_y = current_A2 + current_Ap2 * (tangent_label_x - x2)
        tangent_label_y = min(max(tangent_label_y + 0.35, y_min_common + 0.45), y_max_common - 0.45)
        ax22.text(
            tangent_label_x,
            tangent_label_y,
            rf"$A'({x2:.2f})={current_Ap2:.4f}$",
            ha="left",
            va="center",
            fontsize=14,
            fontweight="semibold",
            color="#b36b00",
            bbox=dict(
                boxstyle="round,pad=0.26,rounding_size=0.16",
                fc="white",
                ec="#ffb347",
                lw=1.0,
                alpha=0.95,
            ),
        )

        ax22.set_title("y=A(x)", fontsize=14)
        ax22.set_xlabel("x")
        ax22.set_ylabel("A(x)")
        ax22.set_xlim(x_min_common, x_max_common)
        ax22.set_ylim(y_min_common, y_max_common)
        add_common_style(ax22)
        st.pyplot(fig22, use_container_width=True)

    with right:
        fig2, ax2 = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
        ax2.plot(xs, ys, linewidth=3.4, label="f(x)", color="#8bbce9")
        draw_to_x_axis(ax2, a2, f(np.array([a2]))[0], "#f2a3c7", linewidth=1.6, marker_size=45)
        ax2.text(
            a2,
            0 + m2_axis_y_offsets[0],
            f"{a2:.2f}",
            **pretty_a_label_kwargs(),
        )
        draw_to_x_axis(ax2, x2, current_f2, "#9bd18b", linewidth=1.6, marker_size=55)
        ax2.text(
            x2,
            0 + m2_axis_y_offsets[1],
            f"{x2:.2f}",
            ha="center",
            va="top",
            fontsize=13,
            bbox=smart_value_bbox(),
        )
        m2_right_xytext = smart_point_xytext(
            x2, current_f2, x_min_common, x_max_common, y_min_common, y_max_common, other_points=[(a2, f(np.array([a2]))[0])]
        )
        ax2.annotate(
            f"({x2:.2f}, {current_f2:.2f})",
            xy=(x2, current_f2),
            xytext=m2_right_xytext,
            textcoords="offset points",
            color="#2f6f4f",
            fontsize=13.2,
            fontweight="semibold",
            bbox=dict(
                boxstyle="round,pad=0.24,rounding_size=0.18",
                fc="white",
                ec="#86c79d",
                lw=1.0,
                alpha=0.96,
            ),
            arrowprops=dict(arrowstyle="-", color="#86c79d", lw=1.0, alpha=0.9),
        )
        ax2.set_title("y=f(x)", fontsize=14)
        ax2.set_xlabel("x")
        ax2.set_ylabel("f(x)")
        ax2.set_xlim(x_min_common, x_max_common)
        ax2.set_ylim(y_min_common, y_max_common)
        ax2.legend()
        add_common_style(ax2)
        st.pyplot(fig2, use_container_width=True)

    components.html(
        """
        <div style="width:100%; padding:0.15rem 0 0 0; background:transparent;">
            <script>
                window.MathJax = {
                    tex: {
                        inlineMath: [['\\(', '\\)']],
                        displayMath: [['\\[', '\\]']]
                    },
                    svg: {fontCache: 'global'}
                };
            </script>
            <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>

            <div style="
                width:100%;
                min-height:520px;
                padding:0.1rem 0 0 0.4rem;
                box-sizing:border-box;
                text-align:left;
                color:#000;
            ">
                <div style="font-size:42px; line-height:1.65; text-align:left;">
                    <div style="margin-left:0px; margin-top:0px;">
                        \[
                        A(x+\Delta x)-A(x)\;\approx\; f(x)\cdot \Delta x
                        \]
                    </div>

                    <div style="margin-left:0px; margin-top:52px;">
                        \[
                        \frac{A(x+\Delta x)-A(x)}{\Delta x}\;\approx\; f(x)
                        \]
                    </div>

                    <div style="margin-left:240px; margin-top:84px;">
                        \[
                        A'(x)\;=\;f(x)
                        \]
                    </div>
                </div>
            </div>
        </div>
        """,
        height=560,
    )

# -----------------------------
# Module 4
# -----------------------------
with module4:
    st.subheader("模組 4：FTC Part 2 幾何意義")
    st.caption("把定積分看成原函數的總改變量，而不是一條要背的公式。")

    if show_formula:
        st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.latex(r"F'(x)=f(x) \quad \Rightarrow \quad \int_a^b f(x)\,dx = F(b)-F(a)")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="soft-control-box">', unsafe_allow_html=True)
    a = st.slider(
        "固定點 a",
        min_value=float(domain_left),
        max_value=float(domain_right),
        value=float(st.session_state.get("m4a", min(max(0.0, domain_left), domain_right))),
        step=0.05,
        key="m4a",
    )
    if st.session_state.get("m4b_raw", min(domain_right, 2.0)) < a:
        st.session_state["m4b_raw"] = float(a)
    b4 = st.slider(
        "選擇右端點 b",
        min_value=float(domain_left),
        max_value=float(domain_right),
        value=float(st.session_state.get("m4b_raw", max(min(domain_right, 2.0), float(a)))),
        step=0.05,
        key="m4b_raw",
        on_change=enforce_m4b_not_below_a,
    )
    b4 = float(max(b4, a))
    st.markdown('</div>', unsafe_allow_html=True)

    Axs_m4 = cumulative_integral(f, a, xs)
    def F_m4(x):
        arr = np.array(x, dtype=float)
        return np.interp(arr, xs, Axs_m4)

    b4_display = max(b4, a)
    exact_area = (F_m4(np.array([b4_display])) - F_m4(np.array([a])))[0]
    Fa = F_m4(np.array([a]))[0]
    Fb = F_m4(np.array([b4_display]))[0]
    m4_axis_positions = [a, b4_display]
    m4_axis_levels = get_axis_label_levels(m4_axis_positions, threshold=0.45)
    m4_axis_y_offsets = [-0.17 - 0.32 * lvl for lvl in m4_axis_levels]

    m4c1, m4c2, m4c3 = st.columns(3)
    m4c1.metric("F(a)", f"{Fa:.4f}")
    m4c2.metric("F(b)", f"{Fb:.4f}")
    m4c3.metric("F(b)-F(a)", f"{exact_area:.4f}")

    left, right = st.columns(2)
    with left:
        Fx = Axs_m4
        fig42, ax42 = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
        ax42.plot(xs, Fx, linewidth=3.4, color="#8bbce9")
        draw_to_x_axis(ax42, a, Fa, "#f2a3c7", linewidth=1.6, marker_size=45)
        ax42.text(
            a,
            0 + m4_axis_y_offsets[0],
            f"{a:.2f}",
            **pretty_a_label_kwargs(),
        )
        draw_to_x_axis(ax42, b4_display, Fb, "#9bd18b", linewidth=1.6, marker_size=55)
        ax42.text(
            b4_display,
            0 + m4_axis_y_offsets[1],
            f"{b4_display:.2f}",
            ha="center",
            va="top",
            fontsize=13,
            bbox=smart_value_bbox(),
        )
        m4_left_xytext_a = smart_point_xytext(
            a, Fa, x_min_common, x_max_common, y_min_common, y_max_common, other_points=[(b4_display, Fb)]
        )
        ax42.annotate(
            f"({a:.2f}, {Fa:.2f})",
            xy=(a, Fa),
            xytext=m4_left_xytext_a,
            textcoords="offset points",
            color="#c45a7a",
            fontsize=13.0,
            fontweight="semibold",
            bbox=dict(
                boxstyle="round,pad=0.22,rounding_size=0.16",
                fc="white",
                ec="#f2b3c8",
                lw=1.0,
                alpha=0.96,
            ),
            arrowprops=dict(arrowstyle="-", color="#f2b3c8", lw=1.0, alpha=0.9),
        )
        m4_left_xytext_b = smart_point_xytext(
            b4_display, Fb, x_min_common, x_max_common, y_min_common, y_max_common, other_points=[(a, Fa)]
        )
        ax42.annotate(
            f"({b4_display:.2f}, {Fb:.2f})",
            xy=(b4_display, Fb),
            xytext=m4_left_xytext_b,
            textcoords="offset points",
            color="#2f6f4f",
            fontsize=13.0,
            fontweight="semibold",
            bbox=dict(
                boxstyle="round,pad=0.22,rounding_size=0.16",
                fc="white",
                ec="#86c79d",
                lw=1.0,
                alpha=0.96,
            ),
            arrowprops=dict(arrowstyle="-", color="#86c79d", lw=1.0, alpha=0.9),
        )
        delta_x = a + 0.55 * (b4_display - a)
        delta_y = Fa + 0.62 * (Fb - Fa) + (0.18 if abs(Fb - Fa) < 0.8 else 0.0)
        ax42.text(
            delta_x,
            delta_y,
            f"ΔF = {exact_area:.2f}",
            ha="center",
            va="center",
            fontsize=13.4,
            fontweight="semibold",
            color="#2f2f2f",
            bbox=smart_area_bbox(),
        )
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
        fa4 = f(np.array([a]))[0]
        fb4 = f(np.array([b4_display]))[0]
        draw_to_x_axis(ax4, a, fa4, "#f2a3c7", linewidth=1.6, marker_size=45)
        ax4.text(
            a,
            0 + m4_axis_y_offsets[0],
            f"{a:.2f}",
            **pretty_a_label_kwargs(),
        )
        draw_to_x_axis(ax4, b4_display, fb4, "#9bd18b", linewidth=1.6, marker_size=55)
        ax4.text(
            b4_display,
            0 + m4_axis_y_offsets[1],
            f"{b4_display:.2f}",
            ha="center",
            va="top",
            fontsize=13,
            bbox=smart_value_bbox(),
        )
        m4_right_xytext_a = smart_point_xytext(
            a, fa4, x_min_common, x_max_common, y_min_common, y_max_common, other_points=[(b4_display, fb4)]
        )
        ax4.annotate(
            f"({a:.2f}, {fa4:.2f})",
            xy=(a, fa4),
            xytext=m4_right_xytext_a,
            textcoords="offset points",
            color="#c45a7a",
            fontsize=13.0,
            fontweight="semibold",
            bbox=dict(
                boxstyle="round,pad=0.22,rounding_size=0.16",
                fc="white",
                ec="#f2b3c8",
                lw=1.0,
                alpha=0.96,
            ),
            arrowprops=dict(arrowstyle="-", color="#f2b3c8", lw=1.0, alpha=0.9),
        )
        m4_right_xytext_b = smart_point_xytext(
            b4_display, fb4, x_min_common, x_max_common, y_min_common, y_max_common, other_points=[(a, fa4)]
        )
        ax4.annotate(
            f"({b4_display:.2f}, {fb4:.2f})",
            xy=(b4_display, fb4),
            xytext=m4_right_xytext_b,
            textcoords="offset points",
            color="#2f6f4f",
            fontsize=13.0,
            fontweight="semibold",
            bbox=dict(
                boxstyle="round,pad=0.22,rounding_size=0.16",
                fc="white",
                ec="#86c79d",
                lw=1.0,
                alpha=0.96,
            ),
            arrowprops=dict(arrowstyle="-", color="#86c79d", lw=1.0, alpha=0.9),
        )
        mask4 = (xs >= a) & (xs <= b4_display)
        fill_area_by_sign(ax4, xs[mask4], ys[mask4], fill_pos_color, fill_neg_color, alpha=0.30)
        ys_mask4 = ys[mask4]
        positive_part4 = ys_mask4[ys_mask4 >= 0]
        negative_part4 = ys_mask4[ys_mask4 < 0]
        area_x_mid4 = a + 0.58 * (b4_display - a)
        if exact_area >= 0:
            if len(positive_part4) > 0:
                area_y_mid4 = 0.50 * np.max(positive_part4)
            else:
                area_y_mid4 = 0.35 * max(y_max_common, 1.0)
        else:
            if len(negative_part4) > 0:
                area_y_mid4 = 0.50 * np.min(negative_part4)
            else:
                area_y_mid4 = 0.35 * min(y_min_common, -1.0)
        ax4.text(
            area_x_mid4,
            area_y_mid4,
            f"{exact_area:.2f}",
            ha="center",
            va="center",
            fontsize=13.4,
            fontweight="semibold",
            color="#2f2f2f",
            bbox=smart_area_bbox(),
        )
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

