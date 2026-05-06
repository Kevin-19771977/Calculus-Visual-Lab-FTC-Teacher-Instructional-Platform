import re
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="微積分基本定理教學平台",
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
        <h1 style="margin-bottom:0.4rem;">📘 微積分基本定理教學平台</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    selected_module = st.radio(
        "模組切換",
        options=[
            "模組 1｜原函數 f(x) 動態生成累積函數 A(x)",
            "模組 2｜累積函數 A(x) 的導函數等於原函數 f(x)",
            "模組 3｜用累積函數 A(x) 的端點差求定積分",
        ],
        index=0,
        key="selected_module",
    )

    if selected_module.startswith("模組 1"):
        selected_module_key = "module1"
    elif selected_module.startswith("模組 2"):
        selected_module_key = "module2"
    else:
        selected_module_key = "module3"

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
    st.markdown("### 函數圖形顯示範圍")
    st.markdown(
        """
        <div style="text-align:center; color:#52667a; font-size:0.9rem; margin-bottom:0.15rem;">
            y 軸上界
        </div>
        """,
        unsafe_allow_html=True,
    )
    y_max_input = st.number_input("y 軸顯示範圍：上界", value=5.0, step=0.5, format="%.2f", label_visibility="collapsed")

    left_input_col, right_input_col = st.columns(2)
    with left_input_col:
        st.markdown('<div style="text-align:center; color:#52667a; font-size:0.9rem; margin-bottom:0.15rem;">x 軸左端點</div>', unsafe_allow_html=True)
        domain_left = st.number_input("x 軸顯示範圍：左端點", value=-3.0, step=0.5, format="%.2f", label_visibility="collapsed")
    with right_input_col:
        st.markdown('<div style="text-align:center; color:#52667a; font-size:0.9rem; margin-bottom:0.15rem;">x 軸右端點</div>', unsafe_allow_html=True)
        domain_right = st.number_input("x 軸顯示範圍：右端點", value=3.0, step=0.5, format="%.2f", label_visibility="collapsed")

    st.markdown(
        """
        <div style="text-align:center; color:#52667a; font-size:0.9rem; margin:0.15rem 0;">
            y 軸下界
        </div>
        """,
        unsafe_allow_html=True,
    )
    y_min_input = st.number_input("y 軸顯示範圍：下界", value=-5.0, step=0.5, format="%.2f", label_visibility="collapsed")

    if domain_right <= domain_left:
        st.warning("右端點必須大於左端點，已暫時使用預設區間 [-3, 3]。")
        domain_left, domain_right = -3.0, 3.0

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

m1_initial_value = float(min(max(1.0, domain_left), domain_right))
if "m1_defaults_initialized" not in st.session_state:
    st.session_state["m1a"] = m1_initial_value
    st.session_state["m1x_raw"] = m1_initial_value
    st.session_state["m1z_raw"] = m1_initial_value
    st.session_state["m1_defaults_initialized"] = True
else:
    if "m1a" not in st.session_state:
        st.session_state["m1a"] = m1_initial_value
    if "m1x_raw" not in st.session_state:
        st.session_state["m1x_raw"] = m1_initial_value
    if "m1z_raw" not in st.session_state:
        st.session_state["m1z_raw"] = m1_initial_value

st.session_state["m1a"] = float(min(max(st.session_state["m1a"], domain_left), domain_right))
st.session_state["m1x_raw"] = float(min(max(st.session_state["m1x_raw"], domain_left), domain_right))
st.session_state["m1z_raw"] = float(min(max(st.session_state["m1z_raw"], domain_left), domain_right))

m2_initial_value = float(min(max(1.0, domain_left), domain_right))
m2_dx_initial_value = 1.0
if "m2_defaults_initialized" not in st.session_state:
    st.session_state["m2a"] = m2_initial_value
    st.session_state["m2x"] = m2_initial_value
    st.session_state["m2dx"] = m2_dx_initial_value
    st.session_state["m2_show_tangent"] = False
    st.session_state["m2_show_secant"] = False
    st.session_state["m2_defaults_initialized"] = True
else:
    if "m2a" not in st.session_state:
        st.session_state["m2a"] = m2_initial_value
    if "m2x" not in st.session_state:
        st.session_state["m2x"] = m2_initial_value
    if "m2dx" not in st.session_state:
        st.session_state["m2dx"] = m2_dx_initial_value

st.session_state["m2a"] = float(min(max(st.session_state["m2a"], domain_left), domain_right))
st.session_state["m2x"] = float(min(max(st.session_state["m2x"], domain_left), domain_right))
st.session_state["m2dx"] = float(min(max(st.session_state["m2dx"], 0.01), 1.0))

if "m4a" not in st.session_state:
    st.session_state["m4a"] = float(min(max(0.0, domain_left), domain_right))
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
if "m2_show_tangent" not in st.session_state:
    st.session_state["m2_show_tangent"] = False
if "m2_show_secant" not in st.session_state:
    st.session_state["m2_show_secant"] = False

# 第一次進入程式時，清除所有先前留下的圖形紀錄；
# 之後學生按下「留下圖形」仍可正常保存，不會在 rerun 時被清除。
if "saved_curves_cleaned_on_start" not in st.session_state:
    st.session_state["m1_saved_a_curves"] = []
    st.session_state["m1_saved_curve_color_idx"] = 0
    st.session_state["m2_saved_a_curves"] = []
    st.session_state["m2_saved_curve_color_idx"] = 0
    st.session_state["m3_saved_a_curves"] = []
    st.session_state["m3_saved_curve_color_idx"] = 0
    st.session_state["saved_curves_cleaned_on_start"] = True

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

a_default = float(st.session_state.get("m1a", min(max(1.0, domain_left), domain_right)))
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
# Module display is selected from the sidebar
# -----------------------------

# -----------------------------
# Module 1
# -----------------------------
if selected_module_key == "module1":
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
        a = st.slider(
            "固定點 a",
            min_value=float(domain_left),
            max_value=float(domain_right),
            value=float(st.session_state.get("m1a", m1_initial_value)),
            step=0.05,
            key="m1a",
        )
        if st.session_state.get("m1x_raw", m1_initial_value) < a:
            st.session_state["m1x_raw"] = float(a)
        x1 = st.slider(
            "向右拖動x",
            min_value=float(domain_left),
            max_value=float(domain_right),
            value=float(st.session_state.get("m1x_raw", max(m1_initial_value, float(a)))),
            step=0.05,
            key="m1x_raw",
            on_change=enforce_m1x_not_below_a,
        )
        x1 = float(max(x1, a))
        if st.session_state.get("m1z_raw", m1_initial_value) > a:
            st.session_state["m1z_raw"] = float(a)
        z1 = st.slider(
            "向左拖動x",
            min_value=float(domain_left),
            max_value=float(domain_right),
            value=float(st.session_state.get("m1z_raw", min(m1_initial_value, float(a)))),
            step=0.05,
            key="m1z_raw",
            on_change=enforce_m1z_not_above_a,
        )
        z1 = float(min(z1, a))

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
    with top_formula_col:
        st.markdown('<div style="padding: 1.2rem 0 0.3rem 0;">', unsafe_allow_html=True)

        show_left_formula = st.checkbox("向左累積算式", value=False, key="m1_show_left_formula")
        if show_left_formula:
            st.latex(
                rf"\Large A({{\color{{green}}{{{z1:.2f}}}}})=\int_{{\color{{red}}{{{a:.2f}}}}}^{{\color{{green}}{{{z1:.2f}}}}} f(t)\,dt"
                rf"={current_Z:.4f}"
            )
        else:
            st.markdown('<div style="height: 2.2rem;"></div>', unsafe_allow_html=True)

        st.markdown('<div style="height: 0.9rem;"></div>', unsafe_allow_html=True)

        show_right_formula = st.checkbox("向右累積算式", value=False, key="m1_show_right_formula")
        if show_right_formula:
            st.latex(
                rf"\Large A({{\color{{green}}{{{x1:.2f}}}}})=\int_{{\color{{red}}{{{a:.2f}}}}}^{{\color{{green}}{{{x1:.2f}}}}} f(t)\,dt"
                rf"={current_A:.4f}"
            )
        else:
            st.markdown('<div style="height: 2.2rem;"></div>', unsafe_allow_html=True)

        button_col_left, button_col_right = st.columns(2, gap="small")
        with button_col_left:
            if st.button("留下圖形", key="m1_save_a_curve", use_container_width=True):
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
            if st.button("清除圖形", key="m1_clear_saved_curves", use_container_width=True):
                st.session_state["m1_saved_a_curves"] = []
                st.session_state["m1_saved_curve_color_idx"] = 0
        show_full_A_curve = st.checkbox("顯示全部圖形", value=False, key="m1_show_full_curve")
        st.markdown('</div>', unsafe_allow_html=True)

    m1_axis_positions = [a]
    m1_axis_keys = ["a"]
    if show_right_formula:
        m1_axis_positions.append(x1)
        m1_axis_keys.append("x")
    if show_left_formula:
        m1_axis_positions.append(z1)
        m1_axis_keys.append("z")
    m1_axis_levels = get_axis_label_levels(m1_axis_positions, threshold=0.45)
    m1_axis_y_offsets = {
        key: -0.15 - 0.32 * lvl
        for key, lvl in zip(m1_axis_keys, m1_axis_levels)
    }

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
            if show_right_formula:
                mask_A_display = (xs >= a) & mask_A
                ax12.plot(xs[mask_A_display], Axs[mask_A_display], linewidth=4.2, color="#8fc9a8")
            if show_left_formula:
                mask_Z_display = (xs >= z1) & (xs <= a)
                ax12.plot(xs[mask_Z_display], Axs[mask_Z_display], linewidth=4.2, color="#8fc9a8")
        draw_to_x_axis(ax12, a, np.interp(a, xs, Axs), "#f2a3c7", linewidth=1.6, marker_size=45)
        ax12.text(
            a,
            0 + m1_axis_y_offsets["a"] - 0.02,
            f"{a:.2f}",
            **pretty_a_label_kwargs(),
        )
        if show_right_formula:
            draw_to_x_axis(ax12, x1, current_A, "#9bd18b", linewidth=1.6, marker_size=55)
            # 顯示 x 的數值（左圖綠色線與 x 軸交點）
            ax12.text(
                x1,
                0 + m1_axis_y_offsets["x"],
                f"{x1:.2f}",
                ha="center",
                va="top",
                fontsize=13,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#d9e7d9", lw=0.6, alpha=0.92),
            )

            x_xytext = smart_point_xytext(
                x1, current_A, x_min_common, x_max_common, y_min_common, y_max_common,
                other_points=[(z1, current_Z)] if show_left_formula else []
            )
            ax12.annotate(
                f"A({x1:.2f})={current_A:.4f}",
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

        if show_left_formula:
            draw_to_x_axis(ax12, z1, current_Z, "#9bd18b", linewidth=1.6, marker_size=55)
            ax12.text(
                z1,
                0 + m1_axis_y_offsets["z"],
                f"{z1:.2f}",
                ha="center",
                va="top",
                fontsize=13,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#d9e7d9", lw=0.6, alpha=0.92),
            )

            z_xytext = smart_point_xytext(
                z1, current_Z, x_min_common, x_max_common, y_min_common, y_max_common,
                other_points=[(x1, current_A)] if show_right_formula else []
            )
            ax12.annotate(
                f"A({z1:.2f})={current_Z:.4f}",
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
            0 + m1_axis_y_offsets["a"] - 0.02,
            f"{a:.2f}",
            **pretty_a_label_kwargs(),
        )
        if show_right_formula:
            draw_to_x_axis(ax11, x1, current_f, "#9bd18b", linewidth=1.6, marker_size=55)
            # 顯示 x 的數值（綠色線與 x 軸交點）
            ax11.text(
                x1,
                0 + m1_axis_y_offsets["x"],
                f"{x1:.2f}",
                ha="center",
                va="top",
                fontsize=13,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#d9e7d9", lw=0.6, alpha=0.92),
            )
        if show_left_formula:
            draw_to_x_axis(ax11, z1, current_fz, "#9bd18b", linewidth=1.6, marker_size=55)
            ax11.text(
                z1,
                0 + m1_axis_y_offsets["z"],
                f"{z1:.2f}",
                ha="center",
                va="top",
                fontsize=13,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#d9e7d9", lw=0.6, alpha=0.92),
            )
        if show_right_formula and x1 >= a:
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
        if show_left_formula and z1 <= a:
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
        <div style="margin-top:0.65rem; font-size:1.18rem; line-height:1.9; color:#42586f;">
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
if selected_module_key == "module2":
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
        m2_left_control_col, m2_right_slider_col = st.columns([0.45, 0.55], gap="large")

        with m2_right_slider_col:
            a2 = st.slider(
                "固定點 a",
                min_value=float(domain_left),
                max_value=float(domain_right),
                value=float(st.session_state.get("m2a", m2_initial_value)),
                step=0.05,
                key="m2a",
            )
            x2 = st.slider(
                "拖動 x",
                min_value=float(domain_left),
                max_value=float(domain_right),
                value=float(st.session_state.get("m2x", m2_initial_value)),
                step=0.05,
                key="m2x",
            )
            dx2 = st.slider(
                "Δx",
                min_value=0.01,
                max_value=1.0,
                value=float(st.session_state.get("m2dx", m2_dx_initial_value)),
                step=0.01,
                key="m2dx",
            )

        components.html(
            """
            <script>
            const repaintModule2Sliders = () => {
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

            repaintModule2Sliders();
            const module2SliderIntervalId = setInterval(repaintModule2Sliders, 500);
            window.addEventListener("beforeunload", () => clearInterval(module2SliderIntervalId));
            </script>
            """,
            height=0,
        )

        Axs_m2 = cumulative_integral(f, a2, xs)
        Aprime_m2 = safe_gradient(Axs_m2, xs)
        current_A2 = np.interp(x2, xs, Axs_m2)
        current_f2 = f(np.array([x2]))[0]
        current_Ap2 = np.interp(x2, xs, Aprime_m2)
        x2_plus = float(min(x2 + dx2, x_max_common))
        current_A2_plus = np.interp(x2_plus, xs, Axs_m2)
        current_f2_plus = f(np.array([x2_plus]))[0]

        with m2_left_control_col:
            m2_check_col_left, m2_check_col_right = st.columns(2, gap="small")
            with m2_check_col_left:
                show_secant_m2 = st.checkbox("割線", key="m2_show_secant")
            with m2_check_col_right:
                show_tangent_m2 = st.checkbox("切線", key="m2_show_tangent")

            m2_button_col_left, m2_button_col_right = st.columns(2, gap="small")
            with m2_button_col_left:
                if st.button("留下圖形", key="m2_save_a_curve", use_container_width=True):
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
                            "show_secant": bool(show_secant_m2),
                            "x_plus": float(x2_plus),
                            "A_plus": float(current_A2_plus),
                            "secant_slope": float((current_A2_plus - current_A2) / max(x2_plus - x2, 1e-9)),
                        }
                    )
            with m2_button_col_right:
                if st.button("清除圖形", key="m2_clear_saved_curves", use_container_width=True):
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

            if show_tangent_m2:
                saved_tangent_half_width = 1.05
                saved_tangent_x = np.linspace(
                    max(x_min_common, saved_x - saved_tangent_half_width),
                    min(x_max_common, saved_x + saved_tangent_half_width),
                    40,
                )
                saved_tangent_y = saved_A + saved_Aprime * (saved_tangent_x - saved_x)
                ax22.plot(saved_tangent_x, saved_tangent_y, linewidth=2.6, color="#ffb347", alpha=0.72)

            if saved_item.get("show_secant", False):
                saved_x_plus = float(saved_item.get("x_plus", saved_x))
                saved_A_plus = float(saved_item.get("A_plus", saved_A))
                saved_secant_slope = float(saved_item.get("secant_slope", 0.0))
                ax22.scatter([saved_x_plus], [saved_A_plus], s=38, color="#9bd18b", zorder=7)
                saved_secant_half_width = 1.05
                saved_secant_center_x = 0.5 * (saved_x + saved_x_plus)
                saved_secant_center_y = 0.5 * (saved_A + saved_A_plus)
                saved_secant_x = np.linspace(
                    max(x_min_common, saved_secant_center_x - saved_secant_half_width),
                    min(x_max_common, saved_secant_center_x + saved_secant_half_width),
                    40,
                )
                saved_secant_y = saved_secant_center_y + saved_secant_slope * (saved_secant_x - saved_secant_center_x)
                ax22.plot(saved_secant_x, saved_secant_y, linewidth=2.6, color="#ffb347", alpha=0.72)

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
        if show_secant_m2:
            m2_left_x_xytext = smart_point_xytext(
                x2,
                current_A2,
                x_min_common,
                x_max_common,
                y_min_common,
                y_max_common,
                other_points=[(a2, np.interp(a2, xs, Axs_m2)), (x2_plus, current_A2_plus)] if show_secant_m2 else [(a2, np.interp(a2, xs, Axs_m2))],
            )
            ax22.annotate(
                f"A({x2:.2f})={current_A2:.4f}",
                xy=(x2, current_A2),
                xytext=m2_left_x_xytext,
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
        if show_secant_m2:
            draw_to_x_axis(ax22, x2_plus, current_A2_plus, "#9bd18b", linewidth=1.6, marker_size=36)
            m2_left_xplus_xytext = smart_point_xytext(
                x2_plus,
                current_A2_plus,
                x_min_common,
                x_max_common,
                y_min_common,
                y_max_common,
                other_points=[(a2, np.interp(a2, xs, Axs_m2)), (x2, current_A2)],
            )
            ax22.annotate(
                f"A({x2_plus:.2f})={current_A2_plus:.4f}",
                xy=(x2_plus, current_A2_plus),
                xytext=m2_left_xplus_xytext,
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

            secant_slope = (current_A2_plus - current_A2) / max(x2_plus - x2, 1e-9)
            secant_half_width = 1.05
            secant_center_x = 0.5 * (x2 + x2_plus)
            secant_center_y = 0.5 * (current_A2 + current_A2_plus)
            secant_x = np.linspace(
                max(x_min_common, secant_center_x - secant_half_width),
                min(x_max_common, secant_center_x + secant_half_width),
                40,
            )
            secant_y = secant_center_y + secant_slope * (secant_x - secant_center_x)
            ax22.plot(secant_x, secant_y, linewidth=3.0, color="#ffb347", zorder=6)

        if show_tangent_m2:
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
        ax2.plot(xs, ys, linewidth=3.4, color="#8bbce9")
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
        if show_secant_m2:
            ax2.plot(
                [x2_plus, x2_plus],
                [0, current_f2_plus],
                linestyle="--",
                linewidth=1.6,
                color="#9bd18b",
                zorder=6,
            )
            ax2.scatter([x2_plus], [current_f2_plus], s=36, color="#9bd18b", zorder=7)

            mask_m2_fill = (xs >= min(x2, x2_plus)) & (xs <= max(x2, x2_plus))
            fill_area_by_sign(ax2, xs[mask_m2_fill], ys[mask_m2_fill], fill_pos_color, fill_neg_color, alpha=0.40)

        if show_secant_m2 or show_tangent_m2:
            m2_right_xytext = smart_point_xytext(
                x2, current_f2, x_min_common, x_max_common, y_min_common, y_max_common, other_points=[(a2, f(np.array([a2]))[0])]
            )
            ax2.annotate(
                f"f({x2:.2f})={current_f2:.2f}",
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
        add_common_style(ax2)
        st.pyplot(fig2, use_container_width=True)

    st.markdown(
        """
        <style>
        .m2-derivation-title {
            font-size: 1.22rem;
            font-weight: 800;
            color: #38506a;
            margin: 0.35rem 0 0.55rem 0;
        }
        .m2-table-header {
            background: #f6f8fb;
            border: 1px solid #dfe6ef;
            border-radius: 12px;
            padding: 0.55rem 0.8rem;
            font-weight: 800;
            color: #38506a;
            text-align: center;
            margin-bottom: 0.35rem;
        }
        .m2-row-gap {
            height: 0.45rem;
        }
        .m2-centered-formula {
            text-align: center;
            width: 100%;
        }
        </style>
        
        """,
        unsafe_allow_html=True,
    )

    delta_A_value = current_A2_plus - current_A2
    rect_area_value = current_f2 * dx2

    mini_dx_actual = max(x2_plus - x2, 1e-6)
    mini_x_fixed_min = 0.0
    mini_x_fixed_max = 1.0
    mini_y_fixed_min = y_min_common
    mini_y_fixed_max = y_max_common

    xs_mini_global = xs[(xs >= x2) & (xs <= x2_plus)]
    ys_mini = ys[(xs >= x2) & (xs <= x2_plus)]

    if len(xs_mini_global) < 2:
        xs_mini_global = np.linspace(x2, x2_plus, 50)
        ys_mini = np.array(f(xs_mini_global), dtype=float)

    xs_mini_local = xs_mini_global - x2

    header_left, header_right = st.columns([0.45, 0.55], gap="large")
    with header_left:
        st.markdown('<div class="m2-table-header">一般推導式</div>', unsafe_allow_html=True)
    with header_right:
        st.markdown('<div class="m2-table-header">當下數值化對照</div>', unsafe_allow_html=True)

    row1_left, row1_right = st.columns([0.45, 0.55], gap="large")
    with row1_left:
        row1_left_card = st.container(border=True)
        with row1_left_card:
            st.latex(r"\Large A(x+\Delta x)-A(x)\;\approx\; f(x)\cdot \Delta x")
    with row1_right:
        row1_right_card = st.container(border=True)
        with row1_right_card:
            st.latex(rf"\Large A({x2:.2f}+{dx2:.2f})-A({x2:.2f})\;\approx\; f({x2:.2f})\cdot {dx2:.2f}")

            compare_col_left, compare_col_right = st.columns(2, gap="small")

            with compare_col_left:
                fig_mini_left, ax_mini_left = plt.subplots(figsize=(3.2, 2.6), constrained_layout=True)
                ax_mini_left.plot(xs_mini_local, ys_mini, linewidth=2.2, color="#8bbce9")
                fill_area_by_sign(
                    ax_mini_left,
                    xs_mini_local,
                    ys_mini,
                    fill_pos_color,
                    fill_neg_color,
                    alpha=0.45,
                )
                ax_mini_left.axhline(0, linewidth=1.2, color="#b0b0b0", zorder=0)
                area_text_x = 0.5 * mini_dx_actual
                if delta_A_value >= 0:
                    positive_part = ys_mini[ys_mini >= 0]
                    if len(positive_part) > 0:
                        area_text_y = 0.52 * np.max(positive_part)
                    else:
                        area_text_y = 0.35 * max(mini_y_fixed_max, 1.0)
                else:
                    negative_part = ys_mini[ys_mini < 0]
                    if len(negative_part) > 0:
                        area_text_y = 0.52 * np.min(negative_part)
                    else:
                        area_text_y = 0.35 * min(mini_y_fixed_min, -1.0)

                ax_mini_left.text(
                    area_text_x,
                    area_text_y,
                    f"{delta_A_value:.4f}",
                    ha="center",
                    va="center",
                    fontsize=11.5,
                    fontweight="semibold",
                    color="#2f2f2f",
                    bbox=dict(
                        boxstyle="round,pad=0.24,rounding_size=0.14",
                        fc="white",
                        ec="#c9d2de",
                        lw=0.9,
                        alpha=0.94,
                    ),
                )
                ax_mini_left.set_xlim(mini_x_fixed_min, mini_x_fixed_max)
                ax_mini_left.set_ylim(mini_y_fixed_min, mini_y_fixed_max)
                ax_mini_left.grid(alpha=0.18)
                for spine in ["top", "right"]:
                    ax_mini_left.spines[spine].set_visible(False)
                ax_mini_left.tick_params(labelsize=8.5)
                st.pyplot(fig_mini_left, use_container_width=True)

            with compare_col_right:
                fig_mini_right, ax_mini_right = plt.subplots(figsize=(3.2, 2.6), constrained_layout=True)
                x_rect = np.linspace(0.0, dx2, 50)
                y_rect = np.full_like(x_rect, float(current_f2))
                ax_mini_right.fill_between(
                    x_rect,
                    0,
                    y_rect,
                    color="#f6b6c8",
                    alpha=0.75,
                )
                ax_mini_right.plot([0.0, dx2], [current_f2, current_f2], color="#d97a9a", linewidth=2.0)
                ax_mini_right.plot([0.0, 0.0], [0, current_f2], color="#d97a9a", linewidth=2.0)
                ax_mini_right.plot([dx2, dx2], [0, current_f2], color="#d97a9a", linewidth=2.0)
                ax_mini_right.axhline(0, linewidth=1.2, color="#b0b0b0", zorder=0)

                rect_text_x = 0.5 * dx2
                rect_text_y = current_f2 / 2.0 if abs(current_f2) > 1e-9 else 0.0
                ax_mini_right.text(
                    rect_text_x,
                    rect_text_y,
                    f"{rect_area_value:.4f}",
                    ha="center",
                    va="center",
                    fontsize=11.5,
                    fontweight="semibold",
                    color="#2f2f2f",
                    bbox=dict(
                        boxstyle="round,pad=0.24,rounding_size=0.14",
                        fc="white",
                        ec="#dcb2bf",
                        lw=0.9,
                        alpha=0.94,
                    ),
                )
                ax_mini_right.set_xlim(mini_x_fixed_min, mini_x_fixed_max)
                ax_mini_right.set_ylim(mini_y_fixed_min, mini_y_fixed_max)
                ax_mini_right.grid(alpha=0.18)
                for spine in ["top", "right"]:
                    ax_mini_right.spines[spine].set_visible(False)
                ax_mini_right.tick_params(labelsize=8.5)
                st.pyplot(fig_mini_right, use_container_width=True)

    st.markdown('<div class="m2-row-gap"></div>', unsafe_allow_html=True)

    row2_left, row2_right = st.columns([0.45, 0.55], gap="large")
    with row2_left:
        row2_left_card = st.container(border=True)
        with row2_left_card:
            st.latex(r"{\huge \frac{A(x+\Delta x)-A(x)}{\Delta x}}\;\approx\;{\LARGE f(x)}")
    with row2_right:
        row2_right_card = st.container(border=True)
        with row2_right_card:
            st.latex(rf"{{\huge \frac{{A({x2:.2f}+{dx2:.2f})-A({x2:.2f})}}{{{dx2:.2f}}}}}\;\approx\;{{\LARGE f({x2:.2f})}}")

            slope_value_m2 = (current_A2_plus - current_A2) / dx2
            slope_value_col_left, slope_value_col_right = st.columns([0.74, 0.26], gap="small")
            with slope_value_col_left:
                st.markdown(
                    f"""
                    <div style="margin-top:0.85rem; text-align:center; font-size:2.1rem; font-weight:800; color:#1f77b4; line-height:1.35;">
                        {slope_value_m2:.4f}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with slope_value_col_right:
                st.markdown(
                    f"""
                    <div style="margin-top:0.85rem; text-align:center; font-size:2.1rem; font-weight:800; color:#d62728; line-height:1.35;">
                        {current_f2:.4f}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="m2-row-gap"></div>', unsafe_allow_html=True)

    row3_left, row3_right = st.columns([0.45, 0.55], gap="large")
    with row3_left:
        row3_left_card = st.container(border=True)
        with row3_left_card:
            st.latex(r"\LARGE A'(x)\;=\;f(x)")
    with row3_right:
        row3_right_card = st.container(border=True)
        with row3_right_card:
            st.latex(rf"\LARGE A'({x2:.2f})\;=\;f({x2:.2f})")


# -----------------------------
# Module 3
# -----------------------------
if selected_module_key == "module3":
    st.subheader("模組 3：用累積函數 A(x) 的端點差求定積分")

    m3_initial_value = float(min(max(1.0, domain_left), domain_right))

    def enforce_m3b_not_above_c():
        c_val = float(st.session_state.get("m3c", m3_initial_value))
        b_val = float(st.session_state.get("m3b", m3_initial_value))
        if b_val > c_val:
            st.session_state["m3b"] = c_val

    def enforce_m3c_not_below_b():
        c_val = float(st.session_state.get("m3c", m3_initial_value))
        b_val = float(st.session_state.get("m3b", m3_initial_value))
        if c_val < b_val:
            st.session_state["m3c"] = b_val

    if "m3_endpoint_defaults_initialized" not in st.session_state:
        st.session_state["m3a"] = m3_initial_value
        st.session_state["m3b"] = m3_initial_value
        st.session_state["m3c"] = m3_initial_value
        st.session_state["m3_endpoint_defaults_initialized"] = True
    else:
        if "m3a" not in st.session_state:
            st.session_state["m3a"] = m3_initial_value
        if "m3b" not in st.session_state:
            st.session_state["m3b"] = float(st.session_state.get("m3a", m3_initial_value))
        if "m3c" not in st.session_state:
            st.session_state["m3c"] = float(st.session_state.get("m3a", m3_initial_value))

    st.session_state["m3a"] = float(min(max(st.session_state["m3a"], domain_left), domain_right))
    st.session_state["m3b"] = float(min(max(st.session_state["m3b"], domain_left), domain_right))
    st.session_state["m3c"] = float(min(max(st.session_state["m3c"], domain_left), domain_right))
    if st.session_state["m3b"] > st.session_state["m3c"]:
        st.session_state["m3b"] = st.session_state["m3c"]

    if "m3_saved_a_curves" not in st.session_state:
        st.session_state["m3_saved_a_curves"] = []
    if "m3_saved_curve_color_idx" not in st.session_state:
        st.session_state["m3_saved_curve_color_idx"] = 0

    if show_formula:
        st.markdown(
            '<div style="text-align:center; padding: 0.6rem 0 0.9rem 0;">',
            unsafe_allow_html=True
        )
        st.latex(r"""
        \Huge
        \begin{aligned}
        A(x)=\int_a^x f(t)\,dt
        &\qquad\Rightarrow\qquad
        A(c)-A(b)=\int_a^c f(t)\,dt-\int_a^b f(t)\,dt\\[0.35em]
        &\qquad\Rightarrow\qquad
        A(c)-A(b)=\int_b^c f(t)\,dt
        \end{aligned}
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    full_width_col = st.container()
    with full_width_col:
        m3_left_control_col, m3_right_slider_col = st.columns([0.45, 0.55], gap="large")

        with m3_right_slider_col:
            a3 = st.slider(
                "固定點 a",
                min_value=float(domain_left),
                max_value=float(domain_right),
                value=float(st.session_state.get("m3a", m3_initial_value)),
                step=0.05,
                key="m3a",
            )
            c3 = st.slider(
                "右端點 c",
                min_value=float(domain_left),
                max_value=float(domain_right),
                value=float(st.session_state.get("m3c", m3_initial_value)),
                step=0.05,
                key="m3c",
                on_change=enforce_m3c_not_below_b,
            )
            b3 = st.slider(
                "左端點 b",
                min_value=float(domain_left),
                max_value=float(domain_right),
                value=float(st.session_state.get("m3b", m3_initial_value)),
                step=0.05,
                key="m3b",
                on_change=enforce_m3b_not_above_c,
            )

        components.html(
            """
            <script>
            const repaintModule3Sliders = () => {
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

            repaintModule3Sliders();
            const module3SliderIntervalId = setInterval(repaintModule3Sliders, 500);
            window.addEventListener("beforeunload", () => clearInterval(module3SliderIntervalId));
            </script>
            """,
            height=0,
        )

        c3 = float(max(c3, b3))
        Axs_m3 = cumulative_integral(f, a3, xs)
        current_A3_b = float(np.interp(b3, xs, Axs_m3))
        current_A3_c = float(np.interp(c3, xs, Axs_m3))
        current_f3_b = float(f(np.array([b3]))[0])
        current_f3_c = float(f(np.array([c3]))[0])
        endpoint_diff_value_m3 = current_A3_c - current_A3_b
        definite_integral_value_m3 = endpoint_diff_value_m3

        m3_axis_positions = [a3, b3, c3]
        m3_axis_keys = ["a", "b", "c"]
        m3_axis_levels = get_axis_label_levels(m3_axis_positions, threshold=0.45)
        m3_axis_y_offsets = {
            key: -0.17 - 0.32 * lvl
            for key, lvl in zip(m3_axis_keys, m3_axis_levels)
        }

        with m3_left_control_col:
            m3_button_col_left, m3_button_col_right = st.columns(2, gap="small")
            with m3_button_col_left:
                if st.button("留下圖形", key="m3_save_a_curve", use_container_width=True):
                    color_idx = int(st.session_state.get("m3_saved_curve_color_idx", 0))
                    curve_color = RAINBOW_COLORS[color_idx % len(RAINBOW_COLORS)]
                    st.session_state["m3_saved_curve_color_idx"] = color_idx + 1
                    st.session_state["m3_saved_a_curves"].append(
                        {
                            "a": float(a3),
                            "b": float(b3),
                            "c": float(c3),
                            "A_b": float(current_A3_b),
                            "A_c": float(current_A3_c),
                            "curve": np.array(Axs_m3, dtype=float),
                            "color": curve_color,
                        }
                    )
            with m3_button_col_right:
                if st.button("清除圖形", key="m3_clear_saved_curves", use_container_width=True):
                    st.session_state["m3_saved_a_curves"] = []
                    st.session_state["m3_saved_curve_color_idx"] = 0

    left, right = st.columns(2, gap="large")
    with left:
        fig32, ax32 = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)

        for saved_item in st.session_state.get("m3_saved_a_curves", []):
            saved_curve = np.array(saved_item["curve"], dtype=float)
            saved_color = saved_item.get("color", "#9fd8b3")
            saved_b = float(saved_item.get("b", a3))
            saved_c = float(saved_item.get("c", a3))
            saved_A_b = float(saved_item.get("A_b", np.interp(saved_b, xs, saved_curve)))
            saved_A_c = float(saved_item.get("A_c", np.interp(saved_c, xs, saved_curve)))
            ax32.plot(xs, saved_curve, linewidth=2.2, color=saved_color, alpha=0.70)
            ax32.scatter([saved_b, saved_c], [saved_A_b, saved_A_c], s=48, color="#9bd18b", zorder=7)

        ax32.plot(xs, Axs_m3, linewidth=3.4, color="#8fc9a8")

        draw_to_x_axis(ax32, a3, np.interp(a3, xs, Axs_m3), "#f2a3c7", linewidth=1.6, marker_size=45)
        ax32.text(
            a3,
            0 + m3_axis_y_offsets["a"],
            f"{a3:.2f}",
            **pretty_a_label_kwargs(),
        )

        draw_to_x_axis(ax32, b3, current_A3_b, "#9bd18b", linewidth=1.6, marker_size=55)
        ax32.text(
            b3,
            0 + m3_axis_y_offsets["b"],
            f"{b3:.2f}",
            ha="center",
            va="top",
            fontsize=13,
            bbox=smart_value_bbox(),
        )
        b_xytext = smart_point_xytext(
            b3,
            current_A3_b,
            x_min_common,
            x_max_common,
            y_min_common,
            y_max_common,
            other_points=[(c3, current_A3_c), (a3, np.interp(a3, xs, Axs_m3))],
        )
        ax32.annotate(
            f"A({b3:.2f})={current_A3_b:.4f}",
            xy=(b3, current_A3_b),
            xytext=b_xytext,
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

        draw_to_x_axis(ax32, c3, current_A3_c, "#9bd18b", linewidth=1.6, marker_size=55)
        ax32.text(
            c3,
            0 + m3_axis_y_offsets["c"],
            f"{c3:.2f}",
            ha="center",
            va="top",
            fontsize=13,
            bbox=smart_value_bbox(),
        )
        c_xytext = smart_point_xytext(
            c3,
            current_A3_c,
            x_min_common,
            x_max_common,
            y_min_common,
            y_max_common,
            other_points=[(b3, current_A3_b), (a3, np.interp(a3, xs, Axs_m3))],
        )
        ax32.annotate(
            f"A({c3:.2f})={current_A3_c:.4f}",
            xy=(c3, current_A3_c),
            xytext=c_xytext,
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

        ax32.set_title("y=A(x)", fontsize=14)
        ax32.set_xlabel("x")
        ax32.set_ylabel("A(x)")
        ax32.set_xlim(x_min_common, x_max_common)
        ax32.set_ylim(y_min_common, y_max_common)
        add_common_style(ax32)
        st.pyplot(fig32, use_container_width=True)

    with right:
        fig3, ax3 = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
        ax3.plot(xs, ys, linewidth=3.4, color="#8bbce9")
        draw_to_x_axis(ax3, a3, f(np.array([a3]))[0], "#f2a3c7", linewidth=1.6, marker_size=45)
        ax3.text(
            a3,
            0 + m3_axis_y_offsets["a"],
            f"{a3:.2f}",
            **pretty_a_label_kwargs(),
        )

        draw_to_x_axis(ax3, b3, current_f3_b, "#9bd18b", linewidth=1.6, marker_size=55)
        ax3.text(
            b3,
            0 + m3_axis_y_offsets["b"],
            f"{b3:.2f}",
            ha="center",
            va="top",
            fontsize=13,
            bbox=smart_value_bbox(),
        )
        draw_to_x_axis(ax3, c3, current_f3_c, "#9bd18b", linewidth=1.6, marker_size=55)
        ax3.text(
            c3,
            0 + m3_axis_y_offsets["c"],
            f"{c3:.2f}",
            ha="center",
            va="top",
            fontsize=13,
            bbox=smart_value_bbox(),
        )

        mask_m3_fill = (xs >= min(b3, c3)) & (xs <= max(b3, c3))
        fill_area_by_sign(ax3, xs[mask_m3_fill], ys[mask_m3_fill], fill_pos_color, fill_neg_color, alpha=0.40)
        if np.any(mask_m3_fill) and abs(c3 - b3) > 1e-9:
            area_x_mid = 0.5 * (b3 + c3)
            ys_mask_m3 = ys[mask_m3_fill]
            if definite_integral_value_m3 >= 0:
                positive_part = ys_mask_m3[ys_mask_m3 >= 0]
                if len(positive_part) > 0:
                    area_y_mid = 0.52 * np.max(positive_part)
                else:
                    area_y_mid = 0.38 * max(y_max_common, 1.0)
            else:
                negative_part = ys_mask_m3[ys_mask_m3 < 0]
                if len(negative_part) > 0:
                    area_y_mid = 0.52 * np.min(negative_part)
                else:
                    area_y_mid = 0.38 * min(y_min_common, -1.0)
            ax3.text(
                area_x_mid,
                area_y_mid,
                f"{definite_integral_value_m3:.4f}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="semibold",
                color="#2f2f2f",
                bbox=smart_area_bbox(),
            )

        ax3.set_title("y=f(x)", fontsize=14)
        ax3.set_xlabel("x")
        ax3.set_ylabel("f(x)")
        ax3.set_xlim(x_min_common, x_max_common)
        ax3.set_ylim(y_min_common, y_max_common)
        add_common_style(ax3)
        st.pyplot(fig3, use_container_width=True)

    st.markdown(
        """
        <style>
        .m3-table-header {
            background: #f6f8fb;
            border: 1px solid #dfe6ef;
            border-radius: 12px;
            padding: 0.55rem 0.8rem;
            font-weight: 800;
            color: #38506a;
            text-align: center;
            margin-bottom: 0.35rem;
        }
        .m3-row-gap {
            height: 0.45rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    header_left, header_right = st.columns([0.45, 0.55], gap="large")
    with header_left:
        st.markdown('<div class="m3-table-header">一般推導式</div>', unsafe_allow_html=True)
    with header_right:
        st.markdown('<div class="m3-table-header">當下數值化對照</div>', unsafe_allow_html=True)

    row_left, row_right = st.columns([0.45, 0.55], gap="large")
    with row_left:
        row_left_card = st.container(border=True)
        with row_left_card:
            st.latex(r"\LARGE A(c)-A(b)=\int_b^c f(t)\,dt")
    with row_right:
        row_right_card = st.container(border=True)
        with row_right_card:
            st.latex(rf"\LARGE A({c3:.2f})-A({b3:.2f})=\int_{{{b3:.2f}}}^{{{c3:.2f}}} f(t)\,dt")
            endpoint_value_col, integral_value_col = st.columns(2, gap="small")
            with endpoint_value_col:
                st.markdown(
                    f"""
                    <div style="margin-top:0.85rem; text-align:center; padding-left:2.2rem; font-size:2.1rem; font-weight:800; color:#1f77b4; line-height:1.35;">
                        {endpoint_diff_value_m3:.4f}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with integral_value_col:
                st.markdown(
                    f"""
                    <div style="margin-top:0.85rem; text-align:center; font-size:2.1rem; font-weight:800; color:#d62728; line-height:1.35;">
                        {definite_integral_value_m3:.4f}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
