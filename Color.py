# """
# ╔══════════════════════════════════════════════════════════════╗
# ║     ChromaVision — Professional HSV Color Detection Suite    ║
# ║     Built by: 30+ Year Computer Vision Expert                ║
# ║     Stack: Python · OpenCV · Streamlit · Plotly              ║
# ╚══════════════════════════════════════════════════════════════╝
# """

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import time
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChromaVision · HSV Detection",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  CUSTOM CSS — Industrial Dark Aesthetic
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700;900&family=Barlow+Condensed:wght@400;700;900&display=swap');

:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #111118;
    --bg-card: #16161f;
    --bg-card-hover: #1c1c28;
    --accent-cyan: #00f5d4;
    --accent-yellow: #f5c400;
    --accent-red: #ff3860;
    --accent-purple: #9b59b6;
    --text-primary: #e8e8f0;
    --text-muted: #6b6b88;
    --border: #2a2a3d;
    --border-bright: #3d3d5c;
    --grid-line: rgba(0, 245, 212, 0.05);
}

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* Grid background */
.stApp {
    background:
        linear-gradient(var(--grid-line) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid-line) 1px, transparent 1px),
        var(--bg-primary) !important;
    background-size: 40px 40px !important;
}

/* Main header */
.main-header {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 900;
    font-size: 3.2rem;
    letter-spacing: -1px;
    background: linear-gradient(135deg, var(--accent-cyan) 0%, #a8edea 50%, var(--accent-yellow) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin: 0;
}

.sub-header {
    font-family: 'Share Tech Mono', monospace;
    color: var(--text-muted);
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 2px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    border: 1px solid var(--accent-cyan);
    color: var(--accent-cyan);
    background: rgba(0, 245, 212, 0.07);
}

/* Metric cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent-cyan);
    padding: 16px 20px;
    border-radius: 4px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, var(--accent-cyan), transparent);
}
.metric-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.4rem;
    font-weight: 900;
    color: var(--accent-cyan);
    line-height: 1;
}
.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 4px;
}
.metric-delta {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent-yellow);
    margin-top: 2px;
}

/* Color swatches */
.color-swatch {
    display: inline-block;
    width: 28px;
    height: 28px;
    border-radius: 3px;
    border: 1px solid var(--border-bright);
    margin: 2px;
    vertical-align: middle;
}

/* Section headers */
.section-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 0.7rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: var(--accent-cyan);
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::before {
    content: '▶';
    font-size: 0.5rem;
    color: var(--accent-yellow);
}

/* Accuracy bar */
.accuracy-bar-bg {
    background: var(--border);
    height: 6px;
    border-radius: 3px;
    overflow: hidden;
    margin-top: 6px;
}
.accuracy-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-yellow));
    transition: width 0.5s ease;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(0, 245, 212, 0.1) !important;
    border-color: var(--accent-yellow) !important;
    color: var(--accent-yellow) !important;
}

/* Sliders */
.stSlider > div > div > div > div {
    background: var(--accent-cyan) !important;
}

/* Info boxes */
.info-box {
    background: rgba(0, 245, 212, 0.05);
    border-left: 3px solid var(--accent-cyan);
    padding: 12px 16px;
    border-radius: 2px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    margin: 8px 0;
    color: var(--text-muted);
}

/* Warning box */
.warn-box {
    background: rgba(245, 196, 0, 0.05);
    border-left: 3px solid var(--accent-yellow);
    padding: 12px 16px;
    border-radius: 2px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    margin: 8px 0;
    color: var(--accent-yellow);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted) !important;
    padding: 12px 24px;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* Upload */
[data-testid="stFileUploadDropzone"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border-bright) !important;
    border-radius: 4px !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Image containers */
.img-frame {
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    background: var(--bg-card);
    padding: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  CORE COLOR PROFILES
# ─────────────────────────────────────────────────────────────
COLOR_PROFILES = {
    "Red": {
        "ranges": [
            ([0, 120, 70], [10, 255, 255]),
            ([170, 120, 70], [180, 255, 255]),
        ],
        "hex": "#FF3860",
        "display": "🔴",
    },
    "Orange": {
        "ranges": [([10, 100, 100], [25, 255, 255])],
        "hex": "#FF8C00",
        "display": "🟠",
    },
    "Yellow": {
        "ranges": [([25, 100, 100], [35, 255, 255])],
        "hex": "#FFD700",
        "display": "🟡",
    },
    "Green": {
        "ranges": [([35, 60, 60], [85, 255, 255])],
        "hex": "#00C851",
        "display": "🟢",
    },
    "Cyan": {
        "ranges": [([85, 60, 60], [105, 255, 255])],
        "hex": "#00F5D4",
        "display": "🩵",
    },
    "Blue": {
        "ranges": [([100, 80, 70], [130, 255, 255])],
        "hex": "#3273DC",
        "display": "🔵",
    },
    "Purple": {
        "ranges": [([130, 50, 50], [160, 255, 255])],
        "hex": "#9B59B6",
        "display": "🟣",
    },
    "Pink": {
        "ranges": [([160, 50, 50], [170, 255, 255])],
        "hex": "#FF69B4",
        "display": "🩷",
    },
    "White": {
        "ranges": [([0, 0, 200], [180, 30, 255])],
        "hex": "#FFFFFF",
        "display": "⬜",
    },
    "Black": {
        "ranges": [([0, 0, 0], [180, 255, 50])],
        "hex": "#1A1A2E",
        "display": "⬛",
    },
}

MORPHOLOGY_OPS = {
    "None": None,
    "Erosion": cv2.MORPH_ERODE,
    "Dilation": cv2.MORPH_DILATE,
    "Opening (Noise Remove)": cv2.MORPH_OPEN,
    "Closing (Fill Gaps)": cv2.MORPH_CLOSE,
    "Gradient": cv2.MORPH_GRADIENT,
}

# ─────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def pil_to_cv(pil_img):
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv_to_pil(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def apply_color_mask(bgr_img, color_name, custom_lower=None, custom_upper=None):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    if custom_lower is not None:
        mask = cv2.inRange(hsv, np.array(custom_lower), np.array(custom_upper))
    else:
        profile = COLOR_PROFILES[color_name]
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in profile["ranges"]:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return mask, hsv


def apply_morphology(mask, op_name, kernel_size):
    op = MORPHOLOGY_OPS[op_name]
    if op is None or kernel_size == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, op, kernel)


def overlay_mask(bgr_img, mask, color_hex, alpha=0.45):
    overlay = bgr_img.copy()
    result = bgr_img.copy()
    h = int(color_hex.lstrip("#")[0:2], 16)
    s = int(color_hex.lstrip("#")[2:4], 16)
    v = int(color_hex.lstrip("#")[4:6], 16)
    overlay[mask > 0] = [v, s, h]  # approximate
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
    # Draw contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 245, 212), 2)
    return result, contours


def compute_metrics(mask, contours):
    total_px = mask.shape[0] * mask.shape[1]
    detected_px = int(np.sum(mask > 0))
    pct = round(detected_px / total_px * 100, 2)
    num_obj = len([c for c in contours if cv2.contourArea(c) > 200])
    areas = sorted([cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 200], reverse=True)
    largest = round(areas[0]) if areas else 0
    return detected_px, pct, num_obj, largest


def extract_dominant_colors(bgr_img, k=6):
    img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    if len(pixels) > 10000:
        idx = np.random.choice(len(pixels), 10000, replace=False)
        pixels = pixels[idx]
    km = KMeans(n_clusters=k, n_init=5, max_iter=100, random_state=42)
    km.fit(pixels)
    counts = np.bincount(km.labels_)
    centers = km.cluster_centers_.astype(int)
    order = np.argsort(-counts)
    return [(centers[i], counts[i]) for i in order]


def build_histogram_plot(hsv_img):
    channels = ["Hue", "Saturation", "Value"]
    colors_hex = ["#00F5D4", "#F5C400", "#FF3860"]
    bins = [180, 256, 256]
    ranges_list = [[0, 180], [0, 256], [0, 256]]

    fig = make_subplots(rows=1, cols=3, subplot_titles=channels)
    for i, (ch, col, b, r) in enumerate(zip(channels, colors_hex, bins, ranges_list)):
        hist = cv2.calcHist([hsv_img], [i], None, [b], r).flatten()
        x = np.linspace(r[0], r[1], b)
        fig.add_trace(
            go.Scatter(x=x, y=hist, fill="tozeroy", line=dict(color=col, width=1.5),
                       fillcolor=col.replace("#", "rgba(") + ",0.15)".replace("(", "").replace("#", "rgba("),
                       name=ch, showlegend=False),
            row=1, col=i + 1,
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono", size=10, color="#6b6b88"),
        margin=dict(l=10, r=10, t=35, b=10),
        height=200,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color="#2a2a3d")
    fig.update_yaxes(showgrid=True, gridcolor="#1c1c28", zeroline=False, color="#2a2a3d")
    return fig


def build_color_wheel_plot(dominant_colors):
    labels, values, colors_hex = [], [], []
    for (rgb, count) in dominant_colors:
        hex_c = "#{:02x}{:02x}{:02x}".format(*rgb)
        labels.append(hex_c.upper())
        values.append(int(count))
        colors_hex.append(hex_c)
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors_hex, line=dict(color="#0a0a0f", width=2)),
        hole=0.55,
        textfont=dict(family="Share Tech Mono", size=9, color="#e8e8f0"),
        showlegend=True,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Share Tech Mono", size=10, color="#6b6b88"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
        legend=dict(font=dict(size=9, color="#6b6b88"), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def build_heatmap(mask):
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (51, 51), 0)
    blurred = blurred / blurred.max() if blurred.max() > 0 else blurred
    # downsample for speed
    h, w = blurred.shape
    blurred_small = cv2.resize(blurred, (min(w, 200), min(h, 150)))
    fig = go.Figure(go.Heatmap(
        z=blurred_small,
        colorscale=[[0, "#0a0a0f"], [0.3, "#00f5d420"], [0.7, "#f5c400aa"], [1, "#ff3860"]],
        showscale=False,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=160,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, autorange="reversed"),
    )
    return fig


def generate_synthetic_image(scene_type):
    """Generate a synthetic test image for demo purposes."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    if scene_type == "Color Blocks":
        colors_bgr = [(0, 0, 200), (0, 200, 0), (200, 0, 0),
                      (0, 200, 200), (200, 200, 0), (200, 0, 200)]
        for i, c in enumerate(colors_bgr):
            x = (i % 3) * 213
            y = (i // 3) * 240
            img[y:y+240, x:x+213] = c
    elif scene_type == "Circles":
        colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255),
                  (0, 200, 0), (255, 0, 0), (180, 0, 180)]
        positions = [(107, 107), (320, 107), (533, 107),
                     (107, 360), (320, 360), (533, 360)]
        img[:] = 40
        for c, p in zip(colors, positions):
            cv2.circle(img, p, 90, c, -1)
            cv2.circle(img, p, 90, (255, 255, 255), 2)
    elif scene_type == "Gradient Rainbow":
        for x in range(640):
            hue = int(x / 640 * 179)
            img[:, x] = cv2.cvtColor(
                np.array([[[hue, 220, 220]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
            )[0][0]
    elif scene_type == "Lego Bricks":
        img[:] = (30, 30, 30)
        brick_colors = [
            (0, 0, 220), (0, 130, 0), (0, 0, 180),
            (0, 200, 255), (0, 255, 0), (200, 200, 0),
            (255, 0, 0), (180, 0, 180), (0, 165, 255),
        ]
        for row in range(3):
            for col in range(3):
                x1, y1 = col * 160 + 20, row * 140 + 20
                x2, y2 = x1 + 120, y1 + 100
                c = brick_colors[row * 3 + col]
                cv2.rectangle(img, (x1, y1), (x2, y2), c, -1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                # studs
                for s in range(3):
                    cx = x1 + 20 + s * 35
                    cv2.circle(img, (cx, y1 + 30), 10,
                               tuple(min(v + 40, 255) for v in c), -1)
    return img


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px'>
        <div style='font-family:"Share Tech Mono",monospace;font-size:0.6rem;letter-spacing:3px;
                    color:#6b6b88;text-transform:uppercase;margin-bottom:4px'>System</div>
        <div style='font-family:"Barlow Condensed",sans-serif;font-size:1.6rem;font-weight:900;
                    color:#00f5d4;letter-spacing:-0.5px'>ChromaVision</div>
        <div style='font-family:"Share Tech Mono",monospace;font-size:0.6rem;color:#6b6b88;
                    margin-top:2px'>v2.0 · HSV Detection Engine</div>
    </div>
    <hr style='margin:8px 0;border-color:#2a2a3d'/>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Detection Mode</div>', unsafe_allow_html=True)
    mode = st.selectbox("", ["Preset Color Profile", "Custom HSV Range", "Multi-Color Scan", "Dominant Color Analysis"], label_visibility="collapsed")

    st.markdown('<div class="section-title" style="margin-top:16px">Color Target</div>', unsafe_allow_html=True)

    if mode == "Preset Color Profile":
        selected_color = st.selectbox("Select Color", list(COLOR_PROFILES.keys()))
        profile = COLOR_PROFILES[selected_color]
        st.markdown(f"""
        <div class='info-box'>
            {profile['display']} {selected_color}<br/>
            HEX: {profile['hex']}<br/>
            Ranges: {len(profile['ranges'])} band(s)
        </div>
        """, unsafe_allow_html=True)

    elif mode == "Custom HSV Range":
        st.markdown("**Hue**")
        h_lo, h_hi = st.slider("H", 0, 179, (35, 85), label_visibility="collapsed")
        st.markdown("**Saturation**")
        s_lo, s_hi = st.slider("S", 0, 255, (60, 255), label_visibility="collapsed")
        st.markdown("**Value**")
        v_lo, v_hi = st.slider("V", 0, 255, (60, 255), label_visibility="collapsed")
        custom_lower = [h_lo, s_lo, v_lo]
        custom_upper = [h_hi, s_hi, v_hi]

        # Live HSV preview
        preview_hsv = np.zeros((40, 200, 3), dtype=np.uint8)
        mid_h = (h_lo + h_hi) // 2
        mid_s = (s_lo + s_hi) // 2
        mid_v = (v_lo + v_hi) // 2
        preview_hsv[:] = [mid_h, mid_s, mid_v]
        preview_bgr = cv2.cvtColor(preview_hsv, cv2.COLOR_HSV2BGR)
        st.image(cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB), caption="Live HSV Preview", use_container_width=True)

    elif mode == "Multi-Color Scan":
        scan_colors = st.multiselect("Colors to Detect", list(COLOR_PROFILES.keys()),
                                     default=["Red", "Green", "Blue"])

    st.markdown('<div class="section-title" style="margin-top:16px">Morphology</div>', unsafe_allow_html=True)
    morph_op = st.selectbox("Operation", list(MORPHOLOGY_OPS.keys()), index=3)
    kernel_sz = st.slider("Kernel Size", 1, 25, 7, step=2)

    st.markdown('<div class="section-title" style="margin-top:16px">Visualization</div>', unsafe_allow_html=True)
    show_contours = st.toggle("Show Contours", value=True)
    show_histogram = st.toggle("HSV Histogram", value=True)
    show_heatmap = st.toggle("Density Heatmap", value=True)
    min_area = st.slider("Min Object Area (px²)", 0, 5000, 200, step=50)

    st.markdown('<div class="section-title" style="margin-top:16px">Processing</div>', unsafe_allow_html=True)
    blur_radius = st.slider("Pre-blur (Gaussian)", 0, 15, 3, step=2)

    st.markdown("""
    <hr style='margin:16px 0;border-color:#2a2a3d'/>
    <div style='font-family:"Share Tech Mono",monospace;font-size:0.55rem;
                color:#3d3d5c;text-align:center;line-height:1.8'>
        CHROMAVISION ENGINE<br/>OPENCV 4.x · SKLEARN<br/>
        30+ YEARS CV EXPERTISE
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("""
    <div style='padding:8px 0 20px'>
        <p class='main-header'>CHROMA<span style='color:#f5c400'>VISION</span></p>
        <p class='sub-header'>HSV Color Detection & Segmentation Suite · Professional Edition</p>
    </div>
    """, unsafe_allow_html=True)
with col_h2:
    st.markdown("""
    <div style='text-align:right;padding-top:12px'>
        <span class='status-badge'>● SYSTEM READY</span><br/><br/>
        <span style='font-family:"Share Tech Mono",monospace;font-size:0.6rem;color:#3d3d5c'>
            ENGINE: OpenCV 4.x<br/>COLOR SPACE: HSV/BGR/RGB
        </span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────
tab_analyze, tab_demo, tab_batch, tab_reference = st.tabs([
    "⬡  ANALYZE IMAGE",
    "◈  DEMO SCENES",
    "⬢  BATCH SCAN",
    "⊞  HSV REFERENCE",
])


# ══════════════════════════════════════════════════════════════
#  TAB 1: ANALYZE IMAGE
# ══════════════════════════════════════════════════════════════
with tab_analyze:
    upload_col, _ = st.columns([2, 1])
    with upload_col:
        uploaded = st.file_uploader(
            "DROP IMAGE HERE — JPG / PNG / WEBP / BMP",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="visible",
        )

    if uploaded:
        t0 = time.time()
        pil_img = Image.open(uploaded)
        bgr = pil_to_cv(pil_img)

        # Pre-blur
        if blur_radius > 0:
            bgr_proc = cv2.GaussianBlur(bgr, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
        else:
            bgr_proc = bgr.copy()

        hsv_img = cv2.cvtColor(bgr_proc, cv2.COLOR_BGR2HSV)

        # ── MULTI-COLOR mode
        if mode == "Multi-Color Scan":
            st.markdown('<div class="section-title">Multi-Color Segmentation</div>', unsafe_allow_html=True)
            if not scan_colors:
                st.markdown('<div class="warn-box">⚠ Select at least one color in the sidebar.</div>', unsafe_allow_html=True)
            else:
                combined_mask = np.zeros(bgr_proc.shape[:2], dtype=np.uint8)
                result_img = bgr_proc.copy()

                per_color_stats = []
                palette_colors = [(0, 245, 212), (245, 196, 0), (255, 56, 96),
                                  (155, 89, 182), (46, 204, 113), (52, 152, 219)]

                for idx, cname in enumerate(scan_colors):
                    mask, _ = apply_color_mask(bgr_proc, cname)
                    mask = apply_morphology(mask, morph_op, kernel_sz)
                    combined_mask |= mask
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    draw_col = palette_colors[idx % len(palette_colors)]
                    big_contours = [c for c in contours if cv2.contourArea(c) > min_area]
                    cv2.drawContours(result_img, big_contours, -1, draw_col, 3)
                    det, pct, nobj, largest = compute_metrics(mask, big_contours)
                    per_color_stats.append((cname, det, pct, nobj, largest,
                                            COLOR_PROFILES[cname]["hex"]))

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Original**")
                    st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                with c2:
                    st.markdown("**Detected Regions**")
                    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)

                st.markdown('<div class="section-title" style="margin-top:20px">Per-Color Statistics</div>', unsafe_allow_html=True)
                stat_cols = st.columns(len(per_color_stats))
                for i, (cname, det, pct, nobj, largest, hex_c) in enumerate(per_color_stats):
                    with stat_cols[i]:
                        st.markdown(f"""
                        <div class='metric-card' style='border-top-color:{hex_c}'>
                            <div style='font-family:"Share Tech Mono",monospace;font-size:0.6rem;
                                        letter-spacing:2px;color:{hex_c}'>{cname.upper()}</div>
                            <div class='metric-value' style='color:{hex_c};font-size:1.8rem'>{pct}%</div>
                            <div class='metric-label'>Coverage</div>
                            <div class='metric-delta'>{nobj} objects · {det:,} px</div>
                        </div>
                        """, unsafe_allow_html=True)

        # ── DOMINANT COLOR ANALYSIS
        elif mode == "Dominant Color Analysis":
            st.markdown('<div class="section-title">Dominant Color Extraction — K-Means Clustering</div>', unsafe_allow_html=True)
            k_val = st.slider("Number of clusters (K)", 2, 12, 6)
            with st.spinner("Running K-Means clustering..."):
                dominant = extract_dominant_colors(bgr_proc, k=k_val)

            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("**Source Image**")
                st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with c2:
                st.markdown("**Color Distribution**")
                fig_pie = build_color_wheel_plot(dominant)
                st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

            st.markdown('<div class="section-title" style="margin-top:20px">Extracted Palette</div>', unsafe_allow_html=True)
            total_count = sum(c for _, c in dominant)
            pal_cols = st.columns(k_val)
            for i, ((rgb, count), col) in enumerate(zip(dominant, pal_cols)):
                hex_c = "#{:02x}{:02x}{:02x}".format(*rgb)
                pct = round(count / total_count * 100, 1)
                hsv_px = cv2.cvtColor(np.array([[rgb]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0][0]
                with col:
                    st.markdown(f"""
                    <div style='background:{hex_c};height:60px;border-radius:3px;
                                border:1px solid #2a2a3d;margin-bottom:6px'></div>
                    <div style='font-family:"Share Tech Mono",monospace;font-size:0.6rem;color:#6b6b88;
                                text-align:center;line-height:1.8'>
                        {hex_c.upper()}<br/>
                        <span style='color:#e8e8f0'>{pct}%</span><br/>
                        H:{hsv_px[0]} S:{hsv_px[1]}<br/>V:{hsv_px[2]}
                    </div>
                    """, unsafe_allow_html=True)

        # ── PRESET or CUSTOM
        else:
            if mode == "Preset Color Profile":
                mask, _ = apply_color_mask(bgr_proc, selected_color)
                target_hex = COLOR_PROFILES[selected_color]["hex"]
                label = selected_color
            else:
                mask, _ = apply_color_mask(bgr_proc, None, custom_lower, custom_upper)
                target_hex = "#00f5d4"
                label = "Custom"

            mask = apply_morphology(mask, morph_op, kernel_sz)
            result_img, contours = overlay_mask(bgr_proc, mask, target_hex)
            big_contours = [c for c in contours if cv2.contourArea(c) > min_area]

            # Contour bounding boxes
            annotated = result_img.copy()
            if show_contours:
                for cnt in big_contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    area = cv2.contourArea(cnt)
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (245, 196, 0), 1)
                    cv2.putText(annotated, f"{int(area)}px²", (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (245, 196, 0), 1)

            proc_time = round((time.time() - t0) * 1000, 1)
            det_px, pct, num_obj, largest = compute_metrics(mask, big_contours)

            # ── METRICS ROW
            m1, m2, m3, m4 = st.columns(4)
            metrics_data = [
                (f"{pct}%", "COLOR COVERAGE", f"{det_px:,} pixels detected"),
                (str(num_obj), "OBJECTS FOUND", f"min area {min_area}px²"),
                (f"{proc_time}ms", "PROC TIME", "per frame latency"),
                (f"{largest:,}", "LARGEST OBJECT", "pixels² area"),
            ]
            for col, (val, lbl, delta) in zip([m1, m2, m3, m4], metrics_data):
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{val}</div>
                        <div class='metric-label'>{lbl}</div>
                        <div class='metric-delta'>{delta}</div>
                        <div class='accuracy-bar-bg'>
                            <div class='accuracy-bar-fill' style='width:{min(float(val.rstrip("%ms").replace(",","")) / 100 * 100 if "%" in val else 60, 100)}%'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── IMAGE PANELS
            img_c1, img_c2, img_c3 = st.columns(3)
            with img_c1:
                st.markdown('<div class="section-title">Original</div>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with img_c2:
                st.markdown('<div class="section-title">Binary Mask</div>', unsafe_allow_html=True)
                st.image(mask, use_container_width=True)
            with img_c3:
                st.markdown('<div class="section-title">Detection Overlay</div>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            # ── ANALYTICS ROW
            if show_histogram or show_heatmap:
                an_c1, an_c2 = st.columns([2, 1])
                if show_histogram:
                    with an_c1:
                        st.markdown('<div class="section-title">HSV Channel Histograms</div>', unsafe_allow_html=True)
                        fig_hist = build_histogram_plot(hsv_img)
                        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
                if show_heatmap:
                    with an_c2:
                        st.markdown('<div class="section-title">Detection Density Heatmap</div>', unsafe_allow_html=True)
                        fig_heat = build_heatmap(mask)
                        st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

            # ── CONTOUR TABLE
            if big_contours:
                st.markdown('<div class="section-title" style="margin-top:8px">Detected Objects — Contour Analysis</div>', unsafe_allow_html=True)
                rows = []
                for i, cnt in enumerate(big_contours[:20]):
                    x, y, w, h = cv2.boundingRect(cnt)
                    area = cv2.contourArea(cnt)
                    perimeter = cv2.arcLength(cnt, True)
                    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                    rows.append({
                        "ID": i + 1,
                        "Area (px²)": int(area),
                        "BBox": f"({x},{y}) {w}×{h}",
                        "Perimeter": round(perimeter, 1),
                        "Circularity": round(circularity, 3),
                        "Centroid": f"({x + w//2}, {y + h//2})",
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)

        # ── EXPORT
        st.markdown("---")
        exp_c1, exp_c2 = st.columns(2)
        with exp_c1:
            if mode not in ["Dominant Color Analysis", "Multi-Color Scan"]:
                buf = io.BytesIO()
                Image.fromarray(mask).save(buf, format="PNG")
                st.download_button("⬇ Download Mask", buf.getvalue(), "mask.png", "image/png")
        with exp_c2:
            if mode not in ["Dominant Color Analysis", "Multi-Color Scan"]:
                buf2 = io.BytesIO()
                cv_to_pil(annotated).save(buf2, format="PNG")
                st.download_button("⬇ Download Overlay", buf2.getvalue(), "overlay.png", "image/png")

    else:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;border:1px dashed #2a2a3d;border-radius:4px;
                    background:rgba(0,245,212,0.02)'>
            <div style='font-size:3rem;margin-bottom:12px'>🎨</div>
            <div style='font-family:"Barlow Condensed",sans-serif;font-size:1.4rem;font-weight:700;
                        color:#6b6b88;letter-spacing:2px'>UPLOAD AN IMAGE TO BEGIN</div>
            <div style='font-family:"Share Tech Mono",monospace;font-size:0.65rem;color:#3d3d5c;
                        margin-top:8px'>Supports JPG · PNG · WEBP · BMP</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TAB 2: DEMO SCENES
# ══════════════════════════════════════════════════════════════
with tab_demo:
    st.markdown('<div class="section-title">Synthetic Test Scenes</div>', unsafe_allow_html=True)
    sc1, sc2 = st.columns([2, 1])
    with sc1:
        scene = st.selectbox("Select Scene", ["Color Blocks", "Circles", "Gradient Rainbow", "Lego Bricks"])
    with sc2:
        demo_color = st.selectbox("Detect Color", list(COLOR_PROFILES.keys()), index=3)

    if st.button("▶  RUN DETECTION"):
        synth = generate_synthetic_image(scene)
        if blur_radius > 0:
            synth_proc = cv2.GaussianBlur(synth, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
        else:
            synth_proc = synth.copy()

        mask_d, _ = apply_color_mask(synth_proc, demo_color)
        mask_d = apply_morphology(mask_d, morph_op, kernel_sz)
        result_d, contours_d = overlay_mask(synth_proc, mask_d, COLOR_PROFILES[demo_color]["hex"])
        big_c = [c for c in contours_d if cv2.contourArea(c) > min_area]

        det, pct, nobj, largest = compute_metrics(mask_d, big_c)

        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown("**Synthetic Scene**")
            st.image(cv2.cvtColor(synth, cv2.COLOR_BGR2RGB), use_container_width=True)
        with d2:
            st.markdown("**Binary Mask**")
            st.image(mask_d, use_container_width=True)
        with d3:
            st.markdown("**Detection Result**")
            st.image(cv2.cvtColor(result_d, cv2.COLOR_BGR2RGB), use_container_width=True)

        dm1, dm2, dm3 = st.columns(3)
        with dm1:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{pct}%</div>
                <div class='metric-label'>Coverage</div></div>""", unsafe_allow_html=True)
        with dm2:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{nobj}</div>
                <div class='metric-label'>Objects</div></div>""", unsafe_allow_html=True)
        with dm3:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{largest:,}</div>
                <div class='metric-label'>Largest (px²)</div></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TAB 3: BATCH SCAN
# ══════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="section-title">Batch Image Processing</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Upload multiple images. ChromaVision will scan all images for the selected color profile and return an aggregated report.</div>', unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Upload Images (multi-select)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
        label_visibility="visible",
    )
    batch_color = st.selectbox("Target Color (Batch)", list(COLOR_PROFILES.keys()), key="batch_color")

    if batch_files and st.button("▶  RUN BATCH"):
        results = []
        prog = st.progress(0)
        status_txt = st.empty()

        for i, f in enumerate(batch_files):
            status_txt.markdown(f'<div class="info-box">Processing [{i+1}/{len(batch_files)}] {f.name}…</div>', unsafe_allow_html=True)
            pil_b = Image.open(f)
            bgr_b = pil_to_cv(pil_b)
            mask_b, _ = apply_color_mask(bgr_b, batch_color)
            mask_b = apply_morphology(mask_b, morph_op, kernel_sz)
            contours_b, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            big_b = [c for c in contours_b if cv2.contourArea(c) > min_area]
            det_b, pct_b, nobj_b, largest_b = compute_metrics(mask_b, big_b)
            h, w = bgr_b.shape[:2]
            results.append({
                "Filename": f.name,
                "Size": f"{w}×{h}",
                "Coverage %": pct_b,
                "Objects": nobj_b,
                "Detected px": det_b,
                "Largest obj": largest_b,
                "Status": "✅ HIT" if pct_b > 0.5 else "⬜ MISS",
            })
            prog.progress((i + 1) / len(batch_files))

        status_txt.empty()
        prog.empty()

        st.markdown('<div class="section-title">Batch Results</div>', unsafe_allow_html=True)
        st.dataframe(results, use_container_width=True, hide_index=True)

        hits = sum(1 for r in results if r["Status"] == "✅ HIT")
        avg_cov = round(sum(r["Coverage %"] for r in results) / len(results), 2)
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            st.markdown(f"""<div class='metric-card'><div class='metric-value'>{hits}/{len(results)}</div>
                <div class='metric-label'>Images with Target Color</div></div>""", unsafe_allow_html=True)
        with bc2:
            st.markdown(f"""<div class='metric-card'><div class='metric-value'>{avg_cov}%</div>
                <div class='metric-label'>Average Coverage</div></div>""", unsafe_allow_html=True)
        with bc3:
            st.markdown(f"""<div class='metric-card'><div class='metric-value'>{len(results)}</div>
                <div class='metric-label'>Total Images Processed</div></div>""", unsafe_allow_html=True)

        # Coverage bar chart
        fig_bar = go.Figure(go.Bar(
            x=[r["Filename"] for r in results],
            y=[r["Coverage %"] for r in results],
            marker=dict(color=[r["Coverage %"] for r in results],
                        colorscale=[[0, "#2a2a3d"], [0.5, "#00f5d4"], [1, "#f5c400"]]),
            text=[f"{r['Coverage %']}%" for r in results],
            textposition="outside",
            textfont=dict(family="Share Tech Mono", size=9, color="#6b6b88"),
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Share Tech Mono", size=10, color="#6b6b88"),
            margin=dict(l=10, r=10, t=20, b=10),
            height=220,
            xaxis=dict(showgrid=False, color="#2a2a3d"),
            yaxis=dict(showgrid=True, gridcolor="#1c1c28", title="Coverage %"),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════
#  TAB 4: HSV REFERENCE
# ══════════════════════════════════════════════════════════════
with tab_reference:
    st.markdown('<div class="section-title">HSV Color Space Reference</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box' style='margin-bottom:16px'>
        HSV (Hue, Saturation, Value) separates color information from brightness — making it far more
        robust to lighting variation than RGB. OpenCV uses H∈[0,179], S∈[0,255], V∈[0,255].
    </div>
    """, unsafe_allow_html=True)

    # Color table
    st.markdown('<div class="section-title">Preset Color HSV Ranges</div>', unsafe_allow_html=True)
    ref_data = []
    for name, p in COLOR_PROFILES.items():
        for lo, hi in p["ranges"]:
            ref_data.append({
                "Color": f"{p['display']} {name}",
                "H Low": lo[0], "H High": hi[0],
                "S Low": lo[1], "S High": hi[1],
                "V Low": lo[2], "V High": hi[2],
                "HEX": p["hex"],
            })
    st.dataframe(ref_data, use_container_width=True, hide_index=True)

    # Hue spectrum visualization
    st.markdown('<div class="section-title" style="margin-top:16px">Hue Spectrum (OpenCV: 0–179)</div>', unsafe_allow_html=True)
    hue_bar = np.zeros((60, 360, 3), dtype=np.uint8)
    for x in range(360):
        hue_bar[:, x] = [x // 2, 220, 200]
    hue_bgr = cv2.cvtColor(hue_bar, cv2.COLOR_HSV2BGR)
    st.image(cv2.cvtColor(hue_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("""
        <div class='section-title'>Key Concepts</div>
        <div style='font-family:"Share Tech Mono",monospace;font-size:0.7rem;color:#6b6b88;line-height:2'>
            <b style='color:#00f5d4'>Hue (H):</b> The color type. 0=Red, 60=Yellow,<br/>
            &nbsp;&nbsp;120=Green, 180=Cyan, 240=Blue (×0.5 in CV)<br/>
            <b style='color:#00f5d4'>Saturation (S):</b> Color intensity / purity<br/>
            &nbsp;&nbsp;0=Gray, 255=Fully saturated<br/>
            <b style='color:#00f5d4'>Value (V):</b> Brightness<br/>
            &nbsp;&nbsp;0=Black, 255=Full brightness<br/>
            <b style='color:#f5c400'>Red wraps:</b> Needs 2 ranges (0–10 + 170–180)
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown("""
        <div class='section-title'>Morphological Operations</div>
        <div style='font-family:"Share Tech Mono",monospace;font-size:0.7rem;color:#6b6b88;line-height:2'>
            <b style='color:#00f5d4'>Erosion:</b> Shrinks white regions (removes noise)<br/>
            <b style='color:#00f5d4'>Dilation:</b> Expands white regions (fills gaps)<br/>
            <b style='color:#00f5d4'>Opening:</b> Erosion → Dilation (noise removal)<br/>
            <b style='color:#00f5d4'>Closing:</b> Dilation → Erosion (fill holes)<br/>
            <b style='color:#00f5d4'>Gradient:</b> Edge detection via dilation-erosion<br/>
            <b style='color:#f5c400'>Tip:</b> Use Opening for noisy scenes,<br/>
            &nbsp;&nbsp;Closing for broken detections.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='warn-box' style='margin-top:16px'>
        ⚡ PRO TIP: For robust real-time detection, always pre-blur with Gaussian (σ=3–5),
        use Closing morphology to fill fragmented masks, and filter contours by minimum area
        to eliminate sensor noise. For green-screen keying, use S>60 + V>60 to exclude
        washed-out or shadowed regions.
    </div>
    """, unsafe_allow_html=True)
