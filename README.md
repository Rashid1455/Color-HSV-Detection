<div align="center">

```
 ██████╗██╗  ██╗██████╗  ██████╗ ███╗   ███╗ █████╗ ██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗
██╔════╝██║  ██║██╔══██╗██╔═══██╗████╗ ████║██╔══██╗██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║
██║     ███████║██████╔╝██║   ██║██╔████╔██║███████║██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║
██║     ██╔══██║██╔══██╗██║   ██║██║╚██╔╝██║██╔══██║╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║
╚██████╗██║  ██║██║  ██║╚██████╔╝██║ ╚═╝ ██║██║  ██║ ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║
 ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝
```

### 🎨 Professional HSV Color Detection Suite · Built on Python + OpenCV + Streamlit

<br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)

<br/>

> **Detect, isolate, and analyze any color in any image — in seconds.**  
> Powered by HSV masking, K-Means clustering, and morphological processing.

<br/>

</div>

---

## ⚡ What is ChromaVision?

ChromaVision is a **production-grade color detection app** with an industrial dark-mode UI. Upload a photo, pick a target color, and get back a binary mask, a highlighted overlay, pixel-accurate metrics, and a dominant color palette — all in one click.

Whether you're inspecting quality control images, processing sports footage, building vision pipelines, or just exploring computer vision, ChromaVision gives you the full toolkit without writing a single line of code.

---

## 🚀 Quick Start

```bash
# 1. Clone or download the project
git clone https://github.com/yourname/chromavision.git
cd chromavision

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch
streamlit run Color.py
```

> App opens at **`http://localhost:8501`** — dark mode, zero config.

---

## 🖼️ Features at a Glance

| Tab | What It Does |
|-----|-------------|
| 🎯 **Detect** | Upload any image → get a color mask, annotated overlay, contour outlines, and pixel metrics in one shot |
| 🧪 **Demo Scenes** | Run detection on synthetic test patterns (Color Blocks, Circles, Rainbow, Lego Bricks) — no image needed |
| 📦 **Batch Scan** | Drop in dozens of images at once → aggregated detection report + coverage bar chart |
| 📚 **HSV Reference** | Full visual HSV spectrum, preset range table, and morphology cheat sheet |

---

## 🎛️ Detection Controls

Dial in precision with the sidebar:

| Control | What It Does |
|---------|-------------|
| **Color Profile** | Choose from 10+ presets: Red, Green, Blue, Yellow, Orange, Cyan, Magenta, White, Black, Skin |
| **Custom HSV Range** | Override any preset with manual Hue / Saturation / Value sliders |
| **Gaussian Blur** | Pre-smooth the image to reduce sensor noise before masking |
| **Morphology Op** | Erosion, Dilation, Opening, Closing, or Gradient — clean up your mask |
| **Kernel Size** | Control the morphology brush radius |
| **Min Contour Area** | Filter out tiny specks below a pixel threshold |

---

## 📊 Output Metrics

Every detection run surfaces:

- **Coverage %** — what fraction of the image is your target color
- **Object Count** — number of distinct blobs detected
- **Largest Object (px²)** — area of the biggest contour
- **Binary Mask** — pure black/white mask for downstream processing
- **Dominant Palette** — K-Means extracted top 5 colors with hex codes and percentages

---

## 🧠 Why HSV?

RGB mixes color and brightness in every channel — making threshold-based detection fragile under different lighting. **HSV separates them:**

```
Hue        → What color it is    (0–179 in OpenCV)
Saturation → How vivid it is     (0 = gray, 255 = pure)
Value      → How bright it is    (0 = black, 255 = full)
```

Result: the same red shirt looks "red" whether you're indoors, outdoors, or under a lamp — your threshold still works.

> 💡 **Pro Tip:** Red wraps around 0° in the hue spectrum. ChromaVision handles this automatically by applying **two ranges** (0–10 and 170–180) for red detection.

---

## 🗂️ Project Structure

```
chromavision/
├── Color.py            ← Main Streamlit app
├── requirements.txt    ← Python dependencies
└── README.md           ← You are here
```

---

## 📦 Dependencies

```txt
streamlit>=1.32.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
plotly
scikit-learn
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🔧 Tuning Guide

| Scene Type | Recommended Settings |
|------------|---------------------|
| Clean studio shot | Blur = 0, Opening morph, Min area = 500 |
| Outdoor / natural light | Blur = 3–5, Closing morph, Min area = 1000 |
| Noisy webcam feed | Blur = 5, Opening morph, Min area = 2000 |
| Green screen keying | Custom HSV: S > 60, V > 60 to exclude washed-out areas |
| Dark shadows | Lower V minimum to ~30 to catch shadowed regions |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">

**Built with 🎨 by a 30+ Year Computer Vision Expert**  
*Python · OpenCV · Streamlit · Plotly · scikit-learn*

</div>
