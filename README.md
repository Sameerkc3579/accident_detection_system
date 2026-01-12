# 👁️ Sentinel: AI Traffic Accident Detection System

<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge)

<p align="center">
  <strong>Real-time computer vision system capable of detecting road accidents with 84% Precision.</strong>
</p>



[View Demo Video](path/to/your/video.mp4) | [Report Bug](https://github.com/sameer3579/accident-detection-system/issues)

</div>

---

## 📖 Overview

**Sentinel** is an automated accident detection system designed to reduce emergency response times. Utilizing **Transfer Learning** on the YOLOv8m architecture, this model detects vehicle collisions in video feeds while filtering out normal traffic flow.

Unlike generic object detectors, this model was engineered specifically to distinguish between "dense traffic" (False Positives) and "actual collisions" (True Positives) using a specialized dataset including negative samples.

## 🚀 Key Features

* **⚡ Real-Time Inference:** Optimized for rapid detection on standard hardware using Ultralytics YOLOv8.
* **🎯 High Precision Engineering:** Achieved **84.2% Precision** through rigorous training on 3,000+ images.
* **🛡️ False Positive Reduction:** Trained on ~1,200 "negative images" (normal traffic/empty roads) to prevent false alarms.
* **📹 Video Pipeline:** Custom Python script (`inference_script.py`) handles frame-by-frame processing, annotation, and confidence thresholding.

## 📊 Engineering Metrics

The model was trained for **69 Epochs** (Early Stopping enabled) on a custom dataset.

| Metric | Score | Significance |
| :--- | :--- | :--- |
| **mAP50** | **78.1%** | Strong ability to detect accidents across various angles/scales. |
| **Precision** | **84.2%** | **High Trust.** When it flags an accident, it is 84% likely to be real. |
| **Recall** | **69.1%** | Conservative detection strategy to minimize false alarms. |
| **Inference Speed** | **~12ms** | Capable of processing ~80 FPS on Tesla T4 GPU. |

<details>
<summary>📈 View Training Graphs (Loss & mAP)</summary>

![Training Graphs](path/to/results.png)

</details>

## 🛠️ Installation

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/sameer3579/accident-detection-system.git](https://github.com/sameer3579/accident-detection-system.git)
   cd accident-detection-system

   ... (previous sections like Usage, Project Structure, etc.) ...

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <h3>👨‍💻 Author</h3>
  <p><strong>Sameer</strong></p>
  <p>AI Engineer & Full Stack Developer</p>
  <p>
    <a href="https://github.com/sameer3579">GitHub Profile</a>
  </p>
  <sub>Built with 💙 by Sameer</sub>
</div>
