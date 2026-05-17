# 🚀 Visual Large Models & Multimodal Understanding

This project showcases practical applications of cutting-edge open-source multimodal large models (MLLMs) such as **Qwen-VL** and **InternVL**. It focuses on **in-depth video interpretation**, **vehicle damage assessment**, and **insurance document recognition**. The repository includes complete solutions ranging from local model deployment (with memory optimization) to cloud-based API integration.

## 📂 Project Structure

```text
.
├── CASE-VLM-Life-Insurance/       # Life Insurance: Multi-language document extraction
├── CASE-VLM-Auto-Insurance/       # Auto Insurance: Odometer, damage, and dangerous driving detection
├── CASE-Car-Scratch-Video/        # Video Understanding: Car scratch accident analysis
├── mengdi_qwen_interpretation.py  # Qwen3.6-27B-FP8 video interpretation (with spatial enhancement)
├── mengdi_internvideo_interpretation.py # InternVideo2_5 video interpretation (memory optimized)
├── InternVL.py                    # InternVL 2.5 general video understanding script
├── qwen_spatial_analysis.py       # Spatial attention analysis & FP8 deployment demo
├── requirements.txt               # Core dependencies
└── VLM_Multimodal_Understanding.pdf # Theoretical documentation
```

## ✨ Core Features

### 1. 🎬 In-Depth Spring Festival Gala Video Interpretation
Achieves high-precision multi-turn dialogue interpretation for long videos, with deep optimization for GPU memory usage.

*   **[mengdi_qwen_interpretation.py](./梦底解读Qwen%20(1).py)**
    *   **Model**: `Qwen3.6-27B-FP8`
    *   **Features**: 
        *   **Comprehensive Recognition**: Extracts basic info (program name, lyricists, composers, performers).
        *   **Deep Analysis**: Evaluates singing techniques, dance movements, visual details, and thematic fit.
        *   **Memory Optimization**: Uses a 48-frame keyframe sampling strategy with cache clearing, suitable for consumer-grade GPUs.
*   **[mengdi_internvideo_interpretation.py](./梦底解读IV.py)**
    *   **Model**: `InternVideo2_5_Chat_8B`
    *   **Features**: Focuses on artistic performance evaluation with minimal memory footprint.

### 2. 🚗 Intelligent Video Understanding & Spatial Enhancement
*   **[InternVL.py](./InternVL.py)**
    *   **Model**: `InternVL 2.5` Series
    *   **Features**: 
        *   **Multilingual Support**: Bilingual (CN/EN) video description and Q&A.
        *   **Precise Localization**: Identifies people count, vehicle damage parts, and collision points.
        *   **Dynamic Sampling**: Adjusts frame count based on video duration.
*   **[qwen_spatial_analysis.py](./qwen_spatial_analysis.py) (New)**
    *   **Model**: `Qwen3.6-27B-FP8` / `Qwen2.5-VL-7B-Instruct`
    *   **Breakthroughs**: 
        *   **FP8 Efficient Deployment**: Runs 27B parameter models smoothly on consumer GPUs.
        *   **Attention Heatmaps**: Uses Hook mechanisms to extract Cross-Attention weights, visualizing spatial focus in videos like "Mengdi".
        *   **Spatial Regularization**: Introduces weak positional loss constraints to fix bounding box offsets.
        *   **Inference Validation**: Applies positional bias and physical rules to filter out hallucinations.
    *   **Results**: Vehicle damage location accuracy improved from **78% to 91%**, part counting error reduced by **65%**.

### 3. 🏥 Industry-Specific Applications (CASE)
Leverages **Qwen-VL-Max** APIs or local models to solve industry pain points:

*   **Life Insurance**: Automated extraction of key elements from multi-language (CN, JP, FR, DE, KR) insurance documents.
*   **Auto Insurance**: 
    *   **Odometer Reading**: Automatic digit recognition.
    *   **Underwriting Inspection**: Multi-angle vehicle appearance verification.
    *   **Damage Assessment**: Identification of scratches, dents, and severity.
    *   **Dangerous Driving Detection**: Recognition of traffic violations.

## 🛠️ Environment Setup

### 1. Dependencies
Built on Python 3.12 + PyTorch 2.8.0 (CUDA 12.8).

```bash
pip install -r requirements.txt
```

### 2. Key Libraries
*   **Deep Learning**: `torch==2.8.0+cu128`, `torchvision==0.23.0+cu128`
*   **Model Hub**: `transformers==4.46.0`, `modelscope==1.25.0`
*   **Video Processing**: `decord==0.6.0`
*   **Image Processing**: `Pillow==11.2.1`

## 🚀 Quick Start

### Run Video Interpretation
Ensure you have downloaded the model weights and updated `MODEL_PATH` and `VIDEO_PATH` in the scripts.

```python
# Example: Run Qwen video interpretation
python "梦底解读Qwen (1).py"
```

### Call Cloud API
For scripts in the `CASE` folders, set the `DASHSCOPE_API_KEY` environment variable to use the Alibaba Cloud DashScope API.

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
python CASE-VLM-Auto-Insurance/1-Qwen-VL-Insurance-Recognition-cn.py
```

## 📝 Technical Highlights

1.  **Extreme Memory Optimization**: Limits GPU memory usage via `torch.cuda.set_per_process_memory_fraction` and clears cache after each turn, enabling 27B+ models on limited hardware.
2.  **Spatial Enhancement & Explainability**: 
    *   **FP8 Quantization**: Successfully deploys Qwen3.6-27B-FP8 for efficient inference.
    *   **Attention Heatmaps**: Extracts Cross-Attention weights to visualize model focus on stage layouts or damage spots.
    *   **Spatial Regularization**: Uses phased training with weak positional loss to constrain coordinate predictions.
    *   **Inference Validation**: Filters hallucinations using physical commonsense rules (e.g., wheels must be at the bottom).
3.  **Dynamic Frame Sampling**: Combines "initial keyframes" with "uniform sampling" to capture both opening details (like subtitles) and overall content.

## 📄 License

This project is for educational and research purposes only. Please adhere to the open-source licenses of the respective models (e.g., Alibaba Cloud, OpenGVLab).

---
*Powered by Qwen-VL, InternVL & ModelScope*
