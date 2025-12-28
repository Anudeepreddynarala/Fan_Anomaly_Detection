# 🏭 Real-Time Fan Anomaly Detection on ESP32

[![ESP-IDF](https://img.shields.io/badge/ESP--IDF-v5.5-blue.svg)](https://github.com/espressif/esp-idf)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow%20Lite-Micro-orange.svg)](https://www.tensorflow.org/lite/microcontrollers)
[![Edge Impulse](https://img.shields.io/badge/Edge%20Impulse-Enabled-green.svg)](https://www.edgeimpulse.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Project Overview

A **production-ready industrial IoT system** that performs real-time anomaly detection on industrial fan sounds using machine learning inference on edge hardware. This project demonstrates advanced embedded systems engineering skills including real-time audio processing, on-device ML inference, and hardware integration.

### Key Technical Achievements

- ✅ **Edge ML Deployment**: TensorFlow Lite Micro with INT8 quantization running on resource-constrained hardware
- ✅ **Real-Time Processing**: 6.2 inferences/second with 162ms total latency
- ✅ **Production Code Quality**: Comprehensive error handling, performance monitoring, and modular architecture
- ✅ **Hardware Integration**: Multi-peripheral system (I2S microphone, I2C display, dual-core processing)
- ✅ **Optimized Performance**: Efficient memory management (~85KB RAM usage)

---

## 📋 Technical Specifications

| Category | Details |
|----------|---------|
| **Platform** | ESP32 (Xtensa LX6, 240MHz dual-core) |
| **Framework** | ESP-IDF v5.5 |
| **ML Framework** | TensorFlow Lite Micro + Edge Impulse SDK |
| **Audio Input** | INMP441 I2S MEMS Microphone @ 16kHz |
| **Display** | SSD1306 128×64 OLED (I2C) |
| **Model Type** | 2D CNN (Convolutional Neural Network) |
| **Input Features** | 13×32 MFCC (Mel-Frequency Cepstral Coefficients) |
| **Model Size** | 35 KB (quantized INT8) |
| **Inference Time** | 55.1ms (measured on hardware) |
| **RAM Usage** | 85 KB (TFLite arena: 63KB, buffers: 22KB) |
| **Flash Usage** | 342 KB application + 800 KB framework = 1.14 MB total |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ESP32 (Dual Core)                       │
│                                                                 │
│  ┌─────────────────────┐         ┌──────────────────────────┐   │
│  │   Core 1 (Audio)    │         │    Core 0 (Inference)    │   │
│  │                     │         │                          │   │
│  │  I2S Microphone     │────────▶│  Signal Processing       │   │
│  │  @ 16kHz            │  DMA    │  (MFCC Extraction)       │   │
│  │  416 samples        │  Queue  │                          │   │
│  │  (~26ms window)     │         │  TFLite Micro Inference  │   │
│  └─────────────────────┘         │  (2D CNN, INT8)          │   │
│                                  │                          │   │
│                                  │  Classification:         │   │
│                                  │  • Normal / Anomaly      │   │
│                                  └──────────┬───────────────┘   │
│                                             │                   │
│                                             ▼                   │
│                                  ┌──────────────────────────┐   │
│                                  │   I2C OLED Display       │   │
│                                  │   • Real-time results    │   │
│                                  │   • Inference time       │   │
│                                  │   • Confidence scores    │   │
│                                  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Machine Learning Pipeline

### 1. **Data Source**
- **Dataset**: MIMII (Malfunctioning Industrial Machine Investigation) Dataset
- **Target**: Fan ID 00, 0dB SNR (realistic factory noise conditions)
- **Samples**: 1,011 normal + 407 abnormal recordings
- **Duration**: 10-second audio files, chunked into 1-second segments

### 2. **Feature Extraction**
- **Method**: Mel-Frequency Cepstral Coefficients (MFCC)
- **Configuration**: 13 coefficients × 32 time frames
- **Window**: 416 samples @ 16kHz = 26ms
- **Processing**: Pre-emphasis → Windowing → FFT → Mel filterbank → DCT

### 3. **Model Architecture**
```
Input (13×32 MFCC)
    ↓
2D Convolutional Layers
    ↓
Pooling & Activation
    ↓
Fully Connected Layers
    ↓
Softmax (2 classes: Normal/Anomaly)
```

### 4. **Optimization**
- **Quantization**: INT8 post-training quantization (reduced from 140KB to 35KB)
- **Optimization**: Custom ESP-IDF porting layer for efficient execution on Xtensa architecture
- **Training Accuracy**: ~87% on validation set
- **Inference Consistency**: ±0.1ms variance (extremely stable performance)

---

## 📊 Performance Metrics

### Real-Time Timing Breakdown (Measured on Hardware)

```
┌──────────────────────────────────────────────────────────────┐
│ Audio Capture:    varies   (26ms - 410ms, async)             │
│ Inference:        55.1 ms  █████████████████░░░░░░░░░  34%   │
│ Display Update:  106.8 ms  ███████████████████████████  66%  │
│ ────────────────────────────────────────────────────────────-│
│ Total Latency:   162.0 ms  per inference cycle               │
│ Throughput:        6.2 inferences/second                     │
└──────────────────────────────────────────────────────────────┘

Performance Summary (from live ESP32):
• Inference Time:    55.1ms (highly consistent ±0.1ms)
• Total Cycle:       162.0ms average
• Throughput:        6.17-6.18 inferences/sec
• Anomaly Detection: Successfully detected anomalies with 99.6% confidence
```

### Memory Footprint

| Component | RAM (SRAM) | Flash (Program) |
|-----------|------------|-----------------|
| TFLite Arena | 63 KB | - |
| Audio Buffer | 832 bytes | - |
| OLED Framebuffer | 1 KB | - |
| I2S DMA | 4 KB | - |
| FreeRTOS Tasks | 12 KB | - |
| Application Code | - | 50 KB |
| Edge Impulse SDK | - | 250 KB |
| TFLite Model | - | 35 KB |
| ESP-IDF Framework | - | 800 KB |
| **Total** | **~85 KB / 520 KB** | **~1.14 MB / 4 MB** |

---

## 🛠️ Hardware Setup

### Pin Configuration

#### INMP441 I2S Microphone
```
INMP441 Pin  →  ESP32 GPIO
────────────────────────
VDD          →  3.3V
GND          →  GND
SD (DOUT)    →  GPIO 32
WS (LRCL)    →  GPIO 15
SCK (BCLK)   →  GPIO 14
L/R          →  GND (Left channel)
```

#### SSD1306 OLED Display (I2C)
```
OLED Pin  →  ESP32 GPIO
────────────────────
VCC       →  3.3V or 5V
GND       →  GND
SDA       →  GPIO 21
SCL       →  GPIO 22
```

### Schematic
```
                     ESP32
          ┌────────────────────────┐
INMP441   │                        │   SSD1306
  ┌───────┤ GPIO 32 (I2S SD)       │
  │       │ GPIO 15 (I2S WS)       │
  │       │ GPIO 14 (I2S SCK)      ├────────┐
  │       │                        │        │
  │       │ GPIO 21 (I2C SDA) ─────┼────────┤
  │       │ GPIO 22 (I2C SCL) ─────┼────────┤
  │       │                        │        │
  └───────┤ 3.3V   GND             ├────────┘
          └────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

1. **ESP-IDF v5.5** - [Installation Guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/)
2. **Hardware**:
   - ESP32 development board
   - INMP441 I2S microphone module
   - SSD1306 128×64 OLED display (I2C)
   - Jumper wires

### Build and Flash

```bash
# Clone the repository
git clone https://github.com/Anudeepreddynarala/Fan_Anomaly_Detection.git
cd Fan_Anomaly_Detection

# Configure ESP-IDF environment
. $HOME/esp/esp-idf/export.sh

# Build the project
idf.py build

# Flash to ESP32 (replace PORT with your serial port)
idf.py -p /dev/ttyUSB0 flash monitor
```

### Expected Output

**Serial Monitor:**
```
I (12549) FAN_ANOMALY: ────────────────────────────────────────────────
I (12549) FAN_ANOMALY: Inference #35 | Result: ANOMALY (99.6%)
I (12559) FAN_ANOMALY:   Audio Capture:  90781 μs (90.8 ms)
I (12559) FAN_ANOMALY:   Inference Time: 55112 μs (55.1 ms) ⚡
I (12569) FAN_ANOMALY:   Display Update: 106851 μs (106.9 ms)
I (12569) FAN_ANOMALY:   Total Cycle:    162030 μs (162.0 ms)
I (12579) FAN_ANOMALY:   Throughput:     6.17 inferences/sec

I (12589) FAN_ANOMALY: ╔═══════════════════════════════════════════════════╗
I (12599) FAN_ANOMALY: ║          10-SECOND PERFORMANCE SUMMARY            ║
I (12609) FAN_ANOMALY: ╠═══════════════════════════════════════════════════╣
I (12629) FAN_ANOMALY: ║ Total Inferences:        35                       ║
I (12629) FAN_ANOMALY: ║ Avg Inference Time:   55.1 ms                     ║
I (12639) FAN_ANOMALY: ║ Avg Total Latency:    162.0 ms                    ║
I (12649) FAN_ANOMALY: ║ Throughput:            6.2 inferences/sec         ║
I (12659) FAN_ANOMALY: ╚═══════════════════════════════════════════════════╝
```

**OLED Display:**
```
![IMG_7135](https://github.com/user-attachments/assets/f15e4092-7ea6-441b-af50-738070ebedd1)


┌─────────────────────┐
│   FAN STATUS        │
├─────────────────────┤
│                     │
│     ANOMALY         │
│                     │
│  N:0  A:100         │
│  55ms               │
└─────────────────────┘
```

---

## 💼 Skills Demonstrated (For Embedded Systems Engineering Roles)

### Core Embedded Systems Skills

#### 1. **Real-Time Systems Design**
- ✅ Multi-core FreeRTOS task management with priority scheduling
- ✅ Interrupt-driven I2S DMA for zero-copy audio streaming
- ✅ Deterministic latency: 162ms end-to-end processing (55ms inference + 107ms display)
- ✅ Race condition prevention with proper mutex/queue handling

#### 2. **Hardware Integration & Driver Development**
- ✅ I2S peripheral configuration for MEMS microphone (16kHz sampling)
- ✅ I2C driver implementation for OLED display
- ✅ Custom bit-banging for pixel-level display control
- ✅ DMA buffer management and circular buffering

#### 3. **Memory Optimization**
- ✅ Static memory allocation (no heap fragmentation)
- ✅ Efficient buffer management: 85KB total RAM usage
- ✅ Flash optimization: 342KB application code
- ✅ Stack size tuning for RTOS tasks

#### 4. **Performance Profiling & Optimization**
- ✅ Microsecond-precision timing using hardware timers (esp_timer_get_time())
- ✅ Per-stage performance breakdown (capture/inference/display)
- ✅ Throughput monitoring (6.2 inferences/sec measured on hardware)
- ✅ CPU utilization analysis and optimization

### Machine Learning Engineering Skills

#### 5. **Edge ML Deployment**
- ✅ TensorFlow Lite Micro integration from scratch
- ✅ Model quantization (FP32 → INT8, 4x size reduction)
- ✅ Custom MFCC feature extraction pipeline
- ✅ Custom porting layer for Xtensa architecture (ei_porting_esp32.cpp)

#### 6. **Signal Processing**
- ✅ Real-time audio preprocessing (pre-emphasis, windowing)
- ✅ FFT implementation for frequency domain analysis
- ✅ Mel-scale filterbank application
- ✅ Feature normalization and quantization

### Software Engineering Skills

#### 7. **Professional Code Quality**
- ✅ Modular architecture with clear separation of concerns
- ✅ Comprehensive error handling (I2S, I2C, inference failures)
- ✅ Extensive logging for debugging and monitoring
- ✅ Clean, documented, maintainable codebase

#### 8. **Build Systems & DevOps**
- ✅ CMake build configuration for complex multi-library project
- ✅ Dependency management (CMSIS-DSP, CMSIS-NN, TFLite)
- ✅ Custom porting layer for platform abstraction
- ✅ Version control and CI/CD ready

#### 9. **Cross-Platform Development**
- ✅ ESP-IDF framework expertise (v5.5)
- ✅ C++ modern features (C++14, templates, STL)
- ✅ Platform abstraction through porting layers
- ✅ Hardware-agnostic algorithm design

---

## 📈 Project Complexity Indicators

### Technical Depth
- **Lines of Code**: 500+ (main application) + 2000+ (SDK integration)
- **Components Integrated**: 8 (I2S, I2C, TFLite, CMSIS, FreeRTOS, Display, Audio, ML)
- **Build Complexity**: 1672 compilation units
- **Third-Party Libraries**: TensorFlow Lite Micro, Edge Impulse SDK, CMSIS-DSP/NN

### Real-World Applicability
- **Industry Standard Dataset**: MIMII (recognized in academic/industrial research)
- **Production-Ready Code**: Error handling, resource management, logging
- **Realistic Constraints**: 0dB SNR (factory noise level), resource-limited hardware
- **Scalable Architecture**: Easy to retrain for different machinery types

---

## 🎓 Key Learning Outcomes

This project showcases expertise in:

1. **Embedded ML Pipeline**: Data collection → Training → Quantization → Deployment → Monitoring
2. **Real-Time Constraints**: Meeting hard deadlines on resource-constrained hardware
3. **Hardware/Software Co-Design**: Optimizing both algorithm and hardware configuration
4. **Production Engineering**: Not just "it works" but "it works reliably and efficiently"
5. **Problem Solving**: Debugging build systems, linker errors, hardware timing issues

---

## 🔄 Future Enhancements

- [ ] Over-the-Air (OTA) updates for model deployment
- [ ] MQTT/HTTP integration for cloud logging
- [ ] Multi-class classification (bearing faults, imbalance, etc.)
- [ ] Anomaly localization (time-frequency attention maps)
- [ ] Adaptive thresholding based on environmental noise
- [ ] Battery-powered operation with deep sleep modes

---

## 📚 References & Resources

### Dataset
- [MIMII Dataset](https://zenodo.org/record/3384388) - Malfunctioning Industrial Machine Investigation and Inspection

### Frameworks & Tools
- [ESP-IDF](https://docs.espressif.com/projects/esp-idf/) - Espressif IoT Development Framework
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers) - ML for embedded systems
- [Edge Impulse](https://www.edgeimpulse.com/) - Embedded ML platform
- [CMSIS](https://www.keil.com/pack/doc/CMSIS/General/html/index.html) - Cortex Microcontroller Software Interface Standard

### Academic Papers
- Purohit et al., "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection" (2019)
- Koizumi et al., "ToyADMOS: A Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection" (2019)

---

## 📝 License

The Edge Impulse SDK and TensorFlow Lite components are subject to their respective licenses.

---

## 👤 Author

**Anudeep Reddy Narala**

🔗 [GitHub](https://github.com/Anudeepreddynarala) | 💼 [LinkedIn](https://linkedin.com/in/anudeep-reddy-narala)

---

## 🌟 Acknowledgments

- Edge Impulse for the embedded ML SDK
- Google TensorFlow team for TensorFlow Lite Micro
- Espressif Systems for ESP-IDF
- MIMII Dataset creators for the industrial sound data
- ARM for CMSIS-DSP and CMSIS-NN libraries


