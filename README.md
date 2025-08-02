# FaceReco: Real-Time Face Recognition System (Under Deveploment)

![License](https://img.shields.io/github/license/dialga-cmd/face_reco)
![C++](https://img.shields.io/badge/language-C++17-blue.svg)
![Qt](https://img.shields.io/badge/framework-Qt5-green.svg)
![ONNXRuntime](https://img.shields.io/badge/inference-ONNXRuntime-orange.svg)

**FaceReco** is a cross-platform, real-time face recognition application developed using **C++**, **Qt5**, **OpenCV**, and **ONNX Runtime**. It performs facial detection and recognition on live camera input using deep learning models in ONNX format.

---

## ✨ Features

- 🎯 Real-time face detection and recognition using ONNX models
- 🖥️ Modern and responsive Qt5 GUI
- 📷 Live webcam integration with video frame processing
- 🧠 ONNX Runtime for high-speed inference
- 🗂️ Modular structure with separate UI, logic, and model layers
- 📁 Result logging and face profile handling
- 🔄 Easily extendable with custom models or preprocessing logic

---

## 🧠 Tech Stack

| Component       | Technology        |
|----------------|-------------------|
| Language        | C++17             |
| GUI Framework   | Qt5               |
| Deep Learning   | ONNX Runtime      |
| Image Processing| OpenCV            |
| Build System    | CMake             |
| Version Control | Git, Git LFS      |

---

## 📁 Folder Structure

```plaintext
face_reco/
│
├── include/             # Header files
├── src/                 # Core face logic and utilities
├── ui/                  # UI classes and interactions
├── models/              # Pretrained ONNX models (Git ignored)
├── third_party/         # External dependencies like ONNX Runtime
├── results/             # Output and logs
├── main.cpp             # Application entry point
├── CMakeLists.txt       # Build config
└── README.md            # Project documentation
```
# ⚙️ Installation

🔧 __Prerequisites (All Platforms)__

    C++17 compiler

    Qt5 development tools

    OpenCV development libraries

    CMake ≥ 3.15

    Git & Git LFS

    ONNX Runtime (or use included submodule)

🐧 __Linux (Ubuntu/Debian)__

```
sudo apt update
sudo apt install build-essential qtbase5-dev cmake libopencv-dev git git-lfs
```
```
git clone https://github.com/dialga-cmd/face_reco.git
cd face_reco
```
```
git lfs install
git lfs pull
```

# Initialize submodules if needed
```
git submodule update --init --recursive
```
```
mkdir build && cd build
cmake ..
make
```
```
./FaceReco
```

🪟 __Windows (MSYS2 / Visual Studio)__

    Install tools:

        CMake

        Qt5 SDK

        OpenCV for Windows

        Git and Git LFS

    Clone repo and build:
```
git clone https://github.com/dialga-cmd/face_reco.git
cd face_reco
```
```
git lfs install
git lfs pull
git submodule update --init --recursive
```
```
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```
    Run the resulting .exe binary from Release/ directory.

🍎 __macOS__

brew install cmake git git-lfs opencv qt
```
git clone https://github.com/dialga-cmd/face_reco.git
cd face_reco
```
```
git lfs install
git lfs pull
git submodule update --init --recursive
```
```
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$(brew --prefix qt)
make
```
```
./FaceReco
```

📄 License

This project is licensed under the [MIT License](https://github.com/dialga-cmd/face_reco/blob/master/LICENSE).
📌 To-Do (Planned Features)

    Add face registration and profile management

    Model benchmarking module

    Export recognition results to CSV/JSON

    Support for multiple cameras
