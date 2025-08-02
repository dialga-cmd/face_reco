#include "include/extractor.hpp"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

Extractor::Extractor() {}

ImageFeatures Extractor::extractFeatures(const QImage& image) {
    ImageFeatures features;

    // Convert QImage to cv::Mat
    cv::Mat mat(image.height(), image.width(), CV_8UC4, (void*)image.bits(), image.bytesPerLine());
    cv::Mat matBGR;
    cv::cvtColor(mat, matBGR, cv::COLOR_BGRA2BGR);

    // --- Face Embedding using ONNX ---
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FaceReco");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        Ort::Session session(env, "models/facenet.onnx", session_options);

        // Preprocess: resize and normalize as required by your model
        cv::Mat resized;
        cv::resize(matBGR, resized, cv::Size(160, 160)); // Example size
        resized.convertTo(resized, CV_32F, 1.0 / 255);

        std::vector<float> inputTensorValues;
        inputTensorValues.assign((float*)resized.datastart, (float*)resized.dataend);

        std::array<int64_t, 4> inputShape{1, resized.channels(), resized.rows, resized.cols};

        Ort::AllocatorWithDefaultOptions allocator;
        const char* inputName = session.GetInputName(0, allocator);
        const char* outputName = session.GetOutputName(0, allocator);

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorValues.size(), inputShape.data(), inputShape.size());

        auto outputTensors = session.Run(Ort::RunOptions{nullptr}, &inputName, &inputTensor, 1, &outputName, 1);
        float* floatArray = outputTensors.front().GetTensorMutableData<float>();
        size_t embeddingSize = outputTensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

        features.faceEmbedding = std::vector<float>(floatArray, floatArray + embeddingSize);
    } catch (...) {
        features.faceEmbedding = {};
    }

    // --- Material Analysis (Simple Texture Detection) ---
    cv::Mat gray;
    cv::cvtColor(matBGR, gray, cv::COLOR_BGR2GRAY);
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    double variance = cv::mean(laplacian.mul(laplacian))[0];
    if (variance < 50)
        features.material = "Smooth";
    else
        features.material = "Textured";

    // --- Color Analysis (Dominant Color) ---
    cv::Scalar meanColor = cv::mean(matBGR);
    int r = static_cast<int>(meanColor[2]);
    int g = static_cast<int>(meanColor[1]);
    int b = static_cast<int>(meanColor[0]);
    features.color = QString("R:%1 G:%2 B:%3").arg(r).arg(g).arg(b);

    return features;
}