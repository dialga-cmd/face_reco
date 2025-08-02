#include "face_embedder.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <numeric>

std::vector<float> extractEmbeddingFromImage(const cv::Mat& inputImage) {
    // Preprocess image
    cv::Mat resized, floatImg;
    cv::resize(inputImage, resized, cv::Size(160, 160));
    resized.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

    // HWC -> CHW
    std::vector<float> inputTensorValues(160 * 160 * 3);
    size_t index = 0;
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < 160; ++i)
            for (int j = 0; j < 160; ++j)
                inputTensorValues[index++] = floatImg.at<cv::Vec3f>(i, j)[c];

    std::array<int64_t, 4> inputShape = {1, 3, 160, 160};

    // ONNX Runtime setup
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FaceEmbedder");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    Ort::Session session(env, "models/facenet.onnx", sessionOptions);
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input/output names (new API)
    auto inputNameAllocated = session.GetInputNameAllocated(0, allocator);
    auto outputNameAllocated = session.GetOutputNameAllocated(0, allocator);
    const char* inputName = inputNameAllocated.get();
    const char* outputName = outputNameAllocated.get();

    // Create input tensor
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorValues.size(),
        inputShape.data(), inputShape.size()
    );

    // Run session
    auto outputTensors = session.Run(
        Ort::RunOptions{nullptr},
        &inputName, &inputTensor, 1,
        &outputName, 1
    );

    // Extract output
    float* floatArray = outputTensors.front().GetTensorMutableData<float>();
    std::vector<float> embedding(floatArray, floatArray + 128);

    // Normalize the vector
    float norm = std::sqrt(std::inner_product(embedding.begin(), embedding.end(), embedding.begin(), 0.0f));
    for (float& val : embedding)
        val /= norm + 1e-10f;

    return embedding;
}
