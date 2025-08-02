#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>

class FaceEmbeddingExtractor {
public:
    FaceEmbeddingExtractor(const std::string& modelPath)
        : env(ORT_LOGGING_LEVEL_WARNING, "FaceNet"), session(nullptr), memoryInfo(nullptr) {
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = Ort::Session(env, modelPath.c_str(), sessionOptions);
        memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    }

    std::vector<float> getEmbedding(const cv::Mat& face) {
        cv::Mat resized;
        cv::resize(face, resized, cv::Size(160, 160)); // FaceNet input size
        resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);

        std::vector<float> inputTensorValues;
        inputTensorValues.reserve(160 * 160 * 3);
        for (int i = 0; i < 160; ++i)
            for (int j = 0; j < 160; ++j)
                for (int c = 0; c < 3; ++c)
                    inputTensorValues.push_back(resized.at<cv::Vec3f>(i, j)[c]);

        std::array<int64_t, 4> inputShape = {1, 160, 160, 3};

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorValues.size(),
            inputShape.data(), inputShape.size()
        );

        const char* inputNames[] = {"input"};
        const char* outputNames[] = {"embeddings"};

        auto output = session.Run(Ort::RunOptions{nullptr},
                                  inputNames, &inputTensor, 1,
                                  outputNames, 1);

        float* floatArray = output[0].GetTensorMutableData<float>();
        size_t numElements = output[0].GetTensorTypeAndShapeInfo().GetElementCount();

        std::vector<float> embedding(floatArray, floatArray + numElements);
        return l2Normalize(embedding);
    }

    float compareEmbeddings(const std::vector<float>& a, const std::vector<float>& b) {
        float dot = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
            dot += a[i] * b[i];
        return dot; // Cosine similarity, as vectors are normalized
    }

private:
    std::vector<float> l2Normalize(const std::vector<float>& vec) {
        float norm = 0.0f;
        for (float v : vec) norm += v * v;
        norm = std::sqrt(norm);
        std::vector<float> normalized(vec.size());
        for (size_t i = 0; i < vec.size(); ++i)
            normalized[i] = vec[i] / norm;
        return normalized;
    }

    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memoryInfo;
};
