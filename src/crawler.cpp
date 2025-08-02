#include "crawler.hpp"
#include "face_embedder.hpp"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <iostream>
#include <curl/curl.h>
#include <regex>
#include <fstream>
#include <nlohmann/json.hpp>
#include <QDir>
#include <QStandardPaths>
#include <QPdfWriter>
#include <QPainter>
#include <QImage>

using json = nlohmann::json;

// Forward declarations
std::vector<float> preprocessFace(const cv::Mat& img);
std::vector<float> getEmbedding(const std::vector<float>& input);
float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);

// Globals
Crawler::Crawler(const std::string& path) : inputImagePath(path), stopFlag(false) {}

Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "faceNet"};
Ort::Session* session = nullptr;
Ort::SessionOptions session_options;
std::vector<float> referenceEmbedding;

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output) {
    size_t totalSize = size * nmemb;
    output->append((char*)contents, totalSize);
    return totalSize;
}

void Crawler::startSearch() {
    std::cout << "Starting web search.....\n";

std::cout << "Trying to load ONNX model from: " << QDir::currentPath().toStdString() << "/models/faceNet.onnx" << std::endl;

        try {
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            session = new Ort::Session(env, "models/faceNet.onnx", session_options);
        } catch (const Ort::Exception& e) {
            std::cerr << "Failed to load ONNX model: " << e.what() << "\n";
            return;
        }

    // Extract reference face embedding
    cv::Mat refImg = cv::imread(inputImagePath);
    if (refImg.empty()) {
        std::cerr << "Failed to load reference image.\n";
        return;
    }

    referenceEmbedding = getEmbedding(preprocessFace(refImg));

    crawlSurfaceWeb();
    crawlDeepWeb();
    crawlDarkWeb();
}

void Crawler::stopSearch() {
    stopFlag = true;
}

// Add getter method for matched images
std::vector<std::pair<std::string, float>> Crawler::getMatchedImages() const {
    return matchedImages;
}

void Crawler::downloadResults(const std::string& outPath) {
    if (matchedImages.empty()) {
        std::cout << "No results to save.\n";
        return;
    }

    QString pdfPath = QString::fromStdString(outPath);
    QPdfWriter writer(pdfPath);
    writer.setPageSize(QPageSize(QPageSize::A4));
    QPainter painter(&writer);

    int y = 0;
    for (const auto& [url, score] : matchedImages) {
        CURL* curl = curl_easy_init();
        std::string buffer;

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
        curl_easy_perform(curl);
        curl_easy_cleanup(curl);

        std::vector<uchar> data(buffer.begin(), buffer.end());
        QImage image = QImage::fromData(QByteArray((const char*)data.data(), data.size()));

        if (!image.isNull()) {
            painter.drawImage(50, y + 30, image.scaledToWidth(300));
            painter.drawText(50, y + image.height() + 40, QString("URL: %1").arg(QString::fromStdString(url)));
            painter.drawText(50, y + image.height() + 60, QString("Similarity: %1").arg(score));
            y += image.height() + 100;

            if (y > 1000) {
                writer.newPage();
                y = 0;
            }
        }
    }

    painter.end();
    std::cout << "✅ Results saved to: " << outPath << "\n";
}

void Crawler::crawlSurfaceWeb() {
    std::cout << "Scanning surface web using Yandex...\n";

    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "CURL init failed.\n";
        return;
    }

    std::ifstream file(inputImagePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open image.\n";
        curl_easy_cleanup(curl);
        return;
    }

    std::vector<char> imageData((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::string response;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: multipart/form-data");

    std::string boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW";
    std::ostringstream postFields;
    postFields << "--" << boundary << "\r\n"
               << "Content-Disposition: form-data; name=\"upfile\"; filename=\"face.jpg\"\r\n"
               << "Content-Type: image/jpeg\r\n\r\n";
    postFields.write(imageData.data(), imageData.size());
    postFields << "\r\n--" << boundary << "--\r\n";

    std::string postData = postFields.str();

    curl_easy_setopt(curl, CURLOPT_URL, "https://yandex.com/images/search");
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postData.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, postData.size());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "Mozilla/5.0");

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::cerr << "Yandex upload failed: " << curl_easy_strerror(res) << "\n";
        return;
    }

    std::smatch match;
    std::regex redirectRegex(R"(https:\\/\\/yandex\\.com\\/images\\/search\\?rpt=imageview[^"]+)");
    if (!std::regex_search(response, match, redirectRegex)) {
        std::cerr << "Failed to extract redirect URL.\n";
        return;
    }

    std::string redirectUrl = std::regex_replace(match[0].str(), std::regex(R"(\\/)"), "/");
    std::string fullUrl = "https://yandex.com" + redirectUrl;

    CURL* curl2 = curl_easy_init();
    if (!curl2) {
        std::cerr << "Failed to init second curl.\n";
        return;
    }

    std::string html;
    curl_easy_setopt(curl2, CURLOPT_URL, fullUrl.c_str());
    curl_easy_setopt(curl2, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl2, CURLOPT_WRITEDATA, &html);
    curl_easy_setopt(curl2, CURLOPT_USERAGENT, "Mozilla/5.0");
    curl_easy_setopt(curl2, CURLOPT_FOLLOWLOCATION, 1L);
    res = curl_easy_perform(curl2);
    curl_easy_cleanup(curl2);

    if (res != CURLE_OK) {
        std::cerr << "Failed to fetch Yandex results page.\n";
        return;
    }

    std::regex imgRegex(R"(img_url=([^&]+))");
    auto begin = std::sregex_iterator(html.begin(), html.end(), imgRegex);
    auto end = std::sregex_iterator();

    int matches = 0;
    for (auto i = begin; i != end && matches < 10; ++i) {
        std::string encodedUrl = (*i)[1];
        std::string imageUrl;

        char* decoded = curl_unescape(encodedUrl.c_str(), encodedUrl.length());
        imageUrl = decoded;
        curl_free(decoded);

        std::cout << "Checking image: " << imageUrl << "\n";
        if (imageMatches(imageUrl)) {
            std::cout << "✅ Match found: " << imageUrl << "\n";
            ++matches;
        }
    }

    if (matches == 0) {
        std::cout << "No matching images found on Yandex.\n";
    }
}

void Crawler::crawlDeepWeb() {
    std::cout << "Scanning deep web for your image...\n";
}

void Crawler::crawlDarkWeb() {
    std::cout << "Scanning dark web for your image...\n";
}

bool Crawler::imageMatches(const std::string& url) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;

    std::string buffer;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    if (res != CURLE_OK) return false;

    std::vector<uchar> data(buffer.begin(), buffer.end());
    cv::Mat img = cv::imdecode(data, cv::IMREAD_COLOR);
    if (img.empty()) return false;

    auto input = preprocessFace(img);
    auto embedding = getEmbedding(input);
    float similarity = cosineSimilarity(referenceEmbedding, embedding);

    std::cout << "Similarity score with " << url << ": " << similarity << "\n";
    
    if (similarity > 0.75f) {
        matchedImages.emplace_back(url, similarity);  // ✅ Save for PDF
        return true;
    }
    return false;
}

std::vector<float> preprocessFace(const cv::Mat& img) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(160, 160));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    std::vector<float> inputTensorValues(160 * 160 * 3);
    int idx = 0;
    for (int i = 0; i < resized.rows; ++i) {
        for (int j = 0; j < resized.cols; ++j) {
            cv::Vec3f pixel = resized.at<cv::Vec3f>(i, j);
            inputTensorValues[idx++] = pixel[0];
            inputTensorValues[idx++] = pixel[1];
            inputTensorValues[idx++] = pixel[2];
        }
    }
    return inputTensorValues;
}

std::vector<float> getEmbedding(const std::vector<float>& input) {
    Ort::AllocatorWithDefaultOptions allocator;

    const int64_t dims[] = {1, 160, 160, 3};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(input.data()), input.size(),
        dims, 4);

    const char* inputNames[] = {"input"};
    const char* outputNames[] = {"embeddings"};

    auto output_tensors = session->Run(
        Ort::RunOptions{nullptr}, inputNames, &input_tensor, 1,
        outputNames, 1);

    float* floatArray = output_tensors[0].GetTensorMutableData<float>();
    return std::vector<float>(floatArray, floatArray + 128);
}

float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0f, normA = 0.0f, normB = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (std::sqrt(normA) * std::sqrt(normB));
}