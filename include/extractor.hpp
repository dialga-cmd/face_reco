#pragma once
#include <QString>
#include <QImage>
#include <vector>

struct ImageFeatures {
    std::vector<float> faceEmbedding;
    QString material;
    QString color;
    // Add more properties as needed
};

class Extractor {
public:
    Extractor();
    ImageFeatures extractFeatures(const QImage& image);
};