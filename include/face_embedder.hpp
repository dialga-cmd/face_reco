#ifndef FACE_EMBEDDER_HPP
#define FACE_EMBEDDER_HPP

#include <opencv2/opencv.hpp>
#include <vector>

std::vector<float> extractEmbeddingFromImage(const cv::Mat& image);

#endif  // FACE_EMBEDDER_HPP
