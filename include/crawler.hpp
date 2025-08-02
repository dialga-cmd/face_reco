#pragma once
#include <string>
#include <vector>
#include <utility>

class Crawler {
public:
    Crawler(const std::string& path);
    void startSearch();
    void stopSearch();
    void downloadResults(const std::string& outPath);
    
    // Add getter method for matched images
    std::vector<std::pair<std::string, float>> getMatchedImages() const;

private:
    std::string inputImagePath;
    bool stopFlag;
    std::vector<std::pair<std::string, float>> matchedImages;
    
    void crawlSurfaceWeb();
    void crawlDeepWeb();
    void crawlDarkWeb();
    bool imageMatches(const std::string& url);
};