#include "crawler_worker.hpp"
#include "crawler.hpp"

CrawlerWorker::CrawlerWorker(const QString& imagePath)
    : imagePath(imagePath) {}

void CrawlerWorker::process() {
    Crawler crawler(imagePath.toStdString());

    // Run the image search
    crawler.startSearch();

    // Fetch the results
    std::vector<std::pair<std::string, float>> resultVec = crawler.getMatchedImages();

    // Convert to QVector for Qt signal
    QVector<std::pair<std::string, float>> resultQtVec(resultVec.begin(), resultVec.end());

    emit resultsReady(resultQtVec);
    emit finished();
}
