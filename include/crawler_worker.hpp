#ifndef CRAWLER_WORKER_HPP
#define CRAWLER_WORKER_HPP

#include <QObject>
#include <QString>
#include <QVector>
#include <utility>
#include <string>

class CrawlerWorker : public QObject {
    Q_OBJECT

public:
    explicit CrawlerWorker(const QString& imagePath);
    void process();

signals:
    void resultsReady(const QVector<std::pair<std::string, float>>& results);
    void finished();

private:
    QString imagePath;
};

#endif // CRAWLER_WORKER_HPP
