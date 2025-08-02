#pragma once
#include "../include/crawler.hpp"
#include "../include/crawler_worker.hpp"
#include "../include/result_data.hpp"
#include <QMainWindow>
#include <QPushButton>
#include <QLabel>
#include <QListWidget>
#include <QString>
#include <QVector>
#include <QThread>

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onUploadImage();
    void onStartScan();
    void onDownloadResults();

private:
    Crawler* crawler = nullptr;
    QThread* crawlerThread = nullptr;
    QPushButton *uploadButton;
    QPushButton *scanButton;
    QPushButton *downloadButton;
    QLabel *inputImageLabel;
    QListWidget *resultList;
    QString inputImagePath;

    QVector<ResultData> results;

    QVector<ResultData> startImageScan(const QString &imagePath);
    bool saveResultsToFolder(const QString &folderPath);
    void startWebCrawl(const QString &imagePath);
    void addResult(const QPixmap &thumb, const QString &url, const QString &platform, const QString &user, double score);
    void saveResultsToFile(const QString &savePath);
    void clearAllData();
    void extractImageFeatures(const QString& filePath);
};