#include <QStatusBar>
#include "mainwindow.hpp"
#include <QVBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QDir>
#include <QListWidgetItem>
#include <QDesktopServices>
#include <QUrl>
#include <QPdfWriter>
#include <opencv2/opencv.hpp>
#include <QPainter>
#include "crawler.hpp"
#include "crawler_worker.hpp"
#include "result_data.hpp"
#include "face_embedder.hpp"

extern std::vector<float> referenceEmbedding;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    uploadButton = new QPushButton("Upload Image", this);
    scanButton = new QPushButton("Scan", this);
    downloadButton = new QPushButton("Download Results", this);
    inputImageLabel = new QLabel(this);
    resultList = new QListWidget(this);

    QWidget *central = new QWidget(this);
    QVBoxLayout *layout = new QVBoxLayout(central);
    layout->addWidget(uploadButton);
    layout->addWidget(scanButton);
    layout->addWidget(downloadButton);
    layout->addWidget(inputImageLabel);
    layout->addWidget(resultList);
    setCentralWidget(central);

    connect(uploadButton, &QPushButton::clicked, this, &MainWindow::onUploadImage);
    connect(scanButton, &QPushButton::clicked, this, &MainWindow::onStartScan);
    connect(downloadButton, &QPushButton::clicked, this, &MainWindow::onDownloadResults);

    downloadButton->setEnabled(false);

    connect(resultList, &QListWidget::itemDoubleClicked, this, [](QListWidgetItem* item){
        QDesktopServices::openUrl(QUrl(item->toolTip()));
    });
}

MainWindow::~MainWindow() {
    if (crawlerThread) {
        crawlerThread->quit();
        crawlerThread->wait();
        delete crawlerThread;
    }
    clearAllData();
}

void MainWindow::onUploadImage()
{
    QString filePath = QFileDialog::getOpenFileName(this, tr("Select Image"), QDir::homePath(), tr("Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.gif)"));
    if (!filePath.isEmpty()) {
        inputImagePath = filePath;
        QPixmap pixmap(filePath);
        inputImageLabel->setPixmap(pixmap.scaled(200, 200, Qt::KeepAspectRatio));
        statusBar()->showMessage("Image loaded successfully.");
        extractImageFeatures(filePath);
        scanButton->setEnabled(true);
    }
}

void MainWindow::extractImageFeatures(const QString& filePath)
{
    cv::Mat img = cv::imread(filePath.toStdString());
    if (img.empty()) {
        QMessageBox::critical(this, "Image Error", "Failed to load image.");
        return;
    }

    referenceEmbedding = extractEmbeddingFromImage(img);
}



void MainWindow::onStartScan()
{
    if (inputImagePath.isEmpty()) {
        QMessageBox::warning(this, tr("No Image"), tr("Please upload an image first."));
        return;
    }

    bool isZero = std::all_of(referenceEmbedding.begin(), referenceEmbedding.end(), [](float v) {
        return std::abs(v) < 1e-6;
    });

    if (isZero) {
        QMessageBox::warning(this, "No Features", "Cannot scan: no valid features detected.");
        statusBar()->showMessage("Scan aborted.");
        return;
    }

    statusBar()->showMessage("Scanning the internet...");
    resultList->clear();
    scanButton->setEnabled(false);

    if (crawlerThread) {
        crawlerThread->quit();
        crawlerThread->wait();
        delete crawlerThread;
        crawlerThread = nullptr;
    }

    CrawlerWorker* worker = new CrawlerWorker(inputImagePath);
    crawlerThread = new QThread;

    worker->moveToThread(crawlerThread);

    connect(crawlerThread, &QThread::started, worker, &CrawlerWorker::process);
    connect(worker, &CrawlerWorker::resultsReady, this, [=](const QVector<std::pair<std::string, float>>& rawResults){
        QVector<ResultData> results;
        for (const auto& pair : rawResults) {
            ResultData data;
            data.url = QString::fromStdString(pair.first);
            data.similarity = pair.second;
            data.description = QString("Match with similarity: %1").arg(pair.second);
            data.image = QPixmap(":/icons/match.png");
            results.append(data);
        }

        this->results = results;
        resultList->clear();

        if (results.isEmpty()) {
            statusBar()->showMessage("No matches found.");
            downloadButton->setEnabled(false);
        } else {
            for (const auto &result : results) {
                QListWidgetItem* item = new QListWidgetItem();
                item->setIcon(QIcon(result.image));
                item->setText(result.description + "\n" + result.url);
                item->setToolTip(result.url);
                resultList->addItem(item);
            }
            downloadButton->setEnabled(true);
            statusBar()->showMessage("Scan complete.");
        }

        scanButton->setEnabled(true);
    });

    connect(worker, &CrawlerWorker::finished, crawlerThread, &QThread::quit);
    connect(worker, &CrawlerWorker::finished, worker, &CrawlerWorker::deleteLater);
    connect(crawlerThread, &QThread::finished, crawlerThread, &QThread::deleteLater);

    crawlerThread->start();
}

void MainWindow::onDownloadResults()
{
    QString folderPath = QFileDialog::getExistingDirectory(this, tr("Select Folder to Save PDF"), QDir::homePath());
    if (!folderPath.isEmpty()) {
        saveResultsToFolder(folderPath);
    }
}

QVector<ResultData> MainWindow::startImageScan(const QString &imagePath)
{
    std::string imgPath = imagePath.toStdString();

    if (crawler) {
        delete crawler;
        crawler = nullptr;
    }

    crawler = new Crawler(imgPath);
    crawler->startSearch();

    QVector<ResultData> results;
    return results;
}

bool MainWindow::saveResultsToFolder(const QString &folderPath)
{
    QString pdfPath = folderPath + "/results.pdf";
    QPdfWriter pdf(pdfPath);
    pdf.setPageSize(QPageSize(QPageSize::A4));
    QPainter painter(&pdf);

    int y = 0;
    for (const auto& result : results) {
        painter.drawPixmap(0, y, 100, 100, result.image);
        painter.drawText(110, y + 20, result.description);
        painter.drawText(110, y + 40, result.url);
        y += 120;
        if (y > pdf.height() - 120) {
            pdf.newPage();
            y = 0;
        }
    }
    painter.end();
    statusBar()->showMessage("PDF saved to: " + pdfPath);
    return true;
}

void MainWindow::clearAllData()
{
    resultList->clear();
}
