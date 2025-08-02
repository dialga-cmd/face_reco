#include <QApplication>
#include <QMetaType>
#include "include/result_data.hpp"
#include "ui/mainwindow.hpp"

int main(int argc, char *argv[]) {
    qRegisterMetaType<QVector<ResultData>>("QVector<ResultData>");
    QApplication app(argc, argv);
    MainWindow w;
    w.show();
    return app.exec();
}