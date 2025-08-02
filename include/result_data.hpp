#pragma once
#include <QString>
#include <QPixmap>

struct ResultData {
    QPixmap image;
    QString url;
    QString description;
    float similarity;
};
