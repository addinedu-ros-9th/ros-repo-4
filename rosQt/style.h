#ifndef STYLE_H
#define STYLE_H

#include <QApplication>
#include <QString>
#include <QMap>

class StyleManager
{
public:
    static void applyStyle(QApplication* app);
    
private:
    static QMap<QString, QString> getColors();
    static QString getRadius();
    static QStringList getQssFiles();
    static QString loadQssFile(const QString& filePath);
    static QString replaceTemplateVariables(const QString& style);
    static QString getDefaultComponentStyles();  // 이 줄 추가
};

#endif // STYLE_H