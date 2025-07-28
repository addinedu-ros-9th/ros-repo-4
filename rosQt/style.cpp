#include "style.h"
#include <QFile>
#include <QTextStream>
#include <QDebug>
#include <QFontDatabase>
#include <QFont>
#include <QDir>

QMap<QString, QString> StyleManager::getColors()
{
    QMap<QString, QString> colors;
    colors["primary"] = "#009999";
    colors["primary_dark"] = "#00696D";
    colors["secondary"] = "#FF9C1D";
    colors["therity"] = "#579ACE";
    colors["primary_light"] = "#CCEBEB";
    colors["green_gray1"] = "#EBF2F3";
    colors["error"] = "#F44336";
    
    return colors;
}

QString StyleManager::getRadius()
{
    return "2px";
}

QStringList StyleManager::getQssFiles()
{
    QStringList files;
    files << "layout.qss"
          << "login.qss"
          << "component/btn.qss"
          << "component/checkbox.qss"
          << "component/radiobox.qss"
          << "component/divider.qss"
          << "component/textfield.qss"
          << "component/select.qss"
          << "component/spinbox.qss"
          << "component/table.qss"
          << "component/scrollbar.qss"
          << "contents/cctv.qss"
          << "contents/dashboard.qss"
          << "contents/detect_log.qss"
          << "utils.qss";
    
    return files;
}

QString StyleManager::loadQssFile(const QString& filePath)
{
    QFile file(filePath);
    if (file.open(QFile::ReadOnly | QFile::Text)) {
        QTextStream stream(&file);
        QString content = stream.readAll();
        file.close();
        return content;
    } else {
        qDebug() << "Failed to load QSS file:" << filePath;
        return QString();
    }
}

QString StyleManager::replaceTemplateVariables(const QString& style)
{
    QString result = style;
    
    // 색상 변수 치환
    QMap<QString, QString> colors = getColors();
    QMap<QString, QString>::const_iterator i = colors.constBegin();
    while (i != colors.constEnd()) {
        QString placeholder = "{{" + i.key() + "}}";
        result.replace(placeholder, i.value());
        ++i;
    }
    
    // radius 변수 치환
    result.replace("{{radius}}", getRadius());
    result.replace("./style/images/", "../style/images/");

    return result;
}

void StyleManager::applyStyle(QApplication* app)
{
    QString qssPath = "../style/qss/";
    QStringList qssFiles = getQssFiles();
    QString combinedStyle;
    
    // 모든 QSS 파일 로드
    for (const QString& file : qssFiles) {
        QString fullPath = qssPath + file;
        // qDebug() << "Trying to load:" << fullPath;  // 디버깅 추가
        
        QString fileContent = loadQssFile(fullPath);
        if (!fileContent.isEmpty()) {
            combinedStyle += fileContent + "\n";
        } else {
            qDebug() << "✗ Failed to load:" << file;
        }
    }
    
    // 로드된 파일이 없으면 기본 컴포넌트 스타일 적용
    if (combinedStyle.isEmpty()) {
        qDebug() << "No QSS files loaded, applying default component styles";
        combinedStyle = getDefaultComponentStyles();
    }
    
    // 템플릿 변수 치환
    combinedStyle = replaceTemplateVariables(combinedStyle);
    
    // 스타일 적용
    app->setStyleSheet(combinedStyle);
    
}

QString StyleManager::getDefaultComponentStyles()
{
    return R"(
        /* 기본 스타일 */
        QMainWindow {
            background-color: #f0f0f0;
            font-family: "Arial", sans-serif;
        }
        
        /* 버튼 스타일 */
        QPushButton {
            background-color: #03B4C8;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: bold;
            min-height: 20px;
        }
        
        QPushButton:hover {
            background-color: #0E9FAF;
        }
        
        QPushButton:pressed {
            background-color: #025a66;
        }
        
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
        
        /* 텍스트 필드 스타일 */
        QLineEdit {
            background-color: white;
            border: 2px solid #ddd;
            border-radius: 4px;
            padding: 8px 12px;
            font-size: 14px;
            color: #333;
        }
        
        QLineEdit:focus {
            border-color: #03B4C8;
            outline: none;
        }
        
        QLineEdit::placeholder {
            color: #999;
        }
        
        /* 체크박스 스타일 */
        QCheckBox {
            font-size: 14px;
            color: #333;
            spacing: 8px;
        }
        
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 2px solid #ddd;
            border-radius: 3px;
            background-color: white;
        }
        
        QCheckBox::indicator:checked {
            background-color: #03B4C8;
            border-color: #03B4C8;
            image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iOSIgdmlld0JveD0iMCAwIDEyIDkiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0xMC42IDEuNEwzLjkgOC4xTDEuNCA1LjYiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cjwvc3ZnPgo=);
        }
        
        QCheckBox::indicator:hover {
            border-color: #03B4C8;
        }
        
        /* 라벨 스타일 */
        QLabel {
            color: #333;
            font-size: 14px;
        }
        
        /* 특정 ID/클래스 스타일 */
        #login_logo {
            font-weight: 900;
            font-size: 50px;
            color: #03B4C8;
        }
        
        #login_bg {
            background-color: #f8f9fa;
        }
    )";
}

// 폰트 로드 함수 (선택사항)
void loadCustomFont(QApplication* app)
{
    int fontId = QFontDatabase::addApplicationFont("./style/fonts/Inter-VariableFont.ttf");
    if (fontId != -1) {
        QStringList fontFamilies = QFontDatabase::applicationFontFamilies(fontId);
        if (!fontFamilies.isEmpty()) {
            QFont font(fontFamilies.first());
            app->setFont(font);
            qDebug() << "Custom font loaded:" << fontFamilies.first();
        }
    } else {
        qDebug() << "Failed to load custom font";
    }
}