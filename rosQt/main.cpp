#include <QApplication>
#include <QStackedWidget>
#include <QMainWindow>
#include <QDebug>

#include "login.h"   
#include "layout.h" 
#include "style.h"

class MainApp : public QMainWindow
{
    Q_OBJECT

public:
    MainApp(QWidget *parent = nullptr);
    
private slots:
    void showLayoutWindow();
    void showLoginWindow();

private:
    QStackedWidget *stackedWidget;
    LoginWindow *loginWindow;
    LayoutWindow *layoutWindow;
};

MainApp::MainApp(QWidget *parent)
    : QMainWindow(parent)
{
    setWindowTitle("Asan Hero");
    setFixedSize(1440, 900);
    
    // QStackedWidget을 central widget으로 설정
    stackedWidget = new QStackedWidget(this);
    setCentralWidget(stackedWidget);
    
    // LoginWindow와 LayoutWindow 생성 (부모 없이)
    loginWindow = new LoginWindow();  
    layoutWindow = new LayoutWindow(); 
    
    // 위젯들을 스택에 추가
    stackedWidget->addWidget(loginWindow);
    stackedWidget->addWidget(layoutWindow);
    
    // 시그널 연결
    connect(loginWindow, &LoginWindow::loginSuccessful, 
            this, &MainApp::showLayoutWindow);
    connect(layoutWindow, &LayoutWindow::logout_successful, 
            this, &MainApp::showLoginWindow);
    
    // 초기 화면: 로그인
    stackedWidget->setCurrentWidget(loginWindow);
}

void MainApp::showLayoutWindow()
{
    qDebug() << "Switching to LayoutWindow";
    stackedWidget->setCurrentWidget(layoutWindow);
    layoutWindow->refresh();
}

void MainApp::showLoginWindow()
{
    qDebug() << "Switching to LoginWindow";
    stackedWidget->setCurrentWidget(loginWindow);
}

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    
    StyleManager::applyStyle(&app);
    
    MainApp mainApp;
    mainApp.show();
    
    return app.exec();
}

#include "main.moc"