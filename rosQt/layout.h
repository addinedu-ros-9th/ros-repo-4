#ifndef LAYOUTWINDOW_H
#define LAYOUTWINDOW_H

#include <QWidget>
#include <QPushButton>
#include <QLabel>
#include <QStackedWidget>
#include <QPoint>
#include <QVBoxLayout>
#include <QHBoxLayout>

class Ui_LayoutWidget;
class Ui_userPopover;
class DashboardWidget;
class LogWidget;

// UserPopover 클래스
class UserPopover : public QWidget
{
    Q_OBJECT

public:
    explicit UserPopover(QWidget *parent = nullptr);
    ~UserPopover();
    
    void refresh();
    void show_at(const QPoint& pos);

signals:
    void logoutRequested();

private:
    void setWidgetClasses();
    
    Ui_userPopover *ui;
};

// LayoutWindow 클래스
class LayoutWindow : public QWidget   // QMainWindow 대신 QWidget
{
    Q_OBJECT

public:
    explicit LayoutWindow(QWidget *parent = nullptr);
    ~LayoutWindow();
    
    void refresh();

signals:
    void logout_successful();

private slots:
    void clickMenu1();
    void clickMenu2();
    void handle_logout();
    void show_user_popover();

private:
    void setWidgetClasses();
    void setupConnections();
    void setupWidgets();
    void updateMenuButtons(int activeMenu);
    
    Ui_LayoutWidget *ui;
    UserPopover *user_popover;
    DashboardWidget *dashboard_widget;
    LogWidget *log_widget;
    QString store_name;
};

#endif // LAYOUTWINDOW_H