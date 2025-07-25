#include "dashboard.h"
#include "ui_dashboard.h"
#include "status.h"
#include "map.h"
#include <QVBoxLayout>
#include <QLabel>
#include <QDebug>

DashboardWidget::DashboardWidget(QWidget *parent) 
    : QWidget(parent)
    , ui(new Ui_DashboardWidget)
    , status_widget(nullptr) 
    , map_widget(nullptr)
{
    ui->setupUi(this);  // UI 파일 설정
    setWidgetClasses();
    setupStatusWidget();
    setupMapWidget();
}

DashboardWidget::~DashboardWidget()
{
    delete ui;
}

void DashboardWidget::setupStatusWidget()
{
    // StatusWidget 생성
    status_widget = new StatusWidget(this);
    
    // StatusWidget 위치 설정 (dashboard UI 내의 특정 영역에 배치)
    status_widget->setGeometry(477, 549, 753, 281);  // 위치와 크기 조정
    
    // StatusWidget 표시
    status_widget->show();
    
}

void DashboardWidget::setupMapWidget()
{
    map_widget = new MapWidget(this);
    
    map_widget->setGeometry(19, 58, 438, 772);  // 위치와 크기 조정
    
    map_widget->show();
}

void DashboardWidget::setWidgetClasses()
{
    if (ui->title_point1) {
        ui->title_point1->setProperty("class", "title_point");
    }
    if (ui->title_point2) {
        ui->title_point2->setProperty("class", "title_point");
    }
    if (ui->title_point3) {
        ui->title_point3->setProperty("class", "title_point");
    }
    if (ui->title1) {
        ui->title1->setProperty("class", "size16 weight700");
    }
    if (ui->title2) {
        ui->title2->setProperty("class", "size16 weight700");
    }
    if (ui->title3) {
        ui->title3->setProperty("class", "size16 weight700");
    }
    if (ui->controlBtn) {
        ui->controlBtn->setProperty("class", "btn outlined primary_dark small");
    }
    if (ui->camera_bg) {
        ui->camera_bg->setProperty("class", "bg green_gray1 radius");
    }
    if (ui->camera_img) {
        ui->camera_img->setProperty("class", "camera_img");
    }
    
    if (ui->status_label) {
        ui->status_label->setProperty("class", "label gray");
    }
}

void DashboardWidget::show_at(const QPoint& pos)
{
    move(pos);
    show();
}

void DashboardWidget::refresh()
{
    if (status_widget) {
        status_widget->refresh();
    }
    if (map_widget) {
        map_widget->refresh();
    }
}