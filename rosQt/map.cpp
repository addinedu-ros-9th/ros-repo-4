#include "map.h"
#include "BasicQtApp_autogen/include/ui_map.h"  // UI 헤더 포함
#include <QDebug>

MapWidget::MapWidget(QWidget *parent) 
    : QWidget(parent)
    , ui(new Ui_MapWidget)
{
    ui->setupUi(this);  // UI 파일 설정
    setWidgetClasses();
}

MapWidget::~MapWidget()
{
    delete ui;
}

void MapWidget::setWidgetClasses()
{
    if (ui->map_img) {
        ui->map_img->setProperty("class", "map_img");
    }
    if (ui->map_robot) {
        ui->map_robot->setProperty("class", "map_robot");
    }
}

void MapWidget::show_at(const QPoint& pos)
{
    move(pos);
    show();
}

void MapWidget::refresh()
{
    qDebug() << "Map widget refresh";
    
}