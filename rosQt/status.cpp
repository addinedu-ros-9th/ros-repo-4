#include "status.h"
#include "BasicQtApp_autogen/include/ui_status.h"  // UI 헤더 포함
#include <QDebug>

StatusWidget::StatusWidget(QWidget *parent) 
    : QWidget(parent)
    , ui(new Ui_StatusWidget)
{
    ui->setupUi(this);  // UI 파일 설정
    setWidgetClasses();
}

StatusWidget::~StatusWidget()
{
    delete ui;
}

void StatusWidget::setWidgetClasses()
{
    if (ui->move_bg) {
        ui->move_bg->setProperty("class", "bg green_gray1");
    }
    if (ui->move_first) {
        ui->move_first->setProperty("class", "move_first");
    }
    if (ui->move_last) {
        ui->move_last->setProperty("class", "move_last");
    }
    if (ui->move_arrow) {
        ui->move_arrow->setProperty("class", "move_arrow");
    }
    if (ui->point1) {
        ui->point1->setProperty("class", "point");
    }
    if (ui->point2) {
        ui->point2->setProperty("class", "point");
    }
    if (ui->point3) {
        ui->point3->setProperty("class", "point");
    }
    if (ui->point4) {
        ui->point4->setProperty("class", "point");
    }
    if (ui->title1) {
        ui->title1->setProperty("class", "size14 weight700 color-gray3");
    }
    if (ui->title2) {
        ui->title2->setProperty("class", "size14 weight700 color-gray3");
    }
    if (ui->title3) {
        ui->title3->setProperty("class", "size14 weight700 color-gray3");
    }
    if (ui->title4) {
        ui->title4->setProperty("class", "size14 weight700 color-gray3");
    }
    if (ui->title5) {
        ui->title5->setProperty("class", "size14 weight700 color-gray3");
    }
    if (ui->title5_box){
        ui->title5_box->setProperty("class", "otl gray6 radius");
    }

    if (ui->title5_box_cell1) {
        ui->title5_box_cell1->setProperty("class", "radius bg primary");
    }
    if (ui->title5_box_cell2) {
        ui->title5_box_cell2->setProperty("class", "radius bg primary");
    }
    if (ui->title5_box_cell3) {
        ui->title5_box_cell3->setProperty("class", "radius bg primary");
    }
    if (ui->title5_box_cell4) {
        ui->title5_box_cell4->setProperty("class", "radius bg primary");
    }
    if (ui->title5_box_cell5) {
        ui->title5_box_cell5->setProperty("class", "radius bg primary");
    }
    if (ui->title5_box_cell6) {
        ui->title5_box_cell6->setProperty("class", "radius bg primary");
    }
    if (ui->title5_box_cell7) {
        ui->title5_box_cell7->setProperty("class", "radius bg primary");
    }
    if (ui->title5_box_cell8) {
        ui->title5_box_cell8->setProperty("class", "radius bg primary");
    }
    if (ui->title5_box_cell9) {
        ui->title5_box_cell9->setProperty("class", "radius bg primary");
    }
    if (ui->title5_box_cell10) {
        ui->title5_box_cell10->setProperty("class", "radius bg primary");
    }

    if (ui->title5_status) {
        ui->title5_status->setProperty("class", "size14 weight700 color-gray3");
    }
    if (ui->title6) {
        ui->title6->setProperty("class", "size14 weight700 color-gray3");
    }
    if (ui->title6_status) {
        ui->title6_status->setProperty("class", "size14 weight700 color-gray3");
    }
    if (ui->title6_box) {
        ui->title6_box->setProperty("class", "otl gray6 radius");
    }
    if (ui->title6_box_cell1) {
        ui->title6_box_cell1->setProperty("class", "radius bg primary_25p");
    }
    if (ui->title6_box_cell2) {
        ui->title6_box_cell2->setProperty("class", "radius bg primary_50p");
    }
    if (ui->title6_box_cell3) {
        ui->title6_box_cell3->setProperty("class", "radius bg primary_75p");
    }
    if (ui->title6_box_cell4) {
        ui->title6_box_cell4->setProperty("class", "radius bg primary");
    }
    if (ui->content1) {
        ui->content1->setProperty("class", "size14 weight500 color-gray3");
    }
    if (ui->content2) {
        ui->content2->setProperty("class", "size14 weight500 color-gray3");
    }
    if (ui->content3) {
        ui->content3->setProperty("class", "size14 weight500 color-gray3");
    }
    if (ui->content4) {
        ui->content4->setProperty("class", "size14 weight500 color-gray3"); 
    }
    if (ui->machine_box) {
        ui->machine_box->setProperty("class", "otl primary_dark radius");
    }
    if (ui->battery_img) {
        ui->battery_img->setProperty("class", "battery_img");
    }
    if (ui->network_img) {
        ui->network_img->setProperty("class", "network_img");
    }
    
}

void StatusWidget::show_at(const QPoint& pos)
{
    move(pos);
    show();
}

void StatusWidget::refresh()
{
    qDebug() << "Status widget refresh";
}