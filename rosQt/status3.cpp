#include "status3.h"
#include "BasicQtApp_autogen/include/ui_status3.h"  // UI 헤더 포함
#include <QDebug>
#include <QStyle>
#include <QPushButton>


Status3Widget::Status3Widget(QWidget *parent) 
    : QWidget(parent)
    , ui(new Ui_Status3Widget)
{
    ui->setupUi(this);  // UI 파일 설정
    setWidgetClasses();
}

Status3Widget::~Status3Widget()
{
    delete ui;
}

void Status3Widget::setWidgetClasses()
{
    if (ui->bg) {
        ui->bg->setProperty("class", "bg graye radius");
    }
    if (ui->title1) {
        ui->title1->setProperty("class", "size20 weight700 color-gray6 bg transparent");
    }
}

void Status3Widget::show_at(const QPoint& pos)
{
    move(pos);
    show();
}

void Status3Widget::refresh()
{
    qDebug() << "Status3 widget refresh";

}