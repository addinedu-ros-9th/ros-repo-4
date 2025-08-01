#include "status2.h"
#include "BasicQtApp_autogen/include/ui_status2.h"  // UI 헤더 포함
#include <QDebug>
#include <QStyle>
#include <QPushButton>


Status2Widget::Status2Widget(QWidget *parent) 
    : QWidget(parent)
    , ui(new Ui_Status2Widget)
{
    ui->setupUi(this);  // UI 파일 설정
    arrowBtns[0] = ui->arrowBtn1;
    arrowBtns[1] = ui->arrowBtn2;
    arrowBtns[2] = ui->arrowBtn3;
    arrowBtns[3] = ui->arrowBtn4;
    arrowBtns[4] = ui->arrowBtn5;
    arrowBtns[5] = ui->arrowBtn6;
    arrowBtns[6] = ui->arrowBtn7;
    arrowBtns[7] = ui->arrowBtn8;
    arrowBtns[8] = ui->arrowBtn9;
    setWidgetClasses();
    setupKeyButton();
    onClickKey5();  // 초기 상태 설정
}

Status2Widget::~Status2Widget()
{
    delete ui;
}

void Status2Widget::setWidgetClasses()
{
    for (int i = 0; i < 9; ++i) {
        if (arrowBtns[i]) {
            arrowBtns[i]->setProperty("class", "arrowBtn");
            arrowBtns[i]->style()->unpolish(arrowBtns[i]);
            arrowBtns[i]->style()->polish(arrowBtns[i]);
            arrowBtns[i]->update();
        }
    }
    if (ui->arrowBtn_key1) {
        ui->arrowBtn_key1->setProperty("class", "arrowBtn-key1");
    }
    if (ui->arrowBtn_key2) {
        ui->arrowBtn_key2->setProperty("class", "arrowBtn-key2");
    }
    if (ui->arrowBtn_key3) {
        ui->arrowBtn_key3->setProperty("class", "arrowBtn-key3");
    }
    if (ui->arrowBtn_key4) {
        ui->arrowBtn_key4->setProperty("class", "arrowBtn-key4");
    }
    if (ui->arrowBtn_key5) {
        ui->arrowBtn_key5->setProperty("class", "arrowBtn-key5");
    }
    if (ui->arrowBtn_key6) {
        ui->arrowBtn_key6->setProperty("class", "arrowBtn-key6");
    }
    if (ui->arrowBtn_key7) {
        ui->arrowBtn_key7->setProperty("class", "arrowBtn-key7");
    }
    if (ui->arrowBtn_key8) {
        ui->arrowBtn_key8->setProperty("class", "arrowBtn-key8");
    }
    if (ui->arrowBtn_key9) {
        ui->arrowBtn_key9->setProperty("class", "arrowBtn-key9");
    }
    
    if (ui->move_bg) {
        ui->move_bg->setProperty("class", "bg green_gray1");
    }
    if (ui->move_first) {
        ui->move_first->setProperty("class", "move_first");
    }

    if (ui->point1) {
        ui->point1->setProperty("class", "point");
    }
    if (ui->title1) {
        ui->title1->setProperty("class", "size14 weight700 color-gray3");
    }
    if (ui->content1) {
        ui->content1->setProperty("class", "size14 weight500 color-gray3");
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

void Status2Widget::setMoveFirstText(const QString& text)
{
    if (ui->move_first) {
        if (text == "정지중") {
            ui->move_first->setText("정지중");
        } else {
            ui->move_first->setText(text + " 이동중");
        }
    }
}

void Status2Widget::setupKeyButton()
{
    if (ui->arrowBtn1) {
        connect(ui->arrowBtn1, &QPushButton::clicked,
                this, &Status2Widget::onClickKey1);
    } 
    if (ui->arrowBtn2) {
        connect(ui->arrowBtn2, &QPushButton::clicked,
                this, &Status2Widget::onClickKey2);
    }
    if (ui->arrowBtn3) {
        connect(ui->arrowBtn3, &QPushButton::clicked,
                this, &Status2Widget::onClickKey3);
    }
    if (ui->arrowBtn4) {
        connect(ui->arrowBtn4, &QPushButton::clicked,
                this, &Status2Widget::onClickKey4);
    }
    if (ui->arrowBtn5) {
        connect(ui->arrowBtn5, &QPushButton::clicked,
                this, &Status2Widget::onClickKey5);
    }
    if (ui->arrowBtn6) {
        connect(ui->arrowBtn6, &QPushButton::clicked,
                this, &Status2Widget::onClickKey6);
    }
    if (ui->arrowBtn7) {
        connect(ui->arrowBtn7, &QPushButton::clicked,
                this, &Status2Widget::onClickKey7);
    }
    if (ui->arrowBtn8) {
        connect(ui->arrowBtn8, &QPushButton::clicked,
                this, &Status2Widget::onClickKey8);
    }
    if (ui->arrowBtn9) {
        connect(ui->arrowBtn9, &QPushButton::clicked,
                this, &Status2Widget::onClickKey9);
    }
}

void Status2Widget::onClickKey1()
{
    qDebug() << "Key 1 clicked";
    for (int i = 0; i < 9; ++i) {
        if (arrowBtns[i]) {
            if (i == 0) {
                arrowBtns[i]->setProperty("class", "arrowBtn active");
            } else {
                arrowBtns[i]->setProperty("class", "arrowBtn");
            }
            arrowBtns[i]->style()->unpolish(arrowBtns[i]);
            arrowBtns[i]->style()->polish(arrowBtns[i]);
            arrowBtns[i]->update();
        }
    }
}

void Status2Widget::onClickKey2()
{
    qDebug() << "Key 2 clicked";
    for (int i = 0; i < 9; ++i) {
        if (arrowBtns[i]) {
            if (i == 1) {
                arrowBtns[i]->setProperty("class", "arrowBtn active");
            } else {
                arrowBtns[i]->setProperty("class", "arrowBtn");
            }
            arrowBtns[i]->style()->unpolish(arrowBtns[i]);
            arrowBtns[i]->style()->polish(arrowBtns[i]);
            arrowBtns[i]->update();
        }
    }
}
void Status2Widget::onClickKey3()
{
    qDebug() << "Key 3 clicked";
    for (int i = 0; i < 9; ++i) {
        if (arrowBtns[i]) {
            if (i == 2) {
                arrowBtns[i]->setProperty("class", "arrowBtn active");
            } else {
                arrowBtns[i]->setProperty("class", "arrowBtn");
            }
            arrowBtns[i]->style()->unpolish(arrowBtns[i]);
            arrowBtns[i]->style()->polish(arrowBtns[i]);
            arrowBtns[i]->update();
        }
    }
}
void Status2Widget::onClickKey4()
{
    qDebug() << "Key 4 clicked";
    for (int i = 0; i < 9; ++i) {
        if (arrowBtns[i]) {
            if (i == 3) {
                arrowBtns[i]->setProperty("class", "arrowBtn active");
            } else {
                arrowBtns[i]->setProperty("class", "arrowBtn");
            }
            arrowBtns[i]->style()->unpolish(arrowBtns[i]);
            arrowBtns[i]->style()->polish(arrowBtns[i]);
            arrowBtns[i]->update();
        }
    }
}
void Status2Widget::onClickKey5()
{
    qDebug() << "Key 5 clicked";
    for (int i = 0; i < 9; ++i) {
        if (arrowBtns[i]) {
            if (i == 4) {
                arrowBtns[i]->setProperty("class", "arrowBtn active");
            } else {
                arrowBtns[i]->setProperty("class", "arrowBtn");
            }
            arrowBtns[i]->style()->unpolish(arrowBtns[i]);
            arrowBtns[i]->style()->polish(arrowBtns[i]);
            arrowBtns[i]->update();
        }
    }
}
void Status2Widget::onClickKey6()
{
    qDebug() << "Key 6 clicked";
    for (int i = 0; i < 9; ++i) {
        if (arrowBtns[i]) {
            if (i == 5) {
                arrowBtns[i]->setProperty("class", "arrowBtn active");
            } else {
                arrowBtns[i]->setProperty("class", "arrowBtn");
            }
            arrowBtns[i]->style()->unpolish(arrowBtns[i]);
            arrowBtns[i]->style()->polish(arrowBtns[i]);
            arrowBtns[i]->update();
        }
    }
}
void Status2Widget::onClickKey7()
{
    qDebug() << "Key 7 clicked";
    for (int i = 0; i < 9; ++i) {
        if (arrowBtns[i]) {
            if (i == 6) {
                arrowBtns[i]->setProperty("class", "arrowBtn active");
            } else {
                arrowBtns[i]->setProperty("class", "arrowBtn");
            }
            arrowBtns[i]->style()->unpolish(arrowBtns[i]);
            arrowBtns[i]->style()->polish(arrowBtns[i]);
            arrowBtns[i]->update();
        }
    }
}
void Status2Widget::onClickKey8()
{
    qDebug() << "Key 8 clicked";
    for (int i = 0; i < 9; ++i) {
        if (arrowBtns[i]) {
            if (i == 7) {
                arrowBtns[i]->setProperty("class", "arrowBtn active");
            } else {
                arrowBtns[i]->setProperty("class", "arrowBtn");
            }
            arrowBtns[i]->style()->unpolish(arrowBtns[i]);
            arrowBtns[i]->style()->polish(arrowBtns[i]);
            arrowBtns[i]->update();
        }
    }
}
void Status2Widget::onClickKey9()
{
    qDebug() << "Key 9 clicked";
    for (int i = 0; i < 9; ++i) {
        if (arrowBtns[i]) {
            if (i == 8) {
                arrowBtns[i]->setProperty("class", "arrowBtn active");
            } else {
                arrowBtns[i]->setProperty("class", "arrowBtn");
            }
            arrowBtns[i]->style()->unpolish(arrowBtns[i]);
            arrowBtns[i]->style()->polish(arrowBtns[i]);
            arrowBtns[i]->update();
        }
    }
}


void Status2Widget::show_at(const QPoint& pos)
{
    move(pos);
    show();
}

void Status2Widget::refresh()
{
    qDebug() << "Status2 widget refresh";

}