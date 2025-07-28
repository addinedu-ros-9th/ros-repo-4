#include "dashboard.h"
#include "ui_dashboard.h"
#include "status.h"
#include "map.h"
#include "udp_image_receiver.h" 
#include "control_popup1.h" 
#include "control_popup2.h" 
#include <QVBoxLayout>
#include <QLabel>
#include <QDebug>

DashboardWidget::DashboardWidget(QWidget *parent) 
    : QWidget(parent)
    , ui(new Ui_DashboardWidget)
    , status_widget(nullptr) 
    , map_widget(nullptr)
    , udp_receiver_(nullptr) 
    , control_popup1_(nullptr) 
    , control_popup2_(nullptr) 
    , status_("대기중")
{
    ui->setupUi(this);  // UI 파일 설정
    setWidgetClasses();
    setupStatusWidget();
    setupMapWidget();
    setupCameraWidget(); 
    setupControlButton();  // 추가
}

DashboardWidget::~DashboardWidget()
{
    if (udp_receiver_) {     
        udp_receiver_->stop();
        delete udp_receiver_;
    }
    if (control_popup1_) {  // 추가
        delete control_popup1_;
    }
    if (control_popup2_) {
        delete control_popup2_;
    }
    delete ui;
}

void DashboardWidget::setStatus(const QString& newStatus)
{
    if (status_ != newStatus) {
        QString oldStatus = status_;
        status_ = newStatus;
        
        qDebug() << "로봇 상태 변경:" << oldStatus << "→" << newStatus;
        
        // 상태가 변경되면 열려있는 팝업들 닫기
        if (control_popup1_ && control_popup1_->isVisible()) {
            control_popup1_->hide();
        }
        if (control_popup2_ && control_popup2_->isVisible()) {
            control_popup2_->hide();
        }
    }
}

QString DashboardWidget::getStatus() const
{
    return status_;
}

void DashboardWidget::setupControlButton()
{
    if (ui->controlBtn) {
        connect(ui->controlBtn, &QPushButton::clicked,
                this, &DashboardWidget::onControlButtonClicked);
    } else {
        qDebug() << "❌ controlBtn을 찾을 수 없습니다!";
    }
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

void DashboardWidget::onControlButtonClicked()
{
    qDebug() << "🎮 Control 버튼 클릭! 현재 상태:" << status_;
    
    if (status_ == "이동중") {
        // 이동 중일 때 - control_popup1 표시
        qDebug() << "이동 중 상태 → ControlPopup1 표시";
        
        // 다른 팝업이 열려있으면 닫기
        if (control_popup2_ && control_popup2_->isVisible()) {
            control_popup2_->hide();
        }
        
        // control_popup1 표시
        if (control_popup1_ && control_popup1_->isVisible()) {
            control_popup1_->raise();
            control_popup1_->activateWindow();
            return;
        }
        
        if (!control_popup1_) {
            control_popup1_ = new ControlPopup1(this);
        }
        
        control_popup1_->show();
        control_popup1_->raise();
        control_popup1_->activateWindow();
        
    } else {
        // 다른 팝업이 열려있으면 닫기
        if (control_popup1_ && control_popup1_->isVisible()) {
            control_popup1_->hide();
        }
        
        // control_popup2 표시 (임시로 popup1과 동일하게, 나중에 교체)
        if (control_popup2_ && control_popup2_->isVisible()) {
            control_popup2_->raise();
            control_popup2_->activateWindow();
            return;
        }
        
        // ControlPopup2 생성 및 표시
        if (!control_popup2_) {
            control_popup2_ = new ControlPopup2(this);  // ← 실제로 ControlPopup2 생성
        }
        
        control_popup2_->show();
        control_popup2_->raise();
        control_popup2_->activateWindow();
    }
}

void DashboardWidget::setupMapWidget()
{
    map_widget = new MapWidget(this);
    
    map_widget->setGeometry(19, 58, 438, 772);  // 위치와 크기 조정
    
    map_widget->show();
}

void DashboardWidget::setupCameraWidget()
{
    // UDP 이미지 수신기 생성
    udp_receiver_ = new UdpImageReceiver("127.0.0.1", 8888, this);
    
    // 시그널 연결
    connect(udp_receiver_, &UdpImageReceiver::imageReceived, 
            this, &DashboardWidget::onImageReceived);
    connect(udp_receiver_, &UdpImageReceiver::connectionError, 
            this, &DashboardWidget::onConnectionError);

    // 연결 성공 시그널 추가 (새로 추가할 예정)
    connect(udp_receiver_, &UdpImageReceiver::connectionEstablished,
            this, &DashboardWidget::onConnectionEstablished);
    
    // 수신 시작
    udp_receiver_->start();
    
    // camera_img에 기본 텍스트 설정
    if (ui->camera_img) {
        ui->camera_img->setText("AI Server 연결 중...\n127.0.0.1:8888");
        ui->camera_img->setAlignment(Qt::AlignCenter);
        ui->camera_img->setScaledContents(true);
        
        // 연결 중 스타일
        ui->camera_img->setStyleSheet("background-color: #333; color: yellow; font-size: 14px;");
    }
}

void DashboardWidget::onImageReceived(const QPixmap& pixmap)
{
    static bool first_image = true;
    if (first_image) {
        qDebug() << "✅ AI Server 연결 성공! 이미지 수신 시작";
        first_image = false;
    }

    if (ui->camera_img) {
        // camera_img에 받은 이미지 표시
        ui->camera_img->setPixmap(pixmap.scaled(
            ui->camera_img->size(), 
            Qt::KeepAspectRatio, 
            Qt::SmoothTransformation
        ));

        // 연결 성공 시 스타일 초기화
        ui->camera_img->setStyleSheet("");
    }
}

void DashboardWidget::onConnectionError(const QString& error)
{
    qDebug() << "❌ AI Server 연결 실패:" << error;
    
    if (ui->camera_img) {
        ui->camera_img->setText("카메라 연결 실패\n" + error);
        ui->camera_img->setAlignment(Qt::AlignCenter);
    }
}

// 새로 추가할 슬롯
void DashboardWidget::onConnectionEstablished()
{
    qDebug() << "🔗 AI Server UDP 소켓 연결됨 (127.0.0.1:8888)";
    
    if (ui->camera_img) {
        ui->camera_img->setText("AI Server 연결됨\n이미지 수신 대기 중...");
        ui->camera_img->setStyleSheet("background-color: #333; color: green; font-size: 14px;");
    }
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
        ui->status_label->setText(status_);
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
