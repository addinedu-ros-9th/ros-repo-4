#include "udp_image_receiver.h"
#include <QDebug>
#include <QImage>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>

UdpImageReceiver::UdpImageReceiver(const QString& ip, int port, QObject *parent)
    : QObject(parent)
    , target_ip_(ip)
    , target_port_(port)
    , socket_fd_(-1)
    , receive_timer_(nullptr)
    , running_(false)
    , connection_established_(false) 
{
    receive_timer_ = new QTimer(this);
    connect(receive_timer_, &QTimer::timeout, this, &UdpImageReceiver::receiveImage);
}

UdpImageReceiver::~UdpImageReceiver()
{
    stop();
}

bool UdpImageReceiver::initialize()
{
    // UDP 소켓 생성
    socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
        emit connectionError("소켓 생성 실패");
        return false;
    }
    
    // 소켓을 non-blocking으로 설정
    int flags = fcntl(socket_fd_, F_GETFL, 0);
    fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK);
    
    // 소켓 옵션 설정
    int opt = 1;
    setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    // 서버 주소 설정
    memset(&server_addr_, 0, sizeof(server_addr_));
    server_addr_.sin_family = AF_INET;
    server_addr_.sin_addr.s_addr = INADDR_ANY;
    server_addr_.sin_port = htons(target_port_);
    
    // 바인드
    if (bind(socket_fd_, (struct sockaddr*)&server_addr_, sizeof(server_addr_)) < 0) {
        emit connectionError(QString("바인드 실패 (포트: %1)").arg(target_port_));
        return false;
    }
    
    qDebug() << "UDP 이미지 수신기 초기화 완료:" << target_ip_ << ":" << target_port_;
    return true;
}

void UdpImageReceiver::start()
{
    if (!initialize()) {
        return;
    }
    
    running_ = true;
    receive_timer_->start(33);  // ~30 FPS (33ms 간격)
    
    qDebug() << "UDP 이미지 수신 시작";
}

void UdpImageReceiver::stop()
{
    running_ = false;
    
    if (receive_timer_) {
        receive_timer_->stop();
    }
    
    if (socket_fd_ >= 0) {
        close(socket_fd_);
        socket_fd_ = -1;
    }
    
    qDebug() << "UDP 이미지 수신 중지";
}

void UdpImageReceiver::receiveImage()
{
    if (!running_ || socket_fd_ < 0) {
        return;
    }
    
    const int buffer_size = 65536;
    std::vector<uint8_t> buffer(buffer_size);
    
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    
    // UDP 패킷 수신 (non-blocking)
    ssize_t received = recvfrom(socket_fd_, buffer.data(), buffer_size, 0,
                               (struct sockaddr*)&client_addr, &client_len);
    
    if (received > 0) {
        // 첫 번째 데이터 수신 시 메시지
        if (!connection_established_) {
            qDebug() << "🎥 AI Server로부터 첫 이미지 데이터 수신됨!";
            connection_established_ = true;
        }
        
        // JPEG 디코딩
        std::vector<uint8_t> jpeg_data(buffer.begin(), buffer.begin() + received);
        cv::Mat image = cv::imdecode(jpeg_data, cv::IMREAD_COLOR);
        
        if (!image.empty()) {
            // 16:9 비율로 자르기
            cv::Mat cropped_image = cropTo16by9(image);
            
            // OpenCV Mat을 QPixmap으로 변환
            QPixmap pixmap = matToQPixmap(cropped_image);
            emit imageReceived(pixmap);
        }
    }
}

// 16:9 비율로 자르는 함수 추가
cv::Mat UdpImageReceiver::cropTo16by9(const cv::Mat& image)
{
    if (image.empty()) {
        return image;
    }
    
    int original_width = image.cols;   // 640
    int original_height = image.rows;  // 480
    
    // 16:9 비율 계산
    double target_ratio = 16.0 / 9.0;  // 1.777...
    double current_ratio = static_cast<double>(original_width) / original_height;  // 640/480 = 1.333...
    
    int crop_width, crop_height;
    int crop_x, crop_y;
    
    if (current_ratio > target_ratio) {
        // 현재 이미지가 더 넓음 → 좌우를 자르기
        crop_height = original_height;  // 480 그대로
        crop_width = static_cast<int>(crop_height * target_ratio);  // 480 * (16/9) = 853
        
        // 하지만 원본이 640이므로 실제로는 640 사용
        crop_width = std::min(crop_width, original_width);
        
        crop_x = (original_width - crop_width) / 2;  // 중앙 정렬
        crop_y = 0;
    } else {
        // 현재 이미지가 더 높음 → 상하를 자르기 (640x480의 경우)
        crop_width = original_width;   // 640 그대로
        crop_height = static_cast<int>(crop_width / target_ratio);  // 640 / (16/9) = 360
        
        crop_x = 0;
        crop_y = (original_height - crop_height) / 2;  // 중앙 정렬
    }
    
    // ROI (Region of Interest) 설정
    cv::Rect crop_rect(crop_x, crop_y, crop_width, crop_height);
    
    // 범위 검사
    crop_rect.x = std::max(0, crop_rect.x);
    crop_rect.y = std::max(0, crop_rect.y);
    crop_rect.width = std::min(crop_rect.width, original_width - crop_rect.x);
    crop_rect.height = std::min(crop_rect.height, original_height - crop_rect.y);
    
    return image(crop_rect);
}

QPixmap UdpImageReceiver::matToQPixmap(const cv::Mat& mat)
{
    if (mat.empty()) {
        return QPixmap();
    }
    
    // OpenCV BGR을 RGB로 변환
    cv::Mat rgb_mat;
    cv::cvtColor(mat, rgb_mat, cv::COLOR_BGR2RGB);
    
    // QImage로 변환
    QImage qimg(rgb_mat.data, rgb_mat.cols, rgb_mat.rows, rgb_mat.step, QImage::Format_RGB888);
    
    // QPixmap으로 변환
    return QPixmap::fromImage(qimg);
}