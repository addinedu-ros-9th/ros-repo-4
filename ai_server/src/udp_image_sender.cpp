#include "ai_server/udp_image_sender.h"
#include <iostream>
#include <cstring>

UdpImageSender::UdpImageSender(const std::string& target_ip, int target_port)
    : socket_fd_(-1), target_ip_(target_ip), target_port_(target_port),
      compression_quality_(80), max_packet_size_(60000), initialized_(false),
      sequence_number_(0)
{
}

UdpImageSender::~UdpImageSender()
{
    stop();
}

bool UdpImageSender::initialize()
{
    std::cout << "UDP Image Sender 초기화 중..." << std::endl;
    
    // UDP 소켓 생성
    socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
        std::cerr << "UDP 소켓 생성 실패!" << std::endl;
        return false;
    }
    
    // 타겟 주소 설정
    memset(&target_addr_, 0, sizeof(target_addr_));
    target_addr_.sin_family = AF_INET;
    target_addr_.sin_port = htons(target_port_);
    
    if (inet_pton(AF_INET, target_ip_.c_str(), &target_addr_.sin_addr) <= 0) {
        std::cerr << "잘못된 IP 주소: " << target_ip_ << std::endl;
        close(socket_fd_);
        return false;
    }
    
    // 소켓 버퍼 크기 설정
    int buffer_size = 1024 * 1024; // 1MB
    setsockopt(socket_fd_, SOL_SOCKET, SO_SNDBUF, &buffer_size, sizeof(buffer_size));
    
    initialized_ = true;
    std::cout << "UDP Image Sender 초기화 완료! (타겟: " << target_ip_ << ":" << target_port_ << ")" << std::endl;
    
    return true;
}

void UdpImageSender::sendImage(const cv::Mat& image, int camera_type)
{
    if (!initialized_ || image.empty()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(send_mutex_);
    
    try {
        // 이미지 압축
        std::vector<uchar> compressed_data = compressImage(image);
        
        if (compressed_data.empty()) {
            std::cerr << "이미지 압축 실패!" << std::endl;
            return;
        }
        
        // 프로토콜 헤더 생성
        std::vector<uchar> header = createPacketHeader(camera_type);
        
        // 헤더와 이미지 데이터 결합
        std::vector<uchar> packet_data;
        packet_data.insert(packet_data.end(), header.begin(), header.end());
        packet_data.insert(packet_data.end(), compressed_data.begin(), compressed_data.end());
        
        // UDP로 전송
        if (!sendData(packet_data)) {
            std::cerr << "UDP 전송 실패!" << std::endl;
        }
        
        // 시퀀스 번호 증가
        sequence_number_++;
        
    } catch (const std::exception& e) {
        std::cerr << "이미지 전송 중 오류: " << e.what() << std::endl;
    }
}

void UdpImageSender::stop()
{
    if (initialized_) {
        initialized_ = false;
        
        if (socket_fd_ >= 0) {
            close(socket_fd_);
            socket_fd_ = -1;
        }
        
        std::cout << "UDP Image Sender 종료됨" << std::endl;
    }
}

void UdpImageSender::setTargetAddress(const std::string& ip, int port)
{
    target_ip_ = ip;
    target_port_ = port;
    
    if (initialized_) {
        // 이미 초기화된 경우 주소 업데이트
        target_addr_.sin_port = htons(target_port_);
        inet_pton(AF_INET, target_ip_.c_str(), &target_addr_.sin_addr);
    }
}

void UdpImageSender::setCompressionQuality(int quality)
{
    compression_quality_ = std::max(1, std::min(100, quality));
}

void UdpImageSender::setMaxPacketSize(int size)
{
    max_packet_size_ = std::max(1024, std::min(65507, size)); // UDP 최대 크기 제한
}

std::vector<uchar> UdpImageSender::compressImage(const cv::Mat& image)
{
    std::vector<uchar> compressed_data;
    
    // JPEG 압축 매개변수
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(compression_quality_);
    
    // 이미지를 JPEG로 압축
    if (!cv::imencode(".jpg", image, compressed_data, compression_params)) {
        std::cerr << "JPEG 압축 실패!" << std::endl;
        return std::vector<uchar>();
    }
    
    static int frame_count = 0;
    frame_count++;
    
    if (frame_count % 30 == 0) { // 30프레임마다 로그
        std::cout << "이미지 압축 완료 - 원본: " << image.cols << "x" << image.rows 
                  << ", 압축 크기: " << compressed_data.size() << " bytes" << std::endl;
    }
    
    return compressed_data;
}

bool UdpImageSender::sendData(const std::vector<uchar>& data)
{
    if (data.size() <= static_cast<size_t>(max_packet_size_)) {
        // 단일 패킷으로 전송 (헤더 없이)
        ssize_t sent = sendto(socket_fd_, data.data(), data.size(), 0,
                             (struct sockaddr*)&target_addr_, sizeof(target_addr_));
        
        if (sent < 0) {
            std::cerr << "UDP 전송 실패: " << strerror(errno) << std::endl;
            return false;
        }
        
        return sent == static_cast<ssize_t>(data.size());
    } else {
        // 다중 패킷으로 분할 전송
        return sendPackets(data);
    }
}

bool UdpImageSender::sendPackets(const std::vector<uchar>& data)
{
    // 간단한 헤더 구조: [패킷 번호(4바이트)][총 패킷 수(4바이트)][데이터 크기(4바이트)][데이터]
    const int header_size = 12;
    const int payload_size = max_packet_size_ - header_size;
    const int total_packets = (data.size() + payload_size - 1) / payload_size;
    
    std::cout << "대용량 이미지 분할 전송: " << data.size() << " bytes -> " 
              << total_packets << " packets" << std::endl;
    
    for (int packet_num = 0; packet_num < total_packets; packet_num++) {
        // 패킷 데이터 준비
        std::vector<uchar> packet_data(header_size);
        
        // 헤더 작성
        *reinterpret_cast<uint32_t*>(&packet_data[0]) = htonl(packet_num);
        *reinterpret_cast<uint32_t*>(&packet_data[4]) = htonl(total_packets);
        
        // 현재 패킷의 데이터 크기
        int start_pos = packet_num * payload_size;
        int current_size = std::min(payload_size, static_cast<int>(data.size()) - start_pos);
        *reinterpret_cast<uint32_t*>(&packet_data[8]) = htonl(current_size);
        
        // 데이터 추가
        packet_data.insert(packet_data.end(), 
                          data.begin() + start_pos, 
                          data.begin() + start_pos + current_size);
        
        // 전송
        ssize_t sent = sendto(socket_fd_, packet_data.data(), packet_data.size(), 0,
                             (struct sockaddr*)&target_addr_, sizeof(target_addr_));
        
        if (sent < 0) {
            std::cerr << "패킷 " << packet_num << " 전송 실패: " << strerror(errno) << std::endl;
            return false;
        }
        
        // 패킷 간 짧은 지연 (네트워크 혼잡 방지)
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    
    return true;
}

std::vector<uchar> UdpImageSender::createPacketHeader(int camera_type)
{
    std::vector<uchar> header;
    
    // 1byte: Start(0xAB)
    header.push_back(0xAB);
    
    // 1byte: 카메라 타입(0x00=front/0x01=back)
    header.push_back(static_cast<uchar>(camera_type));
    
    // 4byte: 시퀀스 번호 (little-endian)
    header.push_back(static_cast<uchar>(sequence_number_ & 0xFF));
    header.push_back(static_cast<uchar>((sequence_number_ >> 8) & 0xFF));
    header.push_back(static_cast<uchar>((sequence_number_ >> 16) & 0xFF));
    header.push_back(static_cast<uchar>((sequence_number_ >> 24) & 0xFF));
    
    // 4byte: 타임스탬프 (현재 시간을 밀리초 단위로, little-endian)
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    
    header.push_back(static_cast<uchar>(millis & 0xFF));
    header.push_back(static_cast<uchar>((millis >> 8) & 0xFF));
    header.push_back(static_cast<uchar>((millis >> 16) & 0xFF));
    header.push_back(static_cast<uchar>((millis >> 24) & 0xFF));
    
    return header;
}
