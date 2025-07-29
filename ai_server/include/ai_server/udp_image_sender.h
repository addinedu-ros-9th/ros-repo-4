#ifndef UDP_IMAGE_SENDER_H
#define UDP_IMAGE_SENDER_H

#include <opencv2/opencv.hpp>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>

class UdpImageSender
{
public:
    UdpImageSender(const std::string& target_ip = "127.0.0.1", int target_port = 8888);
    ~UdpImageSender();
    
    bool initialize();
    void sendImage(const cv::Mat& image);
    void stop();
    
    // 설정 함수들
    void setTargetAddress(const std::string& ip, int port);
    void setCompressionQuality(int quality); // JPEG 압축 품질 (0-100)
    void setMaxPacketSize(int size); // UDP 패킷 최대 크기
    
    bool isInitialized() const { return initialized_; }

private:
    // 소켓 관련
    int socket_fd_;
    struct sockaddr_in target_addr_;
    std::string target_ip_;
    int target_port_;
    
    // 설정
    int compression_quality_;
    int max_packet_size_;
    
    // 상태
    std::atomic<bool> initialized_;
    std::mutex send_mutex_;
    
    // 헬퍼 함수들
    std::vector<uchar> compressImage(const cv::Mat& image);
    bool sendData(const std::vector<uchar>& data);
    bool sendPackets(const std::vector<uchar>& data);
};

#endif // UDP_IMAGE_SENDER_H
