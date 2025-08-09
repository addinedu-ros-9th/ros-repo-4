#include "ai_server/frame_shared_memory.h"
#include <cstring>
#include <iostream>

FrameSharedMemory::FrameSharedMemory(const std::string& name, int width, int height)
    : shm_name_(name), width_(width), height_(height), shm_fd_(-1), shm_ptr_(nullptr), available_(false)
{
    frame_size_ = width_ * height_ * 3; // BGR format
    available_ = createSharedMemory();
}

FrameSharedMemory::~FrameSharedMemory()
{
    destroySharedMemory();
}

bool FrameSharedMemory::createSharedMemory()
{
    // 공유 메모리 생성
    shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    if (shm_fd_ == -1) {
        std::cerr << "공유 메모리 생성 실패: " << shm_name_ << std::endl;
        return false;
    }
    
    // 공유 메모리 크기 설정
    if (ftruncate(shm_fd_, frame_size_) == -1) {
        std::cerr << "공유 메모리 크기 설정 실패" << std::endl;
        close(shm_fd_);
        return false;
    }
    
    // 공유 메모리 매핑
    shm_ptr_ = mmap(nullptr, frame_size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
    if (shm_ptr_ == MAP_FAILED) {
        std::cerr << "공유 메모리 매핑 실패" << std::endl;
        close(shm_fd_);
        return false;
    }
    
    return true;
}

void FrameSharedMemory::destroySharedMemory()
{
    if (shm_ptr_ != nullptr && shm_ptr_ != MAP_FAILED) {
        munmap(shm_ptr_, frame_size_);
        shm_ptr_ = nullptr;
    }
    
    if (shm_fd_ != -1) {
        close(shm_fd_);
        shm_fd_ = -1;
    }
    
    // 공유 메모리 객체 제거
    shm_unlink(shm_name_.c_str());
}

bool FrameSharedMemory::writeFrame(const cv::Mat& frame)
{
    if (!available_ || shm_ptr_ == nullptr) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 프레임 크기 확인
    if (frame.cols != width_ || frame.rows != height_) {
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(width_, height_));
        memcpy(shm_ptr_, resized_frame.data, frame_size_);
    } else {
        memcpy(shm_ptr_, frame.data, frame_size_);
    }
    
    return true;
}

bool FrameSharedMemory::readFrame(cv::Mat& frame)
{
    if (!available_ || shm_ptr_ == nullptr) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    frame = cv::Mat(height_, width_, CV_8UC3);
    memcpy(frame.data, shm_ptr_, frame_size_);
    
    return true;
}

bool FrameSharedMemory::isAvailable() const
{
    return available_;
} 