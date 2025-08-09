#ifndef FRAME_SHARED_MEMORY_H
#define FRAME_SHARED_MEMORY_H

#include <opencv2/opencv.hpp>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <mutex>

class FrameSharedMemory {
public:
    FrameSharedMemory(const std::string& name, int width, int height);
    ~FrameSharedMemory();
    
    bool writeFrame(const cv::Mat& frame);
    bool readFrame(cv::Mat& frame);
    bool isAvailable() const;
    
private:
    std::string shm_name_;
    int width_;
    int height_;
    int shm_fd_;
    void* shm_ptr_;
    size_t frame_size_;
    std::mutex mutex_;
    bool available_;
    
    bool createSharedMemory();
    void destroySharedMemory();
};

#endif // FRAME_SHARED_MEMORY_H 