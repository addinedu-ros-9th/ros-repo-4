#ifndef WEBCAM_STREAMER_H
#define WEBCAM_STREAMER_H

#include <opencv2/opencv.hpp>
#include <functional>
#include <thread>
#include <atomic>

class WebcamStreamer
{
public:
    WebcamStreamer(int camera_id = 0);
    ~WebcamStreamer();
    
    bool initialize();
    void start(std::function<void(const cv::Mat&)> frame_callback);
    void stop();
    
    bool isRunning() const { return running_; }
    bool isInitialized() const { return initialized_; }

private:
    cv::VideoCapture cap_;
    std::atomic<bool> running_;
    std::atomic<bool> initialized_;
    std::thread capture_thread_;
    
    int camera_id_;
    std::function<void(const cv::Mat&)> frame_callback_;
    
    void captureLoop();
};

#endif // WEBCAM_STREAMER_H
