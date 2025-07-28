#include "ai_server/webcam_streamer.h"
#include <iostream>
#include <chrono>

WebcamStreamer::WebcamStreamer(int camera_id)
    : running_(false), initialized_(false), camera_id_(camera_id)
{
}

WebcamStreamer::~WebcamStreamer()
{
    stop();
}

bool WebcamStreamer::initialize()
{
    std::cout << "웹캠 초기화 중... (카메라 ID: " << camera_id_ << ")" << std::endl;
    
    // 웹캠 열기
    cap_.open(camera_id_);
    
    if (!cap_.isOpened()) {
        std::cerr << "웹캠을 열 수 없습니다! (카메라 ID: " << camera_id_ << ")" << std::endl;
        return false;
    }
    
    // 웹캠 설정
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap_.set(cv::CAP_PROP_FPS, 30);
    
    // 버퍼 크기 설정 (최신 프레임을 위해)
    cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    
    initialized_ = true;
    std::cout << "웹캠 초기화 완료!" << std::endl;
    
    return true;
}

void WebcamStreamer::start(std::function<void(const cv::Mat&)> frame_callback)
{
    if (!initialized_) {
        std::cerr << "웹캠이 초기화되지 않았습니다!" << std::endl;
        return;
    }
    
    if (running_) {
        std::cout << "웹캠 스트리머가 이미 실행중입니다." << std::endl;
        return;
    }
    
    frame_callback_ = frame_callback;
    running_ = true;
    
    // 캡처 스레드 시작
    capture_thread_ = std::thread(&WebcamStreamer::captureLoop, this);
    
    std::cout << "웹캠 스트리밍 시작!" << std::endl;
}

void WebcamStreamer::stop()
{
    if (!running_) {
        return;
    }
    
    std::cout << "웹캠 스트리밍 종료중..." << std::endl;
    
    running_ = false;
    
    // 스레드 종료 대기
    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }
    
    // 웹캠 해제
    if (cap_.isOpened()) {
        cap_.release();
    }
    
    initialized_ = false;
    std::cout << "웹캠 스트리밍 종료됨" << std::endl;
}

void WebcamStreamer::captureLoop()
{
    cv::Mat frame;
    int frame_count = 0;
    auto last_fps_time = std::chrono::steady_clock::now();
    
    while (running_) {
        // 프레임 읽기
        if (!cap_.read(frame)) {
            std::cerr << "프레임을 읽을 수 없습니다!" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS
            continue;
        }
        
        if (frame.empty()) {
            std::cerr << "빈 프레임입니다!" << std::endl;
            continue;
        }
        
        // 프레임 콜백 호출
        if (frame_callback_) {
            frame_callback_(frame);
        }
        
        // FPS 계산 및 출력 (매 100프레임마다)
        frame_count++;
        if (frame_count % 100 == 0) {
            auto current_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - last_fps_time).count();
            
            if (duration > 0) {
                double fps = (100.0 * 1000.0) / duration;
                std::cout << "웹캠 FPS: " << fps << std::endl;
            }
            
            last_fps_time = current_time;
        }
        
        // 프레임레이트 제어 (~30 FPS)
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }
}
