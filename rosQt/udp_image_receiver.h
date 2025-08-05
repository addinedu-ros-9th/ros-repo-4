#ifndef UDP_IMAGE_RECEIVER_H
#define UDP_IMAGE_RECEIVER_H

#include <QObject>
#include <QThread>
#include <QTimer>
#include <QPixmap>
#include <opencv2/opencv.hpp>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>
#include <vector>

class UdpImageReceiver : public QObject
{
    Q_OBJECT

public:
    explicit UdpImageReceiver(const QString& ip = "127.0.0.1", int port = 7777, QObject *parent = nullptr);
    ~UdpImageReceiver();
    
    bool initialize();
    void start();
    void stop();
    
signals:
    void imageReceived(const QPixmap& pixmap);
    void connectionError(const QString& error);
    void connectionEstablished(); 

private slots:
    void receiveImage();

private:
    QString target_ip_;
    int target_port_;
    int socket_fd_;
    struct sockaddr_in server_addr_;
    QTimer* receive_timer_;
    bool running_;
    bool connection_established_;
    
    QPixmap matToQPixmap(const cv::Mat& mat);
    cv::Mat cropTo16by9(const cv::Mat& image);
};

#endif // UDP_IMAGE_RECEIVER_H