# 테스트용 파일
#!/usr/bin/env python3
"""
UDP 이미지 수신 테스트 클라이언트
AI Server에서 전송되는 웹캠 이미지를 받아서 화면에 표시합니다.
"""

import socket
import cv2
import numpy as np
import struct

class UdpImageReceiver:
    def __init__(self, port=8888):
        self.port = port
        self.socket = None
        self.buffer_size = 65536
        
    def initialize(self):
        """UDP 소켓 초기화"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('0.0.0.0', self.port))
            self.socket.settimeout(5.0)  # 5초 타임아웃
            print(f"UDP 이미지 수신기 시작됨 (포트: {self.port})")
            return True
        except Exception as e:
            print(f"소켓 초기화 실패: {e}")
            return False
    
    def receive_image(self):
        """이미지 수신 및 디코딩"""
        try:
            # 첫 번째 패킷 수신
            data, addr = self.socket.recvfrom(self.buffer_size)
            
            # 단일 패킷인지 다중 패킷인지 확인
            if len(data) >= 12:  # 헤더가 있는 경우 (다중 패킷)
                # 헤더 파싱해서 실제로 다중 패킷인지 확인
                try:
                    packet_num = struct.unpack('!I', data[0:4])[0]
                    total_packets = struct.unpack('!I', data[4:8])[0]
                    
                    # 합리적인 패킷 수인지 확인 (최대 1000개 패킷)
                    if total_packets > 1000 or packet_num >= total_packets:
                        # 헤더가 아닌 일반 데이터로 간주
                        return self.decode_image(data)
                    else:
                        return self.receive_multi_packet(data, addr)
                except:
                    # 헤더 파싱 실패시 단일 패킷으로 처리
                    return self.decode_image(data)
            else:  # 단일 패킷
                return self.decode_image(data)
                
        except socket.timeout:
            print("수신 타임아웃...")
            return None
        except Exception as e:
            print(f"이미지 수신 오류: {e}")
            return None
    
    def receive_multi_packet(self, first_packet, addr):
        """다중 패킷 이미지 수신"""
        try:
            # 첫 번째 패킷에서 헤더 파싱
            packet_num = struct.unpack('!I', first_packet[0:4])[0]
            total_packets = struct.unpack('!I', first_packet[4:8])[0]
            data_size = struct.unpack('!I', first_packet[8:12])[0]
            
            print(f"다중 패킷 수신 시작: {total_packets}개 패킷")
            
            # 패킷 데이터 저장용
            packets = {}
            packets[packet_num] = first_packet[12:12+data_size]
            
            # 나머지 패킷들 수신
            for _ in range(total_packets - 1):
                data, _ = self.socket.recvfrom(self.buffer_size)
                
                if len(data) >= 12:
                    pkt_num = struct.unpack('!I', data[0:4])[0]
                    pkt_size = struct.unpack('!I', data[8:12])[0]
                    packets[pkt_num] = data[12:12+pkt_size]
            
            # 패킷들을 순서대로 합치기
            complete_data = bytearray()
            for i in range(total_packets):
                if i in packets:
                    complete_data.extend(packets[i])
                else:
                    print(f"패킷 {i} 누락!")
                    return None
            
            return self.decode_image(bytes(complete_data))
            
        except Exception as e:
            print(f"다중 패킷 수신 오류: {e}")
            return None
    
    def decode_image(self, data):
        """바이트 데이터를 OpenCV 이미지로 디코딩"""
        try:
            # JPEG 데이터를 numpy array로 변환
            nparr = np.frombuffer(data, np.uint8)
            
            # OpenCV로 디코딩
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is not None:
                print(f"이미지 디코딩 성공: {image.shape}, 데이터 크기: {len(data)} bytes")
            
            return image
            
        except Exception as e:
            print(f"이미지 디코딩 오류: {e}")
            return None
    
    def stop(self):
        """소켓 종료"""
        if self.socket:
            self.socket.close()
            print("UDP 이미지 수신기 종료됨")

def main():
    print("=== UDP 이미지 수신 테스트 클라이언트 ===")
    print("AI Server에서 전송되는 웹캠 이미지를 수신합니다.")
    print("종료하려면 'q' 키를 누르세요.")
    
    receiver = UdpImageReceiver(8888)
    
    if not receiver.initialize():
        return
    
    try:
        while True:
            # 이미지 수신
            image = receiver.receive_image()
            
            if image is not None:
                # 이미지 표시
                cv2.imshow('UDP Webcam Stream', image)
                
                # 'q' 키로 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("이미지 수신 실패 또는 타임아웃")
                
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")
    
    finally:
        receiver.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
