import cv2

def test_webcam(source=0):
    """간단한 웹캠 테스트 함수"""
    print(f"INFO: Trying to open webcam with source index: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"ERROR: Cannot open webcam with source index {source}.")
        print("Please check the following:")
        print("1. Is the webcam connected and powered on?")
        print("2. Are the correct drivers installed?")
        print("3. Is another application using the webcam?")
        print("4. If you have multiple webcams, try changing the source index (e.g., 1, 2, ...).")
        return

    print("SUCCESS: Webcam opened successfully!")
    print("A window should appear. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame from webcam.")
            break

        # Display the resulting frame
        cv2.imshow('Webcam Test - Press Q to Exit', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    print("INFO: Webcam test finished.")

if __name__ == "__main__":
    # 여러 인덱스를 시도해 볼 수 있습니다.
    test_webcam(0)
    # test_webcam(1) 
    # test_webcam(2) 