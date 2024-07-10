"""
用于测试视频是否能正常播放
"""
import cv2

video_file_path = 'test.mp4'  # 替换为本地视频文件路径

cap = cv2.VideoCapture(video_file_path)

if not cap.isOpened():
    print("Error: Cannot open video file.")
else:
    print("Video file opened successfully.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
