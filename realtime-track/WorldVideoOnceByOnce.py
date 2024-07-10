"""
当运行一次程序时，每检测的一帧都单独放入JSON文件中,包含世界坐标
"""
import cv2
import torch
from ultralytics import RTDETR
import threading
from queue import Queue
from deep_sort_realtime.deepsort_tracker import DeepSort
import json
import os
import numpy as np

def plane_to_world_coordinates(plane_coord, intrinsic_matrix, extrinsic_matrix):
    """
    将平面坐标转换为世界坐标
    :param plane_coord: 平面坐标 (x, y)
    :param intrinsic_matrix: 相机的内参矩阵
    :param extrinsic_matrix: 相机的外参矩阵
    :return: 世界坐标 (X, Y, Z)
    """
    # 将像素坐标转换为齐次坐标
    u, v = plane_coord
    pixel_coord = np.array([u, v, 1.0])

    # 使用内参矩阵计算相机坐标
    cam_coord = np.linalg.inv(intrinsic_matrix) @ pixel_coord

    # 假设Z=1，计算相机坐标
    cam_coord = cam_coord * 1.0

    # 使用外参矩阵计算世界坐标
    world_coord = np.linalg.inv(extrinsic_matrix) @ np.append(cam_coord, 1.0)
    world_coord = world_coord[:3] / world_coord[3]

    return world_coord

def main():
    # 加载类别名称
    class_names = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
        7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
        13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
        26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
        32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
        37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork',
        43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
        50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
        57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv',
        63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
        69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
        76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
    }

    # 加载RT-DETR模型
    model = RTDETR('rtdetr-l.pt')
    video_file_path = 'demo.mp4'  # 本地视频文件路径

    # 初始化DeepSORT
    tracker = DeepSort(max_age=30, nn_budget=100, nms_max_overlap=1.0)

    # 打开本地视频文件
    cap = cv2.VideoCapture(video_file_path)

    # 检查是否成功打开视频文件
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    print("Video file opened successfully.")

    frame_queue = Queue(maxsize=10)
    result_queue = Queue(maxsize=10)

    # 轨迹图数据
    trajectories = {}

    # 相机的内参矩阵（假设值，需要根据实际情况修改）
    intrinsic_matrix = np.array([[1000, 0, 640],
                                 [0, 1000, 360],
                                 [0, 0, 1]])

    # 相机的外参矩阵（假设值，需要根据实际情况修改）
    extrinsic_matrix = np.eye(4)

    # 创建 runs 文件夹（如果不存在）
    if not os.path.exists('oboRuns'):
        os.makedirs('oboRuns')

    def process_frame():
        while True:
            frame = frame_queue.get()
            if frame is None:
                break

            # 进行目标检测，检测人和车辆
            results = model(frame, classes=[0, 2, 3, 5, 7])  # 不需要指定classes，模型会返回所有检测结果

            detections = []
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # 获取检测框信息
                confidences = result.boxes.conf.cpu().numpy()  # 获取置信度
                class_ids = result.boxes.cls.cpu().numpy()  # 获取类别ID

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    class_name = class_names.get(int(class_ids[i]), 'Unknown')  # 从class_names获取类名
                    detections.append(((x1, y1, x2 - x1, y2 - y1), confidences[i], class_name, None))

            result_queue.put((frame, detections))
            frame_queue.task_done()

    def display_frame():
        run_counter = 1  # 用于文件名计数
        while True:
            try:
                frame, detections = result_queue.get()
                if frame is None:
                    break

                # 使用DeepSORT进行跟踪
                tracks = tracker.update_tracks(detections, frame=frame)

                # 显示处理后的视频帧
                person_count = 0
                current_frame_data = {"车": {}, "人": {}}

                for track in tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue

                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)
                    class_name = track.get_det_class()  # 获取检测的类别名
                    if class_name is None:
                        class_name = "Unknown"  # 如果类别名为空，则标记为Unknown
                    label = f"{class_name} {track_id}"

                    if track_id not in trajectories:
                        trajectories[track_id] = []
                    # 计算中心点
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    trajectories[track_id].append((center_x, center_y))

                    # 绘制轨迹
                    for point in trajectories[track_id]:
                        cv2.circle(frame, point, 2, (0, 0, 255), -1)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # 转换为世界坐标
                    world_coord = plane_to_world_coordinates((center_x, center_y), intrinsic_matrix, extrinsic_matrix)

                    # 保存每一帧的ID列表和坐标
                    if class_name == 'person':
                        person_count += 1
                        if track_id not in current_frame_data["人"]:
                            current_frame_data["人"][track_id] = []
                        current_frame_data["人"][track_id].append({
                            "center_plane": [center_x, center_y],
                            "center_world": world_coord.tolist(),
                            "bbox": [x1, y1, x2, y2]
                        })
                    else:
                        if track_id not in current_frame_data["车"]:
                            current_frame_data["车"][track_id] = []
                        current_frame_data["车"][track_id].append({
                            "center_plane": [center_x, center_y],
                            "center_world": world_coord.tolist(),
                            "bbox": [x1, y1, x2, y2]
                        })

                # 显示总数
                cv2.putText(frame, f"Total Persons: {person_count}", (frame.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                result_queue.task_done()

                # 动态生成文件名
                json_filename = os.path.join('oboRuns', f'coord{run_counter}.json')
                run_counter += 1

                json_data = json.dumps({"data": current_frame_data}, ensure_ascii=False, indent=4)
                with open(json_filename, 'w', encoding='utf-8') as f:
                    f.write(json_data)

            except Exception as e:
                print(f"Error in display_frame: {e}")

    # 启动处理线程
    threading.Thread(target=process_frame, daemon=True).start()
    threading.Thread(target=display_frame, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        frame_queue.put(frame)

    frame_queue.put(None)  # 终止处理线程
    result_queue.put(None)  # 终止显示线程

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
