import itertools
import time
from collections import defaultdict
import cv2
import numpy as np
import torch
from ultralytics import YOLO


class MultiTaskVisionSystem:
    def __init__(self):
        # Initialize SMALL models for different tasks
        self.models = {
            'detection': YOLO('yolo11s.pt'),  # Object detection
            'segmentation': YOLO('yolo11s-seg.pt'),  # Segmentation
            'pose': YOLO('yolo11s-pose.pt'),  # Pose estimation
            'tracking': YOLO('yolo11s.pt'),  # Tracking
        }

        # Configuration for different tasks
        self.config = {
            'conf_threshold': 0.3,
            'iou_threshold': 0.3,
            'classes_to_count': [0, 2, 5, 7]  # Persons, cars, buses, trucks
        }

        # Initialize tracker
        self.track_history = defaultdict(lambda: [])
        self.fps = 0
        self.prev_time = 0

        self.trainer = ElectronicTrainer()

    def setup_camera(self, camera_id=0):
        """Setup camera"""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            # Try other camera indices
            for i in range(1, 5):
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    print(f"Camera found at index {i}")
                    break

            if not self.cap.isOpened():
                print("Camera not found, using test video...")
                self.cap = cv2.VideoCapture('test_video.mp4')
                if not self.cap.isOpened():
                    raise ValueError("Failed to open camera or video")

        # Set parameters for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def calculate_fps(self):
        """Calculate FPS"""
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        return self.fps

    def draw_detection_results(self, frame, results):
        """Visualize detection results"""
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = int(box.cls[0])

                    # Draw bounding box
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Label with class and confidence
                    label = f"{result.names[cls]}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def draw_segmentation_results(self, frame, results):
        """Visualize segmentation results"""
        for result in results:
            if result.masks is not None:
                # Use built-in visualization for speed
                annotated_frame = result.plot()
                return annotated_frame

        return frame

    def draw_pose_results(self, frame, results):
        """Visualize pose estimation results"""
        for result in results:
            if result.keypoints is not None:
                # Use built-in visualization for speed
                annotated_frame = result.plot()
                return annotated_frame

        return frame

    def draw_tracking_results(self, frame, results):
        """Visualize tracking results"""
        for result in results:
            if result.boxes is not None and hasattr(result.boxes, 'id') and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)

                for box, track_id, conf, cls in zip(boxes, track_ids, confs, classes):
                    x1, y1, x2, y2 = map(int, box)

                    # Draw bounding box with color by ID
                    color = self.get_color(track_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Label with track ID
                    label = f"ID:{track_id} {result.names[cls]}:{conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Save track history
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    self.track_history[track_id].append(center)

                    # Draw track history (last 20 points)
                    if len(self.track_history[track_id]) > 20:
                        self.track_history[track_id].pop(0)

                    # Draw track line
                    if len(self.track_history[track_id]) > 1:
                        points = np.array(self.track_history[track_id], np.int32)
                        cv2.polylines(frame, [points], False, color, 2)

        return frame

    def get_color(self, track_id):
        """Generate color based on track ID"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 255, 0), (255, 128, 0), (0, 128, 255)
        ]
        return colors[track_id % len(colors)]

    def run_detection(self, frame):
        """Run object detection"""
        results = self.models['detection'](
            frame,
            conf=self.config['conf_threshold'],
            iou=self.config['iou_threshold'],
            verbose=False,
            imgsz=320
        )
        # 1. Отрисовка стандартных боксов
        processed_frame = self.draw_detection_results(frame, results)

        # 2. Проверка дистанции между людьми (центры)
        self.trainer.get_people_distance(processed_frame, results)

        # 3. Контроль выхода за территорию поля
        self.trainer.check_boundary_violation(processed_frame, results)

        return processed_frame

    def run_segmentation(self, frame):
        """Run segmentation"""
        results = self.models['segmentation'](
            frame,
            conf=self.config['conf_threshold'],
            verbose=False,
            imgsz=320
        )
        return self.draw_segmentation_results(frame, results)

    def run_pose_estimation(self, frame):
        results = self.models['pose'](frame, conf=0.3, verbose=False, imgsz=320)
        processed_frame = self.draw_pose_results(frame, results)

        # Новый продвинутый анализ позы
        self.trainer.check_posture_hip_only(processed_frame, results)
        self.trainer.get_leg_distance(processed_frame, results)

        return processed_frame

    def run_tracking(self, frame):
        """Run object tracking"""
        results = self.models['tracking'].track(
            frame,
            conf=self.config['conf_threshold'],
            iou=self.config['iou_threshold'],
            persist=True,
            verbose=False,
            imgsz=320,
            tracker="bytetrack.yaml"  # Use ByteTrack for better tracking
        )
        return self.draw_tracking_results(frame, results)

    def run(self):
        """Main processing loop"""
        print("Starting Computer Vision System with SMALL models...")
        print("Control keys:")
        print("1 - Object Detection")
        print("2 - Segmentation")
        print("4 - Object Tracking")
        print("5 - Pose Estimation")
        print("q - Quit")

        current_mode = 'detection'
        self.prev_time = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to get frame")

                if self.cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                    print("End of video file. Restarting.")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            # Calculate FPS
            fps = self.calculate_fps()

            # Process frame based on current mode
            try:
                if current_mode == 'detection':
                    processed_frame = self.run_detection(frame.copy())
                    mode_text = "Mode: Object Detection"

                elif current_mode == 'segmentation':
                    processed_frame = self.run_segmentation(frame.copy())
                    mode_text = "Mode: Segmentation"


                elif current_mode == 'tracking':
                    processed_frame = self.run_tracking(frame.copy())
                    mode_text = "Mode: Object Tracking"

                elif current_mode == 'pose':
                    processed_frame = self.run_pose_estimation(frame.copy())
                    mode_text = "Mode: Pose Estimation"

                else:
                    # На случай, если current_mode будет иметь неожиданное значение
                    processed_frame = frame.copy()
                    mode_text = "Mode: Unknown"

                # Display current mode and FPS
                cv2.putText(processed_frame, mode_text, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame, "Model: yolo11s", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                cv2.imshow('Multi-Task Vision System (SMALL)', processed_frame)

            except Exception as e:
                print(f"Processing error: {e}")
                cv2.imshow('Multi-Task Vision System (SMALL)', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                current_mode = 'detection'
                print("Switched to: Object Detection")
            elif key == ord('2'):
                current_mode = 'segmentation'
                print("Switched to: Segmentation")

            elif key == ord('4'):
                current_mode = 'tracking'
                print("Switched to: Object Tracking")
            elif key == ord('5'):
                current_mode = 'pose'
                print("Switched to: Pose Estimation")

        self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("System stopped")


def main():
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Using SMALL models for better performance")

    try:
        # Create and run system
        vision_system = MultiTaskVisionSystem()
        vision_system.setup_camera(0)
        vision_system.run()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()


class ElectronicTrainer:

    def __init__(self):
        self.min_dist_people = 100
        self.min_dist_leg = 10
        self.field_rect = None  # Изначально зоны нет

    def detect_red_zone_area(self, frame):
        """
        Ищет самую большую красную зону на кадре.
        Возвращает координаты (x, y, w, h) или None, если зона не найдена.
        """
        # Переводим в формат HSV для лучшего определения цвета
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Диапазон красного цвета №1 (от 0 до 10)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])

        # Диапазон красного цвета №2 (от 170 до 180) - красный в HSV "заворачивается"
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # Создаем маски и объединяем их
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # Убираем шумы (мелкие точки)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        # Ищем контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Берем самый большой красный объект
            largest_contour = max(contours, key=cv2.contourArea)

            # Если объект слишком маленький (шум), игнорируем его
            if cv2.contourArea(largest_contour) > 1000:
                x, y, w, h = cv2.boundingRect(largest_contour)
                return (x, y, x + w, y + h)  # Возвращаем x1, y1, x2, y2

        return None

    def check_boundary_violation(self, frame, results):

        # Сначала ищем зону на текущем кадре
        detected_zone = self.detect_red_zone_area(frame)

        # Если красная зона НЕ найдена - выходим, ничего не рисуем
        if detected_zone is None:
            return

        # Если зона найдена, распаковываем координаты
        fx1, fy1, fx2, fy2 = detected_zone

        # Рисуем найденную зону (зеленая рамка поверх красной разметки для подтверждения)
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
        cv2.putText(frame, "DETECTED ZONE", (fx1, fy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if not results:
            return

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls[0]) == 0:  # person
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Проверяем НОГИ (низ рамки)
                        foot_x = (x1 + x2) // 2
                        foot_y = y2

                        # Логика: точка ног ВНУТРИ найденной зоны?
                        is_inside = (fx1 <= foot_x <= fx2 and fy1 <= foot_y <= fy2)

                        if not is_inside:
                            # ЧЕЛОВЕК ВЫШЕЛ ЗА ПРЕДЕЛЫ
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                            cv2.circle(frame, (foot_x, foot_y), 8, (0, 0, 255), -1)
                            cv2.putText(frame, "OUT OF ZONE!", (x1, y1 - 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    def get_leg_distance(self, frame, results):
        """
        Рассчитывает и отображает расстояние между лодыжками (из режима Pose).
        """
        try:
            if not results or results[0].keypoints is None or results[0].keypoints.xy.shape[0] == 0:
                return

            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            if len(keypoints) < 17:
                return

            left_ankle = keypoints[15]
            right_ankle = keypoints[16]

            if np.all(left_ankle == 0) or np.all(right_ankle == 0):
                return

            distance = np.linalg.norm(left_ankle - right_ankle)
            x1, y1 = map(int, left_ankle)
            x2, y2 = map(int, right_ankle)

            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Distance: {int(distance)} px", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error in get_leg_distance: {e}")

    def calculate_angle(self, a, b, c):
        """Вычисляет угол между тремя точками в градусах."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0: angle = 360 - angle
        return angle

    def check_posture_hip_only(self, frame, results):
        if not results or results[0].keypoints is None:
            return

        for kpts in results[0].keypoints.xy:
            kpts = kpts.cpu().numpy()
            if len(kpts) < 15: continue

            # Точки
            l_shoulder, l_hip, l_knee = kpts[5], kpts[11], kpts[13]
            r_shoulder, r_hip, r_knee = kpts[6], kpts[12], kpts[14]

            # Считаем вертикальные расстояния (разница по Y)
            # Торс: Плечо - Бедро
            torso_h = abs(l_hip[1] - l_shoulder[1])
            # Бедро: Бедро - Колено
            thigh_h = abs(l_knee[1] - l_hip[1])

            # Расчет угла (как и был)
            angle = self.calculate_angle(l_shoulder, l_hip, l_knee)

            # Если thigh_h становится очень маленьким относительно torso_h,
            # значит ноги "ушли" вглубь кадра (человек сел).
            # В норме у стоячего человека это отношение около 0.8 - 1.2.
            # У сидячего фронтально - падает ниже 0.4.
            ratio = thigh_h / torso_h if torso_h != 0 else 1

            if angle < 135 or ratio < 0.45:
                status = "SITTING"
                color = (0, 0, 255)
            else:
                status = "STANDING"
                color = (0, 255, 0)

            # Вывод отладочной информации (поможет настроить порог)
            hip_p = l_hip.astype(int)
            cv2.putText(frame, f"Ang: {int(angle)} Rat: {ratio:.2f}", (hip_p[0]+10, hip_p[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(frame, status, (hip_p[0]+10, hip_p[1]-25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def get_people_distance(self, frame, results):
        """
        Рассчитывает и отображает расстояние между людьми
        """
        people_centers = []
        person_class_id = 0

        # Находим всех людей и их центральные точки
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls[0]) == person_class_id:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        full_center = (center_x, center_y)

                        people_centers.append(full_center)

        # Если есть 2 или более человека, рассчитываем расстояния
        if len(people_centers) >= 2:

            for p1_center, p2_center in itertools.combinations(people_centers, 2):

                # Расчет Евклидова расстояния
                distance = np.linalg.norm(np.array(p1_center) - np.array(p2_center))

                # Визуализация
                color = (0, 255, 0)
                if distance < self.min_dist_people:
                    color = (0, 0, 255)

                cv2.line(frame, p1_center, p2_center, color, 2)

                # Надпись с расстоянием
                mid_point = ((p1_center[0] + p2_center[0]) // 2,
                             (p1_center[1] + p2_center[1]) // 2)

                cv2.putText(frame, f"{int(distance)} px", mid_point,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)






if __name__ == "__main__":
    main()