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
        # ... твой существующий код захвата ...
        self.cap = cv2.VideoCapture('test1.mp4')

        # Получаем параметры исходного видео/камеры
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 20  # На случай, если камера не отдает FPS

        # Настраиваем "записывающее устройство"
        # 'mp4v' — это кодек, 'output_video.mp4' — имя файла
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter('result_training.mp4', fourcc, fps, (width, height))
        print(f"Recording started: result_training.mp4 ({width}x{height} @ {fps}fps)")

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
        """Режим 5: Дуэль (только 2 стоящих человека)"""
        results = self.models['pose'](
            frame,
            conf=0.25,
            verbose=False,
            imgsz=320
        )

        # Мы НЕ используем draw_pose_results для всех,
        # чтобы не рисовать скелеты на тех, кто сидит.

        # Запускаем режим дуэли
        active_players = self.trainer.process_duel_mode(frame, results)

        return frame

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
                self.video_writer.write(processed_frame)  # Записываем кадр в файл
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
        self.cap.release()
        self.video_writer.release() # ОЧЕНЬ ВАЖНО: сохраняем файл на диск
        cv2.destroyAllWindows()
        print("Video saved as result_training.mp4")


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

    def is_standing(self, kpts):
        """Проверяет, стоит ли человек (без отрисовки)"""
        if len(kpts) < 15: return False

        l_shoulder, l_hip, l_knee = kpts[5], kpts[11], kpts[13]
        if np.all(l_hip == 0) or np.all(l_shoulder == 0) or np.all(l_knee == 0):
            return False

        # Используем те же пороги, что и раньше
        angle = self.calculate_angle(l_shoulder, l_hip, l_knee)
        torso_v = abs(l_hip[1] - l_shoulder[1])
        thigh_v = abs(l_knee[1] - l_hip[1])
        ratio = thigh_v / torso_v if torso_v != 0 else 1

        # Возвращаем True, если человек стоит
        return not (angle < 130 or ratio < 0.45)

    def process_duel_mode(self, frame, results):
        """Выбирает двух стоящих людей и анализирует только их"""
        if not results or results[0].keypoints is None:
            return []

        active_players = []

        # 1. Собираем всех, кто стоит
        for i, kpts_obj in enumerate(results[0].keypoints):
            kpts = kpts_obj.xy[0].cpu().numpy()
            conf = kpts_obj.conf[0].cpu().numpy()

            if self.is_standing(kpts):
                # Считаем площадь бокса (чтобы найти тех, кто ближе)
                box = results[0].boxes[i].xyxy[0].cpu().numpy()
                area = (box[2] - box[0]) * (box[3] - box[1])
                active_players.append({
                    'kpts': kpts,
                    'area': area,
                    'box': box,
                    'conf': conf
                })

        # 2. Сортируем по площади (самые крупные/близкие — первые) и берем только двоих
        active_players = sorted(active_players, key=lambda x: x['area'], reverse=True)[:2]

        # 3. Отрисовываем аналитику только для этих двоих
        centers = []
        for player in active_players:
            kpts = player['kpts']
            box = player['box']

            # Отрисовка статуса Standing
            cv2.putText(frame, "ACTIVE", (int(box[0]), int(box[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Точка центра для дистанции
            centers.append(((int(box[0] + box[2]) // 2), (int(box[1] + box[3]) // 2)))

            # Отрисовка шага (между лодыжками)
            l_ankle, r_ankle = kpts[15], kpts[16]
            if not (np.all(l_ankle == 0) or np.all(r_ankle == 0)):
                dist = np.linalg.norm(l_ankle - r_ankle)
                cv2.line(frame, tuple(l_ankle.astype(int)), tuple(r_ankle.astype(int)), (255, 255, 0), 2)
                cv2.putText(frame, f"Step: {int(dist)}", (int(l_ankle[0]), int(l_ankle[1] + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # 4. Дистанция между двумя активными игроками
        if len(centers) == 2:
            p1, p2 = centers
            dist_px = np.linalg.norm(np.array(p1) - np.array(p2))
            color = (0, 0, 255) if dist_px < self.min_dist_people else (0, 255, 0)
            cv2.line(frame, p1, p2, color, 2)
            cv2.putText(frame, f"Dist: {int(dist_px)}px", ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return active_players

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
        cv2.putText(frame, "ZONE", (fx1, fy1 - 10),
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
                            cv2.putText(frame, "Out of Zone!", (x1, y1 - 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    def get_leg_distance(self, frame, results):
        """
        Улучшенный расчет расстояния между стопами для каждого человека.
        """
        if not results or results[0].keypoints is None:
            return

        # Перебираем всех людей, которых нашла нейросеть
        for kpts in results[0].keypoints.xy:
            kpts = kpts.cpu().numpy()

            # Если точек меньше 17, значит скелет неполный — пропускаем
            if len(kpts) < 17:
                continue

            # Точки 15 (левая лодыжка) и 16 (правая лодыжка)
            l_ankle = kpts[15]
            r_ankle = kpts[16]

            # Проверка: если координаты (0,0), значит точка не видна
            if np.all(l_ankle == 0) or np.all(r_ankle == 0):
                continue

            # 1. Расчет расстояния (Евклидова метрика)
            distance = np.linalg.norm(l_ankle - r_ankle)

            # 2. Визуализация
            x1, y1 = map(int, l_ankle)
            x2, y2 = map(int, r_ankle)

            # Рисуем кружки на лодыжках
            cv2.circle(frame, (x1, y1), 5, (255, 255, 0), -1)
            cv2.circle(frame, (x2, y2), 5, (255, 255, 0), -1)

            # Рисуем линию между ними
            color = (255, 255, 0)  # Голубой
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)

            # Вычисляем центр линии для текста
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2

            # Добавляем подложку под текст, чтобы его было видно на любом фоне
            label = f"Step: {int(distance)} px"
            cv2.putText(frame, label, (mid_x - 40, mid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)  # Тень
            cv2.putText(frame, label, (mid_x - 40, mid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    def calculate_angle(self, a, b, c):
        """Вычисляет угол между тремя точками в градусах."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0: angle = 360 - angle
        return angle

    def check_posture_smart(self, frame, results):
        """
        Универсальная проверка: Угол (бок) + Коэффициент сокращения (перед).
        """
        if not results or results[0].keypoints is None:
            return

        for kpts in results[0].keypoints.xy:
            kpts = kpts.cpu().numpy()
            if len(kpts) < 15: continue

            # Точки для анализа (берем левую сторону как эталон, либо обе)
            l_shoulder, l_hip, l_knee = kpts[5], kpts[11], kpts[13]
            r_shoulder, r_hip, r_knee = kpts[6], kpts[12], kpts[14]

            # Проверка видимости
            if np.all(l_hip == 0) or np.all(l_shoulder == 0) or np.all(l_knee == 0):
                continue

            # 1. РАСЧЕТ УГЛА (для бокового вида)
            angle = self.calculate_angle(l_shoulder, l_hip, l_knee)

            # 2. РАСЧЕТ КОЭФФИЦИЕНТА (для фронтального вида)
            # Измеряем вертикальную проекцию торса и бедра
            torso_v = abs(l_hip[1] - l_shoulder[1])
            thigh_v = abs(l_knee[1] - l_hip[1])

            # Отношение высоты бедра к высоте торса
            ratio = thigh_v / torso_v if torso_v != 0 else 1

            # --- ГИБРИДНАЯ ЛОГИКА ---
            # Если угол маленький (бок) ИЛИ если нога сильно укоротилась в 2D (перед)
            if angle < 130 or ratio < 0.45:
                status = "Sitting"
                color = (0, 0, 255)  # Красный
            else:
                status = "Standing"
                color = (0, 255, 0)  # Зеленый

            # Отрисовка данных для тестов
            pos_x, pos_y = int(l_hip[0]), int(l_hip[1])
            cv2.putText(frame, f"Ang: {int(angle)} Rat: {ratio:.2f}", (pos_x + 15, pos_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, status, (pos_x + 15, pos_y - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Рисуем "костную" структуру для визуализации
            cv2.line(frame, tuple(l_shoulder.astype(int)), tuple(l_hip.astype(int)), color, 2)
            cv2.line(frame, tuple(l_hip.astype(int)), tuple(l_knee.astype(int)), color, 2)

    def get_people_distance(self, frame, results):
        """
        Универсальный расчет расстояния между людьми для Detection и Pose режимов.
        """
        people_centers = []

        if not results:
            return

        for result in results:
            # Проверяем наличие боксов (они есть и в Detection, и в Pose)
            if result.boxes is not None:
                for box in result.boxes:
                    # Убеждаемся, что это человек (class 0)
                    cls = int(box.cls[0])
                    if cls == 0:
                        # Берем координаты бокса
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Точный центр человека
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        people_centers.append((center_x, center_y))

        # Если нашли 2+ человека, рисуем линии
        if len(people_centers) >= 2:
            import itertools
            for p1, p2 in itertools.combinations(people_centers, 2):
                dist = np.linalg.norm(np.array(p1) - np.array(p2))

                # Цвет линии: красный если слишком близко, иначе зеленый
                color = (0, 0, 255) if dist < self.min_dist_people else (0, 255, 0)

                cv2.line(frame, p1, p2, color, 2)
                mid_point = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

                # Текст с расстоянием
                cv2.putText(frame, f"{int(dist)} px", mid_point,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)






if __name__ == "__main__":
    main()