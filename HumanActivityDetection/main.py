# main.py

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp
import csv
from utils import blur_or_mosaic_face, get_center, compute_iou, now_str

### ========== PHẦN 1: Tracking + Face Privacy ==========

def run_face_privacy(video_path="fight_1.mp4"):
    input_width, input_height = 1600, 900 
    yolo_model = YOLO("yolov8x.pt")
    tracker = DeepSort(max_age=150, n_init=2, nms_max_overlap=0.7)
    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.40)
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (input_width, input_height))
        frame_flipped = cv2.flip(frame, 1)
        results = yolo_model(frame, classes=[0], conf=0.18)
        results_flip = yolo_model(frame_flipped, classes=[0], conf=0.18)
        detections = []
        person_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (x2 - x1) < 30 or (y2 - y1) < 30:
                    continue
                conf = float(box.conf[0])
                detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person', None))
                person_boxes.append([x1, y1, x2, y2])
        for result in results_flip:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1_new = input_width - x2
                x2_new = input_width - x1
                if (x2_new - x1_new) < 30 or (y2 - y1) < 30:
                    continue
                conf = float(box.conf[0])
                detections.append(([x1_new, y1, x2_new-x1_new, y2-y1], conf, 'person', None))
                person_boxes.append([x1_new, y1, x2_new, y2])
        tracks = tracker.update_tracks(detections, frame=frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_boxes = []
        results_face = face_detection.process(rgb)
        if results_face.detections:
            for detection in results_face.detections:
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x1 = int(bbox.xmin * iw)
                y1 = int(bbox.ymin * ih)
                w = int(bbox.width * iw)
                h = int(bbox.height * ih)
                x2 = x1 + w
                y2 = y1 + h
                for pb in person_boxes:
                    if x1 >= pb[0] and y1 >= pb[1] and x2 <= pb[2] and y2 <= pb[3]:
                        face_boxes.append([x1, y1, x2, y2])
                        break
        frame = blur_or_mosaic_face(frame, face_boxes=face_boxes, method="mosaic", mosaic_size=15)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.imshow("Tracking + Face Privacy (Advanced Optimized)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

### ========== PHẦN 2: Fight Detection ==========

def run_fight_detection(video_path="fight_2.mp4", log_csv="fight_events.csv"):
    WIDTH, HEIGHT = 960, 540
    V_THRESH = 5.0   # speed threshold to start fight
    IOU_THRESH = 0.1   # overlap threshold to start fight
    LOW_SPEED = 2.0   # to end fight
    LOW_IOU = 0.05  # to end fight
    yolo = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=60, n_init=2, nms_max_overlap=0.7)
    cap = cv2.VideoCapture(video_path)
    f = open(log_csv, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["start_time","end_time","attacker_id","victim_id","direction"])
    prev_centers = {}
    state = "Idle"
    attacker = None
    victim = None
    start_time = None
    direction = None
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        results = yolo(frame, classes=[0], conf=0.25)
        dets = []
        for r in results:
            for b in r.boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                if x2-x1<30 or y2-y1<30: continue
                dets.append(([x1,y1,x2-x1,y2-y1], float(b.conf[0]), "person", None))
        tracks = tracker.update_tracks(dets, frame=frame) if dets else []
        centers = {}
        motions = {}
        bboxes = {}
        for t in tracks:
            if not t.is_confirmed(): continue
            tid = t.track_id
            x1,y1,x2,y2 = map(int, t.to_ltrb())
            c = get_center((x1,y1,x2,y2))
            centers[tid] = c
            bboxes[tid] = (x1,y1,x2,y2)
            if tid in prev_centers:
                motions[tid] = c - prev_centers[tid]
            else:
                motions[tid] = np.zeros(2, dtype=np.float32)
        ids = list(centers.keys())
        # Event detection
        if len(ids) == 2:
            a, b = ids
            rel_speed = np.linalg.norm(motions[a] - motions[b])
            iou = compute_iou(bboxes[a], bboxes[b])
            # Start fight
            if state == "Idle":
                if rel_speed > V_THRESH and iou > IOU_THRESH:
                    v_ab = centers[b] - centers[a]
                    projA = np.dot(motions[a],  v_ab)
                    projB = np.dot(motions[b], -v_ab)
                    if projA > projB:
                        attacker, victim = a, b
                        direction = (v_ab/np.linalg.norm(v_ab)).tolist()
                    else:
                        attacker, victim = b, a
                        direction = (-v_ab/np.linalg.norm(v_ab)).tolist()
                    start_time = now_str()
                    state = "Fighting"
            # End fight
            elif state == "Fighting":
                if rel_speed < LOW_SPEED and iou < LOW_IOU:
                    end_time = now_str()
                    w.writerow([start_time, end_time, attacker, victim, direction])
                    state = "Idle"
                    attacker = victim = direction = start_time = None
        for t in tracks:
            if not t.is_confirmed(): continue
            tid = t.track_id
            x1,y1,x2,y2 = map(int, t.to_ltrb())
            if state=="Fighting" and tid==attacker:
                role, color = "victim",   (0,255,0)
            elif state=="Fighting" and tid==victim:
                role, color = "attacker", (0,0,255)
            else:
                role, color = "stranger",(255,255,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"ID {tid} | {role}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        prev_centers = centers.copy()
        cv2.imshow("Fight Detector", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    f.close()
    cv2.destroyAllWindows()

### ========== MAIN ENTRY POINT ==========

if __name__ == "__main__":

    run_face_privacy()   # Hoặc run_fight_detection()
