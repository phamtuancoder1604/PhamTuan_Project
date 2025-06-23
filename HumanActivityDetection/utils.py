# utils.py

import cv2
import numpy as np
import time

def blur_or_mosaic_face(
    image: np.ndarray, 
    face_boxes: list = None, 
    face_landmarks: list = None, 
    method: str = "blur", 
    blur_level: int = 25, 
    mosaic_size: int = 15
):
    img = image.copy()
    if face_boxes:
        for box in face_boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            face_roi = img[y1:y2, x1:x2]
            if face_roi.size == 0: continue
            if method == "blur":
                face_roi = cv2.GaussianBlur(face_roi, (blur_level|1, blur_level|1), 0)
            elif method == "mosaic":
                h, w = face_roi.shape[:2]
                temp = cv2.resize(face_roi, (mosaic_size, mosaic_size), interpolation=cv2.INTER_LINEAR)
                face_roi = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
            img[y1:y2, x1:x2] = face_roi
    if face_landmarks:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for landmarks in face_landmarks:
            points = np.array(landmarks, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
        if method == "blur":
            blurred = cv2.GaussianBlur(img, (blur_level|1, blur_level|1), 0)
            img = np.where(mask[:, :, None] == 255, blurred, img)
        elif method == "mosaic":
            mosaic = img.copy()
            mosaic = cv2.resize(mosaic, (img.shape[1]//mosaic_size, img.shape[0]//mosaic_size), interpolation=cv2.INTER_LINEAR)
            mosaic = cv2.resize(mosaic, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            img = np.where(mask[:, :, None] == 255, mosaic, img)
    return img

def get_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)

def compute_iou(b1, b2):
    xA = max(b1[0], b2[0]); yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2]); yB = min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (area1 + area2 - inter + 1e-6)

def now_str():
    return time.strftime('%Y-%m-%d %H:%M:%S')
