"""Capture desk hand movements and possible interactions with YOLO objects."""
import argparse
import csv
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'models' / 'hand_landmarker.task'
SESSIONS = ROOT / 'data' / 'sessions'
MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
             'hand_landmarker/float16/1/hand_landmarker.task')
ZONE_NAMES = ('Far', 'Middle', 'Near')  # User sits at BOTTOM of image.
DWELL_SECONDS = 0.4
TRACKING_LOSS_SECONDS = 0.75
INTERACTION_SECONDS = 0.4
INTERACTION_GAP_SECONDS = 0.75
PALM_SIZE_LIMIT = 1.5  # Reject unusually large hands after calibration.


def ensure_model():
    if not MODEL.exists():
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        print('Downloading the MediaPipe hand model (first run only)...')
        urllib.request.urlretrieve(MODEL_URL, MODEL)


def choose_desk(cap):
    print('Position the camera. Press SPACE to select the desk, or ESC to exit.')
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('Cannot read camera frames.')
        cv2.putText(frame, 'SPACE: choose desk | ESC: quit', (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('ReachMap - Setup', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyWindow('ReachMap - Setup')
            return None
        if key == 32:
            break

    x, y, w, h = map(int, cv2.selectROI('ReachMap - Setup', frame,
                                        showCrosshair=True, fromCenter=False))
    cv2.destroyWindow('ReachMap - Setup')
    if w < 60 or h < 60:
        raise ValueError('Desk selection too small or cancelled.')
    return x, y, w, h


def palm_position(result, frame_width, frame_height):
    if not result.hand_landmarks:
        return None
    hand = result.hand_landmarks[0]
    indices = (0, 5, 9, 13, 17)
    px = np.mean([hand[i].x for i in indices]) * frame_width
    py = np.mean([hand[i].y for i in indices]) * frame_height
    return float(px), float(py)


def palm_size(result, frame_width, frame_height):
    """Estimate palm size in pixels using stable hand landmarks."""

    if not result.hand_landmarks:
        return None
    hand = result.hand_landmarks[0]

    # Wrist to middle knuckle (0-9) and across knuckles (5-17).
    length = np.hypot((hand[0].x - hand[9].x) * frame_width,
                      (hand[0].y - hand[9].y) * frame_height)
    width = np.hypot((hand[5].x - hand[17].x) * frame_width,
                     (hand[5].y - hand[17].y) * frame_height)
    return float((length + width) / 2)


def desk_position(point, desk):
    x, y, w, h = desk
    px, py = point
    if not (x <= px <= x + w and y <= py <= y + h):
        return None
    return (px - x) / w, (py - y) / h


def get_zone(relative_y):
    return ZONE_NAMES[min(int(relative_y * 3), 2)]


def draw_desk(frame, desk):
    x, y, w, h = desk
    colors = [(80, 110, 255), (90, 210, 235), (100, 210, 110)]
    for index, name in enumerate(ZONE_NAMES):
        top = y + index * h // 3
        bottom = y + (index + 1) * h // 3
        cv2.rectangle(frame, (x, top), (x + w, bottom), colors[index], 2)
        cv2.putText(frame, name, (x + 8, top + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, colors[index], 2)


def detect_objects(frame, desk, fingertip, model):
    """Return detected object classes and classes overlapping the fingertip."""
    result = model.predict(frame, conf=0.4, verbose=False)[0]
    detected = set()
    overlaps = set()

    for box in result.boxes:
        name = result.names[int(box.cls[0])]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        if name == 'person' or desk_position(center, desk) is None:
            continue

        detected.add(name)
        overlapping = (fingertip is not None and
                       x1 <= fingertip[0] <= x2 and y1 <= fingertip[1] <= y2)
        if overlapping:
            overlaps.add(name)

        color = (0, 200, 0) if overlapping else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return detected, overlaps


def run(source):
    ensure_model()
    camera = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f'Camera not accessible: {source}')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1)

    try:
        desk = choose_desk(cap)
        if desk is None:
            return

        print('Loading YOLOv8n object detector...')
        yolo = YOLO('yolov8n.pt')  # Downloads weights on first use.
        SESSIONS.mkdir(parents=True, exist_ok=True)
        output = SESSIONS / f'session_{datetime.now():%Y%m%d_%H%M%S}.csv'
        start = time.monotonic()
        pending_zone = None
        pending_start = start
        stable_zone = None
        last_seen = start
        visits = 0
        interaction_count = 0
        last_timestamp_ms = -1
        overlap_start = {}
        overlap_last = {}
        active_interactions = set()
        reference_size = None  # Set once when user presses C with hand on desk.

        with open(output, 'w', newline='', encoding='utf-8') as handle, \
                mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            writer = csv.writer(handle) 
            writer.writerow(['seconds', 'x', 'y', 'zone', 'visit',
                            'objects', 'interaction_events', 'overlap_objects'])


            while True:
                ok, frame = cap.read()
                if not ok:
                    print('Camera stream ended.')
                    break

                now = time.monotonic()
                seconds = now - start
                timestamp_ms = max(last_timestamp_ms + 1, int(seconds * 1000))
                last_timestamp_ms = timestamp_ms
                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(image, timestamp_ms)

                width, height = frame.shape[1], frame.shape[0]
                point = palm_position(result, width, height)
                size = palm_size(result, width, height)
                too_close = (reference_size is not None and size is not None and
                             size > reference_size * PALM_SIZE_LIMIT)
                relative = desk_position(point, desk) if point else None
                if reference_size is None or too_close:
                    relative = None  # Invalid observations cannot count as visits.

                fingertip = None
                if result.hand_landmarks and reference_size is not None and not too_close:
                    finger = result.hand_landmarks[0][8]  # Index fingertip.
                    candidate = (finger.x * width, finger.y * height)
                    if desk_position(candidate, desk) is not None:
                        fingertip = candidate

                detected, overlaps = detect_objects(frame, desk, fingertip, yolo)

                # Count one event when a class has overlapped long enough.
                events = []
                for name in overlaps:
                    if name not in overlap_start:
                        overlap_start[name] = now
                    overlap_last[name] = now
                    if (name not in active_interactions and
                            now - overlap_start[name] >= INTERACTION_SECONDS):
                        events.append(name)
                        active_interactions.add(name)

                interaction_count += len(events)

                # Short missed detections do not create duplicate events.
                for name in list(overlap_start):
                    if name not in overlaps and now - overlap_last[name] > INTERACTION_GAP_SECONDS:
                        del overlap_start[name]
                        del overlap_last[name]
                        active_interactions.discard(name)

                draw_desk(frame, desk)
                x = y = zone = ''
                visit = 0
                if relative is not None:
                    last_seen = now
                    x, y = relative
                    zone = get_zone(y)
                    if zone != pending_zone:
                        pending_zone = zone
                        pending_start = now
                    if zone != stable_zone and now - pending_start >= DWELL_SECONDS:
                        stable_zone = zone
                        visit = 1
                        visits += 1
                    cx, cy = map(int, point)
                    cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
                    label = f'Hand: {zone}'
                else:
                    pending_zone = None
                    if now - last_seen > TRACKING_LOSS_SECONDS:
                        stable_zone = None
                    if reference_size is None:
                        label = 'Place one hand on desk, press C to calibrate'
                    elif too_close:
                        label = 'Hand too close to camera - ignored'
                    else:
                        label = 'Hand outside desk / not detected'

                # Save EVERY frame, including object detections without a hand.
                writer.writerow([
                    round(seconds, 3),
                    round(x, 4) if x != '' else '',
                    round(y, 4) if y != '' else '',
                    zone,
                    visit,
                    '|'.join(sorted(detected)),
                    '|'.join(sorted(events)),
                    '|'.join(sorted(overlaps))
                ])


                cv2.putText(frame, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)
                cv2.putText(frame, f'Visits: {visits} | Possible interactions: '
                            f'{interaction_count} | Q: finish', (15, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.imshow('ReachMap - Recording', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c') and reference_size is None:
                    if (size is not None and size > 0 and point is not None and
                            desk_position(point, desk) is not None):
                        reference_size = size
                        print(f'Palm calibrated: {reference_size:.1f} pixels')
                    else:
                        print('Calibration failed: hold one hand inside the desk.')
                if key in (ord('q'), 27):
                    break

        print(f'Session saved to: {output}')
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Record one ReachMap session')
    parser.add_argument('--camera', default='0', help='Camera index (0/1) or stream URL')
    args = parser.parse_args()
    run(args.camera)
