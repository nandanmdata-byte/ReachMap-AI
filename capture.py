"""ReachMap AI: record desk hand positions and possible object interactions."""
import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
SESSIONS = ROOT / 'data' / 'sessions'
ZONES = ('Far', 'Middle', 'Near')  # User sits at the bottom of the image.
DWELL_SECONDS = 0.4
LOSS_SECONDS = 0.75
INTERACTION_SECONDS = 0.4
GAP_SECONDS = 0.75
PALM_SIZE_LIMIT = 1.5


def choose_desk(cap):
    """Let the user select the desk rectangle in the camera image."""
    print('Position camera. SPACE: select desk | ESC: quit')
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('Cannot read camera frames.')

        cv2.putText(frame, 'SPACE: select desk | ESC: quit', (15, 30),
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


def palm_position(hand, width, height):
    """Average the wrist and four knuckles to find the palm center."""
    indices = (0, 5, 9, 13, 17)
    x = sum(hand[i].x for i in indices) / len(indices) * width
    y = sum(hand[i].y for i in indices) / len(indices) * height
    return x, y


def palm_size(hand, width, height):
    """Average wrist-to-knuckle length and across-knuckles width in pixels."""
    length = math.hypot((hand[0].x - hand[9].x) * width,
                        (hand[0].y - hand[9].y) * height)
    across = math.hypot((hand[5].x - hand[17].x) * width,
                        (hand[5].y - hand[17].y) * height)
    return (length + across) / 2


def desk_position(point, desk):
    """Convert a pixel position into desk-relative coordinates (0 to 1)."""
    x, y, w, h = desk
    px, py = point
    if not (x <= px <= x + w and y <= py <= y + h):
        return None
    return (px - x) / w, (py - y) / h


def get_zone(relative_y):
    return ZONES[min(int(relative_y * 3), 2)]


def draw_desk(frame, desk):
    x, y, w, h = desk
    colors = [(80, 110, 255), (90, 210, 235), (100, 210, 110)]
    for i, name in enumerate(ZONES):
        top = y + i * h // 3
        bottom = y + (i + 1) * h // 3
        cv2.rectangle(frame, (x, top), (x + w, bottom), colors[i], 2)
        cv2.putText(frame, name, (x + 8, top + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, colors[i], 2)


def detect_objects(frame, desk, fingertip, yolo):
    """Detect YOLO object classes and any box overlapping the index fingertip."""
    result = yolo.predict(frame, conf=0.4, verbose=False)[0]
    detected = set()
    overlaps = set()

    for box in result.boxes:
        name = result.names[int(box.cls[0])]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        center = ((x1 + x2) / 2, (y1 + y2) / 2)

        if name == 'person' or desk_position(center, desk) is None:
            continue

        detected.add(name)
        touching_box = (fingertip is not None and
                        x1 <= fingertip[0] <= x2 and y1 <= fingertip[1] <= y2)
        if touching_box:
            overlaps.add(name)

        color = (0, 200, 0) if touching_box else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return detected, overlaps


def run(source):
    camera = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f'Camera not accessible: {source}')

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        desk = choose_desk(cap)
        if desk is None:
            return

        yolo = YOLO('yolov8n.pt')
        SESSIONS.mkdir(parents=True, exist_ok=True)
        output = SESSIONS / f'session_{datetime.now():%Y%m%d_%H%M%S}.csv'
        start = time.monotonic()

        # Remember zone entries and short tracking losses.
        pending_zone = None
        pending_start = start
        stable_zone = None
        last_seen = start
        visits = 0

        # Remember object overlaps to avoid duplicate interaction events.
        overlap_start = {}
        overlap_last = {}
        active_interactions = set()
        interaction_count = 0
        reference_size = None  # User presses C with their hand on the desk.

        with mp.solutions.hands.Hands(max_num_hands=1) as hands:
            with open(output, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['seconds', 'x', 'y', 'zone', 'visit',
                                 'objects', 'interaction_events', 'overlap_objects'])

                while True:
                    ok, frame = cap.read()
                    if not ok:
                        print('Camera stream ended.')
                        break

                    now = time.monotonic()
                    seconds = now - start
                    height, width = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = hands.process(rgb)

                    hand = None
                    if result.multi_hand_landmarks:
                        hand = result.multi_hand_landmarks[0].landmark

                    point = palm_position(hand, width, height) if hand else None
                    size = palm_size(hand, width, height) if hand else None
                    too_close = (
                        reference_size is not None and size is not None and
                                 size > reference_size * PALM_SIZE_LIMIT
                        )

                    # Uncalibrated/raised hands cannot contribute hand positions.
                    relative = None
                    if point is not None and reference_size is not None and not too_close:
                        relative = desk_position(point, desk)

                    fingertip = None
                    if hand and reference_size is not None and not too_close:
                        finger = hand[8]  # Index fingertip.
                        candidate = (finger.x * width, finger.y * height)
                        if desk_position(candidate, desk) is not None:
                            fingertip = candidate

                    # YOLO still detects desk objects when no valid hand is present.
                    detected, overlaps = detect_objects(frame, desk, fingertip, yolo)

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

                    # A short gap does not turn one overlap into multiple events.
                    for name in list(overlap_start):
                        if name not in overlaps and now - overlap_last[name] > GAP_SECONDS:
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
                        cv2.circle(frame, tuple(map(int, point)), 7, (255, 255, 255), -1)
                        label = f'Hand: {zone}'
                    else:
                        pending_zone = None
                        if now - last_seen > LOSS_SECONDS:
                            stable_zone = None
                        if reference_size is None:
                            label = 'Place hand on desk, press C to calibrate'
                        elif too_close:
                            label = 'Hand too close to camera - ignored'
                        else:
                            label = 'Hand outside desk / not detected'

                    # Preserve columns expected by analysis.py and app.py.
                    writer.writerow([
                        round(seconds, 3),
                        round(x, 4) if x != '' else '',
                        round(y, 4) if y != '' else '',
                        zone, visit,
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
