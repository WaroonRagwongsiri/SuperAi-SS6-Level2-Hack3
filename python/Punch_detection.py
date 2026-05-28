import cv2
import mediapipe as mp
import math
import time

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# =========================
# Adjustable Thresholds
# =========================
PUNCH_DIFF_PERCENT = 20      # ถ้าขนาดมือเปลี่ยนเกิน 20% = punch
PUNCH_COOLDOWN = 0.6         # กันนับซ้ำ ภายใน 0.6 วินาที

ARM_STRAIGHT_ANGLE = 110
MIN_SHOULDER_WRIST_DIST = 0.25
WRIST_NOT_TOO_LOW = 0.15

# =========================
# State Variables
# =========================
previous_hand_area = None
last_punch_time = 0
punch_count = 0


def calculate_angle(a, b, c):
    ab = [a[0] - b[0], a[1] - b[1]]
    cb = [c[0] - b[0], c[1] - b[1]]

    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    mag_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2)

    if mag_ab * mag_cb == 0:
        return 0

    value = dot / (mag_ab * mag_cb)
    value = max(min(value, 1), -1)

    return math.degrees(math.acos(value))


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose, mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_results = pose.process(rgb)
        hand_results = hands.process(rgb)

        punch_detected = False
        reason = "No punch"
        diff_percent = 0

        current_time = time.time()

        # =========================
        # Case 1: Front View
        # Use hand bbox difference
        # =========================
        if hand_results.multi_hand_landmarks:
            largest_hand_area = 0
            largest_hand_landmarks = None
            largest_box = None

            for hand_landmarks in hand_results.multi_hand_landmarks:
                xs = []
                ys = []

                for lm in hand_landmarks.landmark:
                    xs.append(int(lm.x * w))
                    ys.append(int(lm.y * h))

                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                box_w = x_max - x_min
                box_h = y_max - y_min
                hand_area = box_w * box_h

                if hand_area > largest_hand_area:
                    largest_hand_area = hand_area
                    largest_hand_landmarks = hand_landmarks
                    largest_box = (x_min, y_min, x_max, y_max)

            if largest_hand_landmarks is not None:
                x_min, y_min, x_max, y_max = largest_box
                current_hand_area = largest_hand_area

                if previous_hand_area is not None and previous_hand_area > 0:
                    diff_percent = abs(current_hand_area - previous_hand_area) / previous_hand_area * 100

                    if diff_percent > PUNCH_DIFF_PERCENT:
                        if current_time - last_punch_time > PUNCH_COOLDOWN:
                            punch_detected = True
                            punch_count += 1
                            reason = f"Punch by hand diff: {diff_percent:.1f}%"
                            last_punch_time = current_time

                previous_hand_area = current_hand_area

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                cv2.putText(frame, f"Hand Area: {current_hand_area}",
                            (x_min, y_min - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.putText(frame, f"Diff: {diff_percent:.1f}%",
                            (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                mp_draw.draw_landmarks(frame, largest_hand_landmarks, mp_hands.HAND_CONNECTIONS)

        else:
            previous_hand_area = None

        # =========================
        # Case 2: Side View
        # Use arm angle
        # =========================
        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark

            r_shoulder = [
                lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            ]
            r_elbow = [
                lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y
            ]
            r_wrist = [
                lm[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                lm[mp_pose.PoseLandmark.RIGHT_WRIST].y
            ]

            l_shoulder = [
                lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            ]
            l_elbow = [
                lm[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                lm[mp_pose.PoseLandmark.LEFT_ELBOW].y
            ]
            l_wrist = [
                lm[mp_pose.PoseLandmark.LEFT_WRIST].x,
                lm[mp_pose.PoseLandmark.LEFT_WRIST].y
            ]

            arms = [
                ("Right", r_shoulder, r_elbow, r_wrist),
                ("Left", l_shoulder, l_elbow, l_wrist)
            ]

            for side, shoulder, elbow, wrist in arms:
                angle = calculate_angle(shoulder, elbow, wrist)
                arm_dist = distance(shoulder, wrist)
                wrist_too_low = wrist[1] > shoulder[1] + WRIST_NOT_TOO_LOW

                if angle > ARM_STRAIGHT_ANGLE and arm_dist > MIN_SHOULDER_WRIST_DIST and not wrist_too_low:
                    if current_time - last_punch_time > PUNCH_COOLDOWN:
                        punch_detected = True
                        punch_count += 1
                        reason = f"{side} punch by arm angle"
                        last_punch_time = current_time

                cv2.putText(frame, f"{side} Angle: {int(angle)}",
                            (20, 90 if side == "Right" else 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # =========================
        # Display Result
        # =========================
        if punch_detected:
            color = (0, 0, 255)
            text = "PUNCH DETECTED"
        else:
            color = (255, 255, 255)
            text = "NO PUNCH"

        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

        cv2.putText(frame, f"Punch Count: {punch_count}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, reason,
                    (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Punch Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()