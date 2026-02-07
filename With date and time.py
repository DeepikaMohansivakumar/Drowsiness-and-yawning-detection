import cv2
import mediapipe as mp
import numpy as np
import winsound  # For playing sound on Windows
from datetime import datetime

def eye_aspect_ratio(landmarks, points):
    l = landmarks[points[0]]
    r = landmarks[points[3]]
    t = landmarks[points[1]]
    b = landmarks[points[5]]
    hor = np.linalg.norm([r.x - l.x, r.y - l.y])
    ver = np.linalg.norm([t.y - b.y])
    return ver / hor if hor != 0 else 0

def mouth_open_ratio(landmarks):
    top = landmarks[13]  # upper lip
    bottom = landmarks[14]  # lower lip
    return abs(top.y - bottom.y)

# Initialize MediaPipe Face Mesh
mpfm = mp.solutions.face_mesh
face_mesh = mpfm.FaceMesh(static_image_mode=False,
                          max_num_faces=1,
                          refine_landmarks=True,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)

# Landmark indices for eyes
LEFT = [362, 385, 387, 263, 373, 380]
RIGHT = [33, 160, 158, 133, 153, 144]

# Thresholds
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 15
MOUTH_OPEN_THRESHOLD = 0.08

closed_eyes = 0
alarm_on = False

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        ear = (eye_aspect_ratio(lm, LEFT) + eye_aspect_ratio(lm, RIGHT)) / 2.0
        mouth_ratio = mouth_open_ratio(lm)

        # Display EAR and Mouth Ratio
        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"Mouth: {mouth_ratio:.2f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # Drowsiness/Yawning detection
        if ear < EAR_THRESHOLD or mouth_ratio > MOUTH_OPEN_THRESHOLD:
            closed_eyes += 1
        else:
            closed_eyes = 0
            if alarm_on:
                winsound.PlaySound(None, winsound.SND_FILENAME)
                alarm_on = False

        if closed_eyes > CONSEC_FRAMES and not alarm_on:
            alarm_on = True
            winsound.PlaySound("alarm.wav",
                               winsound.SND_FILENAME | winsound.SND_LOOP | winsound.SND_ASYNC)
            cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        if mouth_ratio > MOUTH_OPEN_THRESHOLD:
            cv2.putText(frame, "YAWNING DETECTED", (50, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,165,255), 2)

    else:
        closed_eyes = 0
        if alarm_on:
            winsound.PlaySound(None, winsound.SND_FILENAME)
            alarm_on = False
        cv2.putText(frame, "No face detected", (20,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # Date & Time Display
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text_size = cv2.getTextSize(now, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    x = frame.shape[1] - text_size[0] - 10
    y = 30
    cv2.putText(frame, now, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Drowsiness Detector", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()