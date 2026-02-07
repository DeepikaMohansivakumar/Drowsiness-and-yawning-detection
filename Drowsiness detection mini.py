import cv2
import mediapipe as mp
import numpy as np
import winsound  # Windows-only

def eye_aspect_ratio(landmarks, points):
    left = landmarks[points[0]]
    right = landmarks[points[3]]
    top = landmarks[points[1]]
    bottom = landmarks[points[5]]
    hor = np.linalg.norm([right.x-left.x, right.y-left.y])
    ver = np.linalg.norm([top.x-bottom.x, top.y-bottom.y])
    return ver / hor if hor != 0 else 0

mpfm = mp.solutions.face_mesh
face_mesh = mpfm.FaceMesh(static_image_mode=False, max_num_faces=1,
                          refine_landmarks=True,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)
LEFT = [362,385,387,263,373,380]
RIGHT = [33,160,158,133,153,144]
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 15

closed_eyes = 0  # ← initialize here

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        ear = (eye_aspect_ratio(lm, LEFT) + eye_aspect_ratio(lm, RIGHT)) / 2
        cv2.putText(frame, f"EAR: {ear:.2f}", (20,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        if ear < EAR_THRESHOLD:
            closed_eyes += 1
        else:
            closed_eyes = 0
            winsound.PlaySound(None, winsound.SND_PURGE)

        if closed_eyes > CONSEC_FRAMES:
            winsound.PlaySound("SystemExclamation", 
                               winsound.SND_ALIAS | winsound.SND_LOOP | winsound.SND_ASYNC)
            cv2.putText(frame, "DROWSINESS ALERT!", (50,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    else:
        closed_eyes = 0
        winsound.PlaySound(None, winsound.SND_PURGE)
        cv2.putText(frame, "No face detected", (20,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Drowsiness Detector", frame)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
