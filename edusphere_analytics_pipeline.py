import math
import cv2
import numpy as np
import mediapipe as mp
import csv
import pandas as pd
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# =========================
# Config & thresholds
# =========================
FOCUS_YAW_THRESHOLD_DEG = 20.0
FOCUS_PITCH_THRESHOLD_DEG = 20.0
FOCUS_ROLL_THRESHOLD_DEG = 25.0

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# =========================
# Helpers
# =========================
LANDMARK_IDX = {
    "nose_tip": 1,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
    "chin": 199,
    "forehead_glabella": 6,
}

def estimate_head_angles(landmarks, image_w, image_h):
    def to_xy(idx):
        lm = landmarks[idx]
        return np.array([lm.x * image_w, lm.y * image_h], dtype=np.float32)

    nose = to_xy(LANDMARK_IDX["nose_tip"])
    left_eye = to_xy(LANDMARK_IDX["left_eye_outer"])
    right_eye = to_xy(LANDMARK_IDX["right_eye_outer"])
    chin = to_xy(LANDMARK_IDX["chin"])
    glabella = to_xy(LANDMARK_IDX["forehead_glabella"])

    eye_vec = right_eye - left_eye
    eye_angle = np.degrees(np.arctan2(eye_vec[1], eye_vec[0]))  # roll

    eye_mid = (left_eye + right_eye) / 2.0
    nose_to_eye_mid = eye_mid - nose
    yaw_angle = np.degrees(np.arctan2(nose_to_eye_mid[0], abs(nose_to_eye_mid[1]) + 1e-6))

    vert_vec = chin - glabella
    pitch_angle = np.degrees(np.arctan2(vert_vec[1], abs(vert_vec[0]) + 1e-6)) - 90.0
    pitch_angle = max(min(pitch_angle, 90.0), -90.0)

    roll_angle = eye_angle
    return yaw_angle, pitch_angle, roll_angle

def is_focused(yaw_deg, pitch_deg, roll_deg):
    return (abs(yaw_deg) <= FOCUS_YAW_THRESHOLD_DEG and
            abs(pitch_deg) <= FOCUS_PITCH_THRESHOLD_DEG and
            abs(roll_deg) <= FOCUS_ROLL_THRESHOLD_DEG)

# =========================
# Session function
# =========================
def analyze_session(video_path, session_name):
    cap = cv2.VideoCapture(video_path)

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    face_detector = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.3
    )

    csv_file = open(f"/content/{session_name}_results.csv", mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "present", "focused", "yaw", "pitch", "roll"])

    frame_count = 0
    present_frames = 0
    focused_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections = face_detector.process(rgb)

        present = False
        focused = False
        yaw = pitch = roll = 0.0

        if detections.detections:
            present = True
            det = detections.detections[0]
            bbox = det.location_data.relative_bounding_box
            min_x = int(bbox.xmin * w)
            min_y = int(bbox.ymin * h)
            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)

            face_crop = frame[min_y:min_y+box_h, min_x:min_x+box_w]

            if face_crop.size > 0:
                fh, fw = face_crop.shape[:2]
                if fh < 100 or fw < 100:
                    scale = 2.0
                    face_crop = cv2.resize(face_crop, (int(fw*scale), int(fh*scale)))

                rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                results_face = face_mesh.process(rgb_face)

                if results_face.multi_face_landmarks:
                    face = results_face.multi_face_landmarks[0].landmark
                    yaw, pitch, roll = estimate_head_angles(face, face_crop.shape[1], face_crop.shape[0])
                    focused = is_focused(yaw, pitch, roll)

        if present:
            present_frames += 1
        if focused:
            focused_frames += 1

        csv_writer.writerow([frame_count, present, focused, yaw, pitch, roll])
        frame_count += 1

    cap.release()
    csv_file.close()

    total_frames = frame_count
    presence_percentage = (present_frames / total_frames) * 100 if total_frames > 0 else 0
    focus_percentage = (focused_frames / total_frames) * 100 if total_frames > 0 else 0

    print(f" جلسة {session_name} انتهت")
    print(f"عدد الإطارات الكلي: {total_frames}")
    print(f"إطارات الحضور: {present_frames} ({presence_percentage:.1f}%)")
    print(f"إطارات التركيز: {focused_frames} ({focus_percentage:.1f}%)")
    print(f"تم حفظ النتائج في {session_name}_results.csv")

    return {
        "session": session_name,
        "total_frames": total_frames,
        "present_frames": present_frames,
        "focused_frames": focused_frames,
        "presence_percentage": presence_percentage,
        "focus_percentage": focus_percentage
    }

# =========================
# Run multiple sessions
# =========================
sessions = []
sessions.append(analyze_session("/content/a1.mp4", "session1"))
sessions.append(analyze_session("/content/a2.mp4", "session2")) 

# =========================
# Compare performance
# =========================
df = pd.DataFrame(sessions)
print("\n مقارنة الأداء بين الجلسات:")
print(df)


plt.figure(figsize=(10,5))
plt.bar(df["session"], df["presence_percentage"], alpha=0.6, label="Presence %")
plt.bar(df["session"], df["focus_percentage"], alpha=0.6, label="Focus %")
plt.ylabel("Percentage")
plt.title("Comparison of Presence and Focus across sessions")
plt.legend()
plt.show()
