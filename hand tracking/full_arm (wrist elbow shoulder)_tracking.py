import cv2
import mediapipe as mp
import math

# ===============================
# 🧠 قاعدة بيانات العضلات والأعصاب
# ===============================
muscles = [
    {"muscle": "Biceps brachii", "nerve": "Musculocutaneous nerve (C5-C6)", "motions": ["Elbow flexion", "Forearm supination", "Shoulder flexion"]},
    {"muscle": "Brachialis", "nerve": "Musculocutaneous nerve (C5-C6)", "motions": ["Elbow flexion"]},
    {"muscle": "Brachioradialis", "nerve": "Radial nerve (C5-C6)", "motions": ["Elbow flexion (mid-pronation)"]},
    {"muscle": "Triceps brachii", "nerve": "Radial nerve (C6-C8)", "motions": ["Elbow extension", "Shoulder extension (long head)"]},
    {"muscle": "Anconeus", "nerve": "Radial nerve (C7-C8)", "motions": ["Elbow extension", "Stabilizes elbow joint"]},
    {"muscle": "Pronator teres", "nerve": "Median nerve (C6-C7)", "motions": ["Forearm pronation", "Weak elbow flexion"]},
    {"muscle": "Supinator", "nerve": "Radial (deep branch, C6)", "motions": ["Forearm supination"]},
    {"muscle": "Deltoid", "nerve": "Axillary nerve (C5-C6)", "motions": ["Shoulder abduction", "Flexion", "Extension"]},
    {"muscle": "Pectoralis major", "nerve": "Medial & lateral pectoral nerves (C5-T1)", "motions": ["Shoulder flexion", "Adduction", "Medial rotation"]},
    {"muscle": "Latissimus dorsi", "nerve": "Thoracodorsal nerve (C6-C8)", "motions": ["Shoulder extension", "Adduction", "Medial rotation"]},
    {"muscle": "Teres major", "nerve": "Lower subscapular nerve (C5-C6)", "motions": ["Shoulder adduction", "Medial rotation"]},
    {"muscle": "Coracobrachialis", "nerve": "Musculocutaneous nerve (C5-C7)", "motions": ["Shoulder flexion", "Adduction"]}
]

# ===============================
# 🎥 إعداد MediaPipe
# ===============================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ الكاميرا غير متصلة.")
    exit()

print("✅ تتبع الكتف والكوع مع معلومات العضلات...")

def calc_angle(a, b, c):
    """حساب الزاوية بين 3 نقاط (shoulder, elbow, wrist)"""
    ax, ay = a
    bx, by = b
    cx, cy = c
    ab = (ax - bx, ay - by)
    cb = (cx - bx, cy - by)
    dot = ab[0]*cb[0] + ab[1]*cb[1]
    mag_ab = math.hypot(ab[0], ab[1])
    mag_cb = math.hypot(cb[0], cb[1])
    cos_angle = dot / (mag_ab * mag_cb + 1e-6)
    angle = math.degrees(math.acos(max(-1, min(1, cos_angle))))
    return int(angle)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        h, w, _ = frame.shape

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            def px(p): return int(p.x * w), int(p.y * h)

            # 🔹 يمين
            rs, re, rw = px(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]), px(lm[mp_pose.PoseLandmark.RIGHT_ELBOW]), px(lm[mp_pose.PoseLandmark.RIGHT_WRIST])
            angle_r = calc_angle(rs, re, rw)

            # 🔹 يسار
            ls, le, lw = px(lm[mp_pose.PoseLandmark.LEFT_SHOULDER]), px(lm[mp_pose.PoseLandmark.LEFT_ELBOW]), px(lm[mp_pose.PoseLandmark.LEFT_WRIST])
            angle_l = calc_angle(ls, le, lw)

            # رسم المفاصل
            cv2.line(frame, rs, re, (255, 0, 0), 4)
            cv2.line(frame, re, rw, (255, 0, 0), 4)
            cv2.line(frame, ls, le, (0, 0, 255), 4)
            cv2.line(frame, le, lw, (0, 0, 255), 4)

            for p in [rs, re, rw, ls, le, lw]:
                cv2.circle(frame, p, 8, (255, 255, 255), -1)

            # عرض زاوية الكوع
            cv2.putText(frame, f"Right Elbow: {angle_r} deg", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            cv2.putText(frame, f"Left Elbow:  {angle_l} deg", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

            # 🔍 معلومات تشريحية بسيطة على الشاشة
            active_muscles = ["Biceps brachii", "Brachialis", "Triceps brachii"]
            y_offset = 150
            cv2.putText(frame, "Active muscles (Elbow/Shoulder):", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            for m in active_muscles:
                cv2.putText(frame, f"  - {m}", (40, y_offset + 25 * (active_muscles.index(m) + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,255,200), 2)

        cv2.imshow("Elbow & Shoulder Tracking + Muscles", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
