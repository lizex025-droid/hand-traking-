import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# زاوية بين 3 نقاط
def calc_angle(a, b, c):
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

# Flexion / Extension
def flex_ext(angle):
    if angle < 150:
        return "Flexion"
    elif angle > 175:
        return "Extension"
    else:
        return "Neutral"

# Radial / Ulnar deviation
def radial_ulnar(wx, wy, ix, iy):
    dx = ix - wx   # الاتجاه الأفقي

    threshold = 20  # يمكنك تعديله حسب الدقة

    if dx > threshold:
        return "Radial deviation"
    elif dx < -threshold:
        return "Ulnar deviation"
    else:
        return "Neutral deviation"

cap = cv2.VideoCapture(0)

print("✅ Tracking Flex/Ext + Radial/Ulnar Deviation")

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:

                # النقاط الأساسية
                wrist = handLms.landmark[0]
                index_mcp = handLms.landmark[5]
                pinky_mcp = handLms.landmark[17]

                wx, wy = int(wrist.x * w), int(wrist.y * h)
                ix, iy = int(index_mcp.x * w), int(index_mcp.y * h)
                px, py = int(pinky_mcp.x * w), int(pinky_mcp.y * h)

                # زاوية Flexion/Extension
                wrist_angle = calc_angle((ix, iy), (wx, wy), (px, py))
                flex_ext_txt = flex_ext(wrist_angle)

                # Radial/Ulnar deviation
                deviation_txt = radial_ulnar(wx, wy, ix, iy)

                # رسم خطوط اليد
                cv2.line(frame, (wx, wy), (ix, iy), (0, 255, 0), 3)
                cv2.line(frame, (wx, wy), (px, py), (0, 255, 0), 3)

                for x, y in [(wx, wy), (ix, iy), (px, py)]:
                    cv2.circle(frame, (x, y), 8, (255, 255, 255), -1)

                # النص النهائي
                cv2.putText(frame, f"Wrist: {flex_ext_txt}, {deviation_txt}",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)

                # رسم اليد كاملة
                mp_drawing.draw_landmarks(
                    frame,
                    handLms,
                    mp_hands.HAND_CONNECTIONS
                )

        cv2.imshow("Wrist Motion Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
