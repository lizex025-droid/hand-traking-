import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def angle_between(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 == 0 or mag2 == 0:
        return 0
    cos_a = dot / (mag1 * mag2)
    cos_a = max(-1, min(1, cos_a))
    return math.degrees(math.acos(cos_a))

def flex_ext(angle):
    if angle < 165:
        return "Flexion"
    elif angle > 185:
        return "Extension"
    else:
        return "Neutral"

cap = cv2.VideoCapture(0)
print("🔥 Accurate Wrist Flexion/Extension Tracking...")

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
            for hand in results.multi_hand_landmarks:

                # نقاط مهمة
                wrist = hand.landmark[0]
                index_mcp = hand.landmark[5]
                pinky_mcp = hand.landmark[17]
                middle_mcp = hand.landmark[9]

                wx, wy = wrist.x * w, wrist.y * h
                ix, iy = index_mcp.x * w, index_mcp.y * h
                px, py = pinky_mcp.x * w, pinky_mcp.y * h
                mx, my = middle_mcp.x * w, middle_mcp.y * h

                # محور اليد (يدل على اتجاه الساعد)
                forearm_vec = (px - ix, py - iy)

                # محور المعصم (الاتجاه الحقيقي للمعصم)
                wrist_vec = (mx - wx, my - wy)

                # حساب الزاوية
                angle = angle_between(forearm_vec, wrist_vec)
                motion = flex_ext(angle)

                # عرض الزاوية
                cv2.putText(frame, f"{motion} ({int(angle)} deg)",
                            (int(wx)+20, int(wy)-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)

                # رسم الخطوط
                cv2.line(frame, (int(ix), int(iy)), (int(px), int(py)), (0, 255, 0), 3)
                cv2.line(frame, (int(wx), int(wy)), (int(mx), int(my)), (0, 200, 255), 3)

                # رسم المفاصل
                for x, y in [(wx, wy), (ix, iy), (px, py), (mx, my)]:
                    cv2.circle(frame, (int(x), int(y)), 7, (255, 255, 255), -1)

                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Accurate Wrist Flex/Ext Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
