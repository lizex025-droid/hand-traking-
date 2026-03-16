import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Calculate angle between 3 points
def calc_angle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c

    ab = (ax - bx, ay - by)
    cb = (cx - bx, cy - by)

    dot = ab[0]*cb[0] + ab[1]*cb[1]
    mag_ab = math.hypot(ab[0], ab[1])
    mag_cb = math.hypot(cb[0], cb[1])

    if mag_ab == 0 or mag_cb == 0:
        return 0

    cos_angle = dot / (mag_ab * mag_cb)
    cos_angle = max(-1, min(1, cos_angle))

    angle = math.degrees(math.acos(cos_angle))
    return int(angle)

# Flexion / Extension labeling
def flex_ext(angle):
    if angle < 150:
        return "Flexion"
    elif angle > 175:
        return "Extension"
    else:
        return "Neutral"

cap = cv2.VideoCapture(0)
print("📌 Running Wrist Flexion/Extension Tracking...")

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Camera not returning frames.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:

                # Extract wrist - index MCP - pinky MCP
                wrist = hand.landmark[0]
                index_mcp = hand.landmark[5]
                pinky_mcp = hand.landmark[17]

                wx, wy = int(wrist.x * w), int(wrist.y * h)
                ix, iy = int(index_mcp.x * w), int(index_mcp.y * h)
                px, py = int(pinky_mcp.x * w), int(pinky_mcp.y * h)

                # Compute angle
                angle = calc_angle((ix, iy), (wx, wy), (px, py))
                motion = flex_ext(angle)

                # Draw
                for (x, y) in [(wx, wy), (ix, iy), (px, py)]:
                    cv2.circle(frame, (x, y), 8, (255, 255, 255), -1)

                cv2.line(frame, (wx, wy), (ix, iy), (0, 255, 0), 3)
                cv2.line(frame, (wx, wy), (px, py), (0, 255, 0), 3)

                cv2.putText(frame, f"{motion} ({angle}°)", 
                            (wx + 10, wy - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 255, 255), 2)

                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        else:
            cv2.putText(frame, "No hand detected", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Wrist Flex/Ext Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

