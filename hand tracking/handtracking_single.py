import cv2
import mediapipe as mp

# تهيئة Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# تشغيل الكاميرا
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("⚠️ لم يتم فتح الكاميرا.")
    exit()

print("✅ بدأ تتبع اليد...")

# تتبع يد واحدة فقط
with mp_hands.Hands(
    max_num_hands=1,                # 👈 يد واحدة فقط
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ لم يتم التقاط إطار من الكاميرا.")
            break

        # قلب الصورة أفقياً (عشان تكون مثل المرآة)
        frame = cv2.flip(frame, 1)

        # تحويل الصورة إلى RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # في حال تم اكتشاف اليد
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # عرض الفيديو
        cv2.imshow("Hand Tracking (One Hand)", frame)

        # للخروج اضغط ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
print("انتهى التتبع.")

