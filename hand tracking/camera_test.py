import cv2

print("🔍 جاري فتح الكاميرا...")

# حاول فتح الكاميرا
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ لم يتم فتح الكاميرا. جرّب رقم آخر (1 أو 2).")
    exit()

print("✅ الكاميرا تعمل الآن. سيتم عرض الصورة...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ لم يتم التقاط إطار من الكاميرا.")
        break

    frame = cv2.flip(frame, 1)
    cv2.imshow('Camera Test', frame)

    # اضغط ESC للخروج
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("🟢 تم إغلاق الكاميرا.")
