import cv2
import mediapipe as mp

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Start the live camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame from the camera.")
            break

        frame = cv2.flip(frame, 1)  # Flip the image horizontally for a mirror effect
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

                # Gesture detection
                fingers = [landmarks[i][1] < landmarks[i - 2][1] for i in [8, 12, 16, 20]]
                thumb = landmarks[4][0] > landmarks[3][0]

                # "LIKE" gesture
                if thumb and not any(fingers):
                    cv2.putText(frame, 'LIKE', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # "DISLIKE" gesture (corrected logic)
                thumb_down = landmarks[4][1] > landmarks[3][1]  # Thumb is pointing down
                fingers_closed = all(landmarks[i][1] > landmarks[i - 2][1] for i in [8, 12, 16, 20])  # Fingers are closed
                if thumb_down and fingers_closed:
                    cv2.putText(frame, 'DISLIKE', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # "OK" gesture
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                distance = ((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2) ** 0.5
                if distance < 0.05 and all(landmarks[i][1] < landmarks[i - 2][1] for i in [12, 16, 20]):
                    cv2.putText(frame, 'OK', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # "PEACE" gesture
                if fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
                    cv2.putText(frame, 'PEACE', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # "HEART" gesture
                index_up = landmarks[8][1] < landmarks[6][1]
                middle_up = landmarks[12][1] < landmarks[10][1]
                distance_v = ((landmarks[8][0] - landmarks[12][0]) ** 2 + (landmarks[8][1] - landmarks[12][1]) ** 2) ** 0.5
                if index_up and middle_up and distance_v < 0.05:
                    cv2.putText(frame, 'HEART', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()