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

        total_fingers = 0  # Counter for total extended fingers

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

                # Count extended fingers for one hand
                fingers = [
                    landmarks[8][1] < landmarks[6][1],  # Index finger
                    landmarks[12][1] < landmarks[10][1],  # Middle finger
                    landmarks[16][1] < landmarks[14][1],  # Ring finger
                    landmarks[20][1] < landmarks[18][1],  # Pinky finger
                ]
                thumb = landmarks[4][0] > landmarks[3][0]  # Thumb (right hand)

                # Add the number of extended fingers for this hand
                total_fingers += fingers.count(True) + (1 if thumb else 0)

        # Display the total number of extended fingers
        cv2.putText(frame, f'TOTAL FINGERS: {total_fingers}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Finger Counting', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()