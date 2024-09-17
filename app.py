import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def process_hand_gestur(hand_landmarks, frame):
    """Processes hand landmarks and moves the cursor"""
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # Get the image dimensions
    h, w, _ = frame.shape
    
    # Calculate the cursor position
    x = int(index_tip.x * w)
    y = int(index_tip.y * h)
    
    # Move the cursor using pyautogui
    pyautogui.moveTo(x, y)

    # print(f"Index Tip: {index_tip.x}, {index_tip.y}, {index_tip.z}")
    # print(f"Cursor Position: {x}, {y}")

def app():
    """Main function"""
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                process_hand_gestur(hand_landmarks, frame)
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app()
