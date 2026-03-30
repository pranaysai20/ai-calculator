import cv2
import mediapipe as mp
import time
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------
# Button Class
# -------------------------------
class Button:
    def __init__(self, pos, width, height, value):
        self.pos = pos
        self.width = width
        self.height = height
        self.value = value

    def draw(self, img, hover=False):
        color = (0, 255, 0) if hover else (200, 200, 200)
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
                      color, cv2.FILLED)
        cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
                      (50, 50, 50), 3)
        cv2.putText(img, self.value, (self.pos[0] + 30, self.pos[1] + 70),
                    cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 50), 2)

    def checkHover(self, x, y):
        return self.pos[0] < x < self.pos[0] + self.width and \
               self.pos[1] < y < self.pos[1] + self.height


# -------------------------------
# Create Calculator Buttons
# -------------------------------
buttonListValues = [['7', '8', '9', '/'],
                    ['4', '5', '6', '*'],
                    ['1', '2', '3', '-', 'C'],
                    ['0', '.', '=', '+']]


buttonList = []
for y in range(len(buttonListValues)):
    for x in range(len(buttonListValues[y])):
        xpos = x * 100 + 50   # shift inside the screen
        ypos = y * 100 + 150
        buttonList.append(Button((xpos, ypos), 100, 100, buttonListValues[y][x]))


# -------------------------------
# Initialize Mediapipe Hands
# -------------------------------
DETECTION_RESULT = None

def save_result(result, unused_output_image, timestamp_ms):
    global DETECTION_RESULT
    DETECTION_RESULT = result

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    result_callback=save_result
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # width
cap.set(4, 720)   # height

equation = ""
last_click_time = 0
click_delay = 0.4  # seconds delay between clicks


# -------------------------------
# Main Loop
# -------------------------------
while True:
    ret, img = cap.read()
    if not ret:
        break
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    detector.detect_async(mp_image, time.time_ns() // 1_000_000)

    index_tip = None
    thumb_tip = None

    if DETECTION_RESULT and DETECTION_RESULT.hand_landmarks:
        hand_landmarks = DETECTION_RESULT.hand_landmarks[0]
        h, w, c = img.shape
        index_tip = (int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h))
        thumb_tip = (int(hand_landmarks[4].x * w), int(hand_landmarks[4].y * h))

        # Draw landmarks
        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())

        # Draw fingertips
        cv2.circle(img, index_tip, 10, (0, 0, 255), -1)
        cv2.circle(img, thumb_tip, 10, (255, 0, 0), -1)

        # Pinch detection
        dist = ((index_tip[0] - thumb_tip[0]) ** 2 + (index_tip[1] - thumb_tip[1]) ** 2) ** 0.5
        if dist < 40:  # threshold for pinch
            current_time = time.time()
            if current_time - last_click_time > click_delay:
                for button in buttonList:
                    if button.checkHover(index_tip[0], index_tip[1]):
                        val = button.value
                        if val == '=':
                            try:
                                # Replace operators with Python equivalents
                                expression = equation.replace('^', '**').replace('√', 'math.sqrt')
                                equation = str(eval(expression))
                            except:
                                equation = "Error"
                        elif val == 'C':
                            equation = ''
                        else:
                            equation += val
                        last_click_time = current_time
                        break

    # Draw buttons
    for button in buttonList:
        hover = False
        if index_tip:
            hover = button.checkHover(index_tip[0], index_tip[1])
        button.draw(img, hover)

    # Display equation/result
    cv2.rectangle(img, (50, 50), (750, 130), (200, 200, 200), cv2.FILLED)
    cv2.rectangle(img, (50, 50), (750, 130), (50, 50, 50), 3)
    cv2.putText(img, equation, (60, 120), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

    cv2.imshow("AI Human Calculator", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

detector.close()
cap.release()
cv2.destroyAllWindows()