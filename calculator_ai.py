import cv2
import mediapipe as mp
import time

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
                    ['1', '2', '3', '-'],
                    ['0', '.', '=', '+','C']]
                           

buttonList = []
for y in range(len(buttonListValues)):
    for x in range(len(buttonListValues[y])):
        xpos = x * 100 + 50   # shift inside the screen
        ypos = y * 100 + 150
        buttonList.append(Button((xpos, ypos), 100, 100, buttonListValues[y][x]))


# -------------------------------
# Initialize Mediapipe Hands
# -------------------------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

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
    results = hands.process(imgRGB)

    index_tip = None
    thumb_tip = None

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        h, w, c = img.shape
        index_tip = (int(handLms.landmark[8].x * w), int(handLms.landmark[8].y * h))
        thumb_tip = (int(handLms.landmark[4].x * w), int(handLms.landmark[4].y * h))

        # Draw landmarks
        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

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
                                equation = str(eval(equation))
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
    cv2.rectangle(img, (50, 50), (550, 130), (200, 200, 200), cv2.FILLED)
    cv2.rectangle(img, (50, 50), (550, 130), (50, 50, 50), 3)
    cv2.putText(img, equation, (60, 120), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

    cv2.imshow("AI Human Calculator", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
