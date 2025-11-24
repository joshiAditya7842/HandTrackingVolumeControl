import cv2
import mediapipe as mp
import math
import numpy as np


class handDetector:
    def __init__(self, mode=False, maxHands=2, modelComplex=1,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplex
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=self.modelComplex,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmList = []
        self.results = None

    def _to_ndarray(self, img):
        """Convert any OpenCV image (including UMat) to a proper numpy array."""
        if img is None:
            return None

        # If already ndarray, just ensure dtype and layout
        if isinstance(img, np.ndarray):
            return np.ascontiguousarray(img, dtype=np.uint8)

        # If it's a UMat (or similar), try .get()
        if hasattr(img, "get"):
            img = img.get()

        # Fallback: let numpy try to convert
        img = np.array(img, dtype=np.uint8)
        img = np.ascontiguousarray(img)
        return img

    def findHands(self, img, draw=True):
        """Detect hands and draw landmarks."""

        # Make sure img is a clean numpy array (H, W, 3) uint8
        img = self._to_ndarray(img)
        if img is None:
            return img

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB = self._to_ndarray(imgRGB)  # again ensure ndarray

        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def findPosition(self, img, handNo=0, draw=True):
        """Return list of landmarks [id, x, y] for one hand."""
        self.lmList = []

        if self.results and self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, c = img.shape

            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return self.lmList

    def fingersUp(self):
        """Return list of 0/1 for each finger (up or down)."""
        fingers = []

        if not self.lmList:
            return fingers

        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        """Distance between two landmarks p1 and p2."""
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]
