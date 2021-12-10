<div id="top"></div>
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/ogzmsl/HandDetection/blob/main/ouz-logo.png">
    <img src="/ouz-logo.png" alt="" width="" height="">
  </a>

  <h3 align="center">Python Hand Detection</h3>

  <p align="center">
    Python with OpenCV - MediaPip Framework Hand Detection
    <br />
    <a href="https://pythonrepo.com/repo/cvzone-cvzone"><strong>Explore the docs »</strong></a>
    <br />
    <br /> 
    <a href="https://oguzmuslu.com">Contact Me</a> 
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

[![product-screenshot]](https://pythonrepo.com/repo/cvzone-cvzone)

It is a Computer vision package that makes it easy to operate image processing and AI functions. It mainly uses OpenCV and Google Mediapipe libraries.

Usage areas
* Military Industry (submarine sonic wave scans), underwater imaging.
* Security, criminal laboratories.
* Medicine.
* Clarification of structures such as tumors, vessels, Tomography, Ultrasound.
* Robotics, traffic, astronomy, radar, newspaper and photography industry applications
* Vb..

Here we just do hand identification with a computer camera based on the basics.

<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

Libraries and programming language I use.

* [Python](https://www.python.org/)
* [OpenCV](https://opencv.org/)
* [Mediapip](https://mediapipe.dev/)
* [Numpy](https://numpy.org/)  

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

The materials you need to do this.

### Installation

· Install PIP packages

   ```sh
   ! pip install opencv
   ```
   ```sh
   ! pip install mediapip
   ```
   ```sh
   ! pip install numpy
   ```
<p align="right">(<a href="#top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

### Basic Code Example

```
import cvzone
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = cvzone.HandDetector(detectionCon=0.5, maxHands=1)

while True:
    # Get image frame
    success, img = cap.read()

    # Find the hand and its landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    
    # Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)

```

### Finding How many finger are up
```
if lmList:
fingers = detector.fingersUp()
totalFingers = fingers.count(1)
cv2.putText(img, f'Fingers:{totalFingers}', (bbox[0] + 200, bbox[1] - 30),
            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
```
<p align="right">(<a href="#top">back to top</a>)</p>

### My Hand Detection

![my-handDetection]

```
import mediapipe as mp
import cv2
import numpy as np 

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #print(results)
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(217, 133, 0), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(105, 0, 101), thickness=2, circle_radius=2),)
                cv2.imshow('HandTracking', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
cap.release()
cv2.destroyAllWindows()
mp_drawing.DrawingSpec()
```

<!-- CONTACT -->
## Contact

Twitter - [@filokipatisi](https://twitter.com/filokipatisi) <br>
E-Mail -  [GMAIL](mailto:oguzzmuslu@gmail.com) <br>
Linkedin - [oguzzmuslu](https://www.linkedin.com/in/oguzzmuslu/)


<p align="right">(<a href="#top">back to top</a>)</p>





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[my-handDetection]: https://github.com/ogzmsl/HandDetection/blob/main/detection.png
[product-screenshot]: https://github.com/ogzmsl/HandDetection/blob/main/screenshot.jpg
