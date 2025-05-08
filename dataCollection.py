from os.path import split

import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
from time import time

# Offset percentages
outputFolderPath = 'Dataset/DataCollect'
confidence = 0.8
offsetPercentageW = 10  # Horizontal padding
offsetPercentageH = 20  # Vertical padding
camWidth = 640
save = True
debug = False
camHeight = 480
floatingPoint = 6
blurThreshold = 35
# Initialize camera and detector
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camWidth)
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

while True:
    success, img = cap.read()
    imgOut = img.copy()
    img, bboxes = detector.findFaces(img, draw=False)

    if bboxes:
        for bbox in bboxes:
            x, y, w, h = bbox["bbox"]
            score = bbox["score"][0]
            listBlur = [] # True False Values indicating if the faces are blurred or not
            listInfo = [] # The normalized values and the class name for the label txt file

            #Check score
            if score > confidence:

                # Calculate width padding
                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)

                # Calculate height padding (extra top padding)
                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3.5)

                # To avoid Blurriness
                if x<0:x=0
                if y<0:y=0
                if w < 0: w = 0
                if h < 0: h = 0

                # ---- Find Blurriness ---------
                imageFace = img[y:y + h, x:x + w]
                cv2.imshow("Face", imageFace)
                blurValue = cv2.Laplacian(imageFace,cv2.CV_64F).var()
                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)


                # ----- Normalize values ---------------
                ih, iw,_=img.shape
                xc,yc = x+w/2, y+h/3
                xcn,ycn = round(xc/iw,floatingPoint),round(yc/ih,floatingPoint)
                wn, hn = round(w/iw,floatingPoint),round(h/ih,floatingPoint)

                #To avoid values above 1
                if xcn>1:xcn=1
                if ycn>1:ycn=1
                if wn>1:wn=1
                if hn>1:hn=1

                # Drawing
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                cvzone.putTextRect(imgOut,f"Score: {int(score*100)} %Blur:{blurValue:.2f}",(x,y-20),scale = 2, thickness = 3)
                if debug:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    cvzone.putTextRect(imgOut, f"Score: {int(score * 100)} %Blur:{blurValue:.2f}", (x, y - 20), scale=2,
                                       thickness=3)

            if save:
                if all(listBlur) and listBlur!=[]:
                    timeNow = str(time()).replace('.', '')
                    cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg",img)




    cv2.imshow("Image", imgOut)
    cv2.waitKey(1)
