import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utlis import *
import keras
import sudukoSolver

model = keras.models.load_model('Resources/myModel.h5')
heightImg = 630
widthImg = 630
obj = cv2.VideoCapture(0,cv2.CAP_DSHOW)
while(True):
    tf,img = obj.read()
#### 1. PREPARE THE IMAGE
    tf, img = obj.read()
    img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)# CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgThreshold = preProcess(img)
    cv2.imshow('Live', img)
    cv2.waitKey(1)

# #### 2. FIND ALL COUNTOURS
    imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
#### 3. FIND THE BIGGEST COUNTOUR AND USE IT AS SUDOKU
    biggest, maxArea = biggestContour(contours) # FIND THE BIGGEST CONTOUR biggest is the four corner points
    if biggest.size != 0:
        imgtemp = img.copy()
        cv2.drawContours(imgtemp,contours,-1, (0, 255, 0), 3)
        cv2.imshow('Live',imgtemp)
        cv2.waitKey(1)
        imgBigContour = img.copy()
        biggest = reorder(biggest)
        # cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25) # DRAW THE BIGGEST CONTOUR corners
        pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2) # GER
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
        imgSolvedDigits = imgBlank.copy()
        boxes = splitBoxes(imgWarpColored)
        numbers = getPredection(boxes, model)
        print(numbers)
        imgDetectedDigits = imgBlank.copy()
        imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
        imgDetectedDigits = drawGrid(imgDetectedDigits)
        imgtemp = imgDetectedDigits.copy()
        cv2.putText(imgtemp, 'Press R to Retake', (30, 70), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200),2)
        cv2.imshow('Live', imgtemp)
        if cv2.waitKey(0) & 0xFF == ord('r'):
            cv2.destroyWindow('Live')
            continue
        numbers = np.asarray(numbers)
        posArray = np.where(numbers > 0, 0, 1)

#### 5. FIND SOLUTION OF THE BOARD
        board = np.array_split(numbers,9)
        try:
            sudukoSolver.solve(board)
        except:
            pass
        flatList = []
        for sublist in board:
            for item in sublist:
                flatList.append(item)
        solvedNumbers =flatList*posArray
        imgSolvedDigits= displayNumbers(imgSolvedDigits,solvedNumbers)
        imgtemp = cv2.addWeighted(imgDetectedDigits,.5,imgSolvedDigits,.5,0.0)
        cv2.imshow('Live',imgtemp)
        if cv2.waitKey(0) & 0xFF==ord('q'):
            break
