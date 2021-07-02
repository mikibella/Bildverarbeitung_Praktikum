import matplotlib.pyplot as plt  # libary for showing image
import numpy as np  # numpy libary
from manual import *
from libary import *
import cv2
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

def colorFilter(path):
    img = cv2.imread(PATH)
    height, width = img.shape[:2]
    imgS = cv2.resize(img, (int(width/5), int(height/5)))
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(grayImg,cv2.CV_64F, ksize=5)
    median = cv2.medianBlur(grayImg, 3)
    ret,th = cv2.threshold(median, 30 , 80, cv2.THRESH_BINARY)
    edges = cv2.Canny(median,50, 70 )

    # Red color
    lower_red = (161, 50, 40)  # (0, 40, 100) S->auf130??
    upper_red = (179, 255, 255)  # 70, 125)#für bilder s&v auf 255, 255)#
    lower_lightred = (0, 120, 40)  # (0, 40, 100) S->auf130??
    upper_lightred = (10, 255, 255) 
    lower_yellow = (17, 120, 20)  # (20, 140, 50)
    upper_yellow = (30, 255, 255)



    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv = cv2.medianBlur(hsv, 5)
    #mask = cv2.inRange(hsv, lower_red, upper_red)
    #mask = cv2.add(mask, cv2.inRange(hsv, lower_lightred, upper_lightred))
    #mask = cv2.add(mask, cv2.inRange(hsv, lower_yellow, upper_yellow))
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    res = cv2.bitwise_and(img, img, mask=mask)
    height, width = res.shape[:2]
    imgS = cv2.resize(res, (int(width/5), int(height/5)))
    cv2.imshow('squares', imgS)
    ch = cv2.waitKey()
    # h, s, v1 = cv2.split(res)
    # thresh = cv2.threshold(v1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


   
    # edges = cv2.Canny(thresh,0, 300 )
    # gausBlur = cv2.GaussianBlur(edges, (5, 5), 0)
    return (res)

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 400 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.2:
                        squares.append(cnt)
    return squares

if __name__ == "__main__":
    # Greetings to the World
    print("Moin World")
    michaelMethoden = 0;


    # creating 2D Array
    #array = np.array([[grayValue for i in range(width)]for j in range(height)])
    # print(array)

    # show image with matplotlib as grayscale image
    #plt.imshow(array, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    # Bild einlesen
    PATH = r"C:\Users\bellmi2\Documents\BV-UNI\schilder\vfs_01.jpg   "

    libary = Libary(PATH)
    manual = Manual(PATH)
    img = cv2.imread(PATH)
    imgColor = colorFilter(PATH)
    squares = find_squares(imgColor)
    print(squares)
    if(squares):
        cv2.drawContours( img, squares, -1, (0, 255, 0), 3 )
        cv2.putText(img, 'Vorfahrt', (squares[0][0][0], squares[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 3, (36,255,12), 7)
    else:
        print("Kein Rechteck gefunden.")
    height, width = img.shape[:2]
    imgS = cv2.resize(img, (int(width/5), int(height/5)))
    cv2.imshow('squares', imgS)
    ch = cv2.waitKey()
