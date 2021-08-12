import cv2
import numpy as np
import glob
def constrastLimit(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_hist_equalized)
    channels[0] = cv2.equalizeHist(channels[0])
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized

def smoothImage(image):
    LoG_image = cv2.GaussianBlur(image, (3,3), 0)           # paramter 
    gray = cv2.cvtColor( LoG_image, cv2.COLOR_BGR2GRAY)
    LoG_image = cv2.Laplacian( gray, cv2.CV_8U,3,3,2)       # parameter
    LoG_image = cv2.convertScaleAbs(LoG_image)
    thresh = cv2.threshold(LoG_image,32,255,cv2.THRESH_BINARY)[1]
    #thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    return thresh

def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
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
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                    if max_cos < 0.2:
                        squares.append(cnt)
    return squares

def find_triangle(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    triangle = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 3 and cv2.contourArea(cnt) > 400 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 3], cnt[(i+2) % 3] ) for i in range(3)])
                    if max_cos < 0.65 and max_cos >0.35 :
                        triangle.append(cnt)
    return triangle

def find_stop(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    stop = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 8 and cv2.contourArea(cnt) > 400 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 8], cnt[(i+2) % 8] ) for i in range(8)])
                    if max_cos < 0.85 and max_cos >0.55:
                        stop.append(cnt)
    return stop

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

if __name__ == "__main__": 
    vf_cascade = cv2.CascadeClassifier()
    vfa_cascade = cv2.CascadeClassifier()
    vfs_cascade = cv2.CascadeClassifier()
    stop_cascade = cv2.CascadeClassifier()
    vf_cascade.load(cv2.samples.findFile(r"C:\Users\bellmi2\Documents\BV-UNI\training\trained_vf\cascade.xml"))
    vfa_cascade.load(cv2.samples.findFile(r"C:\Users\bellmi2\Documents\BV-UNI\training\trained_vfa\cascade.xml"))
    vfs_cascade.load(cv2.samples.findFile(r"C:\Users\bellmi2\Documents\BV-UNI\training\trained_vfs\cascade.xml"))
    stop_cascade.load(cv2.samples.findFile(r"C:\Users\bellmi2\Documents\BV-UNI\training\trained_stop\cascade.xml"))

    cascade = [vf_cascade,vfa_cascade,vfs_cascade,stop_cascade]
            # Lila = VF,    Grün = VFA,   Gelb = VFS ,  blau = stop
    color = [(255, 0, 255),(102, 204, 0),(0, 255, 255),(51, 51,255)]



    frame = cv2.imread(r"C:\Users\bellmi2\Documents\BV-UNI\schilder\vfa_01.jpg")
    #frame = cv2.imread(r"C:\Users\bellmi2\Documents\BV-UNI\schilder\mr\vorfahrt_gewaehren_2.jpg")

    lower_red = (161, 50, 40)  # (0, 40, 100) S->auf130??
    upper_red = (179, 255, 255)  # 70, 125)#für bilder s&v auf 255, 255)#
    lower_lightred = (0, 120, 40)  # (0, 40, 100) S->auf130??
    upper_lightred = (10, 255, 255) 
    lower_yellow = (17, 120, 20)  # (20, 140, 50)
    upper_yellow = (30, 255, 255)

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_lightred, upper_lightred)
    yellow_Mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    red_Mask = cv2.bitwise_or(mask1, mask2)


    frame_resized_out = frame

    findings = []
    findings.append([])
    findings.append([])
    findings.append([])
    findings.append([])

    # for i in range(10,51,10):
    #     print(i)
    #     scale_percent = i/100 # percent of original size
    #     width = int(frame.shape[1] * scale_percent)
    #     height = int(frame.shape[0] * scale_percent)
    #     dim = (width, height)
    #     frame_resized = cv2.resize(frame, dim, interpolation = cv2.INTER_CUBIC)

    #     frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    #     frame_gray = cv2.equalizeHist(frame_gray)

    #     for j in range (0,4):
    #         #-- Detect faces
    #         signs = cascade[j].detectMultiScale(frame_gray)
            
    #         if len(signs):

    #             print(signs)
    #             for (x,y,w,h) in signs:
    #                 #x,y,w,h = int(x/scale_percent),int(y/scale_percent), int((x+w)/scale_percent),int((y+h)/scale_percent)



    #                 findings[j].append((x,y,w,h))

    #                 # center = (x + w//2, y + h//2)
    #                 # frame_resized = cv2.ellipse(frame_resized, center, (w//2, h//2), 0, 0, 360, color[j], 4)
    #                 frame_resized_out = cv2.rectangle(frame_resized_out, (x,y),(w,h),color[j], 4)
    #                 #faceROI = frame_gray[y:y+h,x:x+w]
                


    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    
    imgContrast = constrastLimit(frame)
    imgSmoothed = smoothImage(frame)

    arrEdgeFunctions = [find_triangle,find_triangle,find_squares,find_stop]
    for j in range (0,4):
        #-- Detect faces
        signs = cascade[j].detectMultiScale(frame_gray,scaleFactor=1.05)
        
        if len(signs):
            for (x,y,w,h) in signs:
                findings[j].append((x,y,w,h))
                
                
                if j == 2:
                    cropped_image = yellow_Mask[y:y+h, x:x+w]
                else:
                    cropped_image = red_Mask[y:y+h, x:x+w]
                
                hasColor =  np.sum(cropped_image)

                if hasColor > 0 :
                    #Zeichnen
                    frame_resized_out = cv2.rectangle(frame, (x,y),(x+w,y+h),color[j], 4)


                # #Kantendetektion benutzen
                # print(arrEdgeFunctions[j](cropped_image))


                # cv2.imshow('Capture - Face detection', cropped_image)
                # ch = cv2.waitKey()


                
    #False Positves weg bekommen

        #Image auf Bouding Box Croppen
        #Kanten erkennung die wir vorher gemacht haben




    imS = cv2.resize(frame_resized_out, (960, 540)) 
    cv2.imshow('Capture - Face detection', imS)
    ch = cv2.waitKey()