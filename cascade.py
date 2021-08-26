import cv2
import numpy as np

if __name__ == "__main__": 
    #Create Cascade Objects
    vf_cascade = cv2.CascadeClassifier()
    vfa_cascade = cv2.CascadeClassifier()
    vfs_cascade = cv2.CascadeClassifier()
    stop_cascade = cv2.CascadeClassifier()

    #Load Cascade Files
    vf_cascade.load(cv2.samples.findFile(r".\trained_vf\cascade.xml"))
    vfa_cascade.load(cv2.samples.findFile(r".\trained_vfa\cascade.xml"))
    vfs_cascade.load(cv2.samples.findFile(r".\trained_vfs\cascade.xml"))
    stop_cascade.load(cv2.samples.findFile(r".\trained_stop\cascade.xml"))
    
    #Load Image
    frame = cv2.imread(r"C:\Users\bellmi2\Documents\BV-UNI\schilder\vfa_01.jpg")

    #Cascade array for every road sign 
    cascade = [vf_cascade,vfa_cascade,vfs_cascade,stop_cascade]

    #Colors for bounding boxex
            # Lila = VF,    Grün = VFA,   Gelb = VFS ,  blau = stop
    color = [(255, 0, 255),(102, 204, 0),(0, 255, 255),(51, 51,255)]



    #Color Mask for filtering
    lower_red = (161, 50, 40)  
    upper_red = (179, 255, 255)  
    lower_lightred = (0, 120, 40)  
    upper_lightred = (10, 255, 255) 
    lower_yellow = (17, 120, 20)  
    upper_yellow = (30, 255, 255)

    #Use color mask on raw image 
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_lightred, upper_lightred)
    yellow_Mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    red_Mask = cv2.bitwise_or(mask1, mask2)

    #Initializing of output image
    frame_resized_out = frame

    #Convert to greyscale and equalize Histogramm
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    #Main Loop for every road sign 
    for j in range (0,4):
        #Detect Signs with cascade
        signs = cascade[j].detectMultiScale(frame_gray,scaleFactor=1.05)
        
        #If one sign found
        if len(signs):
            for (x,y,w,h) in signs: 
                #Check if red or yellow and use color mask                         
                if j == 2:
                    cropped_image = yellow_Mask[y:y+h, x:x+w]
                else:
                    cropped_image = red_Mask[y:y+h, x:x+w]
                
                #Sum up all pixels
                hasColor =  np.sum(cropped_image)

                #If bigger than 0 one pixel has been red/yellow -> right color, draw bounding box
                if hasColor > 0 :
                    #draw
                    frame_resized_out = cv2.rectangle(frame, (x,y),(x+w,y+h),color[j], 4)


    #Output Image


    cv2.namedWindow('finalImg', cv2.WINDOW_NORMAL)
    cv2.imshow("finalImg",frame_resized_out)
    ch = cv2.waitKey()