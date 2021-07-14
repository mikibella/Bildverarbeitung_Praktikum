import cv2



if __name__ == "__main__": 
    face_cascade = cv2.CascadeClassifier()
    face_cascade.load(cv2.samples.findFile(r"C:\Users\bellmi2\Documents\BV-UNI\Bildverarbeitung_Praktikum\vfa_cascade.xml"))
    frame = cv2.imread(r"C:\Users\bellmi2\Documents\BV-UNI\schilder\vfa_09.jpg")
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    print(faces)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]
    
    imS = cv2.resize(frame, (960, 540)) 
    cv2.imshow('Capture - Face detection', imS)
    ch = cv2.waitKey()