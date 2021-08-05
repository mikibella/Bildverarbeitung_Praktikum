import cv2



if __name__ == "__main__": 
    face_cascade = cv2.CascadeClassifier()
    face_cascade.load(cv2.samples.findFile(r"C:\Users\bellmi2\Documents\BV-UNI\training\trained_vfs\cascade.xml"))
    frame = cv2.imread(r"C:\Users\bellmi2\Documents\BV-UNI\schilder\vfs_06.jpg")
    #frame = cv2.imread(r"C:\Users\bellmi2\Documents\BV-UNI\schilder\mr\vorfahrt_gewaehren_2.jpg")



    for i in range(20,151,10):
        print(i)
        scale_percent = i # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        frame_resized = cv2.resize(frame, dim, interpolation = cv2.INTER_CUBIC)

        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)
        #-- Detect faces
        faces = face_cascade.detectMultiScale(frame_gray)
        
        if len(faces):
            print(faces)
            for (x,y,w,h) in faces:
                center = (x + w//2, y + h//2)
                frame = cv2.ellipse(frame_resized, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
                faceROI = frame_gray[y:y+h,x:x+w]
            break
        
    imS = cv2.resize(frame_resized, (960, 540)) 
    cv2.imshow('Capture - Face detection', imS)
    ch = cv2.waitKey()