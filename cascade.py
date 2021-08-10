import cv2



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


    frame = cv2.imread(r"C:\Users\bellmi2\Documents\BV-UNI\schilder\stop_02.jpg")
    #frame = cv2.imread(r"C:\Users\bellmi2\Documents\BV-UNI\schilder\mr\vorfahrt_gewaehren_2.jpg")

    
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

    for j in range (0,4):
        #-- Detect faces
        signs = cascade[j].detectMultiScale(frame_gray,scaleFactor=1.05)
        
        if len(signs):
            for (x,y,w,h) in signs:
                findings[j].append((x,y,w,h))
                
                #Zeichnen
                frame_resized_out = cv2.rectangle(frame, (x,y),(x+w,y+h),color[j], 4)
                
    #False Positves weg bekommen

        #Image auf Bouding Box Croppen
        #Kanten erkennung die wir vorher gemacht haben




    imS = cv2.resize(frame_resized_out, (960, 540)) 
    cv2.imshow('Capture - Face detection', imS)
    ch = cv2.waitKey()