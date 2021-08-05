import matplotlib.pyplot as plt  # libary for showing image
import numpy as np  # numpy libary
from manual import *
from libary import *
import cv2
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

def colorFilter(img):
    height, width = img.shape[:2]
    #frame = cv2.GaussianBlur(img, (3,3), 0) 
    # imgS = cv2.resize(img, (int(width/5), int(height/5)))
    # grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # laplacian = cv2.Laplacian(grayImg,cv2.CV_64F, ksize=5)
    # median = cv2.medianBlur(grayImg, 3)
    # ret,th = cv2.threshold(median, 30 , 80, cv2.THRESH_BINARY)
    # edges = cv2.Canny(median,50, 70 )

    # Red color
    lower_red = (161, 50, 40)  # (0, 40, 100) S->auf130??
    upper_red = (179, 255, 255)  # 70, 125)#für bilder s&v auf 255, 255)#
    lower_lightred = (0, 120, 40)  # (0, 40, 100) S->auf130??
    upper_lightred = (10, 255, 255) 
    lower_yellow = (17, 120, 20)  # (20, 140, 50)
    upper_yellow = (30, 255, 255)
    lower_white = (0,0,128)
    upper_white = (255,255,255)
    lower_black = (0,0,0)
    upper_black = (170,150,50)



    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_red, upper_red)
    # mask2 = cv2.inRange(hsv, lower_lightred, upper_lightred)
    # mask3 = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # mask4 = cv2.inRange(hsv, lower_white, upper_white)
    # mask5 = cv2.inRange(hsv, lower_black, upper_black)

    # mask = cv2.bitwise_or(mask1, mask2)
    # mask = cv2.bitwise_or(mask, mask3)
    # mask = cv2.bitwise_or(mask, mask4)
    # mask = cv2.bitwise_or(mask, mask5)



    mask = cv2.add(mask, cv2.inRange(hsv, lower_lightred, upper_lightred))
    mask = cv2.add(mask, cv2.inRange(hsv, lower_yellow, upper_yellow))
    mask = cv2.add(mask, cv2.inRange(hsv, lower_white, upper_white))
    mask = cv2.add(mask, cv2.inRange(hsv, lower_black, upper_black))
    #mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    return mask

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

def find_triangle(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    triangle = []
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
                if len(cnt) == 3 and cv2.contourArea(cnt) > 400 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 3], cnt[(i+2) % 3] ) for i in xrange(3)])
                    if max_cos < 0.65 and max_cos >0.35 :
                        triangle.append(cnt)
    return triangle

def find_stop(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    stop = []
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
                if len(cnt) == 8 and cv2.contourArea(cnt) > 400 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 8], cnt[(i+2) % 8] ) for i in xrange(8)])
                    if max_cos < 0.85 and max_cos >0.55:
                        stop.append(cnt)
    return stop

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

def removeSmallComponents(image, threshold):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    img2 = np.zeros((output.shape),dtype = np.uint8)
    #for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2

def psf2otf(psf, outSize=None):
    # Prepare psf for conversion
    data = prepare_psf(psf, outSize)

    # Compute the OTF
    otf = np.fft.fftn(data)

    return np.complex64(otf)

def prepare_psf(psf, outSize=None, dtype=None):
    if not dtype:
        dtype=np.float32

    psf = np.float32(psf)

    # Determine PSF / OTF shapes
    psfSize = np.int32(psf.shape)
    if not outSize:
        outSize = psfSize
    outSize = np.int32(outSize)

    # Pad the PSF to outSize
    new_psf = np.zeros(outSize, dtype=dtype)
    new_psf[:psfSize[0],:psfSize[1]] = psf[:,:]
    psf = new_psf

    # Circularly shift the OTF so that PSF center is at (0,0)
    shift = -(psfSize / 2)
    shift = shift.astype(int)
    psf = circshift(psf, shift)

    return psf

# Circularly shift array
def circshift(A, shift):
    for i in xrange(shift.size):
        A = np.roll(A, shift[i], axis=i)
    return A


def l0_smoothing(img):
    # L0 minimization parameters
    kappa = 2.0
    _lambda = 2e-2

    N, M, D = np.int32(img.shape)
    S = np.float32(img) / 256
     # Compute image OTF
    size_2D = [N, M]
    fx = np.int32([[1, -1]])
    fy = np.int32([[1], [-1]])
    otfFx = psf2otf(fx, size_2D)
    otfFy = psf2otf(fy, size_2D)

    # Compute F(I)
    FI = np.complex64(np.zeros((N, M, D)))
    FI[:,:,0] = np.fft.fft2(S[:,:,0])
    FI[:,:,1] = np.fft.fft2(S[:,:,1])
    FI[:,:,2] = np.fft.fft2(S[:,:,2])

    # Compute MTF
    MTF = np.power(np.abs(otfFx), 2) + np.power(np.abs(otfFy), 2)
    MTF = np.tile(MTF[:, :, np.newaxis], (1, 1, D))

    # Initialize buffers
    h = np.float32(np.zeros((N, M, D)))
    v = np.float32(np.zeros((N, M, D)))
    dxhp = np.float32(np.zeros((N, M, D)))
    dyvp = np.float32(np.zeros((N, M, D)))
    FS = np.complex64(np.zeros((N, M, D)))

    # Iteration settings
    beta_max = 1e5
    beta = 2 * _lambda
    iteration = 0
    # Iterate until desired convergence in similarity
    while beta < beta_max:
        # compute dxSp
        h[:,0:M-1,:] = np.diff(S, 1, 1)
        h[:,M-1:M,:] = S[:,0:1,:] - S[:,M-1:M,:]

        # compute dySp
        v[0:N-1,:,:] = np.diff(S, 1, 0)
        v[N-1:N,:,:] = S[0:1,:,:] - S[N-1:N,:,:]

        # compute minimum energy E = dxSp^2 + dySp^2 <= _lambda/beta
        t = np.sum(np.power(h, 2) + np.power(v, 2), axis=2) < _lambda / beta
        t = np.tile(t[:, :, np.newaxis], (1, 1, 3))

        # compute piecewise solution for hp, vp
        h[t] = 0
        v[t] = 0

        ### Step 2: estimate S subproblem

        # compute dxhp + dyvp
        dxhp[:,0:1,:] = h[:,M-1:M,:] - h[:,0:1,:]
        dxhp[:,1:M,:] = -(np.diff(h, 1, 1))
        dyvp[0:1,:,:] = v[N-1:N,:,:] - v[0:1,:,:]
        dyvp[1:N,:,:] = -(np.diff(v, 1, 0))
        normin = dxhp + dyvp

        FS[:,:,0] = np.fft.fft2(normin[:,:,0])
        FS[:,:,1] = np.fft.fft2(normin[:,:,1])
        FS[:,:,2] = np.fft.fft2(normin[:,:,2])


        # solve for S + 1 in Fourier domain
        denorm = 1 + beta * MTF
        FS[:,:,:] = (FI + beta * FS) / denorm

        # inverse FFT to compute S + 1
        S[:,:,0] = np.float32((np.fft.ifft2(FS[:,:,0])).real)
        S[:,:,1] = np.float32((np.fft.ifft2(FS[:,:,1])).real)
        S[:,:,2] = np.float32((np.fft.ifft2(FS[:,:,2])).real)

        # update beta for next iteration
        beta *= kappa
        iteration += 1

    # Rescale image
    S = S * 256
    return S

def drawBoundingBox(cnt,boundingBoxCoordinates):
    for i in cnt:
        boundRect = cv2.boundingRect(i)
        boundingList = list(boundRect)
        boundingList[0] = max(0, boundingList[0]-int(0.2*boundingList[2]))
        boundingList[1] = max(0, boundingList[1]-int(0.2*boundingList[3]))
        boundingList[2] = int(boundingList[2]*1.4)
        boundingList[3] = int(boundingList[3]*1.4)
        boundRect = tuple(boundingList)
        boundingBoxCoordinates.append(boundRect)
        
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
    # PATH = r"C:\Users\bellmi2\Documents\BV-UNI\schilder\bilder\stop2.png"
    PATH = r"C:\Users\bellmi2\Documents\BV-UNI\schilder\bilder\stop1.png"

    libary = Libary(PATH)
    manual = Manual(PATH)
    img = cv2.imread(PATH)
    scale_percent = 100 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)


    imS = cv2.resize(img, (960, 540)) 
    cv2.imshow('squares', imS)
    ch = cv2.waitKey()


    #imgS = cv2.ximgproc.l0Smooth(img,0.002,1)

    imgContrast = constrastLimit(img)
    imgSmoothed = smoothImage(imgContrast)
    

    # imS = cv2.resize(imgSmoothed, (960, 540)) 
    # cv2.imshow('squares', imS)
    # ch = cv2.waitKey()


    #binary_image = removeSmallComponents(imgSmoothed, 300)
    binary_image = imgSmoothed

    imS = cv2.resize(binary_image, (960, 540)) 
    cv2.imshow('squares', imS)
    ch = cv2.waitKey()
    res = cv2.bitwise_and(binary_image, binary_image, mask=colorFilter(img))



    # imgS = cv2.resize(res, (int(width/5), int(height/5)))
    # cv2.imshow('squares', imgS)
    # ch = cv2.waitKey()
    boundingBoxCoordinates = []
    squares = find_squares(binary_image)
    if(squares):
        drawBoundingBox(squares,boundingBoxCoordinates)
    else:
        print("Kein Rechteck gefunden.")

    triangle = find_triangle(res)
    if(triangle):
        drawBoundingBox(triangle,boundingBoxCoordinates)
    else:
        print("Kein Dreieck gefunden.")
        
    stop = find_stop(res)
    
    if(stop):
        drawBoundingBox(stop,boundingBoxCoordinates)
    else:
        print("Kein Stop gefunden.")
    print(boundingBoxCoordinates)
    boundingBoxCoordinates = filter(None, boundingBoxCoordinates)
    for j in boundingBoxCoordinates:
        print(j)
        cropped_image = img[j[1]:j[1]+j[3], j[0]:j[0]+j[2]]
        # imS = cv2.resize(cropped_image, (960, 540)) 
        # cv2.imshow('squares', cropped_image)
        # ch = cv2.waitKey()
    
        # width = int(cropped_image.shape[1] * 130/cropped_image.shape[1] )
        # height = int(cropped_image.shape[0] * 120 / cropped_image.shape[0])
        # dim = (width, height)
        # cropped_image = cv2.resize(cropped_image, dim, interpolation = cv2.INTER_CUBIC)
        # cv2.imshow('squares', cropped_image)
        # ch = cv2.waitKey() 
        face_cascade = cv2.CascadeClassifier()
        face_cascade.load(cv2.samples.findFile(r"C:\Users\bellmi2\Documents\BV-UNI\training\trained_stop\cascade.xml"))
        frame_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)
        #-- Detect faces
        faces = face_cascade.detectMultiScale(frame_gray)
        print(faces)
        if(len(faces)!=0):
            for (x,y,w,h) in faces:
                center = (x + w//2, y + h//2)
                frame = cv2.ellipse(cropped_image, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
                faceROI = frame_gray[y:y+h,x:x+w]
        cv2.imshow('squares', cropped_image)
        ch = cv2.waitKey() 
    