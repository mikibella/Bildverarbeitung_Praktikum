import os
import cv2
def generate_negative_description_file():
    folder = "vfs"
    # open the output file for writing. will overwrite all existing data in there
    with open('pos_vfs.txt', 'w') as f:
        # loop over all the filenames
        for filename in os.listdir(folder):
            img = cv2.imread(folder+"/"+filename)
            
            f.write(folder+'/' + filename + '  1  0 0 '+str(img.shape[1])+' '+str(img.shape[0]) +'\n')

if __name__ == "__main__": 
    generate_negative_description_file()


    # generate positive samples from the annotations to get a vector file using:
# $ C:/Users/Ben/learncodebygaming/opencv/build/x64/vc15/bin/opencv_createsamples.exe -info pos.txt -w 24 -h 24 -num 1000 -vec pos.vec

# train the cascade classifier model using:
# $ C:/Users/Ben/learncodebygaming/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data cascade/ -vec pos.vec -bg neg.txt -numPos 200 -numNeg 100 -numStages 10 -w 24 -h 24

# my final classifier training arguments:
# C:/Users/bellmi2/Downloads/opencv/build/x64/vc15/bin/opencv_traincascade.exe -data trained_stop/ -vec pos_stop.vec -bg neg.txt -precalcValBufSize 3000 -precalcIdxBufSize 3000 -numPos 500 -numNeg 1000 -numStages 12 -w 24 -h 24 -maxFalseAlarmRate 0.3 -minHitRate 0.999
