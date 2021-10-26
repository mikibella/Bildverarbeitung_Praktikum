

# generate positive samples from the annotations to get a vector file using:
#opencv_createsamples.exe -info pos.txt -w 24 -h 24 -num 1000 -vec pos.vec


#training arguments:
##opencv_traincascade.exe -data trained_stop/-vec pos_stop.vec -bg neg.txt 
#-precalcValBufSize 3000 -precalcIdxBufSize 3000 -numPos 500 -numNeg 1000 -numStages 12 -w 24 -h 24 
# -maxFalseAlarmRate 0.3 -minHitRate 0.999
