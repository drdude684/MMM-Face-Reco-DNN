import face_recognition
import pickle
import time
import cv2
import signal
import os
import numpy
import base64
import math
from datetime import datetime
from utils.image import Image
from utils.arguments import Arguments
from utils.print import Print
from picamera2 import Picamera2
from libcamera import controls
from phue import Bridge

def signalHandler(signal, frame):
    global closeSafe
    closeSafe = True

def checkMonitorPower():
	global hueBridge,hueBridgeActivated, MonitorPowerStatus
	MonitorPowerStatus = False
	try:
		if not hueBridgeActivated:
			hueBridge=Bridge('192.168.88.10')
			hueBridgeActivated = True
		MonitorPowerStatus=hueBridge.get_light('On/Off plug smart mirror','on')
	except:
		pass
	return MonitorPowerStatus
    
def lockExposure():
    global picam2
    global controls_exptime,controls_angain
    Print.printJson("status", "obtaining Exposure value")
    picam2.stop()                        
    newcontrols = {"AeEnable":True,"ExposureValue":-1.0,"AfMode": controls.AfModeEnum.Continuous}
    picam2.configure(picam2.create_preview_configuration(main={"size": (resolution[0], resolution[1]), "format": "XRGB8888"}, controls=newcontrols))
    picam2.start()
    #time.sleep(0.1)
    time.sleep(0.5)
    md=picam2.capture_metadata()
    picam2.stop()                        
    #newcontrols = {"FrameRate":25,"AeEnable":False,"ExposureTime": md['ExposureTime'], "AnalogueGain": md['AnalogueGain']}
    newcontrols = {"AeEnable":False,"ExposureTime": md['ExposureTime'], "AnalogueGain": md['AnalogueGain'],"AfMode": controls.AfModeEnum.Continuous}
    Print.printJson("status", "locking Exposure value")
    Print.printJson("status","ExposureTime %f, AnalogueGain: %f"%(md['ExposureTime'],md['AnalogueGain']))
    controls_exptime=md['ExposureTime']
    controls_angain=md['AnalogueGain']
    #print("adjusting camera settings to:")
    #print(newcontrols)                        
    picam2.configure(picam2.create_preview_configuration(main={"size": (resolution[0], resolution[1]), "format": "XRGB8888"}, controls=newcontrols))
    picam2.start()                        
    time.sleep(0.1)

def updateStats(existing_aggregate, new_value):
    #straight outta Wikipedia
    (count, mean, M2) = existing_aggregate
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2
    return (count, mean, M2)

def currentMinute():
    return datetime.now().minute
        
signal.signal(signal.SIGINT, signalHandler)
closeSafe = False

# prepare console arguments
Arguments.prepareRecognitionArguments()

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
Print.printJson("status", "loading encodings + face detector...")
data = pickle.loads(open(Arguments.get("encodings"), "rb").read())
detector = cv2.CascadeClassifier(Arguments.get("cascade"))

# initialize the video stream
Print.printJson("status", "starting video stream...")
processWidth = Arguments.get("processWidth")
resolution = Arguments.get("resolution").split(",")
resolution = (int(resolution[0]), int(resolution[1]))
Print.printJson("status", "resolution: %dx%d"%(resolution[0],resolution[1]))
Print.printJson("status", "resample to width: %d"%processWidth)

picam2 = Picamera2()

lockExposure()

crop_top=0
crop_bottom=0

if Arguments.get("output") != 1:
    crop_top=0.25
    crop_bottom=0.25
    

# variable for prev names
prevNames = []

# create unknown path if needed
if Arguments.get("extendDataset") is True:
    unknownPath = os.path.dirname(Arguments.get("dataset") + "unknown/")
    try:
        os.stat(unknownPath)
    except:
        os.mkdir(unknownPath)

tolerance = float(Arguments.get("tolerance"))
if Arguments.get("output") != 1:
    lastMonitorPowerCheck=time.time()
    hueBridgeActivated=False
    checkMonitorPower()

decimatedStats = [None,None]
staggeredInitHandled=False
bufferIndex=0
doFaceIdentification = True
lastMotionDetectionTime=0
lastFaceDetectionTime=0
exposureRefreshInterval=2000 # in frames; frames with detected movement do not count
deadFrames=30 # frames before any motion is considered
minChangedPoints=10
bounds = None
statsString=""
count=0
lastDetectionDuration=0
controls_exptime=0
controls_angain=0

lastSaveMinute=61

verbose=False

# loop over frames from the video file stream
while True:

    if verbose: Print.printJson("status", "loop start")
        
    loopstart=time.time()
    
    if Arguments.get("output") != 1:
        if MonitorPowerStatus:
            if time.time()-lastMonitorPowerCheck>10.0:
                if verbose: Print.printJson("status", "monitor was on, performing regular check to confirm it still is")
                lastMonitorPowerCheck=time.time()
                checkMonitorPower()
        else:
            while not MonitorPowerStatus:
                lastMonitorPowerCheck=time.time()
                checkMonitorPower()
                time.sleep(0.25)
                if verbose: Print.printJson("status", "monitor is not on, will sleep and re-check soon")
            if verbose: Print.printJson("status", "monitor has now been switched on, will now continue by restarting camera")
            decimatedStats = [None,None]
            staggeredInitHandled = False
            lockExposure()
    if verbose: Print.printJson("status", "monitor is on")

    # read the frame
    originalFrame = picam2.capture_array()
    
    # adjust image brightness and contrast
    originalFrame = Image.adjust_brightness_contrast(
        originalFrame, Arguments.get("brightness"), Arguments.get("contrast")
    )

    if Arguments.get("rotateCamera") >= 0 and Arguments.get("rotateCamera") <= 2:
        originalFrame = cv2.rotate(originalFrame, Arguments.get("rotateCamera"))
    
    originalFrame=originalFrame[int(resolution[0]*crop_top):int(resolution[0]*(1-crop_bottom)),:]

    # resize image if we wanna process a smaller image
    if processWidth != resolution[0] and processWidth != 0:
        frame = Image.resize(originalFrame, width=processWidth)
    else:
        frame = originalFrame

    if verbose: Print.printJson("status", "frame obtained")
     
    decimationChannelIndex=2
    decimationStepSize=25
    horizontalExtraMargin=int(processWidth/(15*decimationStepSize))
    #movementThreshold=5 # in sigmas 
    movementThreshold=8 # in sigmas 
    undecimatedFrame=numpy.copy(frame)
    bounds=[0,frame.shape[0],frame.shape[1]]
    decimatedFrame=frame[int(decimationStepSize/2):frame.shape[0]-1:decimationStepSize,int(decimationStepSize/2):frame.shape[1]-1:decimationStepSize,decimationChannelIndex].astype(float)
    decimatedFrame+=frame[int(decimationStepSize/2)-1:frame.shape[0]-2:decimationStepSize,int(decimationStepSize/2):frame.shape[1]-1:decimationStepSize,decimationChannelIndex].astype(float)
    decimatedFrame=frame[int(decimationStepSize/2)+1:frame.shape[0]-1:decimationStepSize,int(decimationStepSize/2):frame.shape[1]-1:decimationStepSize,decimationChannelIndex].astype(float)
    decimatedFrame=frame[int(decimationStepSize/2):frame.shape[0]-1:decimationStepSize,int(decimationStepSize/2)-1:frame.shape[1]-2:decimationStepSize,decimationChannelIndex].astype(float)
    decimatedFrame=frame[int(decimationStepSize/2):frame.shape[0]-1:decimationStepSize,int(decimationStepSize/2)+1:frame.shape[1]-1:decimationStepSize,decimationChannelIndex].astype(float)
    doFaceIdentification=True
    if decimatedStats[0] is None:
        # initialize buffer
        decimatedStats[0]=(1,decimatedFrame,numpy.zeros((decimatedFrame.shape[0],decimatedFrame.shape[1]),dtype='float'))
        decimatedStats[1]=(1,decimatedFrame,numpy.zeros((decimatedFrame.shape[0],decimatedFrame.shape[1]),dtype='float'))
    else:
        if verbose: Print.printJson("status", "processing motion detection data")
        (count, mean, M2) = decimatedStats[bufferIndex]
        if not staggeredInitHandled:
            if count>exposureRefreshInterval/2-2:
                decimatedStats[1]=(1,decimatedFrame,numpy.zeros((decimatedFrame.shape[0],decimatedFrame.shape[1]),dtype='float'))
                staggeredInitHandled = True
                
        delta=numpy.absolute(numpy.divide(numpy.subtract(decimatedFrame,mean),numpy.sqrt(M2)+0.0001)*math.sqrt(count))
        statsString=("mean delta: %2.2f, delta range: (%2.2f-%2.2f)"%(numpy.mean(delta),numpy.min(delta),numpy.max(delta))) 
        #print(statsString)
        if Arguments.get("output") != 1:
            delta[numpy.where(delta<=movementThreshold)]=0
        else:
            delta[numpy.where(delta<=numpy.mean(delta)*3)]=0 #this works better in the development setup
        delta[numpy.where(delta>0)]=255
        #draw markers
        undecimatedFrame[int(decimationStepSize/2):frame.shape[0]-1:decimationStepSize,int(decimationStepSize/2):frame.shape[1]-1:decimationStepSize,0]=delta
        undecimatedFrame[int(decimationStepSize/2)-1:frame.shape[0]-1:decimationStepSize,int(decimationStepSize/2):frame.shape[1]-1:decimationStepSize,0]=delta
        undecimatedFrame[int(decimationStepSize/2)+1:frame.shape[0]-1:decimationStepSize,int(decimationStepSize/2):frame.shape[1]-1:decimationStepSize,0]=delta
        undecimatedFrame[int(decimationStepSize/2):frame.shape[0]-1:decimationStepSize,int(decimationStepSize/2)-1:frame.shape[1]-1:decimationStepSize,0]=delta
        undecimatedFrame[int(decimationStepSize/2):frame.shape[0]-1:decimationStepSize,int(decimationStepSize/2)+1:frame.shape[1]-1:decimationStepSize,0]=delta
        undecimatedFrame[int(decimationStepSize/2):frame.shape[0]-1:decimationStepSize,int(decimationStepSize/2):frame.shape[1]-1:decimationStepSize,1]=delta
        undecimatedFrame[int(decimationStepSize/2)-1:frame.shape[0]-1:decimationStepSize,int(decimationStepSize/2):frame.shape[1]-1:decimationStepSize,1]=delta
        undecimatedFrame[int(decimationStepSize/2)+1:frame.shape[0]-1:decimationStepSize,int(decimationStepSize/2):frame.shape[1]-1:decimationStepSize,1]=delta
        undecimatedFrame[int(decimationStepSize/2):frame.shape[0]-1:decimationStepSize,int(decimationStepSize/2)-1:frame.shape[1]-1:decimationStepSize,1]=delta
        undecimatedFrame[int(decimationStepSize/2):frame.shape[0]-1:decimationStepSize,int(decimationStepSize/2)+1:frame.shape[1]-1:decimationStepSize,1]=delta
        undecimatedFrame[int(decimationStepSize/2):frame.shape[0]-1:decimationStepSize,int(decimationStepSize/2):frame.shape[1]-1:decimationStepSize,2]=delta
        undecimatedFrame[int(decimationStepSize/2)-1:frame.shape[0]-1:decimationStepSize,int(decimationStepSize/2):frame.shape[1]-1:decimationStepSize,2]=delta
        undecimatedFrame[int(decimationStepSize/2)+1:frame.shape[0]-1:decimationStepSize,int(decimationStepSize/2):frame.shape[1]-1:decimationStepSize,2]=delta
        undecimatedFrame[int(decimationStepSize/2):frame.shape[0]-1:decimationStepSize,int(decimationStepSize/2)-1:frame.shape[1]-1:decimationStepSize,2]=delta
        undecimatedFrame[int(decimationStepSize/2):frame.shape[0]-1:decimationStepSize,int(decimationStepSize/2)+1:frame.shape[1]-1:decimationStepSize,2]=delta
        if (numpy.count_nonzero(delta) < minChangedPoints) or (count<deadFrames):
            if verbose: Print.printJson("status", "no motion detected")
            if time.time()-lastMotionDetectionTime>1:
                if (count>exposureRefreshInterval):
                    #refresh of camera exposure settings (sometimes) and stats (always)
                    if verbose: Print.printJson("status", "refreshing exposure settings and statistics")
                    #if time.time()-lastMotionDetectionTime>300:
                    if True:
                        lockExposure()
                    decimatedStats[bufferIndex]=(1,decimatedFrame,numpy.zeros((decimatedFrame.shape[0],decimatedFrame.shape[1]),dtype='float'))
                    bufferIndex=1-bufferIndex
                else:                        
                    #update stats
                    if verbose: Print.printJson("status", "updating statistics")
                    decimatedStats[0] = updateStats(decimatedStats[0],decimatedFrame)
                    decimatedStats[1] = updateStats(decimatedStats[1],decimatedFrame)
                    #print(count)
            bounds = None
            doFaceIdentification=False
        else:
            if verbose: Print.printJson("status", "motion detected")
            lastMotionDetectionTime=time.time()
            nzv=numpy.nonzero(delta)
            # horizontal range will only consider top section of affected area, as most people are wider at the bottom
            verticalBoundary=min(nzv[0])+int((max(nzv[0])-min(nzv[0]))*0.4)                
            nzh=numpy.nonzero(delta[0:verticalBoundary,:])
            if len(nzh[1])<3:
                if verbose: Print.printJson("status", "insufficient #points in top segment, will consider whole area")
                nzh=nzv
            if verbose: Print.printJson("status", "boundaries established, step 1 (%d,%d)"%(len(nzh[1]),len(nzv[0])))
            #print(verticalBoundary)
            bounds=[max(0,min(nzv[0])-1)*decimationStepSize,
                min(undecimatedFrame.shape[0]-1,max(nzv[0])*decimationStepSize+decimationStepSize-1),
                max(0,min(nzh[1])-horizontalExtraMargin)*decimationStepSize,
                min(undecimatedFrame.shape[1]-1,max(nzh[1]+horizontalExtraMargin)*decimationStepSize+decimationStepSize-1)];
            #print(bounds)
            if verbose: Print.printJson("status", "boundaries established, step 2")
            frame=frame[bounds[0]:bounds[1],bounds[2]:bounds[3]]
            if verbose: Print.printJson("status", "frame reduced")
            undecimatedFrame[bounds[0]:bounds[1],bounds[2],1]=255
            undecimatedFrame[bounds[0]:bounds[1],bounds[3],1]=255
            undecimatedFrame[bounds[0],bounds[2]:bounds[3],1]=255
            undecimatedFrame[verticalBoundary*decimationStepSize,bounds[2]:bounds[3],1]=0
            undecimatedFrame[bounds[1],bounds[2]:bounds[3],1]=255
            if verbose: Print.printJson("status", "reduced frame drawn on undecimatedframe")
            if time.time()-lastFaceDetectionTime>10:
                if verbose: Print.printJson("status", "motion detected, but last face detection too long ago, adding frame to statistics buffer")
                if (count>exposureRefreshInterval):
                    #refresh of camera exposure settings and stats
                    if verbose: Print.printJson("status", "refreshing exposure settings and statistics")
                    lockExposure()
                    decimatedStats[bufferIndex]=(1,decimatedFrame,numpy.zeros((decimatedFrame.shape[0],decimatedFrame.shape[1]),dtype='float'))
                    bufferIndex=1-bufferIndex
                else:                        
                    #update stats
                    if verbose: Print.printJson("status", "updating statistics")
                    decimatedStats[0] = updateStats(decimatedStats[0],decimatedFrame)
                    decimatedStats[1] = updateStats(decimatedStats[1],decimatedFrame)
                    
    if verbose: Print.printJson("status", "motion detection done")
    
    undecimatedFrame=cv2.flip(undecimatedFrame,1)

    if doFaceIdentification:
        if verbose: Print.printJson("status", "initiating face identificaction")
        if Arguments.get("method") == "dnn":
            # load the input image and convert it from BGR (OpenCV ordering)
            # to dlib ordering (RGB)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input image
            boxes = face_recognition.face_locations(
                rgb, model=Arguments.get("detectionMethod")
            )
        elif Arguments.get("method") == "haar":
            # convert the input frame from (1) BGR to grayscale (for face
            # detection) and (2) from BGR to RGB (for face recognition)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # detect faces in the grayscale frame
            rects = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            # OpenCV returns bounding box coordinates in (x, y, w, h) order
            # but we need them in (top, right, bottom, left) order, so we
            # need to do a bit of reordering
            boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
            
        if len(boxes)>0:
            if verbose: Print.printJson("status", "identified at least one face")
            lastFaceDetectionTime=time.time()
        else:
            if verbose: Print.printJson("status", "no faces identified")

        # compute the facial embeddings for each face bounding box
        encodings = face_recognition.face_encodings(rgb, boxes)
        if verbose: Print.printJson("status", "faces encoded")
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # compute distances between this encoding and the faces in dataset
            distances = face_recognition.face_distance(data["encodings"], encoding)

            minDistance = 1.0
            if len(distances) > 0:
                # the smallest distance is the closest to the encoding
                minDistance = min(distances)

            # save the name if the distance is below the tolerance
            if minDistance < tolerance:
                idx = numpy.where(distances == minDistance)[0][0]
                name = data["names"][idx]
                if verbose: Print.printJson("status", "face recognition: recognized %s"%name)
            else:
                name = "unknown"

            # update the list of names
            names.append(name)
            
        if verbose: Print.printJson("status", "face recognition done")

        if False: # original code
            # loop over the recognized faces
            for (top, right, bottom, left), name in zip(boxes, names):
                # draw the predicted face name on the image
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                txt = name + " (" + "{:.2f}".format(minDistance) + ")"
                cv2.putText(
                    frame, txt, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2                    
                )
        else:
            if verbose: Print.printJson("status", "updating display data")
            for (top, right, bottom, left), name in zip(boxes, names):                
                # draw the predicted face name on the image
                cv2.rectangle(undecimatedFrame, (undecimatedFrame.shape[1]-bounds[2]-right, bounds[0]+top), (undecimatedFrame.shape[1]-bounds[2]-left, bounds[0]+bottom), (0,255,255), 2)
                #y = top - 15 if top - 15 > 15 else top + 15
                y=bottom+20
                txt = name + " (" + "{:.2f}".format(minDistance) + ")"
                cv2.putText(
                    undecimatedFrame, txt, (undecimatedFrame.shape[1]-bounds[2]-right, bounds[0]+y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2
                )
        lastDetectionDuration=(time.time()-loopstart)*1000
    cv2.putText(
        undecimatedFrame, "Time: %3.0f ms, ID time: %3.0f ms, Frame: %d [%d]"%((time.time()-loopstart)*1000,lastDetectionDuration,count,bufferIndex), (20,undecimatedFrame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2
    )
    cv2.putText(
        undecimatedFrame, statsString, (20,undecimatedFrame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2
    )
    cv2.putText(
        undecimatedFrame, "ExpTime: %f , AnGain %f"%(controls_exptime,controls_angain), (20,undecimatedFrame.shape[0]-70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2
    )

    # display the image to our screen
    if verbose: Print.printJson("status", "exporting views")
    if Arguments.get("output") == 1:
        cv2.imshow("Delta", undecimatedFrame)

    if Arguments.get("outputmm") == 1:
        retval, buffer = cv2.imencode('.jpg', undecimatedFrame)
        #retval, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode()
        Print.printJson("camera_image", {"image": jpg_as_text})

    if verbose: Print.printJson("status", "generating login/logout information")
    logins = []
    logouts = []
    # Check which names are new login and which are new logout with prevNames
    for n in names:
        if prevNames.__contains__(n) == False and n is not None:
            logins.append(n)

            # if extendDataset is active we need to save the picture
            if (Arguments.get("extendDataset") is True) and (lastSaveMinute != currentMinute()):
                # set correct path to the dataset
                path = os.path.dirname(Arguments.get("dataset") + "/" + n + "/")

                today = datetime.now()
                cv2.imwrite(
                    path + "/" + n + "_" + today.strftime("%Y%m%d_%H%M%S") + ".jpg",
                    originalFrame,
                )
                lastSaveMinute = currentMinute()
    for n in prevNames:
        if names.__contains__(n) == False and n is not None:
            logouts.append(n)

    # send inforrmation to prompt, only if something has changes
    if logins.__len__() > 0:
        Print.printJson("login", {"names": logins})

    if logouts.__len__() > 0:
        Print.printJson("logout", {"names": logouts})

    # set this names as new prev names for next iteration
    prevNames = names

    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q") or closeSafe == True:
        break
        
    if verbose: Print.printJson("status", "processing took %3.0f ms"%((time.time()-loopstart)*1000))

    time.sleep(Arguments.get("interval") / 1000)

# do a bit of cleanup
picam2.stop()
cv2.destroyAllWindows()
