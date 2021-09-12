import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime

#Getting the face encodes of each image i have in my own dataset
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

#Marking the attendance in the excel sheet
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            nowTime = datetime.now()
            nowDate = datetime.today()
            dttime = nowTime.strftime('%H:%M:%S')
            dtToday = nowDate.strftime('%d:%B:%Y')
            start = '13:00:00'
            end = '14:00:00'
            if dttime > start and dttime < end:
                status = 'On Time'
            elif dttime < start:
                status = "Early!"
            else:
                status = "Late!"
            f.writelines(f'\n{name}, {dtToday}, {dttime}, {status}')


#Making a frame arround the detected face
def makeAFaceFrame(name):
    y1, x2, y2, x1 = faceLoc
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.rectangle(img, (x1, (y2 - 22)), (x2, y2), (0, 0, 255), cv2.FILLED)
    cv2.putText(img, name, ((x1 + 6), (y2 - 6)), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), 2)

path = 'ImageAttendance'
images = []
classNames = []

myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg) #Zeyad Sultan.jpeg
    classNames.append(os.path.splitext(cl)[0]) #Zeyad Sultan

encodeListKnown = findEncodings(images)
print('Encoding finished!')

cap = cv2.VideoCapture(0)

while True:
    succes, img = cap.read()
    #imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    #imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurrLoc = face_recognition.face_locations(img)
    imgCurrEncode = face_recognition.face_encodings(img, facesCurrLoc)

    for encodeFace, faceLoc in zip(imgCurrEncode, facesCurrLoc):
        #matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex] <= 0.5:
            name = classNames[matchIndex].upper()
            print(name)
            makeAFaceFrame(name)
            markAttendance(name)
        else:
            makeAFaceFrame("Unknown")
    cv2.imshow('Camera', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
