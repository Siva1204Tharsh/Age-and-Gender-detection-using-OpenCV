### 01.Importing the libraries
import cv2
import numpy as np

###########################################################################################################################################################################################

### 02.Load the Pre-trained Model for Age and Gender Detection and prediction
face_proto="opencv_face_detector.pbtxt" #Configuration file for face detection that describes the model's architecture
face_model="opencv_face_detector_uint8.pb" # TensorFlow  Pre-trained model for face detection
age_proto="age_deploy.prototxt" # file that describes the model's architecture
age_model="age_net.caffemodel" # Pre-trained Caffe Model file that contains the trained weights
gender_proto="gender_deploy.prototxt" # file that describes the model's architecture
gender_model="gender_net.caffemodel" # Pre-trained Caffe Model file that contains the trained weights

########################################################################################################################################################################################### 

### 03.Define mean values and other parameters for age and gender
#the input images color values are adjusted  to models expected range
MODEL_MEAN_VALUES =(78.4263377603, 87.7689143744, 114.895847746)#mean=BGR (Blue, Green, Red)

###########################################################################################################################################################################################

###  04.Highest probability for age and gender
age_buckets=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list=['Male', 'Female']

###########################################################################################################################################################################################

### 05.Load the face detector and the age and gender models
face_net = cv2.dnn.readNet( face_model,face_proto) #face detection model
age_net = cv2.dnn.readNet (age_model, age_proto) #age prediction model
gender_net = cv2.dnn.readNet( gender_model, gender_proto) #gender prediction model

#Setup video capture
cap = cv2.VideoCapture(0)

###########################################################################################################################################################################################

### 06.Function to detect faces and predict age and gender
def highlight_face(net,frame,conf_threshold=0.7):  ## how many percentage of face about to be detected
    """Dectect faces in the frame and returns the faces rectangles."""
    frame_opencv_dnn=frame.copy() # original frame no changes
    frame_height=frame_opencv_dnn.shape[0] #height of the frame
    frame_width=frame_opencv_dnn.shape[1] #width of the frame
    #resize , normalize and convert to the required 
    #sclae factor-1.0,frame size(300,300),mean subraction values(BGR)-[104, 117, 123],True-convert the image color{BGR - >RGB},Flase-Orginal image demensions is  no change  but the frame is resized
    blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], True, False) # DL -RGB ,OPENCV -BGR

    net.setInput(blob)
    detections = net.forward() # Class ID , Confidence level, Coordinates of the detected faces
    face_boxes = []

    for i in range(detections.shape[2]): # number of faces detected
        confidence = detections[0, 0, i, 2] # confidence of the detected face
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width) #X-coordinate of the top-left corner of the detected face
            y1 = int(detections[0, 0, i, 4] * frame_height)#y-coordinate of the top-left corner of the detected face
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height/150))) #draw rectangles on the frame

    return  frame_opencv_dnn,face_boxes


###########################################################################################################################################################################################

### 07. Predict age and gender
while True:
    #Capture frame-by-frame
    ret, frame = cap.read() #read the video frame
    if not ret:
        break

    #Dectect faces in the frame 
    result_img ,face_boxes = highlight_face(face_net,frame) #face_net -pre-trained model ,frame -original frame
    if not face_boxes:
        print("No face detected")
        # continue
    
    for face_box in face_boxes:
        #Get the face ROI
        face=frame[max(0,face_box[1]):min(face_box[3],frame.shape[0]-1),
                   max(0,face_box[0]):min(face_box[2],frame.shape[1]-1)]
        #face_box[0]: x-coordinate of the top-left corner of the detected face
        #face_box[1]: y-coordinate of the top-left corner of the detected face
        #face_box[2]: x-coordinate of the bottom-right corner of the detected face
        #face_box[3]: y-coordinate of the bottom-right corner of the detected face

        # Prepare blob for age and gender prediction
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),MODEL_MEAN_VALUES,swapRB=False)#swapRB=False-BGR to RGB

        #predict gender
        gender_net.setInput(blob)
        gender_out = gender_net.forward()
        gender = gender_list[gender_out[0].argmax()]

        #predict age
        age_net.setInput(blob)
        age_out = age_net.forward()
        age = age_buckets[age_out[0].argmax()]
        
        
        
        #draw the results on the frame
        cv2.putText(result_img,f'{gender},{age}',(face_box[0],face_box[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
        

    cv2.imshow("Face and Age and Gender Detection", result_img)
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()    
cv2.destroyAllWindows()
  

###########################################################################################################################################################################################