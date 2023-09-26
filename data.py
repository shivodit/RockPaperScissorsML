# this file hold functions for handling the dataset
import os
from sklearn.preprocessing import OneHotEncoder
import numpy as np, pandas as pd
import keras
import config
from cv2 import imread, resize
import cv2
from PIL import Image

class DataGenerator(keras.utils.Sequence) :
  
    def __init__(self, image_filenames, boxes, labels, type, batch_size, has_boxes=0,dataset=config.DATASET_DIR) :
        # pass complete data during init
        self.type = type # which folder to open
        self.image_filenames = image_filenames
        self.boxes = boxes # bounding boxes
        self.labels = labels # classes
        self.batch_size = batch_size
        self.has_boxes = has_boxes # badly named mode select for using different datasets
        self.dataset = dataset
        
    def __len__(self) :
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(int)
    
    
    def __getitem__(self, idx) :
        # selecting batch wise
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y_labels = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

        # bounding box, objectness and class labels
        if self.has_boxes == 0:
            batch_y_boxes = self.boxes[idx * self.batch_size : (idx+1) * self.batch_size]
            Targets = {
                "class_label": batch_y_labels.astype(int),
                "bounding_box": batch_y_boxes,
                "prob":batch_y_labels.any(axis=1).astype(int).to_numpy().reshape(-1,1)}  
        
        # objectness and class labels
        elif self.has_boxes == 1:
            Targets = {
                "class_label": batch_y_labels,
                "prob":batch_y_labels.any(axis=1).to_numpy().reshape(-1,1)}  

        # class labels
        else :
            Targets = batch_y_labels
       
        return np.array([
                resize(imread(os.path.join(self.dataset,self.type, str(file_name))), config.IMAGE_DIM)
                for file_name in batch_x])/255.0,  Targets
    

def get_data(type,ohe=None,dataset= config.DATASET_DIR):
    t = type
    df = pd.read_csv(os.path.join(dataset,t,"_annotations.csv"))

    if not ohe:
        ohe = OneHotEncoder(handle_unknown='ignore')
        ohe.fit(df.iloc[:,5].to_numpy().reshape(-1,1))
        import pickle
        with open('ohencoder.pickle','wb') as f:
            pickle.dump(ohe,f)
    
    df = df.to_numpy()
    labels = ohe.transform(df[:,5].reshape(-1,1)).toarray()
    df = np.delete(df, 5, 1)
    df = pd.DataFrame(np.append(df, labels, axis=1))
 
    for image in os.listdir(os.path.join(dataset,t)):
        # check if the image ends with png
        if (image.endswith(".jpg")):
            # check if the image is already in df 
            # adding some negative examples
            if not df.iloc[:,0].str.contains(image).any():
                df.loc[len(df)] = [image,0,0,0,0,0,0,0]
                

    df = df.sample(frac = 1)
    f_name = df.iloc[:,0]
    # scaling the box dims
    labels = df.iloc[:,5:8].astype(int)
    box = df.iloc[:,1:5].astype('float64')/640

    return f_name,box,labels    

def get_class_data(type,has_boxes=2):
    dataset = '/home/onu/ml/Dataset'
    df = pd.read_csv(os.path.join(dataset,type,"_classes.csv"))
    f,_,labels = get_data(type,dataset="/home/onu/ml/Rock-Paper-Scissors-SXSW-2")
    df3 = pd.DataFrame(pd.concat([f,labels],axis = 1).values,columns=df.columns)
    df = pd.concat([df,df3],axis = 0)
    return DataGenerator(df.iloc[:,0],None,df.iloc[:,1:4].astype(int),type,config.BATCH_SIZE,has_boxes=has_boxes,dataset=dataset)

def start_video_eval(model,ohe,has_boxes=False):

    video = cv2.VideoCapture(0)
    key = 122
    while True:
        _, frame = video.read()
        im = Image.fromarray(frame)

        h,w,_ = frame.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        
        # Blue color in BGR
        color = (255, 0, 0)
        
        # Line thickness of 2 px
        thickness = 2
        im = im.resize(config.IMAGE_DIM)
        img_array = np.array(im)

        # #Our keras model used a 4D tensor, (images x height x width x channel)
        # #So changing dimension 224x224x3 into 1x128x128x3 
        img_array = np.expand_dims(img_array, axis=0)


        # #Calling the predict method on model to predict 
        prediction = model.predict(img_array,verbose=0)
        if has_boxes==0:

            # this is test
            prob, box , t = prediction
            prob, box = prob[0],box[0]
            if prob[0] > 0.7:
                frame = cv2.rectangle(frame,(int(box[0]*w),int(box[1]*h)) ,(int(box[2]*w),int(box[3]*h)) , color, thickness)
                
    
                # org
                org = (int(box[0]*w),int(box[1]*h))
                
                
                # Using cv2.putText() method
                frame = cv2.putText(frame, ohe.inverse_transform(t)[0][0], org, font, 
                                fontScale, color, thickness, cv2.LINE_AA)
   
        else :
            t = prediction 
            org = ((20,20))
            cv2.putText(frame, ohe.inverse_transform(t)[0][0], org, font, 
                            fontScale, color, thickness, cv2.LINE_AA)


        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
    
    video.release()
    cv2.destroyAllWindows()
