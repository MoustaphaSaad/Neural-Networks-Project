
# coding: utf-8

# # Neural-Networks Project
# 
# ## Introduction
# An essential part of the behavior of humans is their ability to recognize objects. Humans are able to recognize
# large numbers of other humans, letters, digits, and so on.
# The object recognition problem can be defined as a labeling problem based on models of known objects. Formally,
# given an image containing one or more objects of interest (and background) and a set of labels corresponding to a set
# of models known to the system, the system should assign correct labels to regions, or a set of regions, in the image.
# 
# ## Objective
# The goal of this project is to build an object recognition system that can pick out and identify objects from an
# inputted camera image, as shown in Figure 1, based on the registered objects.
# 
# ## System Architecture
# 
# Input -> Features Extraction -> Classifier -> Voting -> Output
# 
# ## Features Extraction
# 
# - Use Scale Invariant Feature Transform (SIFT) algorithm [1] to extract features of an image.
# - SIFT describes image features that have many properties that make them suitable for matching differing images of an object or scene.
# - The features are invariant to image scaling and rotation, and partially invariant to change in illumination and 3D camera viewpoint.
# - An important aspect of this approach is that it generates large numbers of features that densely cover the image over the full range of scales and locations.
# - As shown in Figure 3, given an image, SIFT generates a set of keypoints, each keypoint consists of its location, scale, orientation, and a set of 128 descriptors.
# - Keypoints are the samples, and their features are the 128 element feature vector (descriptors) for each keypoint.
# - You can use the implementation of SIFT in VLFeat library [2] or in OpenCV [3].
# 
# ## Requirments
# 
# - The user must be able to insert an input (image) to the application, and the application has to identify objects on the inputted image.
# - Using the test images you will test your classifier. And find the performance of your classifier using the Overall Accuracy (OA) and Confusion Matrix.
# - A comparative study showing the difference in applying the three classification algorithms based on the six evaluation measures mentioned above. Thus, a report template will be provided to you for filling it.
# - Also, the report must be provided showing the different NN architectures and different parameters you used, and their effect on the training and testing results.
# 
# ### Bouns
# Conduct a comparative study of different feature extraction algorithms such as SIFT, PCA-SIFT [5], and
# SURF [6] to show up their effects in improving classification performance of the project’s objective based
# on the six evaluation measures mentioned above. (A report template will be provided for that)

# ## How to use ipython-notebook?
# ipython notebook act as client to the server side that's running on your console write now
# 
# the notebook has cells just like this one ... you can click on this cell to edit it after editing the cell whether it's a "markdown" cell or "code" cell you can "evaluate" or run the cell by pressing "shift+enter" or you can use the play button in the icon bar above
# 
# you can know and change the type of the cell using the dropdown menu from the icon bar above
# 
# you can evaluate each cell in any order you wish when a cell is not evaluated it has an empty brackets like this on it's left side "[]" when the code in this block is running or waiting to be executed the left side indicator will be "[\*]" after finishing executing the cell will have a number indicating it's order in execution "[1]" for first cell, "[2]" for the second ... etc
# 
# you can use *TAB* key to auto-complete code
# 
# you can add *?* then evaluate the cell to get the documentation for example: "np.array?"

# ### Step 01:
# listing the files of training and testing

# In[1]:

import os,sys
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from sklearn import svm
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")


TRAINING_PATH = "Data set/Training/"
TESTING_PATH = "Data set/Testing/"
TRAINING_FILES = [f for f in listdir(TRAINING_PATH) if isfile(join(TRAINING_PATH, f))]
TESTING_FILES = [f for f in listdir(TESTING_PATH) if isfile(join(TESTING_PATH, f))]


# ### Step 02:
# loading images and labels into dictionary

# In[2]:

Labels_Index = {
                "Cat":        np.array([1.0,0.0,0.0,0.0,0.0]),
                "Laptop":     np.array([0.0,1.0,0.0,0.0,0.0]),
                "Apple":      np.array([0.0,0.0,1.0,0.0,0.0]),
                "Car":        np.array([0.0,0.0,0.0,1.0,0.0]),
                "Helicopter": np.array([0.0,0.0,0.0,0.0,1.0])
                }

#given the word this function returns the onehot vector label
def getLabelIndex(word):
    if word in Labels_Index:
        return Labels_Index[word]
    else:
        return np.zeros(5)

#given the onehot vector label this function returns the word
def getLabelWord(index):
    for key in Labels_Index.keys():
        if np.argmax(index) == np.argmax(Labels_Index[key]):
            return key
    return ""

def getIndexWord(index):
    print index
    one_hot = np.array([0,0,0,0,0])
    one_hot[int(index)] = 1.0
    return getLabelWord(one_hot)


# In[3]:

#function given filename it returns a list of labels associated with this file
#it depend on filename that  contains the classes of this image
def getImageLabels(image_filename):
    result = []
    if image_filename.find("Cat") != -1:
        result.append(Labels_Index["Cat"])
    
    if image_filename.find("Laptop") != -1:
        result.append(Labels_Index["Laptop"])
        
    if image_filename.find("Apple") != -1:
        result.append(Labels_Index["Apple"])
        
    if image_filename.find("Car") != -1:
        result.append(Labels_Index["Car"])
        
    if image_filename.find("Helicopter") != -1:
        result.append(Labels_Index["Helicopter"])
        
    return result
        


# ### Data Layout:
# - Data = Dictionary or map of data
# - Key = File name
# - Value = Dictionary {"image": numpy array of image data, "labels": list of onehot vectors that represents the labels associated with this image
# 
# ### How to Iterate over Data?
# ```
# for filename in TrainingData:
#         image = TrainingData[filename]["image"]
#         labels = TrainingData[filename]["labels"]
# ```

# In[4]:

#dictionary that contains the training data
TrainingData = {}
for image in TRAINING_FILES:
    image_filename = join(TRAINING_PATH, image)
    TrainingData[image_filename] = {"image": cv2.imread(image_filename),
                                     "labels": getImageLabels(image_filename)}
#dictionary that contains the testing data
TestingData = {}
for image in TESTING_FILES:
    image_filename = join(TESTING_PATH, image)
    TestingData[image_filename] = {"image": cv2.imread(image_filename),
                                    "labels": getImageLabels(image_filename)}


# Function to View the image given it's image matrix/data

# In[5]:

def view_image(img):
    cv2.startWindowThread()
    cv2.namedWindow("preview")
    cv2.imshow("preview" ,img)


# Get SIFT Features and a function to draw and view the features on the image

# In[6]:

SIFT = cv2.SIFT()
def getKeyPoints(img):
    return SIFT.detect(img,None)

def getKeyDescPoints(img):
    return SIFT.detectAndCompute(img, None)

def viewSIFTPoints(img,points):
    view_image(cv2.drawKeypoints(img,points))


# In[7]:

def logical_or_labels(labels):
    result = None
    for label in labels:
        if result is None:
            result = label
        else:
            result = np.logical_or(result, label).astype(float)
    return result


# In[8]:

SVM = svm.NuSVC(probability=True,decision_function_shape="ovr")
def train_SVM():
    all_descriptions = []
    all_labels = []
    for key in TrainingData.keys():
        img = TrainingData[key]["image"]
        img = cv2.Canny(img, 100, 200)
        points, descriptions = getKeyDescPoints(img)
        label = logical_or_labels(TrainingData[key]["labels"])
        descriptions = np.array(descriptions)
        all_descriptions.append(np.mean(descriptions, axis = 0))
        all_labels.append(np.argmax(label))
    
    SVM.fit(all_descriptions,all_labels)
    
    kmeans = KMeans(n_clusters=3)
    for key in TestingData.keys():
        org_img = TestingData[key]["image"]
        img = cv2.Canny(org_img, 100, 200)
        points, descriptions = getKeyDescPoints(img)
        all_points = [point.pt for point in points]
        kmeans.fit(all_points)
        clusters_indices = kmeans.predict(all_points)
        cluster_pts0 = []
        cluster_pts1 = []
        cluster_pts2 = []
        points0 = []
        points1 = []
        points2 = []
        desc0 = []
        desc1 = []
        desc2 = []
        clusters = [0,0,0]
        for i in xrange(len(clusters_indices)):
            clusters[clusters_indices[i]] += 1
            if clusters_indices[i] == 0:
                cluster_pts0.append(points[i])
                desc0.append(descriptions[i])
                points0.append(points[i].pt)
            elif clusters_indices[i] == 1:
                cluster_pts1.append(points[i])
                desc1.append(descriptions[i])
                points1.append(points[i].pt)
            elif clusters_indices[i] == 2:
                cluster_pts2.append(points[i])
                desc2.append(descriptions[i])
                points2.append(points[i].pt)
        
        points0 = np.array(points0)
        points1 = np.array(points1)
        points2 = np.array(points2)
        
        c1 = SVM.predict(np.mean(np.array(desc0), axis = 0))
        
        c2 = SVM.predict(np.mean(np.array(desc1), axis = 0))
        
        c3 = SVM.predict(np.mean(np.array(desc2), axis = 0))
        
        answer = {}
        answer[getIndexWord(c1[0])] = points0
        answer[getIndexWord(c2[0])] = points1
        answer[getIndexWord(c3[0])] = points2
        
        answer_image = org_img
        
        colors = [(255,0,0), (0,255,0), (0,0,255)]
        color_ix = 0
        print "answers: ", len(answer)
        for ans in answer.keys():
            min_pt = (100000, 100000)
            max_pt = (0, 0)
            for pt in answer[ans]:
                if pt[0] < min_pt[0] or pt[1] < min_pt[1]:
                    min_pt = pt
                elif pt[0] > max_pt[0] or pt[1] > max_pt[1]:
                    max_pt = pt
            cv2.rectangle(answer_image,(int(min_pt[0]), int(min_pt[1])), (int(max_pt[0]), int(max_pt[1])),colors[color_ix])
            cv2.putText(answer_image, ans, (int(min_pt[0]), int(min_pt[1])), cv2.FONT_HERSHEY_DUPLEX, 0.5, colors[color_ix])
            if color_ix == 0:
                print "Blue: ", ans
            elif color_ix == 1:
                print "Green: ", ans
            elif color_ix == 2:
                print "Red: ", ans
                    
            color_ix += 1
            if color_ix > 2:
                color_ix = 0
        view_image(answer_image)
        raw_input("wait")
        


# In[15]:

train_SVM()

