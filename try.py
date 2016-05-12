import network
import os,sys
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import random

def vectorized_result(j):
    e = np.zeros((5, 1))
    e[j] = 1.0
    return e
def devectorize_result(arr):
    for i in range(0,5):
        if arr[i] == 1:
            return i

TRAINING_PATH = "Data set/Training/"
TESTING_PATH = "Data set/Testing/"
TRAINING_FILES = [f for f in listdir(TRAINING_PATH) if isfile(join(TRAINING_PATH, f))]
TESTING_FILES = [f for f in listdir(TESTING_PATH) if isfile(join(TESTING_PATH, f))]

Labels_Index = {
                "Cat":        vectorized_result(0),
                "Laptop":     vectorized_result(1),
                "Apple":      vectorized_result(2),
                "Car":        vectorized_result(3),
                "Helicopter": vectorized_result(4)
                }

#given the word this function returns the onehot vector label
def getLabelIndex(word):
    if word in Labels_Index:
        return Labels_Index[word]
    else:
        return np.zeros(5)

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

def view_image(img):
    cv2.startWindowThread()
    cv2.namedWindow("preview")
    cv2.imshow("preview" ,img)

SIFT = cv2.SIFT()
def getKeyPoints(img):
    return SIFT.detect(img,None)

def getKeyDescPoints(img):
    return SIFT.detectAndCompute(img , None)

def viewSIFTPoints(img,points):
    view_image(cv2.drawKeypoints(img,points))

trainingSamples = []
LablesSamples = []
for trainfile in TRAINING_FILES:
    kp, desc = getKeyDescPoints(TrainingData[join(TRAINING_PATH,trainfile)]["image"])

    #print list(desc[0])
    for i in range(0 , len(desc)):
        trainingSamples.append(np.reshape(desc[i],(128,1)))
        LablesSamples.append(TrainingData[join(TRAINING_PATH,trainfile)]["labels"][0])
training_data = zip(trainingSamples,LablesSamples)
#print LablesSamples

testSamples = []
testlablesSamples = []
for testfile in TESTING_FILES:
    kp, desc = getKeyDescPoints(TestingData[join(TESTING_PATH,testfile)]["image"])
    for i in range(0 , len(desc)):
        testSamples.append(np.reshape(desc[i],(128,1)))
        testlablesSamples.append(devectorize_result(TestingData[join(TESTING_PATH,testfile)]["labels"][0]))

test_data = zip(testSamples,testlablesSamples)

net = network.Network([128, 25, 5])
net.SGD(training_data, 30, 100, 0.5, test_data=test_data)
