import sys
import os

def refineData():
    ROOT_DIR = os.path.dirname(os.path.abspath("RobotOdoData.txt"))
    rawDataFile = open(ROOT_DIR + "/RobotOdoData.txt", "r") 
    dataFile = open(ROOT_DIR + "/RefinedOdoData.txt", "w")
    #dataFile.close()
    #dataFile = open(ROOT_DIR + "/RefinedOdoData.txt", "a")
    
    rawData = rawDataFile.readlines()
    currentFrame = 0
    startTime = -1
    marker = "Data:"
    data = []
    for i in range(len(rawData)):
        if (rawData[i].find(marker) != -1):
            line = rawData[i]
            line = line[rawData[i].find(marker)+len(marker):]
            line = line.split(",")
            if startTime == -1:
                startTime = int(line[3])
            while int((int(line[3])-startTime)/(1000000000/24)) >= currentFrame:
                currentFrame += 1
                dataFile.write(line[0]+","+line[1]+","+line[2])
                data.append(line[0]+","+line[1]+","+line[2])
    
    rawDataFile.close()
    dataFile.close()
    #dataFile = open(ROOT_DIR + "/RefinedOdoData.txt", "r")
    #data = dataFile.read();
    #dataFile.close()
    return data