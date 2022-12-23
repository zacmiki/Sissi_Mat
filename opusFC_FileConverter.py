import os
import opusFC
import numpy as np

def getListOfFiles(dirName):
    filesList = os.listdir(dirName)   # gets the list of all the files in the director
    allFiles = list()

    for entry in filesList:
        fullPath = os.path.join(dirName, entry) #If entry is a Directory then get list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

#dirName = (input("Enter the Directory path name "))
dirName="/Volumes/EMPTY/toConvert/Toma"
listOfFiles = getListOfFiles(dirName)

#print("\n the File loaded is" , fileName, "it contains", dataSets, "datasets \n")


for entry in listOfFiles:
    if opusFC.isOpusFile(entry) == True:
        print(entry)
        dbs = opusFC.listContents(entry)
        dataSets = len(dbs)
        a = np.array(dbs)
        print(a)

        for sets in range(dataSets):
            data = opusFC.getOpusData(entry, dbs[sets])
            for item in dbs:
                suffix = item[0]
                filename = entry + "." + suffix + ".txt"
                spectrum = np.column_stack((data.x, data.y))
                np.savetxt(filename, spectrum, delimiter = ',')
