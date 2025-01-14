import os
import math
import pickle

import numpy as np
from .ProcessInput import *
from .Visuals import *

def classifyFeatureData(directory, writeAddress, modelPath, generateDataReport=False, keepNPZFiles=True,
                               numberOfMusicalExercises=5):

    # Loads the SVM model
    model = pickle.load(open(modelPath, 'rb'))

    # Initializations
    flaggedFileList = np.array([])
    firstPassFilesList = np.array([])
    secondPassFilesList = np.array([])
    secondPassDataDict = np.array([])
    musLengths = np.empty([numberOfMusicalExercises])
    nonMusTLengths = np.empty([numberOfMusicalExercises])
    fileLength = np.empty([1])
    correctSegFileCount = 0
    shortestSegOnlyBool = False

    for entry in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, entry)) and entry[-4:] == '.npz':
            try:
                featOrder, featArr, _, _, blockTimes = fileOpen(directory + '/' + entry)
                rawClassification = model.predict(featArr)
                rawClassification -= 1
                segmentsCount = numberOfMusicalExercises * 2
                postProcessedPreds, correctSegBool = postProc(rawClassification, True, False, False, segmentsCount)

                postProcessSegmentCount = countSegments(postProcessedPreds)

                if correctSegBool:
                    writeText(postProcessedPreds, blockTimes, entry[:-4], writeAddress)
                    firstPassFilesList = np.append(firstPassFilesList, entry)
                    correctSegFileCount += 1

                    nonMusTimeLengths, musTimeLengths, _, _, _, length, _, _, _ = segmentLengths(postProcessedPreds,
                                                                                                 blockTimes)
                    fileLength = np.append(fileLength, length)
                    musLengths = np.vstack((musLengths, musTimeLengths))
                    nonMusTLengths = np.vstack((nonMusTLengths, nonMusTimeLengths))

                    correctSegBool = False

                elif postProcessSegmentCount < 10 or postProcessSegmentCount > 25:
                    flaggedFileList = np.append(flaggedFileList, entry)

                else:
                    secondPassFilesList = np.append(secondPassFilesList, entry)
                    secondPassDataDict = np.append(secondPassDataDict, {"entry": entry, "procPred": postProcessedPreds,
                                                                        "blockTimes": blockTimes})
            except:
                print("Error with " + entry[:-4])
                flaggedFileList = np.append(flaggedFileList, entry)

    # Second pass data calculations
    firstPassSegLengthTimes = musLengths[1:, :]
    firstPassNonMusLengthTimes = nonMusTLengths[1:, :]
    firstPassFullFileLengths = fileLength[1:]
    firstPassSegmentPercents = firstPassSegLengthTimes / firstPassFullFileLengths.T[:, None]
    firstpassNonMusTLengths = firstPassNonMusLengthTimes.mean(0)
    firstPassNonMusTStd = firstPassNonMusLengthTimes.std(0)
    segmentPercMean = firstPassSegmentPercents.mean(0)
    segmentPercStd = firstPassSegmentPercents.std(0)

    ################ Second pass begins ###################

    # If this is true, the shortest segment removal process alone is implemented for the remaining files
    if correctSegFileCount < 25 or (correctSegFileCount/(correctSegFileCount + flaggedFileList.size +
                                                        secondPassFilesList.size)) < 0.25:
        shortestSegOnlyBool = True

        flipShortestUntilSegmentCount = numberOfMusicalExercises * 2

        for i in np.arange(len(secondPassDataDict)):
            try:
                entry = secondPassDataDict[i]["entry"]
                procPred = secondPassDataDict[i]["procPred"]
                blockTimes = secondPassDataDict[i]["blockTimes"]

                pred = processFlipSeg(procPred, flipShortestUntilSegmentCount, segmentPercMean, segmentPercStd,
                                      firstpassNonMusTLengths, firstPassNonMusTStd, blockTimes)

                writeText(pred, blockTimes, entry[:-4], writeAddress)
            except:
                print("Error with " + entry)

    # This includes the informative process
    else:
        flipShortestUntilSegmentCount = (numberOfMusicalExercises * 2) + 2

        for i in np.arange(len(secondPassDataDict)):
            try:
                entry = secondPassDataDict[i]["entry"]
                procPred = secondPassDataDict[i]["procPred"]
                blockTimes = secondPassDataDict[i]["blockTimes"]

                pred = processFlipSeg(procPred, flipShortestUntilSegmentCount, segmentPercMean, segmentPercStd,
                                      firstpassNonMusTLengths, firstPassNonMusTStd, blockTimes)

                writeText(pred, blockTimes, entry[:-4], writeAddress)

            except:
                print("Error with " + entry)
                flaggedFileList = np.append(flaggedFileList, entry)

    # Generate report txt file
    if generateDataReport:
        newDataReport(flaggedFileList, writeAddress, correctSegFileCount, firstPassFilesList, secondPassFilesList,
                      shortestSegOnlyBool)

    if keepNPZFiles != True:
        for entry in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, entry)) and entry[-4:] == '.npz':
                os.remove(directory + "/" + entry)



def postProc(predictions, smoothing=False, remNonMus=False, remMus=False, segCount=10):

    predictions[0] = 0

    if smoothing:
        predDiff = np.diff(predictions)
        predFixed = np.copy(predictions)

        # Fixes 0 1 0
        m3 = np.where(predDiff == 1)[0]
        if np.sum(m3 >= len(predDiff) - 2) > 0:
            if predDiff[-2] == 1 and predDiff[-1] == -1:
                predFixed[-2] = 0.
            if predDiff[-1] == 1:
                predFixed[-1] = 0.
            m3 = m3[:-1]
        m4 = m3[np.where(predDiff[m3 + 1] == -1)[0]] + 1
        predFixed[m4] = 0.0

        # Recalculates diff
        predDiff = np.diff(predFixed)

        # Fixes 1 0 1
        m1 = np.where(predDiff == -1)[0]
        if np.sum(m1 >= len(predDiff) - 2) > 0:
            if predDiff[-2] == -1 and predDiff[-1] == 1:
                predFixed[-2] = 1.
            if predDiff[-1] == -1:
                predFixed[-1] = 1.
            m1 = m1[:-1]
        m2 = m1[np.where(predDiff[m1 + 1] == 1)[0]] + 1
        predFixed[m2] = 1.0

        predictions = predFixed
        correctSegBool = segCount <= countSegments(predictions, [], False) <= (segCount + 1)

    return predictions, correctSegBool

def processFlipSeg(predictions, segCount, dataMean, dataStd, nonMusMean, nonMusStd, blockTimes):

    numSeg = countSegments(predictions)

    nonMusTimeLengths, musTimeLengths, nonMusSectionLengths, musSectionLengths, \
    segPerc, fileLength, musToNonMus, nonMus, nonMusToMus = segmentLengths(predictions, blockTimes)

    if numSeg > (segCount + 1):

        # while(numSeg > 11 or np.min(nonMusSectionLengths[nonMusSectionLengths > 0]) < 2):
        while (numSeg > (segCount + 1)):

            nonMusTimeLengths, musTimeLengths, nonMusSectionLengths, musSectionLengths, \
            segPerc, fileLength, musToNonMus, nonMus, nonMusToMus = segmentLengths(predictions, blockTimes)

            # plotPred(predictions)

            shortMus = np.argmin(musSectionLengths[musSectionLengths > 0])

            musLenLimit = 3
            windowCheckSize = 6


            if musSectionLengths[shortMus] <= musLenLimit:
                startInd = nonMus[shortMus] + 1
                endInd = musToNonMus[shortMus+1] + 1

                if startInd < windowCheckSize + 1:
                    checkLeft = predictions[1:startInd]
                else:
                    checkLeft = predictions[startInd-windowCheckSize:startInd-1]

                if len(predictions) - endInd < windowCheckSize + 1:
                    checkRight = predictions[endInd:-1]
                else:
                    checkRight = predictions[endInd+1:endInd+windowCheckSize]

                if (np.sum(checkLeft) + np.sum(checkRight)) < windowCheckSize:
                    predictions[startInd:endInd] = 0
                    musSectionLengths[shortMus] = np.max(musSectionLengths)
                    numSeg = countSegments(predictions)
                    # plt.clf()
                    continue

            shortNonMus = np.argmin(nonMusSectionLengths[1:]) + 1
            startInd = musToNonMus[shortNonMus] + 1
            endInd = nonMusToMus[shortNonMus] + 1
            predictions[startInd:endInd] = 1
            nonMusSectionLengths[shortNonMus] = np.max(nonMusSectionLengths)
            numSeg = countSegments(predictions)


        #     plt.clf()
        # plotPred(predictions)
        # plt.clf()

        nonMusTimeLengths, musTimeLengths, nonMusSectionLengths, musSectionLengths, \
        segPerc, fileLength, musToNonMus, nonMus, nonMusToMus = segmentLengths(predictions, blockTimes)

    if 12 <= numSeg <= 13:
        segFlip, startInd, endInd, flipTo = calcProbability(predictions, nonMusSectionLengths,
                                                    musSectionLengths, dataMean, dataStd, nonMusMean, nonMusStd,
                                                            blockTimes)

        # plotPred(predictions)
        predictions[startInd:endInd] = flipTo
        # plt.clf()
        # plotPred(predictions)
        # plt.clf()

    return predictions

def countSegments(procPred, segmentsArr=[], use=False):

    diffArr = np.diff(procPred)
    numSeg = (np.sum(np.abs(diffArr))+1)
    if use:
        segmentsArr = np.append(segmentsArr, numSeg)
        return segmentsArr, numSeg
    return numSeg

def segmentLengths(predictions, blockTimes=[]):

    predDiff = np.diff(predictions * 1)
    musToNonMus = np.where(predDiff == -1)[0]
    nonMusToMus = np.where(predDiff == 1)[0]
    nonMus = np.copy(nonMusToMus)
    musToNonMus = np.insert(musToNonMus, 0, 0)
    nonMus = np.append(nonMus, len(predictions) - 1)
    if len(musToNonMus) > len(nonMusToMus):
        nonMusToMus = np.append(nonMusToMus, len(predictions) - 1)
    elif len(nonMusToMus) > len(musToNonMus):
        musToNonMus = np.append(musToNonMus, len(predictions))
        nonMusToMus = nonMusToMus[1:]
        nonMusToMus = np.append(nonMusToMus, musToNonMus[-1])
    else:
        nonMus = nonMus[:-1]

    nonMusSectionLengths = nonMusToMus - musToNonMus
    mus = np.copy(musToNonMus)

    if len(nonMus) > len(musToNonMus)-1 and len(nonMus) > 5:
        nonMus = nonMus[:-1]
    elif len(nonMus) > len(musToNonMus)-1:
        mus = np.append(mus, len(predictions) - 1)

    musSectionLengths = mus[1:] - nonMus

    fileLength = 0
    segPerc = 0
    nonMusTimeLengths = 0
    musTimeLengths = 0


    if blockTimes != []:
        nonMusTimeLengths = blockTimes[nonMusToMus] - blockTimes[musToNonMus]
        musTimeLengths = blockTimes[mus[1:] - 1] - blockTimes[nonMus - 1]
        fileLength = blockTimes[-1]
        segPerc = musSectionLengths / fileLength

    if len(nonMusTimeLengths[1:]) < 5:
        nonMusTimeLengths = np.append(nonMusSectionLengths, 0)

    nonMusTimeLengths = nonMusTimeLengths[1:]


    return nonMusTimeLengths, musTimeLengths, nonMusSectionLengths, musSectionLengths, segPerc, fileLength, musToNonMus, nonMus, nonMusToMus

def calcProbability(predictions, nonMusSectionLengths, musSectionLengths, dataMean, dataStd, nonMusMean, nonMusStd,
                    blockTimes):

    i = 0
    mus = True
    probArray = np.array([])
    predCopy = np.copy(predictions)
    startArr = np.array([])
    endArr = np.array([])

    startInd = 0
    endInd = nonMusSectionLengths[0] + 1

    startArr = np.append(startArr, startInd)
    endArr = np.append(endArr, endInd)

    while (i < len(musSectionLengths) or i < len(nonMusSectionLengths)):
        if not mus:
            if (i > len(nonMusSectionLengths)-1):
                break

            startInd = endInd
            endInd += nonMusSectionLengths[i]

            if (endInd >= len(predictions)):
                break

            mus = True

            predCopy[startInd:endInd + 1] = 1

            nonMusTimeLengths, _, _, _, segPerc, _, _, _, _ = segmentLengths(predCopy, blockTimes)
        else:
            if (i > len(musSectionLengths)-1):
                break

            startInd = endInd
            endInd += musSectionLengths[i]

            if (endInd >= len(predictions)):
                break

            mus = False

            predCopy[startInd:endInd] = 0

            nonMusTimeLengths, _, _, _, segPerc, _, _, _, _ = segmentLengths(predCopy, blockTimes)
            i += 1

        startArr = np.append(startArr, startInd)
        endArr = np.append(endArr, endInd)

        segProbabilities = np.array([])
        for y in np.arange(len(segPerc)):
            musProb = (1/(dataStd[y] * math.sqrt(2*math.pi))) * \
                      math.exp(-0.5 * ((segPerc[y] - dataMean[y]) / dataStd[y])**2)
            nonMusProb = (1/(nonMusStd[y] * math.sqrt(2*math.pi))) * \
                         math.exp(-0.5 * ((nonMusTimeLengths[y] - nonMusMean[y]) / nonMusStd[y])**2)
            newProb = musProb * nonMusProb
            segProbabilities = np.append(segProbabilities, newProb)

        prod = 1
        for x in np.arange(len(segProbabilities)):
            prod *= segProbabilities[x]
        probArray = np.append(probArray, prod)
        predCopy = np.copy(predictions)

    # plotPred(predictions)
    segFlip = np.argmax(probArray) + 1
    if segFlip >= len(startArr):
        segFlip = len(startArr) - 1
    startInd = int(startArr[segFlip])
    endInd = int(endArr[segFlip])
    flipTo = 0
    if (segFlip % 2 == 0):
        flipTo = 1


    return segFlip, startInd, endInd, flipTo

def writeText(array, blockTimes, entry, address):
    newFile = open(address + "/" + entry + ".txt", "w")
    stampArr = getStamps(array, blockTimes)
    outputStr = ""
    for i in np.arange(stampArr.size / 2):
        index1 = int(i * 2)
        index2 = int(i * 2 + 1)
        if index2 >= stampArr.size:
            stampArr = np.append(stampArr, blockTimes[-1])
        outputStr = outputStr + str(np.round(stampArr[index1], 9)) + "\t" + "1\t" + \
                    str(np.round(stampArr[index2] - stampArr[index1], 9)) + "\n"
    newFile.write(outputStr[:-1])
    newFile.close()

def getStamps(array, blockTimes):
    diffArr = np.diff(array)
    changePoints = np.where(diffArr != 0)[0] - 1
    return np.array(blockTimes[changePoints])

def newDataReport(flaggedFileList, writeAddress, correctSegFileCount, firstPassFilesList, secondPassFilesList,
                  shortestSegOnlyBool):
    reportFile = open(writeAddress + "/00DataReport.txt", "w")
    writeStr = ""
    if shortestSegOnlyBool:
        writeStr += "There were not enough files with the correct number of segments after smoothing for the " \
                    "informative process to be viable, so the shortest segment removal process was used for all " \
                    "files. This will most likely decrease prediction accuracy.\n\n"
    writeStr += "Total file count: " + str(flaggedFileList.size + correctSegFileCount + secondPassFilesList.size) + "\n"
    writeStr += "Correct initial file count: " + str(correctSegFileCount) + "\n"
    writeStr += "Second pass file count: " + str(secondPassFilesList.size) + "\n"
    writeStr += "Flagged file count: " + str(flaggedFileList.size) + "\n"
    writeStr += "\nFlagged files:\n"

    for i in np.arange(flaggedFileList.size):
        writeStr += str(flaggedFileList[i][:-4]) + "\n"

    writeStr += "\nPrediction confidence:\n"

    for j in np.arange(firstPassFilesList.size):
        writeStr += str(firstPassFilesList[j][:-4]) + " 1.0\n"

    writeStr += "\n"

    for x in np.arange(secondPassFilesList.size):
        writeStr += str(secondPassFilesList[x][:-4]) + " 0.5\n"

    reportFile.write(writeStr[:-1])
    reportFile.close()

if __name__ == '__main__':
    feature_dir = "/home/yding402/fba-data/test_old_feature"
    model_path = "/home/yding402/fba-data/MIG-MusicPerformanceAnalysis-Code/src/auto_segmentation/Models/2017ABAI.sav"
    output = classifyFeatureData(
        directory=feature_dir,
        writeAddress=0,
        modelPath=model_path
    )
    print(output)