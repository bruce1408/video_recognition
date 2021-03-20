import os
orgPath = "/home/bruce/bigVolumn/Datasets/personHead/selfLabel_voc_1"
dataPath = os.path.join(orgPath, 'data')
labelPath = os.path.join(orgPath, 'label')

dataPathList = os.listdir(dataPath)
for i in dataPathList:
    print()