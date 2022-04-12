import numpy as np
import sys
import os
import os.path
import shutil
import argparse

# ta skripta prebere 3 matrike .npy in jih pripravi kot CSV (primerjava z drugimi)

# parametri
sys.path += ["../"]
try:
    import KPnet_config as config
except ModuleNotFoundError as e:
    configPathMissing = os.path.abspath(sys.path[-1])
    print('\nERROR: KPnet_config.py missing in: ' + configPathMissing + '\n\n')
    print(e)
    raise e

import commonlib.filelist as filelist


sep = ','


def fileWriter(matrix, currImFile, fileEnding, headerLine):
    ''' prebere npy matriko (matrix) in jo shrani kot CSV v csvDir/imeSlike...txt po vrsticah za vsako točko posebej
    TODO: naredi bolj logično
    TODO: naredi teste
    
    v glavno srani širina,višina,filename
    npr:
    512,512,A00011_1__08__2019_A_1
    
    v ostale vrstice se shrani
    indeks stolpca, indeks vrstice, indeks točke, kratko ime točke
    npr:
    412,386,0,Ar
    '''
    target = os.path.join(csvDir, currImFile+fileEnding)
    f = open(target, 'w')
    f.write(headerLine)
    for pnt in range(0, pointCnt):
        x = int(matrix[im, pnt, 1])
        y = int(matrix[im, pnt, 0])
        symbol = LMsymbols[pnt]
        line = str(x) + sep + str(y) + sep + str(pnt) + sep + symbol + '\n'
        #print(line, end='')
        f.write(line)
    f.flush()
    f.close()


def fileWriterProb(matrix, currImFile, fileEnding, headerLine):
    '''
    prebere matriko "predikcij" in shrani vrednosti prob (probablitly) v CSV po vsticah po vrsticah za vsako vrstico posebej
    
    matrix je resMatrix
    
    v glavno srani širina,višina,filename
    
    v ostale vrstice se shrani
    prob, indeks točke, kratko ime točke
    npr:
    0.35,0,Ar
    '''
    target = os.path.join(csvDir, currImFile+fileEnding)
    f = open(target, 'w')
    f.write(headerLine)
    for pnt in range(0, pointCnt):
        prob = matrix[im, pnt, 2]
        symbol = LMsymbols[pnt]
        # line = str(prob) + sep + str(pnt) + sep + symbol + '\n'
        line = f'{prob}{sep}{pnt}{sep}{symbol}\n'
        f.write(line)
    f.flush()
    f.close()


if __name__ == '__main__':
    print('npy to csv extractor')
    # 'manual' configuration
    predsEnding = '_preds.txt'
    GTsEnding = '_GTs.txt'
    diffsEnding = '_diffs.txt'
    probsEnding = '_preds_prob.txt'
    
    # reading from config file
    imWidth = config.imWidth
    imHeight = config.imHeight
    imTestCnt = config.imTestCnt
    pointCnt = config.pointCnt
    analysisImagesFile = config.analysisImagesFile
    resMatrixFile = config.resMatrixFile
    GTmatrixFile = config.GTmatrixFile
    diffMatrixFile = config.diffMatrixFile
    landmark_symbolsFile = config.landmark_symbolsFile
    testImagesFile = config.testImagesFile
    configFileDir = config.configFileDir
    csvDir = config.csvDir
    
    dirName = None
    imCnt = None
    # parsing config
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dirName',
                        help=f'Use named directory (source, from 3_npys). Just name, not path. Folder (name) will be used also as output. Default: no subdir is used (path from config)',
                        default=dirName)
    parser.add_argument('-N', '--imNumber',
                        help=f'set (limit) number of images to be predicted. Default: All images in analsysis list (result_img.list)',
                        type=int,
                        default=imCnt)
    # parsing
    argRes = parser.parse_args()
    dirName = argRes.dirName
    imCnt = argRes.imNumber

    if dirName is not None:
        print(f'using modified dir to READ/WRITE npys: {dirName}')
        resMatrixFile = config.getModifiedPath(resMatrixFile, dirName, autoCreate=False)
        GTmatrixFile = config.getModifiedPath(GTmatrixFile, dirName, autoCreate=False)
        diffMatrixFile = config.getModifiedPath(diffMatrixFile, dirName, autoCreate=False)
        analysisImagesFile = config.getModifiedPath(analysisImagesFile, dirName, autoCreate=False)
        csvDir = os.path.join(csvDir, dirName)
    
    # read the points from matrices files
    resMatrix = np.load(resMatrixFile)
    GTmatrix = np.load(GTmatrixFile)
    diffMatrix = np.load(diffMatrixFile)

    # read landmark (point) symbols (names)
    LMsymbols = open(landmark_symbolsFile, 'r').read().split('\n')
    # remove last new line
    del LMsymbols[-1]

    analysisImages = filelist.read_file_list(analysisImagesFile)

    if not (resMatrix.shape[0]
            == GTmatrix.shape[0]
            == diffMatrix.shape[0]
            == len(analysisImages)):
        raise Exception(f'number of result entries {resMatrix.shape[0]}, {GTmatrix.shape[0]}, {diffMatrix.shape[0]} and images in list does not match {len(analysisImages)}')

    #create otput directory
    if not os.path.exists(csvDir):
        os.makedirs(csvDir)
    filesNum = len(analysisImages)
    if imCnt is not None:
        filesNum = imCnt
        # hack za Martinove podatke, kjer ima v strukturi vse slike (ne samo testne)
        # TODO: naredi filtriranje tako kot je v 5_statsBoxPlot.py z uporabo filelist
        resMatrix = resMatrix[-imCnt:]
        GTmatrix = GTmatrix[-imCnt:]
        diffMatrix = diffMatrix[-imCnt:]
        analysisImages = analysisImages[-imCnt:]
    '''
    for i in analysisImages[0:3]:
        print(i)
    print('')
    for i in analysisImages[-3:]:
        print(i)
    sys.exit()
    '''
    
    print(f'writing {filesNum} files to:\n{csvDir}')
    for im in range(filesNum):
        currImPath = analysisImages[im]
        currImFile, currImExt = os.path.splitext(os.path.basename(currImPath))

        #headerLine = str(imWidth) + sep + str(imHeight) + sep + currImFile + '\n'
        # spremenil - imwidth bi moral biti originalni, pa bom sedaj dal -1, sicer bi moral vsaki file prebrati
        headerLine = str(-1) + sep + str(-1) + sep + currImFile + '\n'

        # predictions
        fileWriter(resMatrix, currImFile, predsEnding, headerLine)
        fileWriter(GTmatrix, currImFile, GTsEnding, headerLine)
        fileWriter(diffMatrix, currImFile, diffsEnding, headerLine)
        fileWriterProb(resMatrix, currImFile, probsEnding, headerLine)
        
        # copy the image file
        if os.path.exists(currImPath):
            #target = os.path.join(csvDir, currImFile + '.png')
            target = os.path.join(csvDir, currImFile + currImExt)
            shutil.copy(currImPath, target)
        else:
            print('image file not exists. wont copy')
            
            print(currImPath)
            import ipdb
            ipdb.set_trace()
    print('ok')
