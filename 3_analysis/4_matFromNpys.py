import numpy as np
import sys
import os
import os.path
import shutil
import argparse
import scipy.io.matlab as matlab

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

if __name__ == '__main__':
    print('npy to hdf5 converter')
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
    configFileDir = config.configFileDir
    testImagesListFile = config.testImagesFile
    trainImagesListFile = config.trainImagesFile
    
    dirName = None
    dirPath = None
    imCnt = None
    # parsing config
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dirName',
                        help=f'Use named directory (source, from 3_npys). Just name, not path. Folder (name) will be used also as output. Default: no subdir is used (path from config)',
                        default=dirName)
    parser.add_argument('-P', '--dirPath',
                        help=f'Path to results directory, contaning diffMatrix.npy, GTmatrix.npy, resMatrix.npy, result_img.list, ... Part of the path is also used to generate an ouput path for saving.',
                        default=dirPath)
    parser.add_argument('-N', '--imNumber',
                        help=f'set (limit) number of images to be predicted. Default: All images in analsysis list (result_img.list)',
                        type=int,
                        default=imCnt)
    # parsing
    argRes = parser.parse_args()
    dirName = argRes.dirName
    dirPath = argRes.dirPath
    imCnt = argRes.imNumber

    if dirPath is not None:
        print(f'using input path for npys: {dirPath}')
        npysDir, dirName = dirPath.split(os.path.sep, 1)
        if not os.path.samefile(npysDir, config.npysDir):
            raise Exception(f'path {dirPath} is not a subpath of npys dir: {config.npysDir}')

    if dirName is not None:
        print(f'using modified dir to READ/WRITE npys: {dirName}')
        resMatrixFile = config.getModifiedPath(resMatrixFile, dirName, autoCreate=False)
        GTmatrixFile = config.getModifiedPath(GTmatrixFile, dirName, autoCreate=False)
        diffMatrixFile = config.getModifiedPath(diffMatrixFile, dirName, autoCreate=False)
        analysisImagesFile = config.getModifiedPath(analysisImagesFile, dirName, autoCreate=False)
        matOutFile = os.path.join(os.path.dirname(resMatrixFile), 'results.mat')
    
    # read the points from matrices files
    resMatrix = np.load(resMatrixFile)
    GTmatrix = np.load(GTmatrixFile)
    diffMatrix = np.load(diffMatrixFile)

    # read landmark (point) symbols (names)
    LMsymbols = open(landmark_symbolsFile, 'r').read().split('\n')
    # remove last new line
    del LMsymbols[-1]

    testImagesList = filelist.read_file_list(testImagesListFile)
    testImagesList = [os.path.abspath(p) for p in testImagesList]

    trainImagesList = filelist.read_file_list(trainImagesListFile)
    trainImagesList = [os.path.abspath(p) for p in trainImagesList]

    analysisImages = filelist.read_file_list(analysisImagesFile)

    testImagesInds = filelist.compare_file_lists(analysisImages, testImagesList)
    trainImagesInds = filelist.compare_file_lists(analysisImages, trainImagesList)

    testFlags = np.array([(i in testImagesInds) for i in range(len(analysisImages))])

    if not (resMatrix.shape[0]
            == GTmatrix.shape[0]
            == diffMatrix.shape[0]
            == len(analysisImages)):
        raise Exception(f'number of result entries {resMatrix.shape[0]}, {GTmatrix.shape[0]}, {diffMatrix.shape[0]} and images in list does not match {len(analysisImages)}')

    #create otput directory
    filesNum = len(analysisImages)
    if imCnt is not None:
        filesNum = imCnt
        # hack za Martinove podatke, kjer ima v strukturi vse slike (ne samo testne)
        # TODO: naredi filtriranje tako kot je v 5_statsBoxPlot.py z uporabo filelist
        resMatrix = resMatrix[-imCnt:]
        GTmatrix = GTmatrix[-imCnt:]
        diffMatrix = diffMatrix[-imCnt:]
        analysisImages = analysisImages[-imCnt:]

    img_names = [os.path.splitext(os.path.basename(path))[0] for path in analysisImages]
    pts_names = []

    matlab.savemat(matOutFile, {
        'imgNames': img_names,
        'ptsNames': LMsymbols,
        'GT':GTmatrix[..., [1, 0]],
        'Predictions':resMatrix[..., [1, 0]],
        'probabilities':resMatrix[..., 2],
        'testFlags':testFlags.reshape(-1, 1),
        })

