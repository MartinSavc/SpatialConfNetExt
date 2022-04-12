# -*- coding: utf-8 -*-
#from __future__ import print_function # za python2???
import numpy as np
from sklearn.preprocessing import normalize
import skimage.feature
import sys
import os
import os.path
import matplotlib
import json
import argparse

makeTestImages = True  # overwritten by  argparse
makeTrainImages = False  # overwritten by  argparse
interaktivno = False  # for ipython

pltColTxtGood = '#006400' #'Dark Green'
pltColTxtMiddle = '#FF8C00' #'Dark Orange'
pltColTxtBad = '#8B0000' #'Dark Red'


np.set_printoptions(precision=3)
os.environ["OPENBLAS_NUM_THREADS"] = "3"


if interaktivno:
    matplotlib.use("Qt4Agg")
    matplotlib.interactive(True)
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
# lepi tekst
import matplotlib.patheffects as PathEffects


# parametri
sys.path += ["../"]
try:
    import KPnet_config as config
    from commonlib.cephdataloaders import CephDataLoader, DataResizeLoader
except ModuleNotFoundError as e:
    configPathMissing = os.path.abspath(sys.path[-1])
    print('\nERROR: KPnet_config.py missing in: ' + configPathMissing + '\n\n')
    print(e)
    raise e

from commonlib.plots import (
        drawImageWithPoints,
        drawImageWithPointsIdividually,
        )

imTestCnt = config.imTestCnt
# imTestCnt = 10
imTrainCnt = config.imTrainCnt
#imTrainCnt = 10
pointsCnt = config.pointCnt
imWidth = config.imWidth
imHeight = config.imHeight

netProtoFile = config.netProtoFile
netModelFile = config.netModelFile

resMatrixFile = config.resMatrixFile
GTmatrixFile = config.GTmatrixFile

landmark_symbolsFile = config.landmark_symbolsFile
analysisImagesFile = config.analysisImagesFile
testImagesFile = config.testImagesFile
trainImagesFile = config.trainImagesFile
statsJsonFile = config.statsJsonFile

LMDBdir = config.LMDBdir
imagesResDir = config.imagesResDir


def softmax(z, axis=-1):
    return np.exp(z)/np.sum(np.exp(z),axis, keepdims=True)
def sigmoid(x):
    return np.power(1+np.exp(-x), -1)

dirName = None
imCnt = None
saveFullImage = True
savePerPoints = True
saveA4cropedFigure = False
savePng = False
savePngDpi = 200

parser = argparse.ArgumentParser()
parser.add_argument('-D', '--dirName',
                    help=f'Use named directory (source, from 3_npys). Just name, not path. Folder (name) will be used also as output. Default: no subdir is used (path from config)',
                    default=dirName)
parser.add_argument('-N', '--imNumber',
                    help=f'set (limit) number of images to be predicted. Default: All images in analsysis list (result_img.list)',
                    type=int,
                    default=imCnt)
parser.add_argument('--filesList',
                    help=f'set location of lmdb/hdf5 files.txt. Mandatory when dirname(?)',
                    default=None)
parser.add_argument('--noFullImage',
                    help=f'DONT save svg plot of full image (drawImageWithPoints). Default off', action='store_true')
parser.add_argument('--noPerPoints',
                    help=f'DONT save svg plot of induvidual points (drawImageWithPointsIdividually). Default off', action='store_true')
parser.add_argument('--fullImageA4crop',
                    help=f'ALSO save svg A4 plot of image with crop only points (dont use --noFullImage). Default off', action='store_true')
parser.add_argument('--savePng',
                    help=f'ALSO save png (+ svg) of "full image" and "per points" in subfolder "pngs" . Warning - big files (dpi=200 many MB) curr dpi {savePngDpi}. Default off', action='store_true')
parser.add_argument('-T', '--useTraining',
                    help=f'use training dataset instead of testing (default off)', action='store_true')
parser.add_argument('--kpListFile', type=str,
                    help=f'(temporary) path to alternative file with ordered short landmark names. Default: {landmark_symbolsFile}',
                    default=landmark_symbolsFile)

# parsing
argRes = parser.parse_args()
dirName = argRes.dirName
imCnt = argRes.imNumber  # default None
filesListParse = argRes.filesList  # images list
landmark_symbolsFile = argRes.kpListFile  # landmark list

saveFullImage = not argRes.noFullImage
savePerPoints = not argRes.noPerPoints
saveA4cropedFigure = argRes.fullImageA4crop
savePng = argRes.savePng

if argRes.useTraining is False:
    # make test
    makeTestImages = True
    makeTrainImages = False
else:
    # make train
    makeTestImages = False
    makeTrainImages = True


if __name__ == '__main__':
    # v kolikor je prisoten ditName potem morajo obstajati subfolderji z imeni "dirName" v projektu, kjer dobi podatke
    # npr: 3_analysis/3_npys/diffMatrix.npy -> 3_analysis/3_npys/DS06-gauss-7px-iter100000-test/diffMatrix.npy
    if dirName is not None:
        print(f'using modified dir to READ/WRITE npys: {dirName}')
        resMatrixFile = config.getModifiedPath(resMatrixFile, dirName, autoCreate=False)
        GTmatrixFile = config.getModifiedPath(GTmatrixFile, dirName, autoCreate=False)
        analysisImagesFile = config.getModifiedPath(analysisImagesFile, dirName, autoCreate=False)
        statsJsonFile = config.getModifiedPath(statsJsonFile, dirName, autoCreate=False)

        # WIP: if using "dirName" full path to test/train ImagesFile must be set via
        if filesListParse is None:
            raise Exception(f'using dirName, filesListParse is not set. set it via parameter --filesList')

        # flags from argparse
        if makeTestImages is True:
            # targetDirTest: 3_analysis/6_images/DS06-gause...iter100000-train/
            targetDirTest = os.path.join(imagesResDir, dirName)
            os.makedirs(targetDirTest, exist_ok=True)
            testImagesFile = filesListParse
            
        if makeTrainImages is True:
            targetDirTrain = os.path.join(imagesResDir, dirName)
            os.makedirs(targetDirTrain, exist_ok=True)
            trainImagesFile = filesListParse
    else:
        # targetDirTest: 3_analysis/6_images/testImages
        targetDirTest = os.path.join(imagesResDir, 'testImages')
        targetDirTrain = os.path.join(imagesResDir, 'trainImages')

    resMatrix = np.load(resMatrixFile)
    GTmatrix = np.load(GTmatrixFile)
    testImages = []
    trainImages = []

    if makeTestImages:
        if not os.path.exists(targetDirTest):
                os.mkdir(targetDirTest)

        testImages = open(testImagesFile, 'r').read().split('\n')
        #testImages = [os.path.basename(i).split('.')[0]
        #              for i in testImages if i]
        # sedej: spremenil iz split('.') v splitext saj za primer "Q00663_29__07__2019_A_1.jpg_aug-003.jpg" zgornji ne deluje
        testImages = [os.path.splitext(os.path.basename(i))[0]
                       for i in testImages if i]
    
    if makeTrainImages:
        if not os.path.exists(targetDirTrain):
                os.mkdir(targetDirTrain)
        #print('\n\n\n')
        #import pdb;pdb.set_trace()
        trainImages = open(trainImagesFile, 'r').read().split('\n')
        #trainImages = [os.path.basename(i).split('.')[0]
        #               for i in trainImages if i]
        # sedej: glej zgornji komentar
        trainImages = [os.path.splitext(os.path.basename(i))[0]
                       for i in trainImages if i]


    # read landmark (point) symbols (names)
    LMsymbols = open(landmark_symbolsFile, 'r').read().split('\n')
    # remove last new line
    del LMsymbols[-1]
    # add index number
    for i in range(0, len(LMsymbols)):
        LMsymbols[i] = str(i) + ': ' + LMsymbols[i]

    coordRes = np.zeros((pointsCnt, 4))

    data_loader = CephDataLoader.fromFiles(analysisImagesFile)

    # tukaj bom naložil stats.JSON
    stat_struct = ''
    with open(statsJsonFile, 'r') as f:
        stat_struct = json.load(f)

    # loop trough images
    if imCnt is None:
        imCnt = len(data_loader)
    for i in range(imCnt):

        file_name = os.path.basename(data_loader.get_file_name(i)[0])
        file_name = file_name.rsplit('.', 1)[0]

        outFileWhole = f'{file_name}.svg'
        outFileInd = f'{file_name}_points.svg'
        
        # preveri ce stat_struct obstaja in nalozi podatke o statistiki
        if not stat_struct == '':
            mean = stat_struct["images"][i]["mean"]
            std = stat_struct["images"][i]["std"]
            med = stat_struct["images"][i]["med"]
            # relative - just mean
            meanRel = stat_struct["images"][i]["meanRel"] * 100  # in prec
            stats_data = f'mean={mean:.2f}({meanRel:.2f}%), std={std:.2f}, med={med:.2f}'
        else:
            stats_data = ''

        if file_name in testImages and makeTestImages:
            outFileWhole = os.path.join(targetDirTest, outFileWhole)
            outFileInd = os.path.join(targetDirTest, outFileInd)
        elif file_name in trainImages and makeTrainImages:
            outFileWhole = os.path.join(targetDirTrain, outFileWhole)
            outFileInd = os.path.join(targetDirTrain, outFileInd)
        else:
            import pdb;pdb.set_trace()
            continue

        print(f'img: {i}/{imCnt}: {file_name}')

        coordRes[:, :2] = GTmatrix[i, :, :2]
        coordRes[:, 2:4] = resMatrix[i, :, :2]
        predProbablitys = resMatrix[i, :, 2]

        # vse
        img, _ = data_loader[i]
        img = np.squeeze(img)

        # izrise celo sliko in jo shrani
        if saveFullImage:
            plt.close('all')
            drawImageWithPoints(
                img,
                coordRes,
                save=True,
                savePng=savePng,
                savePngDpi=savePngDpi,
                saveA4cropedFigure=saveA4cropedFigure,
                saveFName=outFileWhole,
                imInd=i, imFName=file_name, stats=stats_data,
                LMsymbols=LMsymbols, predProbablitys=predProbablitys,
                interactive=interaktivno)

        # narise "tiles" za vsako tocko posebej
        if savePerPoints:
            plt.close('all')
            drawImageWithPointsIdividually(
                img, coordRes, pointsCnt=pointsCnt,
                save=True,
                savePng=savePng,
                savePngDpi=savePngDpi,
                saveFName=outFileInd,
                LMsymbols=LMsymbols,
                predProbablitys=predProbablitys,
                interactive=interaktivno)
