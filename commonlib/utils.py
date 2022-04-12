import os
import sys
import shutil
import numpy as np
import h5py
import datetime
import matplotlib.pyplot as plt

sys.path += ["../"]
try:
    import KPnet_config as config
    from commonlib.resize_and_labels_fun import *
except ModuleNotFoundError as e:
    configPathMissing = os.path.abspath(sys.path[-1])
    print('\nERROR: KPnet_config.py missing in: ' + configPathMissing + '\n\n')
    print(e)
    raise e


# kopiranje slik iz seznama in folderja v drugi folder
def kopiraj_fajle_iz_list(fileList, sourceDir, targetDir):
    for fileName in fileList:
        filePath = os.path.join(sourceDir, fileName)
        print(fileName)
        
        # kopiraj image
        #filePathDest = os.path.join(targetDir, fileName)
        #shutil.copy2(filePath, filePathDest)
        
        # kopiraj points txt
        filePathTxtSource = getPointsFileFromImg(filePath)
        txtName = os.path.split(filePathTxtSource)[-1]
        filePathTxtDest = os.path.join(targetDir, txtName)
        shutil.copy2(filePathTxtSource, filePathTxtDest)


def hdfAnalyse(hdf, title=None, dataInd=0, labInd=0, gammaVal=0.2, useBlobCenter=False):
    '''
    the function analizes the H5py file
    it detects: width, height of image, number of labels, 
    max value in image and label, size of label, unique values in label
    
    it plots 4 images in subplots
    1 image, one label, labels with gamma, and pol of label (single row) 
    also saves .svg figure in home foler if title is gicen
    
    inputs:
        hdf: a hdf5 file handle - open with h5py.File('...')
        useBlobCenter: center of segmentation
    '''
    data = hdf['dataRaw'][dataInd]
    dataInHdfCnt = hdf['dataRaw'].shape[0]
    labs, h, w = data.shape
    im = data[0]
    lab = data[labInd+1]

    maxRowInd = int(lab.argmax()/w)
    maxColInd = lab.argmax()%w
    if useBlobCenter:
        import scipy.ndimage as ndimage
        gaussLabel = ndimage.filters.gaussian_filter(lab, 2)
        maxRowInd = int(gaussLabel.argmax()/w)
        maxColInd = gaussLabel.argmax()%w
    imMaxVal = im.max()
    labMaxVal = lab.max()
    row = lab[maxRowInd]
    rowNonZero = row[np.nonzero(row)]
    labSize = len(rowNonZero)
    labSum = lab.sum()

    s2 = int(np.ceil((labSize/2)))
    labOnly = lab[(maxRowInd-s2):(maxRowInd+s2), (maxColInd-s2):(maxColInd+s2)]
    uv, uc = np.unique(labOnly, return_counts=True)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    imHandle = ax1.imshow(im, cmap='gray')
    f.colorbar(imHandle, ax=ax1)
    if title is not None:
        ax1.set_title(title)

    imHandle = ax2.imshow(labOnly)
    f.colorbar(imHandle, ax=ax2)

    imHandle = ax3.imshow(labOnly**gammaVal, cmap='gray')
    f.colorbar(imHandle, ax=ax3)
    ax3.set_title(f'labOnly**{gammaVal} (gammaVal)')

    ax4.plot(rowNonZero, '.')
    ax4.grid(which='both')
    ax4.set_title(np.array_str(rowNonZero, precision=3, suppress_small=True))

    infoPlot = f'hdf: {dataInHdfCnt} ims, w:{w}, h:{h}, labs:{labs-1}\n \
        imMax: {imMaxVal}, labMax: {labMaxVal}, labSize: {labSize}, unique: {len(uc)}, sum: {labSum:.1f}'
    ax2.set_title(infoPlot)

    f.tight_layout()

    if title is not None:
        f.savefig(title+'.svg')

    # statistika
    print('max row values:')
    print(np.array_str(rowNonZero, precision=6, suppress_small=True))
    print('')
    print(f'no. of data in hdf: {dataInHdfCnt}')
    print(f'w:{w}, h:{h}, labs:{labs-1}')
    print(f'imMaxVal:{imMaxVal}')
    print(f'labMaxVal:{labMaxVal}')
    print(f'max ind row:{maxRowInd}, col:{maxColInd}')
    print(f'labSize: {labSize} (width (nonzeros in maxRow)')
    print(f'unique: {len(uc)} ( len(unique(labOnly)) )')
    print(f'labSum: {labSum:.3f}')


def hdfReadTest(hdf, iterative=False):
    # read all
    timeStart = datetime.datetime.now()
    if iterative is True:
        size = hdf['dataRaw'].shape[0]
        for i in range(size):
            data = hdf['dataRaw'][i]
    else:
        data = hdf['dataRaw'][:]
    timeEnd = datetime.datetime.now()
    
    dataInHdfCnt = hdf['dataRaw'].shape[0]
    timeFromStart = timeEnd - timeStart
    secPerIm = timeFromStart.total_seconds()/dataInHdfCnt
    print(f'perf: {secPerIm:.3f} s/im (hdf:{dataInHdfCnt} ims)')
    return timeFromStart.total_seconds(), dataInHdfCnt


def hdfReadTestFolder(fold, iterative=False):
    files = [os.path.join(fold, fil) for fil in os.listdir(fold) if '.h5' in fil]
    files.sort()
    totalSec = 0
    totalIms = 0
    for f in files:
        print(f'file: {f}')
        hdff = fTr = h5py.File(f, 'r')
        secs, ims = hdfReadTest(hdff, iterative=iterative)
        totalSec += secs
        totalIms += ims
        hdff.close()
    print(f'{(totalSec/totalIms):.3f} s/im')


def appendSymbolToPointsFile(pointsFile, symbolsFile, csvChar=','):
    '''
    quick patch
    '''
    if not os.path.isfile(pointsFile):
        raise Exception(f'not a file pointsFile:\n{pointsFile}')
    if not os.path.isfile(symbolsFile):
        raise Exception(f'not a file symbolsFile:\n{symbolsFile}')

    with open(symbolsFile, 'r') as f:
        symbols = f.readlines()
        symbols = [l.rstrip() for l in symbols]

    with open(pointsFile, 'r') as f:
        lines = f.readlines()
        lines = [l.rstrip() for l in lines]

    # v drugi vrstici preverim število elementov. 3 brez imena tocke. 4 z
    if len(lines[1].split(csvChar)) != 3:
        raise Exception(f'line not have 3 elements:\n{lines[1]}')

    for i in range(len(symbols)):
        lines[i+1] += ',' + symbols[i]

    #print(lines)
    with open(pointsFile, 'w') as f:
        for l in lines:
            f.write(l + os.linesep)


# labelsDir = '/home/gsedej/Delo/CephBot/KPnet-CephBot/24_DS09_aug/KPnet/1_data/2_labels'
# symbolsFile = '/home/gsedej/Delo/CephBot/KPnet-CephBot/24_DS09_aug/KPnet/landmark_symbols.txt'
def appendSybolToAllAugmented(labelsDir, symbolsFile):
    '''
    quick patch
    '''
    items = os.listdir(labelsDir)
    items.sort()
    items = [i for i in items if 'aug' in i]
    for i in items:
        print(i)
        pointsFile = os.path.join(labelsDir, i)
        appendSymbolToPointsFile(pointsFile, symbolsFile)
