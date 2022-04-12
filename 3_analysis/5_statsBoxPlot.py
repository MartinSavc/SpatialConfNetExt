import numpy as np
import matplotlib
import sys
import os
import json
import cv2
import argparse
import PIL.Image

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
from commonlib.resize_and_labels_fun import getImageSizeFromFile
from commonlib.plots import (
        stdEucl, meanEucl, medianEucl,
        calcMeansStdsMeds_byPoints,
        setPlot,
        boxPlotImages,
        boxPlotImagesRelative,
        boxPlotPoints,
        boxPlotModel,
        drawImageWithPoints,
        )


def createStatsStructure(diffMatrix, imCnt, pointCnt,
                         datasetName='', fileNames=[], LMsymbols=[],
                         imHeights=[], imWidths=[], diffMatrixRelative=[]):
    '''
    creates lists/dicts structure for kpnet "standard" for statistics
    will be written to json file

    filenames: list - list of filenames (without paths)
    LMsymbols: list - short point names
    imHeights: list - optional, image heights in pixels
    imWidths: list - optional, image widths in pixels
    diffMatrixRelative: matrix - optional

    returns
        stat_struct: dict
    '''
    stat_struct = {}
    stat_struct['dataset'] = datasetName
    images, points = [], []
    # used for "global" statistics (all images/points)
    means, stds, meds = [], [], []
    # trough images
    for i in range(0, imCnt):
        mean = meanEucl(diffMatrix[i, :, :2])
        std = stdEucl(diffMatrix[i, :, :2])
        med = medianEucl(diffMatrix[i, :, :2])

        image = {}
        image['fileName'] = fileNames[i]
        image['mean'] = mean
        image['std'] = std
        image['med'] = med
        image['ind'] = i
        # check if we also have heights, widths and diffMatrixRelative
        if imHeights:
            meanRel = meanEucl(diffMatrixRelative[i, :, :2])
            stdRel = stdEucl(diffMatrixRelative[i, :, :2])
            medRel = medianEucl(diffMatrixRelative[i, :, :2])
            image['meanRel'] = meanRel
            image['stdRel'] = stdRel
            image['medRel'] = medRel
            image['height'] = imHeights[i]
            image['width'] = imWidths[i]
        images.append(image)

        # also save for global stats
        means.append(mean)
        stds.append(std)
        meds.append(med)
    stat_struct['images'] = images

    # trough points
    for i in range(0, pointCnt):
        mean = meanEucl(diffMatrix[:, i, :2])
        std = stdEucl(diffMatrix[:, i, :2])
        med = medianEucl(diffMatrix[:, i, :2])

        point = {}
        point['LMsymbol'] = LMsymbols[i]
        point['mean'] = mean
        point['std'] = std
        point['med'] = med
        point['ind'] = i
        # check if we also have heights, widths and diffMatrixRelative
        if imHeights:
            meanRel = meanEucl(diffMatrixRelative[:, i, :2])
            stdRel = stdEucl(diffMatrixRelative[:, i, :2])
            medRel = medianEucl(diffMatrixRelative[:, i, :2])
            point['meanRel'] = meanRel
            point['stdRel'] = stdRel
            point['medRel'] = medRel
        points.append(point)
    stat_struct['points'] = points

    # global stats
    l2_errs = np.sum(diffMatrix[:, :, :2]**2, 2)**0.5
    l2_errs_all = l2_errs.ravel()
    l2_errs_q1, l2_errs_med, l2_errs_q3, l2_errs_p90, l2_errs_p99 = np.percentile(
        l2_errs_all,
        [25, 50, 75, 90, 99],
        )

    globString = 'globalStats'
    globalStats = {}
    globalStats['mean'] = np.mean(l2_errs_all)
    globalStats['std'] = np.std(l2_errs_all)
    globalStats['med'] = l2_errs_med
    globalStats['q1'] = l2_errs_q1
    globalStats['q3'] = l2_errs_q3
    globalStats['perc_90'] = l2_errs_p90
    globalStats['perc_99'] = l2_errs_p99
    stat_struct[globString] = globalStats
    # check if we also have heights, widths and diffMatrixRelative
    if imHeights:
        globalStatsRel = {}
        l2_errs_rel = np.sum(diffMatrixRelative[:, :, :2]**2, 2)**0.5
        l2_errs_all_rel = l2_errs_rel.ravel()
        l2_errs_q1_rel, l2_errs_med_rel, l2_errs_q3_rel, l2_errs_p90_rel, l2_errs_p99_rel = np.percentile(
            l2_errs_all_rel,
            [25, 50, 75, 90, 99],
            )
        globalStatsRel['mean'] = np.mean(l2_errs_all_rel)
        globalStatsRel['std'] = np.std(l2_errs_all_rel)
        globalStatsRel['med'] = l2_errs_med_rel
        globalStatsRel['q1'] = l2_errs_q1_rel
        globalStatsRel['q3'] = l2_errs_q3_rel
        globalStatsRel['perc_90'] = l2_errs_p90_rel
        globalStatsRel['perc_99'] = l2_errs_p99_rel
        stat_struct[globString+'-relative'] = globalStatsRel

        globalStatsExtra = {}
        factor512 = np.sqrt(2)*512
        globalStatsExtra['mean'] = globalStatsRel['mean'] * factor512
        globalStatsExtra['std'] = globalStatsRel['std'] * factor512
        globalStatsExtra['med'] = globalStatsRel['med'] * factor512
        globalStatsExtra['q1'] = globalStatsRel['q1'] * factor512
        globalStatsExtra['q3'] = globalStatsRel['q3'] * factor512
        globalStatsExtra['perc_90'] = globalStatsRel['perc_90'] * factor512
        globalStatsExtra['perc_99'] = globalStatsRel['perc_99'] * factor512
        stat_struct[globString+'-512px'] = globalStatsExtra

        globalStatsExtra = {}
        factor1024 = np.sqrt(2)*1024
        globalStatsExtra['mean'] = globalStatsRel['mean'] * factor1024
        globalStatsExtra['std'] = globalStatsRel['std'] * factor1024
        globalStatsExtra['med'] = globalStatsRel['med'] * factor1024
        globalStatsExtra['q1'] = globalStatsRel['q1'] * factor1024
        globalStatsExtra['q3'] = globalStatsRel['q3'] * factor1024
        globalStatsExtra['perc_90'] = globalStatsRel['perc_90'] * factor1024
        globalStatsExtra['perc_99'] = globalStatsRel['perc_99'] * factor1024
        stat_struct[globString+'-1024px'] = globalStatsExtra

    return stat_struct


def getAllErrStats_diffMatrix(diffMatrix):
    ''' calculates mean, std and median from diffMatrix
    '''
    # diffMatrix[imInd, pintInd, (y, x, probValue)]
    mean = meanEucl(diffMatrix[:, :, :2])
    std = stdEucl(diffMatrix[:, :, :2])
    med = medianEucl(diffMatrix[:, :, :2])
    return mean, std, med


def printImageErrs(diffMatrix, imCnt=8, imagesFilename=None):
    # za diffMatrix po slikah izpiše mean std in med
    means, stds, meds = [], [], []
    for i in range(0, imCnt):
        mean = meanEucl(diffMatrix[i, :, :2])
        std = stdEucl(diffMatrix[i, :, :2])
        med = medianEucl(diffMatrix[i, :, :2])
        means.append(mean)
        stds.append(std)
        meds.append(med)
        fname = ''
        if imagesFilename is not None:
            fname = f' ({imagesFilename[i]})'
        print(f'image {i:3}: mean={mean:05.2f}, std={std:05.2f}, med={med:05.2f}{fname}')
    print('--------')
    print(f'all images: mean={np.mean(means):05.2f}, std={np.mean(stds):05.2f}, med={np.mean(meds):05.2f}')
    print('--------')


def printPointErrs(diffMatrix, pointCnt=72, LMsymbols=None):
    # po točkah
    means, stds, meds = [], [], []
    for i in range(0, pointCnt):
        mean = meanEucl(diffMatrix[:, i, :2])
        std = stdEucl(diffMatrix[:, i, :2])
        med = medianEucl(diffMatrix[:, i, :2])
        means.append(mean)
        stds.append(std)
        meds.append(med)
        symbol = ''
        if LMsymbols is not None:
            symbol = f' ({LMsymbols[i]})'
        print(f'point: {i:2}: mean={mean:05.2f}, std={std:05.2f}, mean={med:05.2f}{symbol}')
    print('--------')
    print(f'all points: mean={np.mean(means):05.2f}, std={np.mean(stds):05.2f}, med={np.mean(meds):05.2f}')
    print('--------')



def getImageSizeFromFiles(imagePaths, backend='PIL'):
    ''' calls getImageSizeFromFile in loop
    imagePaths - list with relativ/absolute paths to image files (jpgs/pngs)

    returns:
        heights - list
        widths - list
    '''
    widths = []
    heights = []
    for i in range(len(imagePaths)):
        w, h = getImageSizeFromFile(imagePaths[i], backend=backend)
        heights.append(h)
        widths.append(w)
    return heights, widths


def relativeImageErrors(diffMatrix, heights, widths):
    ''' convert actuall diffMatrix to relative diffMatrix.
    errors are relative to image diagonal. value 1.0 is full diagonal.

    diffMatrix - matrix I*P*3 (imgInd, pointInd, (rowInd, colInd, probValue))
    heights - list of hights in pixels
    widths - list of widths in pixels

    returns:
        diffMatrixRelative - matrix I*P*3
    '''
    # copy the matrix to presve probablity
    diffMatrixRelative = diffMatrix.copy()
    # get all diagonals
    for i in range(len(heights)):
        h = heights[i]
        w = widths[i]
        # diagonal using pitagora  c^2 =  a^2 + b^2
        diag = (h**2 + w**2)**0.5
        diffMatrixRelative[i, :, :2] = diffMatrix[i, :, :2]/diag
    return diffMatrixRelative


def saveStatsTxt(diffMatrix, statsFile, imgCnt, pointsCnt, LMsymbols=None, imagesFilename=None):
    '''
    saves stats to txt file using existing print methods (printImageErrs() / printPointErrs()).
    '''
    statsTxt = os.path.join(statsFile)
    terminalOut = sys.stdout
    sys.stdout = open(statsTxt, 'w')
    printImageErrs(diffMatrix, imgCnt, imagesFilename=imagesFilename)
    printPointErrs(diffMatrix, pointsCnt, LMsymbols=LMsymbols)
    # back to normal output
    sys.stdout = terminalOut


def errsVsProbs(diffMatrix, imagesFilename, LMsymbols, diffMatrixRelative,
                worstProbCnt=20, worstRelErrCnt=200):
    # DONT USE FOR "PRODUCTION"
    # IF MORE USAGE EXPECTED:
    # - MAKE AS LIBRARY
    # - MAKE UNIT TESTS
    import pandas
    
    # spodaj je kopija/modifikacija kode iz 5_multiModelStatsBoxPlot
    landmark_symbols = LMsymbols
    landmark_symbols = [lm for lm in landmark_symbols if lm]
    lmInds = np.arange(len(landmark_symbols))
    statsImagesList = imagesFilename
    diffMatrixSelection = diffMatrix

    errs = (diffMatrixSelection[:, :, :2]**2).sum(2)**0.5
    #if diffMatrixRelative is not None:
    errsRel = (diffMatrixRelative[:, :, :2]**2).sum(2)**0.5

    #pandas.options.display.width=200
    data_frames_list = []
    for r, image_path in enumerate(statsImagesList):
        data_frames_list.append(pandas.DataFrame(data={
            'image': os.path.basename(image_path),
            'II': r,
            'landmark': landmark_symbols,
            'LI': lmInds,
            'L2_error': errs[r, :],
            'L2_error_rel': errsRel[r, :],
            'probability': diffMatrixSelection[r, :, 2]
            }))
    multi_model_results_df = pandas.concat(data_frames_list)

    sortedRelErr = multi_model_results_df.sort_values(by='L2_error_rel', ascending=False)
    sortedProb = multi_model_results_df.sort_values(by='probability')

    print('worst probs (no Th\')')
    print(sortedProb[sortedProb.landmark != 'Th\''][:worstProbCnt])
    
    print('worst rel errs (no Th\')')
    print(sortedRelErr[sortedRelErr.landmark != 'Th\''][:worstRelErrCnt].to_string())
    
    # izpis statistik glede na verjetnosti od 0 od 1
    print('errors (stats) by probablity min probabilitys')
    minProbs = np.array([.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    print(f'min probs: {minProbs}')
    resol = config.imWidth
    print(f'errors scaled to resolution: {resol}')
    scale = resol * 1.41
    allCnt = len(sortedRelErr)
    
    for minP in minProbs:
        selection = sortedRelErr.loc[sortedRelErr['probability'] > minP]['L2_error_rel']
        worstVals = np.array(selection[:4]*scale)
        np.set_printoptions(precision=1)
        cnt = len(selection)
        if cnt == 0:
            print('cnt=0, breaking')
            break
        me = np.mean(selection) * scale
        md = np.median(selection) * scale
        percN = 95
        percVal = np.percentile(selection, 99) * scale
        print(f'min probablity: {minP:.2f}: mean: {me:.2f}, median: {md:.2f}, count: {cnt:5} ({(cnt/allCnt)*100:6.2f}%) perc{percN}: {percVal:4.1f}, {worstVals}')
    
    # izpis statistik glede na precentile
    # na ta način bi lahko rekli, če zanemarimo 20% najslabših rezultatov
    print(f'sorted by probablity. (scaled to resolution {resol})')
    percentIndices = np.int32(minProbs*allCnt)
    for minInd in percentIndices:
        selection = sortedProb[minInd:]['L2_error_rel']
        worstVals = np.sort(selection)[::-1][:4]*scale
        cnt = len(selection)
        if cnt == 0:
            print('cnt=0, breaking')
            break
        me = np.mean(selection) * scale
        md = np.median(selection) * scale
        percN = 95
        percVal = np.percentile(selection, 99) * scale
        print(f'inds from: {minInd:6}:end mean: {me:.2f}, median: {md:.2f}, count: {cnt:5} ({(cnt/allCnt)*100:6.2f}%) perc{percN}: {percVal:4.1f}, {worstVals}')


def getSortedByType(stat_struct, imOrPt='image', statMethod='mean'):
    '''
    returns sorted substructure (image or point)

    imOrPt: 'images' or 'points'
    statMethod: 'mean', 'med' or 'std'
        also for images: 'meanRel', 'stdRel', 'medRel'
    returns:
        sorted sub-structure (either images or points)
        if error, return None
    '''
    structList = []
    if imOrPt == 'images':
        name = "fileName"
    elif imOrPt == 'points':
        name = "LMsymbol"
    else:
        raise Exception(f'imOrPt={imOrPt} <- images or points')
    datas = stat_struct[imOrPt]
    values = []
    for d in datas:
        # check if statistic method exists in image/point (ex. meanRel)
        if statMethod not in d:
            return None
        values.append(d[statMethod])
    meanSortInd = np.argsort(values)
    for i in meanSortInd:
        structList.append(stat_struct[imOrPt][i])
    return structList


def saveStatsTxtSorted(stat_struct, statsFileSorted):
    '''
    saves sorted statistics images and points from stats.json file
    to text file stats.txt-sorted.txt
    '''

    def fWriteFor(f, res, method, resType="fileName", prec=2):
        '''
        writes sorted result res
        to file f
        using method method (mean, std, median, menRel, ...)
        result type resType ("fileName" or "LMsymbol")
        print float precision prec

        if res is None does noting
        '''
        if res is None:
            return
        for r in res:
            f.write(f'{method}: {r[method]:.{prec}f} ind: {r["ind"]:3} name: {r[resType]} \n')
        f.write('\n')

    with open(statsFileSorted, 'w') as f:
        # images

        method = 'mean'
        f.write('images by mean:\n')
        res = getSortedByType(stat_struct, imOrPt='images', statMethod=method)
        fWriteFor(f, res, method, resType="fileName")

        method = 'med'
        f.write('images by median:\n')
        res = getSortedByType(stat_struct, imOrPt='images', statMethod=method)
        fWriteFor(f, res, method, resType="fileName")

        method = 'std'
        f.write('images by standard deviation:\n')
        res = getSortedByType(stat_struct, imOrPt='images', statMethod=method)
        fWriteFor(f, res, method, resType="fileName")

        f.write('\n\n')
        # points

        method = 'mean'
        f.write('points by mean:\n')
        res = getSortedByType(stat_struct, imOrPt='points', statMethod=method)
        fWriteFor(f, res, method, resType="LMsymbol")

        method = 'med'
        f.write('points by median:\n')
        res = getSortedByType(stat_struct, imOrPt='points', statMethod=method)
        fWriteFor(f, res, method, resType="LMsymbol")

        method = 'std'
        f.write('points by standard deviation:\n')
        res = getSortedByType(stat_struct, imOrPt='points', statMethod=method)
        fWriteFor(f, res, method, resType="LMsymbol")

        f.write('\n\n')

        # relative images
        method = 'meanRel'
        f.write('images by relative mean:\n')
        res = getSortedByType(stat_struct, imOrPt='images', statMethod=method)
        fWriteFor(f, res, method, resType="fileName", prec=4)

        method = 'medRel'
        f.write('images by relative median:\n')
        res = getSortedByType(stat_struct, imOrPt='images', statMethod=method)
        fWriteFor(f, res, method, resType="fileName", prec=4)

        method = 'stdRel'
        f.write('images by relative standard deviation:\n')
        res = getSortedByType(stat_struct, imOrPt='images', statMethod=method)
        fWriteFor(f, res, method, resType="fileName", prec=4)

        f.write('\n\n')
        # points

        method = 'meanRel'
        f.write('points by relative mean:\n')
        res = getSortedByType(stat_struct, imOrPt='points', statMethod=method)
        fWriteFor(f, res, method, resType="LMsymbol", prec=4)

        method = 'medRel'
        f.write('points by relative median:\n')
        res = getSortedByType(stat_struct, imOrPt='points', statMethod=method)
        fWriteFor(f, res, method, resType="LMsymbol", prec=4)

        method = 'stdRel'
        f.write('points by standard relative deviation:\n')
        res = getSortedByType(stat_struct, imOrPt='points', statMethod=method)
        fWriteFor(f, res, method, resType="LMsymbol", prec=4)


if __name__ == '__main__':
    # CONFIGS
    interactive = False
    # enables relative errors (error/diagonal). slow! needs to load each image
    enableRelativeErrs = True
    # caffe resized
    enableResizedResult = False
    resizedSubDir = 'resized'
    saveResults = True
    enablePlotting = True
    enableStatsByProbablity = False

    imagesListFile = config.testImagesFile  # default list is test list from config
    analysisImagesFile = config.analysisImagesFile
    resMatrixFile = config.resMatrixFile
    GTmatrixFile = config.GTmatrixFile
    diffMatrixFile = config.diffMatrixFile
    diffMatrixResizedFile = config.diffMatrixResizedFile  # za caffe (pomanjšane)
    landmark_symbolsFile = config.landmark_symbolsFile
    statsDir = config.statsDir
    statsFile = config.statsFile
    statsJsonFile = config.statsJsonFile
    landmark_symbolsFile = config.landmark_symbolsFile

    # parsing helper variables
    dirName = None
    dirPath = None
    imCnt = None
    enableImagesBoxPlot = True
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
    parser.add_argument('-L', '--imagesListFile',
                        help=f'set location of images list. Default: {imagesListFile}',
                        default=imagesListFile)
    parser.add_argument('--enableResized',
                        help=f'enable resized (small) images/matrices. Will check npys if exists. Default: {enableResizedResult}',
                        dest='enableResized', action='store_true')
    parser.add_argument('--disableResized',
                        help=f'disable resized (small) images/matrices. Default: {not enableResizedResult}',
                        dest='disableResized', action='store_false')
    parser.add_argument('--interactive', '-I',
                        choices=['yes', 'no'],
                        help=f'Display interactive plots, allow modifications before saving. Default: {["no","yes"][interactive]}',
                        dest='interactive',
                        default=['no', 'yes'][interactive])
    parser.add_argument('--save', '-S',
                        choices=['yes', 'no'],
                        help=f'save generated statistics and images. Default: {["no","yes"][saveResults]}',
                        dest='saveResults',
                        default=['no', 'yes'][saveResults])
    parser.add_argument('--plot', '-p',
                        choices=['yes', 'no'],
                        help=f'enable generating plots. Default: {["no","yes"][enablePlotting]}',
                        dest='enablePlotting',
                        default=['no', 'yes'][enablePlotting])
    parser.add_argument('--relativeErrs',
                        choices=['yes', 'no'],
                        help=f'enable calculation of relative errors (fast, only reads image headers). Default: {["no","yes"][enableRelativeErrs]}',
                        dest='enableRelativeErrs',
                        default=['no', 'yes'][enableRelativeErrs])
    parser.add_argument('--statsByProb',
                        choices=['yes', 'no'],
                        help=f'enable print of statistics by probablility. Needs relative erors. Default: {["no","yes"][enableStatsByProbablity]}',
                        dest='enableStatsByProbablity',
                        default=['no', 'yes'][enableStatsByProbablity])
    parser.add_argument('--enableImagesBoxPlot',
                        choices=['yes', 'no'],
                        help=f'enable BoxPlot per images (can be disabled due to >200 images. slow, unreadable). Default: {["no","yes"][enableImagesBoxPlot]}',
                        dest='enableImagesBoxPlot',
                        default=['no', 'yes'][enableImagesBoxPlot])
    # temporary landmark names from file (for limited number of landmarks)
    parser.add_argument('--kpListFile', type=str,
                        help=f'(temporary) path to alternative file with ordered short landmark names. Default: {landmark_symbolsFile}',
                        default=landmark_symbolsFile)

    # parsing
    argRes = parser.parse_args()
    dirName = argRes.dirName
    dirPath = argRes.dirPath
    imCnt = argRes.imNumber  # default None
    landmark_symbolsFile = argRes.kpListFile

    imagesListFile = argRes.imagesListFile
    interactive = {'yes': True, 'no': False}[argRes.interactive]
    saveResults = {'yes': True, 'no': False}[argRes.saveResults]
    enablePlotting = {'yes': True, 'no': False}[argRes.enablePlotting]
    enableRelativeErrs = {'yes': True, 'no': False}[argRes.enableRelativeErrs]
    enableStatsByProbablity = {'yes': True, 'no': False}[argRes.enableStatsByProbablity]
    enableImagesBoxPlot = {'yes': True, 'no': False}[argRes.enableImagesBoxPlot]

    # check if "resize" parameter was used - test non-default options
    if argRes.enableResized is True:
        enableResizedResult = True
    if argRes.disableResized is False:
        enableResizedResult = False

    # OTHER CONFIGS
    if interactive:
        matplotlib.use("Qt4Agg")
        matplotlib.interactive(True)
    else:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

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
        if enableResizedResult:
            diffMatrixResizedFile = config.getModifiedPath(diffMatrixResizedFile, dirName, autoCreate=False)

        statsFile = config.getModifiedPath(statsFile, dirName, autoCreate=True)
        statsJsonFile = config.getModifiedPath(statsJsonFile, dirName, autoCreate=True)
        statsDir = os.path.join(statsDir, dirName)

    # read the diffMatrix
    resMatrix = np.load(resMatrixFile)
    GTmatrix = np.load(GTmatrixFile)
    diffMatrix = np.load(diffMatrixFile)

    # read landmark (point) symbols (names)
    LMsymbols = open(landmark_symbolsFile, 'r').read().split('\n')
    # remove last new line
    del LMsymbols[-1]
    # add index number
    LMsymbolsFullText = []
    for i in range(0, len(LMsymbols)):
        LMsymbolsFullText.append(f'{i}: {LMsymbols[i]}')

    # getting (test) images filenames from list "txt" file
    imagesList = filelist.read_file_list(imagesListFile)
    for i in range(len(imagesList)):
        imagesList[i] = os.path.abspath(imagesList[i])
    # limit list if argument imCnt imNumber is given
    if imCnt is not None:
        imagesList = imagesList[:imCnt]

    # get filename (without path)
    # 3_images_resize/78_Image0001_Type17.png -> 78_Image0001_Type17.png
    # if i -> ignore empty line
    imagesFilename = [os.path.basename(i) for i in imagesList if i]

    # TODO: duplication of LMsymbols (remove LMsymbols?)
    landmark_symbols = open(landmark_symbolsFile, 'r').read().split('\n')

    analysisImages = filelist.read_file_list(analysisImagesFile)

    testImagesInds = filelist.compare_file_lists(analysisImages, imagesList, only_file_name=True)

    # po vrsti glede na...
    resMatrix = resMatrix[testImagesInds, :]
    GTmatrix = GTmatrix[testImagesInds, :]
    diffMatrix = diffMatrix[testImagesInds, :]

    imgCnt, pointsCnt, _ = diffMatrix.shape
    #import pdb
    #pdb.set_trace()

    # needed for plotting functions to save results
    os.chdir(statsDir)

    # prints to file
    if saveResults:
        print('saving stats txt and json')
        saveStatsTxt(diffMatrix, statsFile, imgCnt, pointsCnt,
                     LMsymbols=LMsymbols, imagesFilename=imagesFilename)

    def on_image_pick(ind):
        print(f'image file:{imagesList[ind]}')
        img = plt.imread(imagesList[ind])
        coord_res = np.zeros((pointsCnt, 4))
        coord_res[:, :2] = GTmatrix[ind, :, :2]
        coord_res[:, 2:] = resMatrix[ind, :, :2]

        drawImageWithPoints(
                img,
                coord_res,
                imInd=ind,
                imFName=imagesList[ind],
                LMsymbols=LMsymbolsFullText,
                )

    if enablePlotting:
        if enableImagesBoxPlot:
            print('images BoxPlot')
            boxPlotImages(
                    diffMatrix,
                    imgCnt,
                    fileNames=imagesFilename,
                    save=saveResults,
                    interactive=interactive,
                    image_pick_handler=on_image_pick,
                    )
        print('points BoxPlot')
        boxPlotPoints(diffMatrix, pointsCnt, LMsymbolsFullText, save=saveResults, interactive=interactive)

        print('model BoxPlot')
        boxPlotModel(diffMatrix, save=saveResults, interactive=interactive)

    diffMatrixRelative, heights, widths = [], [], []
    if enableRelativeErrs:
        print("reading image files to get diagonas -> relative Error")
        origImageFolder = os.path.join(config.projectRootDir, '1_data', '1_images')
        heights, widths = getImageSizeFromFiles(imagePaths=imagesList, backend='PIL')
        diffMatrixRelative = relativeImageErrors(diffMatrix, heights, widths)

        if enablePlotting:
            if enableImagesBoxPlot:
                print('images BoxPlot relative')
                boxPlotImagesRelative(diffMatrixRelative, imgCnt, fileNames=imagesFilename, save=saveResults, interactive=interactive)

    if saveResults:
        # saves JSON stats file
        print('saving stats.json')
        # additionaly accept "relative" information including relativDiffMatrix, list of image widths and heights
        stat_struct = createStatsStructure(
            diffMatrix, imCnt=imgCnt, pointCnt=pointsCnt,
            fileNames=imagesFilename, LMsymbols=landmark_symbols,
            diffMatrixRelative=diffMatrixRelative, imHeights=heights, imWidths=widths)

        with open(statsJsonFile, 'w') as f:
            json.dump(stat_struct, f, indent=1)

    if saveResults:
        # print sorted
        statsFileSorted = os.path.splitext(statsFile)[0] + '-sorted.txt'
        saveStatsTxtSorted(stat_struct, statsFileSorted)

    # caffe resized results
    #if os.path.exists(diffMatrixResizedFile):
    if enableResizedResult:
        print("resized statistics (512px)")
        diffMatrixResized = np.load(diffMatrixResizedFile)
        # CurDir is statsDir (3_analysis/5_stats_graphs)
        # create subdir (3_analysis/5_stats_graphs/resized)
        statsResizedDir = os.path.join(statsDir, resizedSubDir)
        if not os.path.exists(statsResizedDir):
            os.makedirs(statsResizedDir)
        if saveResults:
            statsFileResized = os.path.join(statsResizedDir, 'statsResize.txt')
            saveStatsTxt(diffMatrixResized, statsFileResized, imgCnt, pointsCnt)

        # needed for plotting functions to save results
        os.chdir(statsResizedDir)
        if enablePlotting:
            if enableImagesBoxPlot:
                print('images BoxPlot resized')
                boxPlotImages(diffMatrixResized, imgCnt, fileNames=imagesFilename, save=saveResults, interactive=interactive)
            print('points BoxPlot resized')
            boxPlotPoints(diffMatrixResized, pointsCnt, LMsymbols, save=saveResults, interactive=interactive)

        if saveResults:
            with open(statsFileResized, 'r') as f:
                lines = [l.strip() for l in f.readlines()]
                print('statistics for resized (512px) images')
                for l in lines:
                    if 'all images' in l:
                        print(l)
                    if 'all points' in l:
                        print(l)
        if enableStatsByProbablity:
            errsVsProbs(diffMatrix, imagesList, LMsymbols, diffMatrixRelative)

    if saveResults:
        # output statistics to terminal
        with open(statsFile, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
            print('statistics for original sized images')
            for l in lines:
                if 'all images' in l:
                    print(l)
                if 'all points' in l:
                    print(l)

        print('\n\nNEW satistics:')
        print('original size (different sizes):')
        globString = 'globalStats'
        
        def printTestStats(statExtra, prec=2):
            print(f'mean={statExtra["mean"]:.{prec}f}, std={statExtra["std"]:.{prec}f}, med={statExtra["med"]:.{prec}f}')
            print(f'precentiles (25 50 75 90 99): {statExtra["q1"]:.{prec}f}, {statExtra["med"]:.{prec}f}, {statExtra["q3"]:.{prec}f}, {statExtra["perc_90"]:.{prec}f}, {statExtra["perc_99"]:.{prec}f}')

        glob = stat_struct[globString]
        printTestStats(glob)

        if 'globalStats-relative' in stat_struct:
            print('\nrelative (1.0 is image diagonal):')
            statExtra = stat_struct['globalStats-relative']
            printTestStats(statExtra, prec=4)
        subStructText = 'globalStats-512px'
        if subStructText in stat_struct:
            print('\nerrors @ 512x512:')
            statExtra = stat_struct[subStructText]
            printTestStats(statExtra, prec=2)
        subStructText = 'globalStats-1024px'
        if subStructText in stat_struct:
            print('\nerrors @ 1024x1024:')
            statExtra = stat_struct[subStructText]
            printTestStats(statExtra, prec=2)
