import numpy as np
import argparse
import sys
import os
import json
import PIL.Image
'''
generates SDR (Success detection rates) json and txt files
for given diffMatrix (or folder)
and creates barplot (optionaly)
'''

# TODO: dati v config file?
targetFolder = '5_benchResults'
SDRfileName = 'SDR.txt'
SDRfile = os.path.join(targetFolder, SDRfileName)
SRDjsonName = 'SDR.json'
SDRjson = os.path.join(targetFolder, SRDjsonName)


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
import commonlib.plots as plots
from commonlib.resize_and_labels_fun import getImageSizeFromFile



def plotBench(PzList, ticks, colors, plotFile=None,
              plotTitle=None, enableXkcd=True, figsize=(3.5, 6),
              suggestedYplotMin=40, useXkcdPlot=True):
    '''
    plot results in Wang 2016 barplot style
    and optionaly saves (both as png and svg)
    
    PzList - array, values in percents (for z ranges)
    ticks - array, "z" ranges in mm
    colors - array, strings, colors for matplotlib plot
    plotFile - path, partial filename without file extention (will be addes svg and png). Will overwite! If None, won't save
    plotTitle - string, sets the title of plot (method name)
    enableXkcd - bool, use xkcd style, similar to Wang 2016 barplots
    figsize - size of figure
    '''
    import matplotlib
    matplotlib.use("Qt5Agg")
    #matplotlib.interactive(True)
    import matplotlib.pyplot as plt
    if useXkcdPlot:
        matplotlib.pyplot.xkcd(scale=0.5)
    fig, ax = plt.subplots(figsize=(3.5, 6))
    if plotTitle is not None:
        ax.set_title(plotTitle)
        
    #_barplot(ax, ticks, PzList, colors, suggestedYplotMin)
    plots.barplot_SDR(ax, ticks, PzList, colors, suggestedYplotMin)
    plt.tight_layout()
    plt.show()

    if plotFile is not None:
        svgFile = plotFile + '.svg'
        pngFile = plotFile + '.png'
        fig.savefig(svgFile)
        print(f'saved svg: {svgFile}')
        fig.savefig(pngFile)
        print(f'saved svg: {pngFile}')


def transform_diffMatrix_to_bench_image_size(diffMatrix, analysisImages, mode=0, hBench=2400, wBench=1935):
    '''
    transforms diffMatrix to match benchmark-sized images (2400 x 1935)

    mode: int - how to calculate error (diff) multiplier
        0 - diagonal
        1 - height
        2 - width

    returns transformed diffMatrix (use it as for detectionRate() )

    relative sizes
    - get im size via PIL
    - calculate 100% from (?diagonal, width, height?)
    - calculate back to simulate error on 2400x200

    '''
    diff_matrix_transf = diffMatrix.copy()
    diag_bench = (hBench**2 + wBench**2)**0.5

    # per images
    for i in range(diffMatrix.shape[0]):
        h, w = getImageSizeFromFile(analysisImages[i])
        
        if mode == 0:
            diag = (h**2 + w**2)**0.5
            multiplier = diag_bench / diag
        else:
            raise Exception(f'mode {mode} not implemented (yet)')
        
        diff_matrix_transf[i, :, :2] = diffMatrix[i, :, :2] * multiplier
    return diff_matrix_transf


def detectionRate(diffMatrix, realMult=10, zList=[2, 2.5, 3, 4]):
    '''
    calculates the SDR - Success detection rates for 2mm, 2.5mm, 3mm, 4mm (by default)
    diffMatrix - diffMatrix for orignial ~2000x2400px (?[imgs, points, w, h])
    realMult - multilier (divider) - how many pixels is 1mm ("dpi")
    zList - ranges in milimers
    
    returns
    MRE - scalar, mean radial error (in mm)
    SD - scalar, standard deviation (in mm)
    PzList - array of scalar - SDRs (in %)
    '''
    imCnt = diffMatrix.shape[0]
    pointCnt = diffMatrix.shape[1]
    n = imCnt*pointCnt

    R = np.sum(diffMatrix[:, :, :2]**2, -1)**0.5
    # scaling
    R = R/realMult

    # Mean Radial Error
    MRE = R.sum()/n
    # Standard deviation
    SD = np.sqrt(np.sum(np.power(R-MRE, 2))/(n-1))

    PzList = []
    for z in zList:
        Pz = np.count_nonzero(R < z)
        PzList.append((Pz/n)*100.0)

    return MRE, SD, PzList


def saveResults(SDRfile, SDRjson, MRE, SD, PzList, zList=[2, 2.5, 3, 4],
                doPrint=True, methodName=None, dirName=None):
    '''
    saves results in human readable SDRfile and SDRjson file
    MRE, SD, PzList, zList - see detectionRate()
    methodName - name of metod
    '''
    with open(SDRfile, 'w') as f:
        if methodName is not None:
            f.write(f'method name: {methodName}\n')
        if dirName is not None:
            f.write(f'dirName: {dirName}\n')
        f.write(f'MRE: {MRE:.3f} mm\n')
        f.write(f'SD:  {SD:.3f} mm\n')
        for i, z in enumerate(zList):
            f.write(f'PZ {z} mm:\t{PzList[i]:.2f} %\n')
        print(f'saved txt: {SDRfile}')
    with open(SDRjson, 'w') as f:
        infoString = \
            '''MRE - scalar, mean radial error (in mm)
SD - scalar, standard deviation (in mm)
PzItems - array of:
  z - scalar: range in mm, 
  Pz - scalar: SDRs in % (Success Detection Rates)'''
        stat_struct = {}
        stat_struct['info'] = infoString
        #if methodName is None:
        #    stat_struct['methodName'] = ""
        #else:
        #    stat_struct['methodName'] = methodName
        stat_struct['methodName'] = methodName
        stat_struct['dirName'] = dirName
        stat_struct['MRE'] = MRE
        stat_struct['SD'] = SD
        PzItems = []
        for i, z in enumerate(zList):
            PzItem = {}
            PzItem['z'] = float(z)
            PzItem['Pz'] = float(PzList[i])
            PzItems.append(PzItem)
        stat_struct['PzItems'] = PzItems
        json.dump(stat_struct, f, indent=1)
        print(f'saved JSON: {SDRjson}')
    if doPrint:
        print(f'method name: {methodName}')
        print(f'MRE: {MRE:.3f} mm')
        print(f'SD:  {SD:.3f} mm')
        for i, z in enumerate(zList):
            print(f'PZ {z} mm:\t{PzList[i]:.2f} %')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    diffMatrixFile = None
    dirName = None
    dirPath = None
    enablePlotting = True
    plotTitle = None
    useXkcdPlot = True
    useSizeTransform = True

    # kao da ne bi prišlo do prepisovanja - sploh smiselno?
    appendDirName = None

    imagesListFile = config.testImagesFile  # default list is test list from config
    diffMatrixFile = config.diffMatrixFile
    analysisImagesFile = config.analysisImagesFile

    parser.add_argument('-f', '--diffMatrixFile',
                        help=f'path to a diffMatrix.npy',
                        default=diffMatrixFile)
    parser.add_argument('-D', '--dirName',
                        help=f'Use named directory (source, from 3_npys). Just name, not path. Folder (name) will be used also as output. Default: no subdir is used (path from config)',
                        default=dirName)
    parser.add_argument('-P', '--dirPath',
                        help=f'Path to results directory, contaning diffMatrix.npy, GTmatrix.npy, resMatrix.npy, result_img.list, ... Part of the path is also used to generate an ouput path for saving.',
                        default=dirPath)
    parser.add_argument('-L', '--imagesListFile',
                        help=f'set location of images list. Default: {imagesListFile}',
                        default=imagesListFile)
    parser.add_argument('--appendDirName',
                        help=f'Appends dirName (subfolder) with extra string to avoid overwriting',
                        default=appendDirName,)
    parser.add_argument('--plot', '-p',
                        choices=['yes', 'no'],
                        help=f'enable generating plots. Default: {["no","yes"][enablePlotting]}',
                        dest='enablePlotting',
                        default=['no', 'yes'][enablePlotting])
    parser.add_argument('--plotTitle', type=str,
                        help=f'RECOMMENDED! Add optional plot title with method name. Will be aslo aded in JSON for comparison. None if not given',
                        default=plotTitle)
    parser.add_argument('--methodName', type=str,
                        help=f'Same as --plotTitle (alias)',
                        dest='plotTitle',
                        default=plotTitle)
    parser.add_argument('--useXkcd',
                        choices=['yes', 'no'],
                        help=f'enable xkcd/Wang plot style. Warning: svg big size! Default: {["no","yes"][useXkcdPlot]}',
                        dest='useXkcdPlot',
                        default=['no', 'yes'][useXkcdPlot])
    parser.add_argument('--useSizeTransform',
                        choices=['yes', 'no'],
                        help=f'Transform errors (diffMatrix) to match benchmark data size (2400 x 1935) Default: {["no","yes"][useSizeTransform]}',
                        default=['no', 'yes'][useSizeTransform])


    argRes = parser.parse_args()
    dirName = argRes.dirName
    dirPath = argRes.dirPath
    diffMatrixFile = argRes.diffMatrixFile
    imagesListFile = argRes.imagesListFile
    appendDirName = argRes.appendDirName
    enablePlotting = {'yes': True, 'no': False}[argRes.enablePlotting]
    plotTitle = argRes.plotTitle  # also method name!
    useXkcdPlot = argRes.useXkcdPlot
    useSizeTransform = {'yes': True, 'no': False}[argRes.useSizeTransform]

    if dirPath is not None:
        print(f'using input path for npys: {dirPath}')
        npysDir, dirName = dirPath.split(os.path.sep, 1)
        if not os.path.samefile(npysDir, config.npysDir):
            raise Exception(f'path {dirPath} is not a subpath of npys dir: {config.npysDir}')

    dirNameOutpt = dirName
    if appendDirName is not None:
        dirNameOutpt = dirName+appendDirName

    if dirName is not None:
        print(f'using modified dir to READ/WRITE npys: {dirName}')
        diffMatrixFile = config.getModifiedPath(diffMatrixFile, dirName, autoCreate=False)
        analysisImagesFile = config.getModifiedPath(analysisImagesFile, dirName, autoCreate=False)
        
        # writing
        SDRfile = config.getModifiedPath(SDRfile, dirNameOutpt, autoCreate=True)
        SDRjson = config.getModifiedPath(SDRjson, dirNameOutpt, autoCreate=True)
        targetFolder = os.path.join(targetFolder, dirNameOutpt)
        print(f'DEBUG: subfolder.\ntargetFolder:{targetFolder}\nSDRfile:{SDRfile}')
    else:
        if not os.path.exists(targetFolder):
            os.makedirs(targetFolder)
            print('DEBUG: ceating folder non-subfolders')

    if diffMatrixFile is None:
        raise Exception('diffMatrixFile is None. Either set diffMatrixFile or dirName')

    diffMatrix = np.load(diffMatrixFile)

    print(f'analyzing images in list file {imagesListFile}')
    imagesList = filelist.read_file_list(imagesListFile)
    for i in range(len(imagesList)):
        imagesList[i] = os.path.abspath(imagesList[i])

    # get filename (without path)
    # 3_images_resize/78_Image0001_Type17.png -> 78_Image0001_Type17.png
    # if i -> ignore empty line
    imagesFilename = [os.path.basename(i) for i in imagesList if i]
    analysisImages = filelist.read_file_list(analysisImagesFile)
    testImagesInds = filelist.compare_file_lists(analysisImages, imagesList, only_file_name=True)

    diffMatrix = diffMatrix[testImagesInds, :]

    if useSizeTransform:
        diffMatrix = transform_diffMatrix_to_bench_image_size(diffMatrix, imagesList)

    # MRE - scalar, mean radial error (in mm)
    # SD - scalar, standard deviation (in mm)
    # PzList - array of scalar - SDRs (in %)
    MRE, SD, PzList = detectionRate(diffMatrix)

    saveResults(SDRfile, SDRjson, MRE, SD, PzList, doPrint=True, methodName=plotTitle, dirName=dirName)
    
    if enablePlotting:
        ticks = ['2 mm', '2.5 mm', '3 mm', '4 mm']
        colors = ['#fff794', '#c6f5c9', '#c6e3f5', '#f6c5c9']
        # make plotfile (name, path) out of SDRfile
        # without file extentions - function will add both .png and .svg
        plotFile = os.path.splitext(SDRfile)[0]

        plotBench(PzList, ticks, colors, plotFile=plotFile,
                  plotTitle=plotTitle, figsize=(3.5, 6),
                  useXkcdPlot=useXkcdPlot)

