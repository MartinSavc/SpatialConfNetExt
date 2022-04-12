import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import argparse
from shutil import copyfile
# import pandas
# import functools

# global varibles
enbaleCheckingMatrixShapes = True
# distance between modelst for stripping

sys.path += ["../"]
try:
    import KPnet_config as config
    import commonlib.filelist as filelist
except ModuleNotFoundError as e:
    configPathMissing = os.path.abspath(sys.path[-1])
    print('\nERROR: KPnet_config.py missing in: ' + configPathMissing + '\n\n')
    print(e)
    raise e


def printErrStats3(errs1, errs2, errs3):
    errsMean1 = np.mean(errs1)
    errsMean2 = np.mean(errs2)
    errsMean3 = np.mean(errs3)
    std1 = np.std(errs1)
    std2 = np.std(errs2)
    std3 = np.std(errs3)
    print(f'mean1: {errsMean1:5.2f} +- {std1:5.2f}')
    print(f'mean2: {errsMean2:5.2f} +- {std2:5.2f}')
    print(f'meanM: {errsMean3:5.2f} +- {std3:5.2f}')
    
    def printPercs(percVals, data):
        percs = np.percentile(data, percVals)
        for j in range(len(percVals)):
            print('{:2}={:5.2f}, '.format(percVals[j], percs[j]), end='')
        print('')

    percVals = [50, 90, 99]

    print('perc1: ', end='')
    printPercs(percVals, errs1)
    print('perc2: ', end='')
    printPercs(percVals, errs2)
    print('percM: ', end='')
    printPercs(percVals, errs3)


def funBegin(res1, diff1, res2, diff2):
    if enbaleCheckingMatrixShapes:
        if res1.shape != res2.shape:
            raise Exception(f'shapes not equal. 1: {res1.shape}, 2: {res2.shape}')
    res3 = np.zeros(res1.shape)
    diff3 = np.zeros(res1.shape)
    ims, pos, _ = res1.shape

    errs1 = (diff1[:, :, :2]**2).sum(2)**0.5
    errs2 = (diff2[:, :, :2]**2).sum(2)**0.5
    return res3, diff3, ims, pos, errs1, errs2


def mergeMinDiff(res1, diff1, res2, diff2):
    ''' merges 2 results by choosing smallest errors for each point
    '''
    res3, diff3, ims, pos, errs1, errs2 = funBegin(res1, diff1, res2, diff2)
    
    count1 = 0
    count2 = 0
    for i in range(ims):
        for p in range(pos):
            if errs1[i, p] <= errs2[i,p]:
                res3[i, p] = res1[i, p]
                diff3[i, p] = diff1[i, p]
                count1 += 1
            else:
                res3[i, p] = res2[i,p]
                diff3[i, p] = diff2[i,p]
                count2 += 1
    print(f'count1: {count1}, count2: {count2}')

    errs3 = (diff3[:, :, :2]**2).sum(2)**0.5
    printErrStats3(errs1, errs2, errs3)
    return res3, diff3


def mergeNormalizePerPoint(res1, diff1, res2, diff2):
    '''
    use normalization mean,std per point (all images)
    '''
    res3, diff3, ims, pos, errs1, errs2 = funBegin(res1, diff1, res2, diff2)

    count1 = 0
    count2 = 0
    for p in range(pos):
        probs1 = res1[:, p, 2]
        probMean1 = np.mean(probs1)
        probStd1 = np.std(probs1)
        
        # podobno za drugi pristop
        probs2 = res2[:, p, 2]
        probMean2 = np.mean(probs2)
        probStd2 = np.std(probs2)
        
        for i in range(ims):
            x1 = probs1[i]
            x1_out = (x1 - probMean1)/probStd1
            
            x2 = probs2[i]
            x2_out = (x2 - probMean2)/probStd2
            
            if x1_out >= x2_out:
                res3[i, p] = res1[i, p]
                diff3[i, p] = diff1[i, p]
                count1 += 1
            else:
                res3[i, p] = res2[i, p]
                diff3[i, p] = diff2[i, p]
                count2 += 1
    print(f'count1: {count1}, count2: {count2}')
    errs3 = (diff3[:, :, :2]**2).sum(2)**0.5

    printErrStats3(errs1, errs2, errs3)
    return res3, diff3


def mergeNormalizePerImage(res1, diff1, res2, diff2):
    '''
    use normalization mean,std per image (all points)
    '''
    res3, diff3, ims, pos, errs1, errs2 = funBegin(res1, diff1, res2, diff2)

    count1 = 0
    count2 = 0
    for i in range(ims):
        probs1 = res1[i, :, 2]
        probMean1 = np.mean(probs1)
        probStd1 = np.std(probs1)
        
        # podobno za drugi pristop
        probs2 = res2[i, :, 2]
        probMean2 = np.mean(probs2)
        probStd2 = np.std(probs2)
        
        for p in range(pos):
            x1 = probs1[p]
            x1_out = (x1 - probMean1)/probStd1
            
            x2 = probs2[p]
            x2_out = (x2 - probMean2)/probStd2
            
            if x1_out >= x2_out:
                res3[i, p] = res1[i, p]
                diff3[i, p] = diff1[i, p]
                count1 += 1
            else:
                res3[i, p] = res2[i,p]
                diff3[i, p] = diff2[i,p]
                count2 += 1
    print(f'count1: {count1}, count2: {count2}')
    errs3 = (diff3[:, :, :2]**2).sum(2)**0.5

    printErrStats3(errs1, errs2, errs3)
    return res3, diff3


def mergeNormalize(res1, diff1, res2, diff2):
    '''
    Za vsak pristop normaliziraj te verjetnosti kot (x - povprečje)/standardni_odklon. Tako dobiš dva seznama normaliziranih verjetnosti. Nato združi rezultate tako, da pri vsaki kefalometrični točki vzameš rezultat, kjer je bila ta normalizirana verjetnost višja.
    '''
    res3, diff3, ims, pos, errs1, errs2 = funBegin(res1, diff1, res2, diff2)
    
    # matrika MxN verjetnosti za M slik in N točk za prvi pristop
    probs1 = res1[:, :, 2]
    # povprečna verjetnost čez vse slike in točke za prvi pristop
    probMean1 = np.mean(probs1)
    # standardni odklon čez vse slike in točke za prvi pristop
    probStd1 = np.std(probs1)
    
    # podobno za drugi pristop
    probs2 = res2[:, :, 2]
    probMean2 = np.mean(probs2)
    probStd2 = np.std(probs2)
    
    count1 = 0
    count2 = 0
    for i in range(ims):
        for p in range(pos):
            x1 = probs1[i, p]
            x1_out = (x1 - probMean1)/probStd1
            
            x2 = probs2[i, p]
            x2_out = (x2 - probMean2)/probStd2
            
            if x1_out >= x2_out:
                res3[i, p] = res1[i, p]
                diff3[i, p] = diff1[i, p]
                count1 += 1
            else:
                res3[i, p] = res2[i,p]
                diff3[i, p] = diff2[i,p]
                count2 += 1
                
    print(f'count1: {count1}, count2: {count2}')

    errs3 = (diff3[:, :, :2]**2).sum(2)**0.5

    printErrStats3(errs1, errs2, errs3)
    return res3, diff3


def mergeByProb(res1, diff1, res2, diff2, prob):
    '''
    merge resMatrix 1 and 2 and diffMatrix 1 and 2
    if resMatrixFile1 point probablility [i,p,2] is greater than prob
    matrix1 will be used, else matrix 2
    1 should be matrix with small labels (11px), 2 should be with big lables (120px)

    returns merged resMatrix and diffMatrix
    '''
    res3, diff3, ims, pos, errs1, errs2 = funBegin(res1, diff1, res2, diff2)
    
    count1 = 0
    count2 = 0
    for i in range(ims):
        for p in range(pos):
            #print(res1[i, p, 2])
            if res1[i, p, 2] > prob:
                res3[i, p] = res1[i, p]
                diff3[i, p] = diff1[i, p]
                count1 += 1
            else:
                res3[i,p] = res2[i,p]
                diff3[i,p] = diff2[i,p]
                count2 += 1
                
    print(f'count1: {count1}, count2: {count2}')
    errs1 = (diff1[:, :, :2]**2).sum(2)**0.5
    errs2 = (diff2[:, :, :2]**2).sum(2)**0.5
    errs3 = (diff3[:, :, :2]**2).sum(2)**0.5

    printErrStats3(errs1, errs2, errs3)
    return res3, diff3


def stripByDistance(res1, diff1, res2, diff2, dist=200):
    '''
    zavrže tiste rezultate kjer je razdalja med obema napovema velika (200px)
    '''
    res3, diff3, ims, pos, errs1, errs2 = funBegin(res1, diff1, res2, diff2)

    
    diff1a = diff1.copy()
    diff2a = diff2.copy()
    dists = ((res1[:, :, :2] - res2[:, :, :2])**2).sum(2)**0.5
    countNan = 0
    for i in range(ims):
        for p in range(pos):
            if dists[i, p] > dist:
                countNan += 1
                diff1a[i, p, :] = np.nan
                diff2a[i, p, :] = np.nan
    
    errs1 = (diff1[:, :, :2]**2).sum(2)**0.5
    errs2 = (diff2[:, :, :2]**2).sum(2)**0.5
    errsMean1 = np.mean(errs1)
    errsMean2 = np.mean(errs2)
    std1 = np.std(errs1)
    std2 = np.std(errs2)


    errs1a = np.sum((diff1a[:, :, :2]**2), axis=2)**0.5
    errs2a = np.sum((diff2a[:, :, :2]**2), axis=2)**0.5
    errsMean1a = np.nanmean(errs1a)
    errsMean2a = np.nanmean(errs2a)
    std1a = np.nanstd(errs1a)
    std2a = np.nanstd(errs2a)
    
    def printPercs(percVals, data):
        #percs = np.percentile(data, percVals)
        percs = np.nanpercentile(data, percVals)
        for j in range(len(percVals)):
            print('{:2}={:5.2f}, '.format(percVals[j], percs[j]), end='')
        print('')
    percVals = [25, 50, 90, 99]
    
    print(f'stripping distance = {dist} px')
    print(f'stripped count: {countNan} (of {ims*pos} = {100*countNan/(ims*pos):.1f}%)')
    print('\nmodel1')
    print(f'mean1: {errsMean1:5.2f} +- {std1:5.2f}')
    print('perc1: ', end='')
    printPercs(percVals, errs1)
    print('stripped')
    print(f's_me1: {errsMean1a:5.2f} +- {std1a:5.2f}')
    print('s_pe1: ', end='')
    printPercs(percVals, errs1a)
    
    print('\nmodel2')
    print(f'mean2: {errsMean2:5.2f} +- {std2:5.2f}')
    print('perc2: ', end='')
    printPercs(percVals, errs2)
    print('stripped')
    print(f's_me2: {errsMean2a:5.2f} +- {std2a:5.2f}')
    print('s_pe2: ', end='')
    printPercs(percVals, errs2a)

    #import pdb
    #pdb.set_trace()


def TMPcomparePlot2models():
    # %pylab qt5
    diff12 = np.load('/home/gsedej/Delo/CephBot/KPnet-CephBot/29_DS12/KPnet/3_analysis/3_npys/DS12-aug-gauss-11px-s1.5_g0.5-m0.5-ss20000_kpn-tr-h5-num128-LW1e-4-po72_120px-FT31px-FT11px_20200603-092649_iter_250000-test/diffMatrix.npy')

    diff09 = np.load('/home/gsedej/Delo/CephBot/KPnet-CephBot/24_DS09_aug/KPnet/3_analysis/3_npys/DS09-aug-gauss-11px-s1.5_g0.5-m0.5-ss20000_kpn-tr-h5-num128-LW1e-4_120px-FT31px-FT11pxAug_20200219-172924_iter_200000-test/diffMatrix.npy')

    diff12all = diff12.copy()
    diff12 = diff12all[:200,:,:]

    l2_09 = np.sum(diff09[:, :, :2]**2, 2)**0.5
    l2_12 = np.sum(diff12[:, :, :2]**2, 2)**0.5

    plt.plot(np.mean(l2_12, axis=1), 'g.')
    plt.plot(np.mean(l2_09, axis=1), 'b.')

    mean12 = np.mean(l2_12)

    mean09 = np.mean(l2_09)

    plt.title(f'KPNet. L2 error for landmarks, average by images. Blue DS09, green DS12 (200 images). Mean: DS09={mean09:.2f}, DS12={mean12:.2f}')
    plt.title(f'KPNet. L2 error for images, average by landmarks. Blue DS09, green DS12 (200 images). Mean: DS09={mean09:.2f}, DS12={mean12:.2f}')
    _ = plt.xticks(np.arange(0, 200), rotation=90)


if __name__ == '__main__':
    resMatrixFile = config.resMatrixFile
    GTmatrixFile = config.GTmatrixFile
    diffMatrixFile = config.diffMatrixFile

    enablePlotting = False
    #newDirMerged = 'merged'
    newDirMerged = ''
    dirName1 = ''
    dirName2 = ''
    prob = 0.1
    enableN1 = False  # mergeMinDiff
    enableN2 = False  # mergeByProb
    enableN3 = False  # mergeNormalize
    enableN4 = False  # mergeNormalizePerImage
    enableN5 = False  # mergeNormalizePerPoint
    
    enableS1 = False  # stripByDistance
    stripDist = 200
    
    # help=f'Use named directory (source, from 3_npys). Just name, not path. Folder (name) will be used also as output. Default: no subdir is used (path from config)',
    parser = argparse.ArgumentParser()
    parser.add_argument('-D1', '--dirName1',
                        help='model name or better model (small labels)',
                        default=dirName1)
    parser.add_argument('-D2', '--dirName2',
                        help='worse model (big labels)',
                        default=dirName2)
    # parser.add_argument('-DX', '--dirNames', help='multiple models')

    
    parser.add_argument('-N1', '--enableN1',
                        choices=['yes', 'no'],
                        help=f'Enable normalization 1: mergeMinDiff (always use minimum) Default: {["no","yes"][enableN1]}',
                        default=['no', 'yes'][enableN1])
    parser.add_argument('-N2', '--enableN2',
                        choices=['yes', 'no'],
                        help=f'Enable normalization 1: mergeByProb. use --prob (!!!) probs that are <(--prob) from model1 will be replaced with values from model2 Default: {["no","yes"][enableN2]}',
                        default=['no', 'yes'][enableN2])
    parser.add_argument('-p', '--prob', help='probablility limit',
                        type=float, default=prob)
    parser.add_argument('-N3', '--enableN3',
                        choices=['yes', 'no'],
                        help=f'Enable normalization 3: mergeNormalize (images and points) Default: {["no","yes"][enableN3]}',
                        default=['no', 'yes'][enableN3])
    parser.add_argument('-N4', '--enableN4',
                        choices=['yes', 'no'],
                        help=f'Enable normalization 4: mergeNormalizePerImage Default: {["no","yes"][enableN4]}',
                        default=['no', 'yes'][enableN4])
    parser.add_argument('-N5', '--enableN5',
                        choices=['yes', 'no'],
                        help=f'Enable normalization 4: mergeNormalizePerPoint Default: {["no","yes"][enableN5]}',
                        default=['no', 'yes'][enableN5])
    
    parser.add_argument('-S1', '--enableS1',
                        choices=['yes', 'no'],
                        help=f'Enable stripping 1: stripByDistance Default: {["no","yes"][enableS1]}',
                        default=['no', 'yes'][enableS1])
    #stripDist
    parser.add_argument('--dist', help='distance for strip',
                        type=int, default=stripDist)
    
    
    parser.add_argument('-M', '--mergedDir',
                        help='Name of merged model (enable saving). empty wont save. only one merge!',
                        default=newDirMerged)
    parser.add_argument('--enablePlotting',
                        choices=['yes', 'no'],
                        help=f'Dnable generating plots (see code). Will be saved in subdir. Default: {["no","yes"][enablePlotting]}',
                        default=['no', 'yes'][enablePlotting])
    parser.add_argument('--enableMatrixCheck', '-C',
                        choices=['yes', 'no'],
                        help=f'Enable checking if 2 matrices have same shapes Default: {["no","yes"][enbaleCheckingMatrixShapes]}',
                        default=['no', 'yes'][enbaleCheckingMatrixShapes])

    # parsing
    argRes = parser.parse_args()
    dirName1 = argRes.dirName1
    dirName2 = argRes.dirName2
    prob = float(argRes.prob)
    newDirMerged = argRes.mergedDir
    
    stripDist = argRes.dist
    
    enablePlotting = {'yes': True, 'no': False}[argRes.enablePlotting]
    enableN1 = {'yes': True, 'no': False}[argRes.enableN1]
    enableN2 = {'yes': True, 'no': False}[argRes.enableN2]
    enableN3 = {'yes': True, 'no': False}[argRes.enableN3]
    enableN4 = {'yes': True, 'no': False}[argRes.enableN4]
    enableN5 = {'yes': True, 'no': False}[argRes.enableN5]
    enableS1 = {'yes': True, 'no': False}[argRes.enableS1]
    enbaleCheckingMatrixShapes = {'yes': True, 'no': False}[argRes.enableMatrixCheck]


    GTmatrixFile = config.getModifiedPath(GTmatrixFile, dirName1, autoCreate=False)
    # small labels
    resMatrixFile1 = config.getModifiedPath(resMatrixFile, dirName1, autoCreate=False)
    diffMatrixFile1 = config.getModifiedPath(diffMatrixFile, dirName1, autoCreate=False)
    # big labels
    resMatrixFile2 = config.getModifiedPath(resMatrixFile, dirName2, autoCreate=False)
    diffMatrixFile2 = config.getModifiedPath(diffMatrixFile, dirName2, autoCreate=False)

    parentDir = os.path.abspath(os.path.join(resMatrixFile1, os.pardir))
    modelName1 = os.path.split(os.path.abspath(os.path.join(resMatrixFile1, os.pardir)))[-1]
    modelName2 = os.path.split(os.path.abspath(os.path.join(resMatrixFile2, os.pardir)))[-1]

    GT = np.load(GTmatrixFile)
    res1 = np.load(resMatrixFile1)
    res2 = np.load(resMatrixFile2)
    diff1 = np.load(diffMatrixFile1)
    diff2 = np.load(diffMatrixFile2)
    
    # martin npy hack (last 200 are test)
    martinHack = True
    if martinHack is True:
        res1 = res1[-200:]
        diff1 = diff1[-200:]
        res2 = res2[-200:]
        diff2 = diff2[-200:]

    
    
    
    #mergeMinDiff(res1, diff1, res2, diff2)
    #resMatrixMerged, diffMatrixMerged = mergeNormalizePerImage(res1, diff1, res2, diff2)
    print(f'modelName1:\n{modelName1}')
    print(f'modelName2:\n{modelName2}')
    if enableN1:
        print('\nmergeMinDiff')
        resMatrixMerged, diffMatrixMerged = mergeMinDiff(
            res1, diff1, res2, diff2)
    if enableN2:
        print(f'\nmergeByProb, prob: {prob}')
        resMatrixMerged, diffMatrixMerged = mergeByProb(
            res1, diff1, res2, diff2, prob)
    if enableN3:
        print('\nmergeNormalize')
        resMatrixMerged, diffMatrixMerged = mergeNormalize(
            res1, diff1, res2, diff2)
    if enableN4:
        print('\nmergeNormalizePerImage')
        resMatrixMerged, diffMatrixMerged = mergeNormalizePerImage(
            res1, diff1, res2, diff2)
    if enableN5:
        print('\nmergeNormalizePerPoint')
        resMatrixMerged, diffMatrixMerged = mergeNormalizePerPoint(
            res1, diff1, res2, diff2)
    if enableS1:
        print('\nstripByDistance')
        stripByDistance(res1, diff1, res2, diff2, dist=stripDist)

    
        # save to new folder
    if newDirMerged is not '':
        print(f'saving to folder {newDirMerged} (overwriting)')
        newFolderPath = os.path.join(parentDir, os.pardir, newDirMerged)
        os.makedirs(newFolderPath, exist_ok=True)

        # new files
        GTMatrixNewFile = os.path.join(newFolderPath, "GTmatrix.npy")
        resMatrixMergedFile = os.path.join(newFolderPath, "resMatrix.npy")
        diffMatrixMergedFile = os.path.join(newFolderPath, "diffMatrix.npy")

        np.save(GTMatrixNewFile, GT)
        np.save(resMatrixMergedFile, resMatrixMerged)
        np.save(diffMatrixMergedFile, diffMatrixMerged)

        copyfile(os.path.join(parentDir, 'result_img.list'), os.path.join(newFolderPath, 'result_img.list'))

    # statistics ???
    # todo     errs1 = (diff1[:, :, :2]**2).sum(2)**0.5
    probs1rav = res1[:, :, 2].ravel()
    probs2rav = res2[:, :, 2].ravel()
    errs1 = (diff1[:, :, :2]**2).sum(2)**0.5
    errs2 = (diff2[:, :, :2]**2).sum(2)**0.5

    # percentiles
    percs = [0.1, 0.2, 0.5, 0.8, 0.9]
    perc1 = np.percentile(probs1rav, percs)
    perc1 = np.percentile(probs2rav, percs)

    if enablePlotting:
        os.makedirs('tmp_plots', exist_ok=True)
        
        fig = plt.figure(figsize=(19.20, 10.80))
        plt.plot(errs1, errs2, '.', color='b')
        plt.xlabel(modelName1)
        plt.ylabel(modelName2)
        plt.tight_layout()
        plt.savefig('tmp_plots/model_1vs2.png')
        plt.xlim([-2, 100])
        plt.ylim([-2, 100])
        plt.tight_layout()
        plt.savefig('tmp_plots/model_1vs2_100px.png')

        fig = plt.figure(figsize=(19.20, 10.80))
        n, bins, patches = plt.hist(probs1rav, bins=100)
        plt.xlabel('Probablitly')
        plt.title(f'model: {modelName1}')
        plt.tight_layout()
        plt.savefig('tmp_plots/hist1.png')

        fig = plt.figure(figsize=(19.20, 10.80))
        n, bins, patches = plt.hist(probs2rav, bins=100)
        plt.xlabel('Probablitly')
        plt.title(f'model: {modelName2}')
        plt.tight_layout()
        plt.savefig('tmp_plots/hist2.png')
        
        # martin predlog
        '''
        x-os: napaka povprecja napovedi obeh modelov
        y-ox: razdalja med napovedmi obeh modelov
        
        res1-res2 <- razdalja
        gt - (res1+res2)/2 <- napaka povprecja

        '''
        xData = np.mean([errs1, errs2], axis=0)
        #yData = 
        #diff1-diff2
        
        #errs3 = ((diff2[:, :, :2] - diff1[:, :, :2])**2).sum(2)**0.5
        #yData = errs3
        
        #yData = ((diff2[:, :, :2] - diff1[:, :, :2])**2).sum(2)**0.5
        #xData = 1/2 * ((diff2[:, :, :2] + diff1[:, :, :2])**2).sum(2)**0.5
        yData = ((res1[:, :, :2] - res2[:, :, :2])**2).sum(2)**0.5
        yData = yData.ravel()
        xData = GT - (res1[:, :, :2] + res2[:, :, :2])/2
        xData = ((xData**2).sum(2)**0.5).ravel()
        
        fig = plt.figure(figsize=(19.20, 10.80))
        plt.plot(xData, yData, '.', color='b')
        plt.xlabel('napaka povprecja napovedi obeh modelov')
        plt.ylabel('razdalja med napovedmi obeh modelov')
        plt.title(f'model1: {modelName1}, model2: {modelName2}')
        plt.tight_layout()
        plt.savefig('tmp_plots/martinPlot.png')
        plt.xlim([-5, 200])
        plt.ylim([-5, 200])
        plt.tight_layout()
        plt.savefig('tmp_plots/martinPlot_200px.png')
        plt.xlim([-2, 50])
        plt.ylim([-2, 50])
        plt.tight_layout()
        plt.savefig('tmp_plots/martinPlot_50px.png')
        

        # hist per point
        #errs1 = (diff1[:, :, :2]**2).sum(2)**0.5
        '''
        for i in range(38):
            #errs1[:,i]
            #n, bins, patches = plt.hist(errs1[:,i], bins=100)
            figure()
            n, bins, patches = plt.hist(res1[:,i,2], bins=10)
            #plt.xlabel(f'point: {i}')
            plt.title(i)
        '''
    '''
    for i in range(38):
        data = res1[:, i, 2]
        me = np.mean(data)
        st = np.std(data)
        med = np.median(data)
        percVals = [10, 20, 80, 90]
        percs = np.percentile(data, percVals)
        print(i)
        print('med: {:5.2f} +- {:2.2f}'.format(med,st))
        print('perceniles: ', end='')
        for j in range(len(percVals)):
            print('{:2}={:2.2f}, '.format(percVals[j], percs[j]), end='')
        print('')
    '''
