import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import KPnet_config as config


def savePlotPngSubfolder(fig, saveFNameSvg, dpi=200):
    '''
    saves plot fig as png in pngs subfolder
    saveFNameSvg: original savename (svg). will be given png extension
    dpi: dpi=100-> 1920*1080 (~1MB) dpi=200 -> 3840x2160 (megabytes!)
    '''
    pathSplit = os.path.split(saveFNameSvg)
    pathPngFold = os.path.join(pathSplit[0], 'pngs')
    os.makedirs(pathPngFold, exist_ok=True)
    pathPngFile = os.path.join(pathSplit[0], 'pngs', pathSplit[1]+'.png')
    fig.savefig(pathPngFile, dpi=dpi)


def getMinMaxXYCoordRes(coordRes):
    '''
    returns the minimum and maximum width/height of points in the coordRes
    '''
    min_w = np.finfo(np.float32).max  # x
    min_h = np.finfo(np.float32).max  # y
    max_w = np.finfo(np.float32).min  # x
    max_h = np.finfo(np.float32).min  # y
    
    for currPointInd in range(len(coordRes)):
        x = coordRes[currPointInd, 1]
        y = coordRes[currPointInd, 0]
        if x < min_w:
            min_w = x
        if x > max_w:
            max_w = x
        if y < min_h:
            min_h = y
        if y > max_h:
            max_h = y
    return min_w, min_h, max_w, max_h


def drawImageWithPoints(
        img, coordRes,
        save=False,
        savePng=False,
        savePngDpi=100,
        saveA4cropedFigure=False,
        saveFName='allPoints.svg',
        imInd=-1,
        imFName='',
        stats='',
        LMsymbols=[],
        predProbablitys=None,
        interactive=False):
    """
    na en sliki prikaze vse pare tock (GT in napovedi)

    :param img: image to display
    :param coordRes: koordinate za vse tocke [N, 4 (praviY, praviX, predicY, preditctX) ]
    """
    fig = plt.figure(figsize=(19.20, 10.80))
    im = plt.imshow(img, cmap=plt.cm.gray, interpolation='none')
    plt.plot(coordRes[:, 1], coordRes[:, 0], '.b')  # GT
    plt.plot(coordRes[:, 3], coordRes[:, 2], 'xr')  # prediction
    # še povežem med sabo da se vidi kam spada ko je daleč...
    plt.plot(coordRes[:, (1, 3)].T, coordRes[:, (0, 2)].T, 'g')
    plt.axis('off')
    plt.axis('image')
    #plotTitleText = 'test image id=' + str(imInd) + ', w=' + str(imWidth) + ', h=' + str(imHeight) \
    #    + ', Fname=' + imFName + ' ' + stats + ", blue:GT, red:pred"
    plotTitleText = f'origRes=h:{img.shape[0]},w:{img.shape[1]}, caffe=h:{config.imHeight},w:{config.imWidth},id={imInd}, {stats}, {imFName} (blue:GT, red:pred)'
    plt.title(plotTitleText)

    #napisi poleg label
    linewidth = .0
    fontsize = 4
    ax1 = plt.gca()
    for currPointInd in range(len(coordRes)):
        txt = LMsymbols[currPointInd]
        if predProbablitys is not None:
            prob = predProbablitys[currPointInd]
            txt = f'{txt} {prob:.2f}'
        ax1.annotate(
            txt,
            xy=(coordRes[currPointInd, 1]-0, coordRes[currPointInd, 0]-3),
            color='hotpink',
            path_effects=[PathEffects.withStroke(linewidth=linewidth, foreground="w")],
            verticalalignment='bottom',
            fontsize=fontsize,
            )
        # ce se tiska povecavo tako da so ravno vse tocke na A4 "portrait"
        # potem je fontsize 3 in barva hotpink najmanjse da je prebrati z dobrimi ocmi (svetla/temna podlaga)
        # deeppink je tezko na crni
        # stroke/linewidth ne pomaga pri tako majhnem fontu
        # UPDATE: font 3 je pramajhen. -> 4

    plt.tight_layout()
    fig.subplots_adjust(left=0, right=1, top=0.97, bottom=0, wspace=0.01, hspace=0.01)
    plt.draw()
    if interactive:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
        plt.waitforbuttonpress(1)
    if save:
        plt.savefig(saveFName)
        if savePng:
            savePlotPngSubfolder(fig=plt, saveFNameSvg=saveFName, dpi=savePngDpi)

    if saveA4cropedFigure:
        # fit-a (zoom) vse tocke v A4 portrait sliko
        # preveri min in max po visini in sirini za vse tocke -> 4 tocke.
        # med min im max razdalja -> zmanjsaj za 10% da dobis malo roba
        fig.set_size_inches(8.27, 11.69)  # A4 portrait

        min_w, min_h, max_w, max_h = getMinMaxXYCoordRes(coordRes)

        # procenti: po visini imamo vec prostora, zato lahko vec pikslov slike pokazemo
        proc_h = 0.2
        pad_h = int(((max_h - min_h)*proc_h)/2)
        max_h = max_h + pad_h
        min_h = min_h - pad_h

        proc_w = 0.05
        pad_w = int(((max_w - min_w)*proc_w)/2)
        max_w = max_w + pad_w
        min_w = min_w - pad_w

        ax1.set_xlim(min_w, max_w)
        ax1.set_ylim(max_h, min_h)

        plotTitleText = f'origRes=h:{img.shape[0]},w:{img.shape[1]}, caffe=h:{config.imHeight},w:{config.imWidth},id={imInd},\n {stats}, {imFName} (blue:GT, red:pred)'
        plt.title(plotTitleText)
        
        plt.tight_layout()
        fig.subplots_adjust(left=0, right=1, top=0.97, bottom=0, wspace=0.01, hspace=0.01)
        plt.draw()
        
        dirName = os.path.dirname(saveFName)
        fileName = os.path.basename(saveFName)
        subdir = 'A4print'
        subdir = os.path.join(dirName, subdir)
        os.makedirs(subdir, exist_ok=True)
        a4FName = os.path.join(subdir, fileName)
        plt.savefig(a4FName)


def getPlotTxtColors(
        errRelProc,
        prob,
        useColorText,
        pltColTxtGood='#006400',
        pltColTxtBad='#FF8C00',
        pltColTxtMiddle='#8B0000',
        ):
    '''
    TODO: napisi boljso dokumentacijo
    vrne 4 barve glede na vrednosti relativne napake (euclid) (errRelProc) in verjetnosti (prob). kombinacije
    '''
    if not useColorText:
        return 'k', 'k', 'k', ''
    # err
    # green:good,low relErr: <.5%
    # red:bad, high relErr: >1.0%
    # orange:middle, medimum relErr
    txtColErr = ''
    if errRelProc < .5:
        txtColErr = pltColTxtGood
    elif errRelProc > 1.0:
        txtColErr = pltColTxtBad
    else:
        txtColErr = pltColTxtMiddle
    # prob
    # green:good,high prob: >.7
    # red:bad,low prob: <.3
    # orange:middle, medimum prob
    txtColProb = ''
    #txtCol=pltColTxtGood if prob>0.8 elif
    if prob > .7:
        txtColProb = pltColTxtGood
    elif prob < .3:
        txtColProb = pltColTxtBad
    else:
        txtColProb = pltColTxtMiddle

    # SPECIAL
    # if high error and high prob!
    # or h err &  m prob
    # or m err & h prob

    txtSpec = ''
    colSpec = ''
    if (txtColErr == pltColTxtBad) & (txtColProb == pltColTxtGood):
        txtSpec = "!hi err hi prob"
        colSpec = pltColTxtBad
    if (txtColErr == pltColTxtMiddle) & (txtColProb == pltColTxtGood):
        txtSpec = 'mid err hi prbo'
        colSpec = pltColTxtMiddle
    if (txtColErr == pltColTxtBad) & (txtColProb == pltColTxtMiddle):
        txtSpec = 'hi err mid prbo'
        colSpec = pltColTxtMiddle

    # bolji
    if (txtColErr == pltColTxtGood) & (txtColProb == pltColTxtGood):
        txtSpec = 'great'
        colSpec = pltColTxtGood
    if (txtColErr == pltColTxtMiddle) & (txtColProb == pltColTxtGood):
        txtSpec = 'mid err hi prob'
        colSpec = pltColTxtGood
    if (txtColErr == pltColTxtGood) & (txtColProb == pltColTxtMiddle):
        txtSpec = 'low err mid prob'
        colSpec = pltColTxtGood
    return txtColErr, txtColProb, colSpec, txtSpec

def getDiagonal(h, w):
    return np.sqrt(np.power(h, 2) + np.power(w, 2))

def drawImageWithPointsIdividually(
        img,
        coordRes,
        pointsCnt,
        save=False,
        savePng=False,
        savePngDpi=100,
        saveFName='perPoint.svg',
        LMsymbols=[],
        predProbablitys=-1,
        useColorText=True,
        interactive=False,
        ):
    """
    prikze kompozicijo posameznih 19 tock na trenutni (testni) sliki
    izrise tudi evklidsko napako (err)

    :param net: net objekt od caffeja (po net.forward)
    :param coordRes: koordinate za vse tocke [N, 4 (praviY, praviX, predicY, preditctX) ]
    """
    rows = 6; cols = 12  # fiksno določeno
    winWidth = 50
    fig = plt.figure(figsize=(19.20, 10.80))
    #ax1 = plt.subplot(rows, cols, 1)
    h = img.shape[0]
    w = img.shape[1]
    imDiag = getDiagonal(h=h, w=w)
    linewidth = 0
    fontsize = 8

    for r in range(0, rows):
        for c in range(0, cols):
            currPointInd = r*cols+c
            # do makismane tocke, sicer zaključi
            if currPointInd >= pointsCnt:
                break
            ax1 = plt.subplot(rows, cols, currPointInd+1)
            # plots landmark symbol over the GT point
            ax1.annotate(LMsymbols[currPointInd],
                         xy=(coordRes[currPointInd, 1]-10, coordRes[currPointInd, 0]-35),
                         color='g', path_effects=[PathEffects.withStroke(linewidth=linewidth, foreground="w")],
                         verticalalignment='bottom', fontsize=fontsize)
            plt.plot([0,1])
            # izris labele in slike z alpha
            # imLabel =plt.imshow(net.blobs['label'].data[0,currPointInd], cmap=plt.cm.gray, interpolation='none', vmin=0., vmax=1.)
            # imXray = plt.imshow(net.blobs['data']. data[0,0], cmap=plt.cm.gray, interpolation='none', alpha=.5, vmin=0., vmax=1.)
            # samo slika
            imXray = plt.imshow(img, cmap=plt.cm.gray, interpolation='none')
            # set GT point
            plt.plot(coordRes[currPointInd, 1], coordRes[currPointInd, 0], '.b')
            plt.axis('off')
            plt.axis('image')
            # prediction
            plt.plot(coordRes[currPointInd, 3], coordRes[currPointInd, 2], 'xr')
            errPx = ((coordRes[currPointInd, :2]-coordRes[currPointInd, 2:])**2).sum()**.5
            # relative error: ex: errPx: 8px, diag: 3100 = 0.0025 * 100 = 0.25%
            errRelProc = (errPx / imDiag)*100

            prob = predProbablitys[currPointInd]
            pointInfo = f'err={errPx:.2f}px\nrel={errRelProc:.2f}%'
            pointInfo2 = f'prob:{prob:.2f}'
            
            # color of text depending on probablility (prob)
            txtColErr, txtColProb, colSpec, txtSpec = getPlotTxtColors(errRelProc, prob, useColorText)
            
            
            ax1.annotate(pointInfo,
                         xy=(coordRes[currPointInd, 1]-30, coordRes[currPointInd, 0]+20),
                         color=txtColErr, path_effects=[PathEffects.withStroke(linewidth=linewidth, foreground="w")],
                         verticalalignment='bottom', fontsize=fontsize)
            ax1.annotate(pointInfo2,
                         xy=(coordRes[currPointInd, 1]-30, coordRes[currPointInd, 0]+30),
                         color=txtColProb, path_effects=[PathEffects.withStroke(linewidth=linewidth, foreground="w")],
                         verticalalignment='bottom', fontsize=fontsize)

            if not txtSpec is '':
                ax1.annotate(txtSpec,
                            xy=(coordRes[currPointInd, 1]-40, coordRes[currPointInd, 0]+45),
                            color=colSpec, path_effects=[PathEffects.withStroke(linewidth=linewidth, foreground="w")],
                            verticalalignment='bottom', fontsize=fontsize)
            
            plt.axis('off')
            ax1.set_xlim(coordRes[currPointInd,1]-winWidth,coordRes[currPointInd,1]+winWidth)
            ax1.set_ylim(coordRes[currPointInd,0]+winWidth,coordRes[currPointInd,0]-winWidth)

    plt.tight_layout()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01)
    plt.draw()
    if interactive:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
        plt.waitforbuttonpress(1)
    if save:
        plt.savefig(saveFName)
        if savePng:
            savePlotPngSubfolder(fig=plt, saveFNameSvg=saveFName, dpi=savePngDpi)


# funkcija za train set
def getCoordsTraining(net, coordRes, pointsCnt):
    """
    iz modela dobi GT in napoved v obliki koordinat
    """
    for pointNo in range(0, pointsCnt):
        resultLabel = net.blobs['label'].data[0,pointNo]
        # resy = resultLabel.argmax()/label_w # h, navzdol
        # resx = resultLabel.argmax()%label_w # w, desno
        resy, resx = np.unravel_index(resultLabel.argmax(), resultLabel.shape)
        coordRes[pointNo, 0] = resy
        coordRes[pointNo, 1] = resx
        resultPred = net.blobs['conv_classifier'].data[0, pointNo]
        # predy = resultPred.argmax()/label_w # h, navzdol
        # predx = resultPred.argmax()%label_w # w, desno
        predy, predx = np.unravel_index(resultPred.argmax(), resultLabel.shape)
        coordRes[pointNo, 2] = predy
        coordRes[pointNo, 3] = predx
        # todo daj še tukaj verjetnost shrani
        # 
        [pointNo] = resultPred.max()

def setPlot(ax, index, infoText, symbols, xlabel='Points', ylabel='Error in px'):
    plt.title(infoText)
    plt.xlabel(xlabel)
    plt.xticks(rotation=90)
    ax.yaxis.grid(True, linestyle=':')
    if symbols:
        plt.xticks(index, symbols)
    else:
        plt.xticks(index, index)
    plt.ylabel(ylabel)

def stdEucl(matrix):
    return np.std(np.nansum(matrix**2, -1)**0.5)
def meanEucl(matrix):
    return np.mean(np.nansum(matrix**2, -1)**0.5)
def medianEucl(matrix):
    return np.median(np.nansum(matrix**2, -1)**0.5)

BOXPLOT_INFO_TEXT = "orangeLine:median, greenTriagle:mean, box:25-75%, wisker:last point<'Q3 + 1.5*(Q3-Q1)', circles: outliers>formula"

def boxPlotImages(
        diffMatrix,
        imCnt=8,
        fileNames=[],
        save=True,
        interactive=False,
        image_pick_handler=None,
        ):
    # :2 - only xerr and yerr without prob
    mat = diffMatrix[:,:,:2]
    errs = np.sqrt((mat**2).sum(axis=-1))
    # TODO: izboljšaj
    errs = np.nan_to_num(errs)
    errs = errs.T  # transpose 22i * 72p -> 72*22

    means=[]; stds=[]; meds=[]
    for i in range(0,imCnt):
        mean = meanEucl(diffMatrix[i,:,:2])
        std = stdEucl(diffMatrix[i,:,:2])
        med = medianEucl(diffMatrix[i,:,:2])
        means.append(mean)
        stds.append(std)
        meds.append(med)
    # resized images
    #infoText = 'euclid error per images' + ' (imWidth: ' + str(imWidth) + 'px, imHeight: ' + str(imHeight) + "px)\n" \
    #    + 'mean: {:05.2f}'.format(np.mean(means)) + ', std={:05.2f}'.format(np.mean(stds)) + ', med={:05.2f}'.format(np.mean(meds)) + '\n' \
    #    + BOXPLOT_INFO_TEXT
    # original image
    infoText = 'euclid error per images (variable size of original images)\n' \
        + 'mean: {:05.2f}'.format(np.mean(means)) + ', std={:05.2f}'.format(np.mean(stds)) + ', med={:05.2f}'.format(np.mean(meds)) + '\n' \
        + BOXPLOT_INFO_TEXT

    fig, ax = plt.subplots(figsize=(19.20, 10.80))
    plt.title(infoText)
    index = np.arange(imCnt)

    bp_handles = plt.boxplot(errs, showmeans=True)


    if interactive and image_pick_handler is not None:
        for artist in bp_handles['boxes']:
            artist.set_picker(True)

        def handle_event(event):
            ind = bp_handles['boxes'].index(event.artist)
            print(f'{ind} : {fileNames[ind]}')
            image_pick_handler(ind)

        fig.canvas.mpl_connect('pick_event', handle_event)


    index = np.arange(imCnt+1)
    fileNames2 = fileNames.copy()
    for i in range(0, imCnt):
        # fileNames2[i] = str(index[i]) + '\n' + fileNames2[i]
        fileNames2[i] = str(index[i]) + ': ' + fileNames2[i]
    fileNames2 .insert(0, '')
    setPlot(ax, index, infoText, fileNames2, xlabel='Images')
    ax.set_xlim(0, imCnt+1)

    if interactive:
        figManager = plt.get_current_fig_manager()

    fig.subplots_adjust(left=0.04, right=0.995, top=0.93, bottom=0.25, wspace=0.01, hspace=0.01)
    ax.set_ylim(bottom=0)
    plt.draw()
    if interactive:
        plt.show(block=True)
    if save:
        fig.savefig('images_err_boxplot.svg')
        fig.savefig('images_err_boxplot.png')
        ax.set_ylim(bottom=0, top=20)
        fig.savefig('images_err_ylim_boxplot.svg')
        fig.savefig('images_err_ylim_boxplot.png')
    if interactive:
        plt.close('all')

def boxPlotImagesRelative(diffMatrixRelative, imCnt=8, fileNames=[], save=True, interactive=False):
    # :2 - only xerr and yerr without prob
    mat = diffMatrixRelative[:,:,:2]
    errs = np.sqrt((mat**2).sum(axis=-1))
    # TODO: izboljšaj
    errs = np.nan_to_num(errs)
    errs = errs.T  # transpose 22i * 72p -> 72*22

    means=[]; stds=[]; meds=[]
    for i in range(0,imCnt):
        mean = meanEucl(diffMatrixRelative[i,:,:2])
        std = stdEucl(diffMatrixRelative[i,:,:2])
        med = medianEucl(diffMatrixRelative[i,:,:2])
        means.append(mean)
        stds.append(std)
        meds.append(med)
    # resized images
    #infoText = 'euclid error per images' + ' (imWidth: ' + str(imWidth) + 'px, imHeight: ' + str(imHeight) + "px)\n" \
    #    + 'mean: {:05.2f}'.format(np.mean(means)) + ', std={:05.2f}'.format(np.mean(stds)) + ', med={:05.2f}'.format(np.mean(meds)) + '\n' \
    #    + BOXPLOT_INFO_TEXT
    # original image
    infoText = 'relative euclid error per images (1.0 is diagonal of image)\n' \
        + 'mean: {:01.4f}'.format(np.mean(means)) + ', std={:01.4f}'.format(np.mean(stds)) + ', med={:01.4f}'.format(np.mean(meds)) + '\n' \
        + BOXPLOT_INFO_TEXT

    fig, ax = plt.subplots(figsize=(19.20, 10.80))
    plt.title(infoText)
    index = np.arange(imCnt)

    a = plt.boxplot(errs, showmeans=True)

    index = np.arange(imCnt+1)
    fileNames2 = fileNames.copy()
    for i in range(0, imCnt):
        # fileNames2[i] = str(index[i]) + '\n' + fileNames2[i]
        fileNames2[i] = str(index[i]) + ': ' + fileNames2[i]
    fileNames2 .insert(0, '')
    setPlot(ax, index, infoText, fileNames2, xlabel='Images', ylabel='Relative error (1.0=image_diagonal)')
    ax.set_xlim(0, imCnt+1)

    if interactive:
        figManager = plt.get_current_fig_manager()
        #figManager.window.showMaximized()
        #plt.waitforbuttonpress(1)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.93, bottom=0.25, wspace=0.01, hspace=0.01)
    ax.set_ylim(bottom=0)
    plt.draw()
    if interactive:
        plt.show(block=True)
    if save:
        fig.savefig('images_err_boxplot_rel.svg')
        fig.savefig('images_err_boxplot_rel.png')
        ax.set_ylim(bottom=0, top=0.02)
        fig.savefig('images_err_ylim_boxplot_rel.svg')
        fig.savefig('images_err_ylim_boxplot_rel.png')
    if interactive:
        plt.close('all')

def calcMeansStdsMeds_byPoints(diffMatrix, pointCnt):
    means=[]; stds=[]; meds=[]
    for i in range(0, pointCnt):
        mean = meanEucl(diffMatrix[:,i,:2])
        std = stdEucl(diffMatrix[:,i,:2])
        med = medianEucl(diffMatrix[:,i,:2])
        means.append(mean)
        stds.append(std)
        meds.append(med)
    return means, stds, meds

def boxPlotPoints(diffMatrix, pointCnt=72, symbols=[], save=True, interactive=False):
    # :2 - only xerr and yerr without prob
    mat = diffMatrix[:,:,:2]
    errs = np.sqrt((mat**2).sum(axis=-1))
    # TODO: izboljšaj
    errs = np.nan_to_num(errs)

    means, stds, meds = calcMeansStdsMeds_byPoints(diffMatrix, pointCnt)
    # resized images
    #infoText = 'euclid error per landmarks (points)' + ' (imWidth: ' + str(imWidth) + 'px, imHeight: ' + str(imHeight) + "px)\n" \
    #    + 'mean: {:05.2f}'.format(np.mean(means)) + ', std={:05.2f}'.format(np.mean(stds)) + ', med={:05.2f}'.format(np.mean(meds)) + '\n' \
    #    + BOXPLOT_INFO_TEXT
    # oirignal images
    infoText = 'euclid error per landmarks (variable size of original images)\n' \
        + 'mean: {:05.2f}'.format(np.mean(means)) + ', std={:05.2f}'.format(np.mean(stds)) + ', med={:05.2f}'.format(np.mean(meds)) + '\n' \
        + BOXPLOT_INFO_TEXT
    fig, ax = plt.subplots(figsize=(19.20, 10.80))

    a = plt.boxplot(errs, showmeans=True)

    index = np.arange(pointCnt+1)
    symbols2 = symbols.copy()
    symbols2.insert(0, '') #???
    setPlot(ax, index, infoText, symbols2)

    fig.subplots_adjust(left=0.04, right=0.995, top=0.93, bottom=0.1, wspace=0.01, hspace=0.01)
    if interactive:
        plt.show(block=True)
    if save:
        fig.savefig('points_err_box.svg')
        fig.savefig('points_err_box.png')
        ax.set_ylim(bottom=0, top=20)
        fig.savefig('points_err_box_ylim.svg')
        fig.savefig('points_err_box_ylim.png')
    if interactive:
        plt.close('all')

def boxPlotModel(
    diffMatrix,
    save=True,
    interactive=False,
    ):

    mat = diffMatrix[:, :, :2]
    errs = np.sqrt((mat**2).sum(axis=-1))
    errs = errs.ravel()
    errs = errs[~np.isnan(errs)]

    err_mean = np.mean(errs)
    err_q1, err_q2, err_q3, err_q99 = np.percentile(errs, [25, 50, 75, 99])

    infoText = 'euclid error for model \n' \
        + f'mean: {err_mean:05.2f}, med={err_q2:05.2f}, Q1={err_q1:05.2f} Q3={err_q3:05.2f}\n' \
        + BOXPLOT_INFO_TEXT

    fig, ax = plt.subplots(figsize=(19.20, 10.80))

    ax.boxplot(errs, showmeans=True, autorange=True)
    plt.title(infoText)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.93, bottom=0.25, wspace=0.01, hspace=0.01)
    ax.set_ylim(bottom=0, top=err_q99)

    plt.draw()
    if interactive:
        plt.show(block=True)
    if save:
        fig.savefig('err_boxplot.svg')
        fig.savefig('err_boxplot.png')
    if interactive:
        plt.close('all')


def autolabel_barplot(rects, ax):
    """
    Attach a text label above each bar displaying its height
    orig: https://stackoverflow.com/a/42498711
    
    rects - return of plt.bar()
    ax - axis of plot/subplot
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                f'{height:.2f} %',
                ha='center', va='bottom', fontsize=10)


def barplot_SDR(ax, ticks, PzList, colors, suggestedYplotMin):
    '''
    "private" function
    draws bar plot on ax using colors, ticks (string list)
    
    ax - axis of plot/subplot
    ticks - list for plt.bar() (['2 mm', '2.5 mm', ...]
    PzList - array, values in percents (for z ranges)
    colors - plt colors for color of bars
    suggestedYplotMin - emmm...
    '''
    barlist = ax.bar(ticks, PzList)
    barlist[0].set_color(colors[0])
    barlist[1].set_color(colors[1])
    barlist[2].set_color(colors[2])
    barlist[3].set_color(colors[3])
    ax.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
    # y minimum: 40
    if np.min(PzList) >= suggestedYplotMin:
        ax.set_ylim(suggestedYplotMin, 100)
    autolabel_barplot(barlist, ax)
