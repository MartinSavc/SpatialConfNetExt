import os
import sys
import numpy as np
import scipy.ndimage as ndimage
import PIL.Image
import skimage.transform

def getPointsFileFromImg(fnameImageOrig, txtDirName='2_labels', extension=None):
    '''
    Returns the file containing the points for cephalometric image fnameImageOrig (original size)

    fnameImageOrig - string
        Path to cepahalometric image in dataset.
        e.g.: .../1_images/K00116_3__07__2019_A_1.png

    txtDirName - string
        Directory containing the points data.

    extension - string
        Extension of points data, added to image file name.
        if None, it will auto set set extension either .jpg-points.txt or .png-points.txt
        OLD: extension='.png-points.txt'

    Returns:
        string
        path to keypoint text file

    '''
    '''
    TODO:
    original default extension='.png-points.txt'
    spremeni v None
    
    preveri ce je None -> potem dobi png/jpg in dodaj "-points.txt"
    sicer pa uporabi tistega ki ga dobis iz argumenta funkcije
    
    naredi unit teste za vse kombinacije
    '''
    basename = os.path.basename(fnameImageOrig)
    basename = os.path.splitext(basename)[0]
    if extension is None:
        imgExt = os.path.splitext(os.path.basename(fnameImageOrig))[1]  # ".png" or ".jpg"
        #print(f'imgExt: {imgExt}')
        extension = imgExt + '-points.txt'  # .png-points.txt
        
    pardir = os.path.abspath(os.path.join(fnameImageOrig, os.pardir, os.pardir))
    fnameTxt = os.path.join(pardir, txtDirName, basename + extension)
    # .../2_labels/K00116_3__07__2019_A_1.png-points.txt
    return fnameTxt


def readPointsInTxt(fnameTxt, csvChar=' ', pointCount=72, firstLineMetadata=True, byListFname=None):
    '''
    Reads CephBot CSV .txt file and returns array
    vrne np array, (2, 10), (yyyy, xxxx), y ind vrstice, x ind stolpca
    (CehpBenchmark ima drugace!)

    fnameTxt - string
        path to  keypoint text file

    csvChar - string
        column separator used in file

    pointCount - int
        number of points, if None it is determined from the file
        if None, will be set as actual points in fnameTxt

    firstLineMetadata - bool
        is the first line in the file the columns header?
        If True the first line is skipped.

    byListFname - string
        path to file containing ordered short keypoints names
        if None (by default), same order as in fnameTxt

    returns:
        np.ndarray <2xN>
        Array with N points.
    '''
    with open(fnameTxt, 'r') as f:
        lines = f.readlines()
        lines = [item.rstrip() for item in lines]
        #line example:
        # 4024,3697,2,Me
    if firstLineMetadata:
        del lines[0]

    if pointCount is None:
        pointCount = len(lines)

    # if byListFname path is given, read file and sort "lines" by it
    if byListFname is not None:
        with open(byListFname, 'r') as f:
            kpList = f.readlines()
            kpList = [item.rstrip() for item in kpList]
        linesNew = []
        for keypoint in kpList:
            # search each keypoint from keypointList File
            found = False
            for l in lines:
                currPoint = l.split(csvChar)[3]
                if currPoint == keypoint:
                    linesNew.append(l)
                    found = True
            if found is False:
                raise Exception(f'keypoint  {keypoint}  not found in {fnameTxt}')
        pointCount = len(linesNew)
        lines = linesNew

    if (len(lines) < pointCount) and (byListFname is None):
        txt1 = "ERROR: point count from arg (program) is bigger than point count from data (file)"
        txt2 = f'arg vs file: {len(lines)} vs {pointCount}'
        raise Exception(txt1+'\n'+txt2)
    readVals = np.zeros((2, len(lines)))
    for i in range(0, pointCount):
        line = lines[i].split(csvChar)
        # cephal benchmark has x,y (width, height)
        # we work with y,x
        readVals[1, i] = float(line[0])
        readVals[0, i] = float(line[1])
    return readVals


def resizeAndPadImage(fileName, h=360, w=480):
    '''
    Loads Cephalometric image as grayscale. The datatype is as in file (8bit or 16bit)
    sprejme fileName za hires xray sliko (png), cv2 prebere, resiza in pada. vrne sivinsko sliko

    fileName - string
        path to a Cephalometric image

    h - int
        target height of image

    w - int
        target width of image

    returns:
        np.ndarray <h*w>   shape(h, w)
        2D numpy array of resized image
    '''
    im = np.array(PIL.Image.open(fileName))
    if im.ndim == 3:
        #im = np.mean(im, 2, dtype=im.dtype) # buggy convertion
        im = np.array(np.mean(im, 2), dtype=im.dtype)

    return resizeAndPadImageArray(im, h, w)


def calcResizeRatio(size_orig, size_dest):
    '''
    Calc the resize ratio that will scale the image of size_orig
    using uniform scaling to fit withing size_dest.

    Returns the scaling ratio for the image.
    '''
    h_o, w_o = size_orig
    h_d, w_d = size_dest

    ratio_dest = h_d/w_d
    ratio_orig = h_o/w_o

    if ratio_orig > ratio_dest:
        resize_ratio = h_d/h_o
    else:
        resize_ratio = w_d/w_o

    return resize_ratio


def resizeAndPadImageArray(im, h=360, w=480):
    '''
    Resizes the image to given height and width. Uses proper Downsample
    '''
    # print('im dtpye:' + str(im.dtype) + ' shape:' + str(im.shape) )
    # im = color.rgb2gray(im)
    rateMy = h/float(w)
    rateIm = im.shape[0]/float(im.shape[1])
    if (rateIm > rateMy):
        # =np.ceil(im.shape[0]/rateMy)
        targetW = int(np.ceil(im.shape[0]/rateMy))
        diffW = targetW-im.shape[1]
        imAdded = np.hstack((im, np.zeros((im.shape[0], int(diffW)), dtype=im.dtype)))
    else:
        # if rateIm < rateMY
        # targetH=im.shape[1]*rateMy
        targetH = int(np.ceil(im.shape[1]*rateMy))
        diffH = targetH-im.shape[0]
        imAdded = np.vstack((im, np.zeros((int(diffH), im.shape[1]), dtype=im.dtype)))
    imRes = properDownsample(imAdded, h, w)
    # print("im " + str(im.shape) + str(im.dtype))
    # print("imAdded " + str(imAdded.shape) + str(imAdded.dtype))
    # print("imRes " + str(imRes.shape) + str(imRes.dtype))
    return imRes

def properDownsample(imgGray, h=360, w=480):
    '''
    Uses skimage.transform.resize mode='constant' to "down" resize image

    imgGray - 2D ndarray
        grayscale image 2D np array

    h - int
        target height of image

    w - int
        target width of image

    returns:
        np.ndarray <h*w>   shape(h, w)
        2D numpy array of resized image
    '''
    # opencv ohrani tip tabele
    #imRes = cv2.resize(imgGray, (w, h))

    imRes = skimage.transform.resize(imgGray, (h, w), mode='constant', order=0)
    # vrne med 0 in 1, nazaj spraviti uint8 oz uint16
    # float ne sme biti!
    if issubclass(imgGray.dtype.type, np.integer):
        imRes = imRes * np.iinfo(imgGray.dtype).max
    imRes = imRes.astype(imgGray.dtype)
    return imRes

def resizePoints(imOrig, pointsResol, h=360, w=480):
    '''
    Changes location of points for resized Cephalometric image
    '''
    rate = calcResizeRatio(imOrig.shape, (h, w))
    if isinstance(pointsResol, list):
        pointsResize = [(y*rate, x*rate) for y, x in pointsResol]
    else:
        pointsResize = pointsResol * rate # TODO: preveri
    return pointsResize


def radialExp(pointSize, sigma):
    ''' generate "patch" with radial exponential kernel in center
    pointSize - (odd number!) size of "label" (kernel)
    sigma=lambda
    sigma should not be None
    
    returns: kernel (label) - non normalized!!!
    '''
    if sigma is None:
        raise Exception('Sigma should not be None. set it or calculate via getSigmaForRadialExp() or getSimpleSigmaForRadialExp()')

    X, Y = np.meshgrid(np.arange(pointSize), np.arange(pointSize), )
    X = np.float32(X) - np.floor(pointSize/2)
    Y = np.float32(Y) - np.floor(pointSize/2)
    R = (X**2+Y**2)**0.5
    kernel = sigma*np.exp(-sigma*R)
    return kernel


def getSimpleSigmaForRadialExp(pointSize):
    '''
    simple method to calculate sigma for radial exponential function by the given pointSize
    check getSigmaForRadialExp() for better results
    '''
    # the number is totally experimental
    # hardcoded values
    if pointSize == 30:
        sigma = 0.3
    elif pointSize == 90:
        sigma = 0.09
    else:
        # 12/30 = 0.4
        # 12/90 = 0.13
        sigma = 12/pointSize
    return sigma


def radialExpWithQuantization(pointSize, sigma=None, levelCnt=256):
    '''
    calls radialExp function and quantisizes result
    used to reduce levels in label (kernel), so the nerual network might have less work to do
    
    returns: kernel (label) normalized and quansized
    '''
    levelCnt -= 1 # value 2 makes 3 levels...
    res = radialExp(pointSize, sigma)
    res /= res.max()
    res = np.round(res*levelCnt)/levelCnt
    return res


def plotRadialExp(pointSize, sigma, figsize=(12, 10)):
    res = radialExp(pointSize, sigma)
    res /= res.max()
    f = figure(figsize=figsize)
    f.suptitle(f'pointSize: {pointSize}, sigma: {pointSize}')
    imshow(res, vmin=0, vmax=1)


def testRadialExp(pointSize):
    '''
    helper function for use in ipython qtconsole...
    '''
    close('all')
    #pointSize=90
    for i in range(1,7):
        print(f'i:{i}')
        s = 1/i**2
        print(f'sigma:{s}')
        res = radialExp(pointSize, s)
        res /= res.max()
        f=figure()
        f.suptitle(f'mask:{pointSize}, sigma:{s}, max:{res.max()}')
        imshow(res)
        pyplot.colorbar()


def testQuantRadialExp(pointSize, levelCnt, sigma, figsize=(12, 10)):
    res = radialExpWithQuantization(pointSize=pointSize, sigma=sigma, levelCnt=levelCnt)
    f = figure(figsize=figsize)
    f.suptitle(f'mask:{pointSize}, levelCnt:{levelCnt}, sigma:{sigma}, max:{res.max()}')
    imshow(res, vmin=0, vmax=1)
    pyplot.colorbar()


def getSigmaForRadialExp(pointSize, startSigma=1.0, levelCnt=None, doPrint=False, doPlot=False, maxSteps=100, devVal=2):
    '''
    get sigma value for radexp that edge value is almost zero
    starting sigma gets devided by devVal (2) each step
    levelCnt: if set, it quatisized... (RECOMENDED)
    '''
    sigma = startSigma
    # sigmaLast = sigma
    halfSize = int(np.round(pointSize/2))
    for i in range(maxSteps):
        sigma /= devVal
        res = radialExp(pointSize, sigma)
        res /= res.max()
        if levelCnt is not None:
            res = np.round(res*levelCnt)/levelCnt
        # tests if edge is more than zero
        if doPrint:
            print(f'step: {i}, new sigma: {sigma}, edge val: {res[0, halfSize]}, unique vals: {len(np.unique(res))}')
        if res[0, halfSize] != 0:
            if doPrint:
                print(f'step {i}, reached 0 at edge, sigma: {sigma}')
            break
        # sigmaLast = sigma
    if doPlot:
        if levelCnt is None:
            # plotRadialExp(pointSize, sigmaLast)
            plotRadialExp(pointSize, sigma)
        else:
            # testQuantRadialExp(pointSize, levelCnt, sigmaLast)
            testQuantRadialExp(pointSize, levelCnt, sigma)
    return sigma  # sigmaLast


def makeLabels(pointsResize, labCnt=72, h=360, w=480, pointSize=30, sigma=None, labelType='gauss', useFloat=False, labelQuantizationLevel=None):
    ''' vstavljanje tock v labele
    arguments:
        labelType: valid 'gauss', 'radexp'
        pointsResize: np.ndarray, [2, N]  resized points
        pointSize: size of "patch" used for image-label description of point (pointResize)
        sigma: the value of sigma(gauss) or lambda (radexp)
        useFloat: generates label with max=1.0. False: max=255.0
        labelQuantizationLevel: radialExp, when not None, it will qanitisize. see radialExpWithQuantization()
    returns:
        labels: list of N  ndarrays [h, w] containing "gaussian" labels (max=1.0 or 255.0)
    '''
    if labCnt > len(pointsResize[0]):
        txt1 = f'labCnt parameter bigger than size of pointsResize (file)'
        txt2 = f'labCnt vs len(pointsResize): {labCnt} vs {len(pointsResize[0])}'
        raise Exception(txt1+'\n'+txt2)
    labels = []
    center = int(np.floor(pointSize/2))
    if labelType == 'gauss':
        mask = np.zeros((pointSize, pointSize), dtype=np.float32)
        mask[center, center] = 1
        # avtomatsko računanje sigme glede na velikost točke (patch-a)
        if sigma is None:
            sigma = pointSize/8
        # TODO: gauss label quantization!
        pointPatch = ndimage.filters.gaussian_filter(mask, sigma)
        pointPatch /= pointPatch.max()
        #if not useFloat:
        if useFloat is False:
            pointPatch = pointPatch * 255  # med 0 in 255
    
    elif labelType == 'radexp':
        #pointPatch = radialExp(pointSize)
        # get sigma if None
        if sigma is None:
            if labelQuantizationLevel is not None:
                sigma = getSigmaForRadialExp(pointSize, levelCnt=labelQuantizationLevel)
            else:  # no qanization
                sigma = getSimpleSigmaForRadialExp(pointSize)
        if labelQuantizationLevel is None:
            pointPatch = radialExp(pointSize, sigma)
            # pointPatch is expeted normalized
            pointPatch /= pointPatch.max()
        else:
            pointPatch = radialExpWithQuantization(pointSize, sigma=sigma, levelCnt=labelQuantizationLevel)
        
        if useFloat is False:
            pointPatch = pointPatch * 255  # uint med 0 in 255
        
    else:
        raise Exception(f'unknown labelType: {labelType}. use gauss or radexp')
        
    
    for i in range(labCnt):
        # i=0 # ita tocka
        y = int(round(pointsResize[0][i]))
        x = int(round(pointsResize[1][i]))
        # check if outside
        if y < 0 or y >= h or x < 0 or x >= w:
            print('WARNING!!!\npoint is outside of image (using empty label)')
            #print('x:' + str(x) + ' y:' + str(y) + ' h:' + str(h) + ' w:' + str(w))
            print(f'pointId:{i}, x: {x:3} y: {y:3} h: {h:4} w: {w:4}')
            labels.append(np.zeros((h, w)))
            continue
        # ignore "nevidnih" tock
        # TODO: deprecated? CephBot - all labels need to be OK
        if (x < 0.1):
            if(y < 0.1):
                print('WARNING!!!\n y or y < 0.1 (using empty label)')
                print(f'pointId:{i}, x: {x}, y: {y}')
                labels.append(np.zeros((h, w)))
                continue
            
        # povečaj velikost slike + velikost labele da ne bo zunaj
        # label_pad = np.pad(label, center, mode='constant')
        #print('x' + str(x) + 'y' + str(y) + 'h' + str(h) + 'w' + str(w))
        label_pad = np.zeros((h + pointSize, w + pointSize))
        label_pad[y:y+pointSize, x:x+pointSize] = pointPatch.copy() # TODO: copy needed?
        # izreži samo del kjer je slika
        label = label_pad[center:center+h, center:center+w]
        labels.append(label)
    # return labels and optionaly calculated sigma
    return labels, sigma

def getResultJoined(imResized, labelsResized, disp=False):
    imResult = imResized.copy()
    imResult = imResult/imResult.max()
    imResult = imResult * 0.8
    # print(imResult.max())
    # print(labelsResized[3].max())
    for i in range(0, len(labelsResized)):
        # seštevanje label med sabo...
        # imResult = imResult + labelsResized[i]/255.0
        # lepši preview
        labTmp = labelsResized[i]/labelsResized[i].max()
        imResult = np.maximum(labTmp, imResult)
    if disp:
        import matplotlib.pyplot as pyplot

        pyplot.figure()
        pyplot.imshow(imResult, cmap='gray')
        pyplot.draw()
        pyplot.show()
        pyplot.waitforbuttonpress(4)
        pyplot.close('all')
    return imResult

def rotate_and_scale_im_points(im, points, theta, scaleFactor):
    ''' rotates and scales image and points according to theta in degrees and scale factor
    arguments:
        im: grayscale image
        points: np.ndarray, [2, N] 
            N points in image im, each point is the pair (y, x) 
            or (row, column)
        theta: float
            rotation angle in degrees
        scaleFactor: float
            scale factor, >1 data iz zoomed in, <1 data is zoomed out
    result:
        imRes: rotated and scaled image
        Ps_res: rotated and scale points
        warning: points may be outside the image!
    '''
    # conversion in radinas
    theta_r = np.radians(theta)
    # sine and cosine values
    c, s = np.cos(theta_r), np.sin(theta_r)
    # the image has zero in uppoer left corner, so we invert sine value
    # s = s*-1.
    # rotation matrix
    R_ = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    # scaling matrix
    S_ = np.array([[scaleFactor, 0, 0], [0, scaleFactor, 0], [0, 0, 1]])
    # combining rotation and scaling with matrix multiplciation
    RS_ = R_.dot(S_)
    # Operations work in 0,0, so we have to translate image center to 0,0
    # make the manilupation and translate back to original position.
    # Translation in half of lenght and hight. also shape gives H,W -> W,H
    # update: potrebno je odšteti ena
    tr = np.array(im.shape[::-1])-1
    tr = tr/2  # TODO: kaj pri polovičkah??? (lihe?)
    # translation matrix
    T_ = np.array([[1, 0, tr[0]], [0, 1, tr[1]], [0, 0, 1]])
    # the last translation in invese opeation
    TRST_ = T_.dot(RS_).dot(np.linalg.inv(T_))
    # to transfom all pixels in image we use affine transfrom from ndimage
    # mora biti inverz transformacije !!!
    # prelikavanje iz ciljne v izvor (?)
    #    The inverse coordinate transformation matrix
    #imRes = ndimage.affine_transform(im, np.linalg.inv(TRST_), mode='constant')
    # nearest: better because of "random" for NN training
    imRes = ndimage.affine_transform(im, np.linalg.inv(TRST_), mode='nearest')

    # convert the points in collumn matrix (3, 72) with ones in the end
    Ps_ = np.ones((3, points.shape[1]))
    Ps_[0, :] = points[0, :]
    Ps_[1, :] = points[1, :]
    # convert
    Ps_res = TRST_.dot(Ps_)

    return imRes, Ps_res[:2, :]

def translate_im_points(im, points, tr_vec):
    '''translates image and points by tr_vec
    arguments:
        im: grayscale image
        points: np.ndarray, [2, N] 
            N points in image im, each point is the pair (y, x) 
            or (row, column)
        tr_vec: np.ndarray, [2, ]
    result:
        imRes: translated image
        Ps_res: translated points
        warning: points may be outside the image!
    '''

    Ps_res = points.copy()
    Ps_res[0, :] += tr_vec[0]
    Ps_res[1, :] += tr_vec[1]

    imRes = ndimage.shift(im, tr_vec, mode='nearest')

    return imRes, Ps_res

def check_points_inside_image(img, points):
    ''' check if all points are inside the image
    (inside the image region)
    Returns:
        True: all points are OK
        False: one point (printed) is outside
    '''
    im_h = img.shape[0]
    im_w = img.shape[1]
    #  points[:,71] -> (y,x) (row_ind, col_ind )
    for i in range(len(points[0])):
        status = True
        if points[0, i] < 0:
            status = False
        if points[1, i] < 0:
            status = False
        if points[0, i] >= im_h:
            status = False
        if points[1, i] >= im_w:
            status = False
        if status is False:
            print('BAD: ind: {}, y: {}, x {}'.format(i, points[0, i], points[1, i]))
            return False
    return True

def augument_image_and_points(img, points, rot_min, rot_max, scale_min, scale_max, num, disp=False):
    ''' augumentation of image and it's points. rotation (in degres) and scaling (1.1 in 110% size)
    a uniform random is used
    args:
        img: grayscale image
        points: (2, N) list of points. point(y,x) or (row_ind, col_ind)
        num: number of generated data
    returns:
        res_images: list of augmented images
        res_points: list of augumented points
        if error: [],[] is returned
    '''
    res_images = []
    res_points = []
    # show original
    if disp:
        import matplotlib.pyplot as pyplot

        fig = pyplot.figure()
        fig.canvas.set_window_title('orig')
        pyplot.imshow(img, cmap='gray')
        pyplot.plot(points[1], points[0], 'o')
    for i in range(0, num):
        # check if points gone out of image - repeat generation (augumentation)
        points_are_OK = False
        failCnt = 0
        while points_are_OK is False:
            # get rotation and scaling factors using unform scaling
            rot = np.random.uniform(rot_min, rot_max)
            sc = np.random.uniform(scale_min, scale_max)
            if disp:
                print('r:{:.2f}, s:{:.2f}'.format(rot, sc))
            im_r, po_r = rotate_and_scale_im_points(img, points, rot, sc,  False)
            # test if points are inside image (point outside image not allowed)
            points_are_OK = check_points_inside_image(im_r, po_r)
            if points_are_OK is False:
                failCnt = failCnt + 1
                if disp:
                    print('points outside, retry')
            if failCnt > 30:
                # assume it's infinite loop
                print('ERROR: failCnt > 30. finishing!')
                return [], []
        # all points ARE inside image
        res_images.append(im_r)
        res_points.append(po_r)
        if disp:
            fig = pyplot.figure()
            fig.canvas.set_window_title('res {}: rot={:.2f}, sc={:.2f}'.format(i, rot, sc))
            pyplot.imshow(im_r, cmap='gray')
            pyplot.plot(po_r[1], po_r[0], 'o')
    return res_images, res_points


def OLD_learn_rate_graph(base_lr=0.1, gamma=0.7, stepSize=10000, maxIt=None, doPrint=True, doPlot=True):
    lrs = []
    its = []
    if maxIt is None:
        maxIt = stepSize*10
        
    for i in range(0, maxIt, stepSize):
        lr = base_lr * (gamma ** (int(i/stepSize)))
        lrs.append(lr)
        its.append(i)
        if doPrint:
            print(f'i={i:6}: {base_lr}*{gamma}^{int(i/stepSize):2} = {lr:.5f} (lr)')
    
    if doPlot:
        import matplotlib.pyplot as plt
        
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('itaration')
        
        color1 = 'tab:blue'
        ax1.set_ylabel('lr (linear)', color=color1)
        ax1.plot(its, lrs, '-o', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        # grid for linare scale
        plt.grid(True, which="both")
        
        plt.title(f'learn_rate vs iteration.\nbase_lr={base_lr}, gamma={gamma}, stepSize={stepSize}')
        
        # instantiate a second axes that shares the same x-axis
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        # we already handled the x-label with ax1
        ax2.set_ylabel('lr (log)', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.plot(its, lrs, '-o', color=color2)
        # enable log scale for ax2
        ax2.set_yscale('log')
        
        # otherwise the right y-label is slightly clipped
        fig.tight_layout()
        plt.show()
        return fig


def learn_rate_calculate_limit(limit=0.0005, base_lr=0.1, gamma=0.7, stepSize=10000, maxIt=200000, doPrint=True):
    '''
    helper funtion that limits learn_rate values for given parameter (base_lr, gamma, stepSize=
    it also stops if maxIt has been reached.
    '''
    lrs = []
    its = []
        
    #for i in range(0, maxIt, stepSize):
    i = 0
    while True:
        lr = base_lr * (gamma ** (int(i/stepSize)))
        lrs.append(lr)
        its.append(i)
        if doPrint:
            print(f'i={i:6}: {base_lr}*{gamma}^{int(i/stepSize):2} = {lr:.5f} (lr)')
        # test if lr is smaller than limit
        if lr < limit:
            if doPrint:
                print(f'limit {limit} reached in {i/stepSize} steps. Final LR: {lr}. maximum iteration: {i}')
            break
        i += stepSize
        if i > maxIt:
            if doPrint:
                print(f'maximum iteration {maxIt} reached in {i/stepSize} steps. Final LR: {lr}. Limit was {limit}')
                print('\n!!!check final LR !!!\n\n')
            break
    return lrs, its


def learn_rate_calculate(base_lr=0.1, gamma=0.7, stepSize=10000, maxIt=None, doPrint=True):
    lrs = []
    its = []
    if maxIt is None:
        maxIt = stepSize*10
        
    for i in range(0, maxIt, stepSize):
        lr = base_lr * (gamma ** (int(i/stepSize)))
        lrs.append(lr)
        its.append(i)
        if doPrint:
            print(f'i={i:6}: {base_lr}*{gamma}^{int(i/stepSize):2} = {lr:.5f} (lr)')
    return lrs, its


def learn_rate_graph(base_lr=0.1, gamma=0.7, stepSize=10000, maxIt=None, doPrint=True):
    lrs, its = learn_rate_calculate(base_lr, gamma, stepSize, maxIt, doPrint)
    
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('itaration')
    plt.title(f'learn_rate vs iteration.\nbase_lr={base_lr}, gamma={gamma}, stepSize={stepSize}')
    #plt.grid(True, which="both")
    ax1.grid(True, which="major", color='0')
    ax1.grid(True, which="minor", ls="-.", color='0.8')
    
    # instantiate a second axes that shares the same x-axis
    color2 = 'tab:red'
    # we already handled the x-label with ax1
    ax1.set_ylabel('lr (log)', color=color2)
    ax1.tick_params(axis='y', labelcolor=color2)
    ax1.plot(its, lrs, '-o', color=color2)
    # enable log scale for ax2
    ax1.set_yscale('log')
    
    # otherwise the right y-label is slightly clipped
    fig.tight_layout()
    plt.show()
    return fig, ax1


def learn_rate_graph():
    fig1 = learn_rate_graph(gamma=0.80, maxIt=200000, stepSize=10000)
    fig2 = learn_rate_graph(gamma=0.70, maxIt=200000, stepSize=10000)
    Lax1 = fig1.axes[0]
    Lax2 = fig2.axes[0]
    grpObj = Lax1.get_shared_y_axes()
    grpObj.join(Lax1, Lax2)


def getImageSizeFromFile(imagePath, backend='PIL'):
    ''' reads image from disk and gets hight and width
    backend:
        PIL - gets width and height without reading actual pixel data. fastest
        CV2 - use cv2
        MPL - use matplotlib. slowest
    returns:
        tuple (height, width)
    '''
    if not os.path.exists(imagePath):
        raise Exception(f'imagePath, path not exists: {imagePath} \n {os.path.abspath(imagePath)}')
    if backend == 'MPL':
        im = plt.imread(imagePath)
        h = im.shape[0]
        w = im.shape[1]
    elif backend == 'PIL':
        img = PIL.Image.open(imagePath)
        h = img.height
        w = img.width
    elif backend == 'CV2':
        im = cv2.imread(imagePath)
        h = im.shape[0]
        w = im.shape[1]
    else:
        raise Exception(f'unknown backend: {backend}, use "PIL", "CV2" or "MPL"')
    return h, w

