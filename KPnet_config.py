import os
import sys
# configuraton file for KPnet project
# used by other python scripts in subfolders

# GENERAL CONFIGURATION
# set the number of ceph keypoints (labels) per image
pointCnt = 19
# number of all caph images in dataset, e.g. 532
imCnt = 150+250
# number of training images in dataset, e.g. 400
imTrainCnt = 150
# number of testing images in dataset, e.g. 132
imTestCnt = 250

# set the caffe image width, e.g.: 512
imWidth = -1  # 512
# set the caffe image height, e.g.: 512
imHeight = -1 # 512
# name (+path) of prototxt that included net architecture. Will search in .2_caffe/Models
netProtoFileName = '' #'kpnet_train.prototxt'
# name (+path) of caffe model (weights). Will search in .2_caffe/Models
netModelFileName = '' #'KPnet-CephBot_iter_100000.caffemodel'
# location of segnet vesion of caffe binnary


# OTHER VARIABLES (mostly paths)


configFileDir = os.path.abspath(os.path.join(os.path.realpath(__file__), '..'))
projectRootDir = configFileDir

# file with simbols (names, labels) for each landmark (point) sequentually
landmark_symbolsFile = os.path.join(configFileDir, '1_data/1_images/split/landmark_symbols-bench.txt')

# paths in 3_analysis/
analysisDir = os.path.join(configFileDir, '3_analysis')
# numpy matricies results for original images (original size)
npysDir = os.path.join(analysisDir, '3_npys')
resMatrixFile = os.path.join(npysDir, 'resMatrix.npy')
GTmatrixFile = os.path.join(npysDir, 'GTmatrix.npy')
diffMatrixFile = os.path.join(npysDir, 'diffMatrix.npy')
analysisImagesFile = os.path.join(npysDir, 'result_img.list')
# other folders
csvDir = os.path.join(analysisDir, '4_CSV')
statsDir = os.path.join(analysisDir, '5_stats_graphs')
imagesResDir = os.path.join(analysisDir, '6_images')
resLabelsDir = os.path.join(analysisDir, '7_resLabels')
# statsistics file for avg error on images and points (landmarks)
statsFile = os.path.join(statsDir, 'stats.txt')
statsJsonFile = os.path.join(statsDir, 'stats.json')

# sequentually files for training (first file has index 0)
LMDBdir = os.path.join(configFileDir, '1_data', '6_hdf5')
testImagesFile = os.path.join(configFileDir, '1_data', '1_images', 'split', 'testAllFilesList-bench.txt')
trainImagesFile = os.path.join(configFileDir, '1_data', '1_images', 'split', 'trainFilesList-bench.txt')


# common functions
def getModifiedPath(filePath, dirName, autoCreate=True):
    '''
    inserts dirName in filePath as lastest dir. (autoCreate) also crates if non-exising
    example
        filePath /1/2/3/file.txt
        dirName 'abc'
        result /1/2/3/abc/file.txt
    '''
    # TODO: move fun to common dir/lib
    dirOld = os.path.dirname(filePath)
    fileName = os.path.basename(filePath)

    newDir = os.path.join(dirOld, dirName)
    if autoCreate:
        # creates new dir if not exisitng
        os.makedirs(newDir, exist_ok=True)
    if not os.path.exists(newDir):
        raise Exception(f'generated path not exists: {newDir}\nuse autoCreate=True to create when writing')

    newFilePath = os.path.join(newDir, fileName)
    return newFilePath
