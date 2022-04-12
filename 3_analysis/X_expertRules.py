import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import argparse

sys.path += ["../"]
try:
    import KPnet_config as config
    import commonlib.filelist as filelist
except ModuleNotFoundError as e:
    configPathMissing = os.path.abspath(sys.path[-1])
    print('\nERROR: KPnet_config.py missing in: ' + configPathMissing + '\n\n')
    print(e)
    raise e

import commonlib.filelist as filelist

# TODO make unit test!
def AbelowB(Arc, Brc, Aname, Bname, filename):
    Arow = Arc[0]
    Brow = Brc[0]
    # greater row is below
    if not Arow > Brow:
        print(filename)
        print(f'{Aname} not below {Bname}')
        print(f'{Arow} vs {Brow}')


def check_R3(mat, LMsymbols, imagesFilename):
    ''' check landmark R3
    '''
    R3Ind = LMsymbols.index('R3')
    #print(f'Ar symbol index: {lmInd}')
    ArInd = LMsymbols.index('Ar')
    PNSInd = LMsymbols.index('PNS')
    p6cInd = LMsymbols.index('+6c')
    
    for imInd in range(len(imagesFilename)):
        #imInd = 0
        
        #print(f'R3: {mat[imInd, lmInd]}')
        
        #mat [image index, lm ind, (row-down, column-right)]
        R3Row = mat[imInd, R3Ind, 0]  # up, down
        R3Col = mat[imInd, R3Ind, 1]  # left, right
        
        # compare
        # Ar - left of R3
        ArRow = mat[imInd, ArInd, 0]
        ArCol = mat[imInd, ArInd, 1]
        if not ArCol < R3Col:
            print(imagesFilename[imInd])
            print(f'R3 not left of Ar')
            print(f'{R3Col} vs {ArCol}')
        #print('ok\n') if ArCol < lmCol else print('PROBLEM!\n\n')
        
        # PNS - right of R3
        PNSRow = mat[imInd, PNSInd, 0]
        PNSCol = mat[imInd, PNSInd, 1]
        if not PNSCol > R3Col:
            print(imagesFilename[imInd])
            print(f'R3 not right of PNS')
            print(f'{R3Col} vs {PNSCol}')
        
        # +6c - under R3
        p6cRow = mat[imInd, p6cInd, 0]
        p6cCol = mat[imInd, p6cInd, 1]
        if not p6cRow > R3Row:
            print(imagesFilename[imInd])
            print(f'R3 not above +6c')
            print(f'{R3Row} vs {p6cRow}')


def check_Ar(mat, LMsymbols, imagesFilename):
    ''' check landmark Ar - Articulare
    '''
    ArInd = LMsymbols.index('Ar')  # Articulare

    SInd = LMsymbols.index('S')  # sella
    PNSInd = LMsymbols.index('PNS')

    for imInd in range(len(imagesFilename)):
        ArRow = mat[imInd, ArInd, 0]  # up, down
        ArCol = mat[imInd, ArInd, 1]  # left, right
        
        # Sella - above, right of Ar
        SRow = mat[imInd, SInd, 0]
        SCol = mat[imInd, SInd, 1]
        # S above Ar
        if not SRow < ArRow:
            print(imagesFilename[imInd])
            print(f'Ar not below S')
            print(f'{ArRow} vs {SRow}')
        
        # S right of Ar
        if not SCol > ArCol:
            print(imagesFilename[imInd])
            print(f'Ar left of S')
            print(f'{ArCol} vs {SCol}')

        # PNS - under, right of Ar
        PNSRow = mat[imInd, PNSInd, 0]
        PNSCol = mat[imInd, PNSInd, 1]
        
        # PNS under AR
        ARrc = mat[imInd, ArInd]
        PNSrc = mat[imInd, PNSInd]
        AbelowB(PNSrc, ARrc, 'PNS', 'Ar', imagesFilename[imInd])
        #if not PNSRow > ArRow:
        #    print(imagesFilename[imInd])
        #    print(f'Ar above PNS')
        #    print(f'{ArRow} vs {PNSRow}')
            
        # PNS right of Ar
        if not ArCol < PNSCol:
            print(imagesFilename[imInd])
            print(f'Ar left of PNS')
            print(f'{ArCol} vs {PNSCol}')


def setAPocc(mat, LMsymbols, imagesFilename):
    '''
    calculates new APocc in the middle of -1i and +1i
    
    TODO: result?
    '''
    APoccIndOLD = LMsymbols.index('APocc')

    p1iInd = LMsymbols.index('+1i')
    m1iInd = LMsymbols.index('-1i')

    #for imInd in range(len(imagesFilename)):
    for imInd in range(3):
        APoccOLDRow = mat[imInd, APoccIndOLD, 0]  # up, down
        APoccOLDCol = mat[imInd, APoccIndOLD, 1]  # left, right

        p1iRow = mat[imInd, p1iInd, 0]  # up, down
        p1iCol = mat[imInd, p1iInd, 1]  # left, right
        
        m1iRow = mat[imInd, m1iInd, 0]  # up, down
        m1iCol = mat[imInd, m1iInd, 1]  # left, right
        
        APoccNEWRow = int((p1iRow + m1iRow)/2)
        APoccNEWCol = int((p1iCol + m1iCol)/2)
        
        









if __name__ == '__main__':
    resMatrixFile = config.resMatrixFile
    GTmatrixFile = config.GTmatrixFile
    diffMatrixFile = config.diffMatrixFile
    landmark_symbolsFile = config.landmark_symbolsFile
    imagesListFile = config.testImagesFile  # default list is test list from config


    #enablePlotting = False
    dirName = None

    
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dirName',
                    help=f'Use named directory (source, from 3_npys). Just name, not path. Folder (name) will be used also as output. Default: no subdir is used (path from config)',
                    default=dirName)

    argRes = parser.parse_args()
    dirName = argRes.dirName
    
    if dirName is not None:
        print(f'using modified dir to READ/WRITE npys: {dirName}')
        resMatrixFile = config.getModifiedPath(resMatrixFile, dirName, autoCreate=False)
        GTmatrixFile = config.getModifiedPath(GTmatrixFile, dirName, autoCreate=False)
        diffMatrixFile = config.getModifiedPath(diffMatrixFile, dirName, autoCreate=False)


    
    landmark_symbolsFile = landmark_symbolsFile # argRes.kpListFile
    #imCnt = argRes.imNumber  # default None - limit
    #imagesListFile = argRes.imagesListFile
    
    LMsymbols = open(landmark_symbolsFile, 'r').read().split('\n')
    # remove last new line
    del LMsymbols[-1]
    
    GTmatrix = np.load(GTmatrixFile)
    resMatrix = np.load(resMatrixFile)
    
    imagesList = filelist.read_file_list(imagesListFile)
    # get filename (without path)
    # 3_images_resize/78_Image0001_Type17.png -> 78_Image0001_Type17.png
    # if i -> ignore empty line
    imagesFilename = [os.path.basename(i) for i in imagesList if i]
    
    #print(GTmatrix.shape)
    #print(LMsymbols)
    print('-----\nGT matrix\n-----')
    check_R3(GTmatrix, LMsymbols, imagesFilename)
    check_Ar(GTmatrix, LMsymbols, imagesFilename)
    print('-----\nresMatrix\n-----')
    check_R3(resMatrix, LMsymbols, imagesFilename)
    check_Ar(resMatrix, LMsymbols, imagesFilename)
    


