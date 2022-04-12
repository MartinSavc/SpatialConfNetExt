'''
reads multiple SDR json files and generates one multi-subplot
bar plt (Wang/xkcd) style
'''

import os
import json
import sys
import argparse
import numpy as np

sys.path += ["../"]
import commonlib.plots as plots


def multiPlotBench(SDRjsons, suggestedYplotMin=70, useXkcdPlot=True,
                   targetFolder=None, fileName='multiplot.svg'):
    '''
    draws one figure with multi subplots for each SDRjson in Wang/xkcd barplot style and saves

    SDRjsons - list of paths (strings) to a SDR.json files
    suggestedYplotMin - int, Y minimum on plot (if data has lower it will go lower)
    fileName - filename for savefig
    '''
    import matplotlib
    matplotlib.use("Qt5Agg")
    #matplotlib.interactive(True)
    import matplotlib.pyplot as plt
    if useXkcdPlot:
        matplotlib.pyplot.xkcd(scale=0.5)

    struct_all = []
    for SDRjson in SDRjsons:
        if not os.path.isfile(SDRjson):
            raise Exception(f'not a file: {SDRjson}')
        with open(SDRjson) as f:
            struct = json.load(f)
            struct_all.append(struct)

    figsize = (3.5*len(struct_all), 6)
    fig, axes = plt.subplots(ncols=len(struct_all), sharey=True, figsize=figsize)
    PzMin = 100
    for i, struct in enumerate(struct_all):
            title = ""
            if 'methodName' in struct:
                title = struct['methodName']
            elif 'dirName' in struct:
                title = struct['dirName']
            MRE = struct['MRE']
            SD = struct['SD']
            title = title + f'\nMRE:{MRE:.2f} +- {SD:.2f} mm'
            ax = axes[i]
            ax.set_title(title)
            PzList = []
            for item in struct['PzItems']:
                PzList.append(item['Pz'])
            if np.min(PzList) < PzMin:
                PzMin = np.min(PzList)
            # TODO: globals... ?
            ticks = ['2 mm', '2.5 mm', '3 mm', '4 mm']
            colors = ['#fff794', '#c6f5c9', '#c6e3f5', '#f6c5c9']

            #_barplot(ax, ticks, PzList, colors, suggestedYplotMin)
            plots.barplot_SDR(ax, ticks, PzList, colors, suggestedYplotMin=0)
    if PzMin < suggestedYplotMin:
        ax.set_ylim(PzMin, 100)
    plt.tight_layout()
    plt.show()
    if targetFolder is not None:
        os.makedirs(targetFolder, exist_ok=True)
        
        outputFile = os.path.join(targetFolder, fileName)
        # just in case, split extentin of user given 
        outputFile = os.path.splitext(outputFile)[0]
        extensons = ['.svg', '.png']
        for ext in extensons:
            fig.savefig(outputFile+ext)
            print(f'saved {ext}: {outputFile+ext}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    jsonFiles = None
    useXkcdPlot = True
    plotMainTitle = None
    targetFolder = '5_benchMultiPlot'
    fileName = 'multiplot'

    # default arguments - could be better code...
    parser.add_argument('files', type=str, nargs='*',
                        default=jsonFiles,
                        help='path to json files')
    parser.add_argument('--jsonFiles', '--compareJsonFiles',
                        help=f'path to json files',
                        nargs='+', type=str,
                        default=jsonFiles)
    parser.add_argument('--useXkcd',
                        choices=['yes', 'no'],
                        help=f'enable xkcd/Wang plot style. Warning: svg big size! Default: {["no","yes"][useXkcdPlot]}',
                        dest='useXkcdPlot',
                        default=['no', 'yes'][useXkcdPlot])
    parser.add_argument('--targetFolder', type=str,
                        help=f'Target (sub)folder (will create). Default: {targetFolder}',
                        default=targetFolder)
    parser.add_argument('--fileName', type=str,
                        help=f'Plot filename without path and extention (auto add .png and .svg). Default: {fileName}',
                        default=fileName)

    argRes = parser.parse_args()
    #print(f'files: {argRes.files}')
    #print(f'jsonFiles: {argRes.jsonFiles}')
    argFiles = argRes.files
    jsonFiles = argRes.jsonFiles
    useXkcdPlot = {'yes': True, 'no': False}[argRes.useXkcdPlot]
    targetFolder = argRes.targetFolder
    fileName = argRes.fileName
    
    if jsonFiles is None:
        jsonFiles = argFiles
    if jsonFiles is not None:
        multiPlotBench(jsonFiles, targetFolder=targetFolder, fileName=fileName, useXkcdPlot=useXkcdPlot)
    else:
        print('add json files as parameter')
