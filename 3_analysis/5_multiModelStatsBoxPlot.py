import os
import sys
import argparse
import numpy as np
import pandas
import matplotlib.pyplot as pyplot
import functools

sys.path += ["../"]
try:
    import KPnet_config as config
    import commonlib.filelist as filelist
    import commonlib.dataframe_visualization as df_vis
except ModuleNotFoundError as e:
    configPathMissing = os.path.abspath(sys.path[-1])
    print('\nERROR: KPnet_config.py missing in: ' + configPathMissing + '\n\n')
    print(e)
    raise e


if __name__ == '__main__':
    enableStatsByProbablity = False
    enablePlotting = True
    worstCountPrint = 20
    landmark_symbolsFile = config.landmark_symbolsFile
    #imagesListFile = config.testImagesFile
    
    parser = argparse.ArgumentParser(
        description='Plot boxplots and print statistics for multiple models.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    

    parser.add_argument('-D', '--dirNames',
                        help=f'Names of model result subdirectories, relative to KPnet_config.npysDir',
                        nargs='+',
                        default=[],
                        )
    parser.add_argument('-P', '--dirPaths',
                        help=f'Full valid paths to model results directories. Must be subdirs of KPnet_config.npysDir.',
                        nargs='+',
                        default=[],
                        )
    parser.add_argument('-L', '--imagesListFile',
                        help=f'Path fo file with target list of images used in analysis.',
                        )
    parser.add_argument('--imagesListType',
                        help=f'Predetermined list of target files to use in analysis, train referes to KPnet_config.trainImagesFile, and test refers to KPnet_config.testImagesFile.',
                        choices=['train', 'test'],
                        default='test',
                        )
    parser.add_argument('--statsByProb',
                        choices=['yes', 'no'],
                        help=f'enable print of statistics by worst probablility and error.',
                        dest='enableStatsByProbablity',
                        default=['no', 'yes'][enableStatsByProbablity])
    parser.add_argument('--worstCount', help='number of worst results to be printed',
                        type=int, default=worstCountPrint)

    parser.add_argument('--plot', '-p',
                        choices=['yes', 'no'],
                        help=f'enable generating plots.',
                        dest='enablePlotting',
                        default=['no', 'yes'][enablePlotting])
    # temporary landmark names from file (for limited number of landmarks)
    parser.add_argument('--kpListFile', type=str,
                        help=f'Path to file with predetermined ordered short landmark names',
                        default=landmark_symbolsFile)

    argRes = parser.parse_args()
    dirNamesList = argRes.dirNames
    dirPathsList = argRes.dirPaths
    enableStatsByProbablity = {'yes': True, 'no': False}[argRes.enableStatsByProbablity]
    enablePlotting = {'yes': True, 'no': False}[argRes.enablePlotting]
    landmark_symbolsFile = argRes.kpListFile
    worstCountPrint = argRes.worstCount

    if argRes.imagesListFile:
        imagesListFile = argRes.imagesListFile
    elif argRes.imagesListType == 'test':
        imagesListFile = config.testImagesFile
    elif argRes.imagesListType == 'train':
        imagesListFile = config.trainImagesFile
    else:
        raise Exception('No target images list file was selected to generate statistics.')

    for dirPath in dirPathsList:
        npysDir, dirName = dirPath.split(os.path.sep, 1)
        if not os.path.samefile(npysDir, config.npysDir):
            raise Exception(f'path {dirPath} is not a subpath of npys dir: {config.npysDir}')

        dirNamesList.append(dirName)


    statsImagesList = filelist.read_file_list(imagesListFile)

    landmark_symbols = open(landmark_symbolsFile, 'r').read().split('\n')
    landmark_symbols = [lm for lm in landmark_symbols if lm]

    diffMatrixFile = config.diffMatrixFile
    analysisImagesFile = config.analysisImagesFile

    data_frames_list = []
    for dirName in dirNamesList:
        diffMatrixFileMod = config.getModifiedPath(diffMatrixFile, dirName, autoCreate=False)
        analysisImagesFileMod = config.getModifiedPath(analysisImagesFile, dirName, autoCreate=False)
        analysisImagesList = filelist.read_file_list(analysisImagesFileMod)
        testImagesInds = filelist.compare_file_lists(analysisImagesList, statsImagesList, only_file_name=True)

        # a file in statsImagesList was not found in analysisImagesList
        if None in testImagesInds:
            notFoundImagesList = [name for ind, name in zip(testImagesInds, statsImagesList) if ind is None]
            raise Exception(f'Files in target images list not found in results: {notFoundImagesList}')

        diffMatrix = np.load(diffMatrixFileMod)
        diffMatrixSelection = diffMatrix[testImagesInds, :]

        errs = (diffMatrixSelection[:, :, :2]**2).sum(2)**0.5
        probs = diffMatrixSelection[:, :, 2]
        for r, image_path in enumerate(statsImagesList):
            data_frames_list.append(pandas.DataFrame(data={
                'model': dirName,
                'image': os.path.basename(image_path),
                'landmark': landmark_symbols,
                'L2_error': errs[r, :],
                'probability': probs[r, :],
                }))

    multi_model_results_df = pandas.concat(data_frames_list)

    if enablePlotting:
        def plot_fun(
                ax, values, labels, subplot_ids,
                value_names, label_names, subplot_names,
                ):
            l2_errors = values['L2']
            ax.boxplot(l2_errors, labels=labels)
            ax.set_title(f'{subplot_ids}')

        df_vis_gen = df_vis.group_and_plot_dataframe(
            multi_model_results_df,
            plot_fun,
            values={'L2':'L2_error'},
            inplot_groups=['model'],
            )
        df_vis.plot_and_show(df_vis_gen)
        
        df_vis_gen = df_vis.group_and_plot_dataframe(
            multi_model_results_df,
            plot_fun,
            values={'L2':'L2_error'},
            inplot_groups=['landmark'],
            subplot_groups=['model'],
            gen_subplots_fun=functools.partial(
                df_vis.init_subplots_bestfit,
                share_x=True,
                share_y=True,
                tall=True,
                )
            )
        df_vis.plot_and_show(df_vis_gen)
    

    # experimental code
    if enableStatsByProbablity:
        # prob hack code
        sortedProb = multi_model_results_df.sort_values(by='probability')
        sortedErr = multi_model_results_df.sort_values(by='L2_error', ascending=False)

        # im = 'M00108_5__08__2019_A_1.jpg'
        # lm = 'N\''
        # sortedProb.loc[(list(sortedProb['image'] == im) and list(sortedProb['landmark'] == lm))]
        # sel1 = sortedProb.loc[sortedProb['image'] == im]
        # print(sel1.loc[sel1['landmark'] == lm])
    
        # by prob sort
        imsByProb = []
        lmsByProb = []
        print(f'\n\nsort by prob (worst {worstCountPrint})\n')
        for i in range(worstCountPrint):
            imsByProb.append(sortedProb.iloc[i].image)
            lmsByProb.append(sortedProb.iloc[i].landmark)
        for i in range(worstCountPrint):
            im = imsByProb[i]
            lm = lmsByProb[i]
            sortedProb.loc[(list(sortedProb['image'] == im) and list(sortedProb['landmark'] == lm))]
            sel1 = sortedProb.loc[sortedProb['image'] == im]
            print(sel1.loc[sel1['landmark'] == lm])
            
        # by err sort
        imsByErr = []
        lmsByErr = []
        print(f'\n\nsort by err (worst {worstCountPrint})\n')
        for i in range(worstCountPrint):
            imsByErr.append(sortedErr.iloc[i].image)
            lmsByErr.append(sortedErr.iloc[i].landmark)
        for i in range(worstCountPrint):
            im = imsByErr[i]
            lm = lmsByErr[i]
            sortedErr.loc[(list(sortedErr['image'] == im) and list(sortedErr['landmark'] == lm))]
            sel1 = sortedErr.loc[sortedErr['image'] == im]
            print(sel1.loc[sel1['landmark'] == lm])

