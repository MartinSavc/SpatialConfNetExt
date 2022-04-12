import os

def read_file_list(file_path, relative_paths=False):
    '''
    Reads list of files from a source file. Each line in the source
    file is considered one file path.

    The functions returns a list of absolute or relative file paths.
    As long a relative folder structure between source file and the 
    files listed is preserved, the paths in the list are valid.


    file_path - str
        Path to a file containing the list.

    relative_paths - bool
        Should the function return paths relative to current work. dir. ?
        By default the paths are absolute.
        

    returns:
    files_list - tuple of str
        Tuple of file names from the file.
    '''
    file_dir = os.path.dirname(file_path)
    with open(file_path, 'r') as in_file:
        files_list = in_file.readlines()
    files_list = [f.rstrip() for f in files_list]
    files_list = [
        os.path.join(file_dir, f)
        for f in files_list
        ]

    if relative_paths:
        files_list = [
            os.path.relpath(f)
            for f in files_list]

    return files_list

def write_file_list(file_path, files_list):
    '''
    Writes list of files to a destination file. These will later be
    readable using read_file_list. It's expected that the files
    in files_list exist and are accesible. But checks are not made.

    As long as the relative folder structure between file_path and
    files in files_list is preserved, the files in the read file list
    will be valid.
    '''

    target_dir = os.path.dirname(file_path)

    with open(file_path, 'w') as out_file:
        for file_path in files_list:
            file_path_rel = os.path.relpath(file_path, target_dir)
            out_file.write(f'{file_path_rel}\n')


def compare_file_lists(source_list, target_list, only_file_name=False):
    '''
    For each file in target_list find the index of the same file in source list.
    Returns a list of indexes inds.

    target_list[n] = source_list[inds[n]]

    If an item in target_list does not match any in source_list, None is inserted
    into inds.

    If an item in target_list matches multiple items in source_list, only the 
    index of the first is inserted into inds.

    If only_file_name is set to true, only file names are compared. Otherwise 
    full paths are normalized (with os.path.normpath) and compared.
    '''

    if only_file_name:
        source_norm_list = [os.path.basename(i).split('.')[0] for i in source_list]
        target_norm_list = [os.path.basename(i).split('.')[0] for i in target_list]
    else:
        source_norm_list = [os.path.abspath(os.path.normpath(i)) for i in source_list]
        target_norm_list = [os.path.abspath(os.path.normpath(i)) for i in target_list]

    inds = []
    for target in target_norm_list:
        if target in source_norm_list:
            inds.append(source_norm_list.index(target))
        else:
            inds.append(None)
    return inds
