'''
Helper functions for plotting pandas.DataFrame.
'''
from typing import List, Callable, Any, Tuple, Dict, Iterator
import matplotlib.pyplot as pyplot
import numpy as np
import pandas

def plot_and_store_to_pdf(
        pdf_file_path: str,
        plots_generator: Iterator,
        ):

    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(pdf_file_path) as pdf:
        for _ in plots_generator:
            pdf.savefig()
            pyplot.close()

def plot_and_show(
        plots_generator: Iterator
        ):
    for _ in plots_generator:
        pyplot.show()

def group_and_plot_dataframe(
        dataframe: pandas.DataFrame,
        plot_fun: Callable[[Any], None],
        values: Dict[str, str],
        inplot_groups: List[str] = None,
        subplot_groups: List[str] = None,
        figure_groups: List[str] = None,
        fig_title_fun: Callable[[any], str] = None,
        gen_subplots_fun: Callable[[Tuple[int]], Tuple[pyplot.Figure, np.ndarray]] = None,
        ):
    '''
    Plot data over multiple figures and subplots with
    one function and data groups. Returns a plot generator - 
    each item in the generator is a plotted figure.

    If no plot_fun is given, the generator returns 
    arguments for each call of the plot_fun - for each
    axes in the plot. Each item is an axes to be ploted,
    and each figure once all it's axes are completed.

    dataframe - pandas.DataFrame
        Data to plot.

    plot_fun - 
        Callable function, that will be called for each plot.
        The plot function has the form:
            def fun(ax, values, labels, subplot_ids, 
                    value_names, label_names, subplot_names)
                    

    values - 
        Values to pass onto the plot function. The same keys are 
        used in the dictionary values passed onto plot. The values
        are the column names that should be extracted from dataframe.

    inplot_groups - [str]
    subplot_groups - [str]
    figure_groups - [str]
        Lists of column names of dataframe, used to group data.
        Each grouping represents a plotting level.

        The figure_groups are used to divide the data across
        multiple figures. This can be one or multiple columns, but 
        must be a list of column names. The values are used
        separately to add a suptitle to the figure.

        The subplot_groups are used to divide the data 
        across subplots of each figure. These can be a list of column
        names, each name distributed over its own subplot dimension.
        Or it can be lists of lists, where names within each list share
        a dimension. The value of the group is passed on to the plotting
        function as subplot_ids to be used in the axes title.

        The inplot_groups are used to pass on data within each plot.
        The divisions are passed as lists within values, while
        their values are passed on as labels.

    fig_title_fun -
        Callable used to generate a figure suptitle from
        the figure_groups value.
        Receives the figure grouping values as tuple and should 
        return the string that will be used as figure suptitle.

    gen_subplots_fun - 
        Callable used to generate a figure and subplots based
        on the shape determined from subplot_groups.
        Receives a tuple of integers, should return a pyplot.Figure
        and table of pyplot.Axes. The number of axes should 
        be equal to the shapes volume (product of received integers).

        Any organization is possible, however two common ones 
        are implemented.

        init_subplots
        
        A simple wrapper to pyplot.subplots, used to generate 2D
        subplot grids. This must receive a 2D subplot shape. 
        This is the default.

        init_subplots_bestfit

        A good fit pyplot.subplots wrapper, that attempts to fit
        the number of axes in a square-like arrangement, dropping 
        any unneeded axes. It linearizes the shape, calculates
        the appropirate number of rows and columns that will contain
        enough axes, calls pyplot.subplots, removes redundant axes,
        reshapes the resulting axes table to the required shape.
    '''
    if gen_subplots_fun is None:
        gen_subplots_fun = init_subplots

    if fig_title_fun is None:
        fig_title_fun = lambda fig_id: f'{fig_id}'

    if figure_groups:
        fig_grp_obj = dataframe.groupby(figure_groups)
        for fig_id, fig_df in fig_grp_obj:
            yield from __gpdf_figure__(
                fig_id, fig_df,
                values,
                subplot_groups, inplot_groups,
                plot_fun, fig_title_fun, gen_subplots_fun,
                )
    else:
        yield from __gpdf_figure__(
            '', dataframe,
            values,
            subplot_groups, inplot_groups,
            plot_fun, fig_title_fun, gen_subplots_fun,
            )

def __gpdf_figure__(
        fig_id: Any,
        fig_df: pandas.DataFrame,
        values: Dict[str, str],
        subplot_groups: List[str],
        inplot_groups: List[str],
        plot_fun,
        fig_title_fun,
        gen_subplots_fun,
        ):
    if subplot_groups:
        subplots_shape = tuple(fig_df[sbp_grp].drop_duplicates().shape[0]
                               for sbp_grp in subplot_groups)

        subplot_grp_gen = enum_multigroupby_gen(fig_df, subplot_groups)
    else:
        subplots_shape = [1, 1]
        subplot_grp_gen = (((0, 0), (None, None), fig_df),)

    fig, ax_table = gen_subplots_fun(subplots_shape)
    for ax_cur in ax_table.ravel():
        ax_cur.set_axis_off()

    fig.suptitle(fig_title_fun(fig_id))

    for subplot_inds, subplot_ids, subplot_df in subplot_grp_gen:
        ax_cur = ax_table[(*subplot_inds,)]
        subplot_ret = __gpdf_subplot__(
            ax_cur,
            subplot_ids,
            subplot_df,
            values,
            inplot_groups,
            subplot_groups,
            plot_fun,
            )
        if plot_fun is None:
            yield subplot_ret
    yield fig


def __gpdf_subplot__(
        ax_cur,
        subplot_ids: Any,
        subplot_df: pandas.DataFrame,
        values: Dict[str, str],
        inplot_groups: List[str],
        subplot_groups: List[str],
        plot_fun
        ):
    ax_cur.set_axis_on()

    if inplot_groups:
        inplot_grp_obj = subplot_df.groupby(inplot_groups)

        labels = []
        values_dict = {val_id:[] for val_id in values}
        for l, inplot_df in inplot_grp_obj:
            labels.append(l)
            for val_id, val_ind in values.items():
                values_dict[val_id].append(inplot_df[val_ind].to_numpy())


    else:
        labels = (None,)

        values_dict = {val_id:subplot_df[val_ind].to_numpy() for val_id, val_ind in values}
        inplot_groups = None

    args = (ax_cur, values_dict, labels, subplot_ids)
    kwargs = {'value_names':values, 'label_names':inplot_groups, 'subplot_names':subplot_groups}
    if plot_fun is None:
        return args, kwargs
    return plot_fun(*args, **kwargs)

def enum_multigroupby_gen(
        dataframe: pandas.DataFrame,
        group_ids: List[str]):
    '''
    Generator for multiple successive groupby calls to dataframe dataframe with
    enumeration.

    dataframe: pandas.Dataframe
        the dataframe used to group
    group_ids: List[str]
        list of columns from dataframe to groupby over.
        The items are grouped in the order given.

    Each item in the generator consists of:
        (tuple of indices, tuple of grouped values, dataframe)

    '''
    group_id = group_ids[0]
    group_ids = group_ids[1:]

    for ind, (group_val, groupped_df) in enumerate(dataframe.groupby(group_id)):
        if not group_ids:
            yield (ind,), (group_val,), groupped_df
        else:
            for inds, subgroup_vals, subgroupped_df \
               in enum_multigroupby_gen(groupped_df, group_ids):
                yield  (ind, *inds), (group_val, *subgroup_vals), subgroupped_df

def init_subplots(
        shape: Tuple[int, int],
        share_x: bool = False,
        share_y: bool = False,
    ) -> Tuple[pyplot.Figure, np.ndarray]:
    '''
    Simple wrapper for pyplot.subplots, initializes
    a rectangular arrangement of subplots of given shape.
    The given shape must be 2D.
    '''
    squeeze = False
    if len(shape) == 1:
        rows, = shape
        cols = 1
        squeeze = True
    elif len(shape) == 2:
        rows, cols = shape
    else:
        raise Exception(f'cannot generate subplots in shape: {shape}')
    fig, ax_table = pyplot.subplots(rows, cols, share_x, share_y, squeeze=squeeze)

    set_tight_layout_with_labels(ax_table)

    return fig, ax_table

def init_subplots_bestfit(
        shape: Tuple[int, ...],
        share_x: bool = False,
        share_y: bool = False,
        tall: bool = False,
        row_hint: int = None,
        col_hint: int = None,
    ) -> Tuple[pyplot.Figure, np.ndarray]:
    '''
    Wrapper for pyplot.subplots, takes the given shape, 
    calculates the number of axis and attemps a close to
    square arrangement of of subplots. Any additional subplots
    are removed. The table of axes is reshaped to the required
    shape before returned.

    tall - bool
        If a tall rather than a wide arrangement is prefered.
        For 11 will generate 4x3 instead of 3x4 subplot grid.

    row_hint - int
    col_hint - int
        Manually determine/force the number of rows or columns.
        Given both, only one will be followed.
    '''
    total_len = np.prod(shape)

    if col_hint:
        cols = col_hint
        rows = total_len//cols
        if total_len%cols != 0:
            rows += 1
    else:
        if row_hint:
            rows = row_hint
        else:
            rows = int(total_len**0.5+0.5)

        cols = total_len//rows
        if total_len%rows != 0:
            cols += 1

    if tall:
        if rows < cols:
            rows, cols = cols, rows

    fig, ax_table = pyplot.subplots(
        rows,
        cols,
        share_x,
        share_y,
        squeeze=False,
        )

    ax_table = ax_table.ravel()

    for axes in ax_table[total_len:]:
        axes.remove()
    ax_table = ax_table[:total_len].copy().reshape(shape)
    set_tight_layout_with_labels(ax_table)

    return fig, ax_table

def set_tight_layout_with_labels(ax_table):
    '''
    Attempts to configure a tight layout for the figure in
    advance.

    For a given np.ndarray of pyplot.Axes,
    each axes is set a placeholder title and axes labels.

    Then figure.tight_layout is called for
    the axes with enough space for a figure suptitle.

    The placeholder labels are afterward removed.
    '''
    ax_table = ax_table.ravel()
    fig = ax_table[0].figure

    fig.suptitle('suptitle')
    for ax in ax_table:
        ax.set_title('title')
        ax.set_ylabel('Y (f(x))')
        ax.set_xlabel('X (x)')

    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    fig.suptitle('')
    for ax in ax_table:
        ax.set_title('')
        ax.set_ylabel('')
        ax.set_xlabel('')

import unittest
class TestCase(unittest.TestCase):
    def test_group_and_plot_dataframe(self):

        x = np.linspace(0, 2*np.pi, 10)
        plot_test_df = pandas.concat([
            pandas.DataFrame(data={
                'figure':'fig1',
                'row': 0,
                'col': 0,
                'function':'sin',
                'x':x,
                'y':np.sin(x),
                'nuisance': np.random.permutation(x),
                }),
            pandas.DataFrame(data={
                'figure':'fig1',
                'row': 0,
                'col': 0,
                'function':'cos',
                'x':x+np.pi/2,
                'y':np.cos(x+np.pi/2),
                'nuisance': 'a',
                }),
            pandas.DataFrame(data={
                'figure':'fig1',
                'row': 0,
                'col': 1,
                'function':'square',
                'x':x,
                'y':np.square(x),
                'nuisance': np.random.permutation(x),
                }),
            pandas.DataFrame(data={
                'figure':'fig1',
                'row': 1,
                'col': 1,
                'function':'sqrt',
                'x':x,
                'y':np.sqrt(x),
                }),
            pandas.DataFrame(data={
                'figure':'fig2',
                'row': 0,
                'col': 0,
                'function':'lin',
                'x':x,
                'y':x,
                }),
            pandas.DataFrame(data={
                'figure':'fig2',
                'row': 1,
                'col': 0,
                'function':'inv',
                'x':x,
                'y':-x,
                }),
            ], ignore_index=True, verify_integrity=True)

        def plot_fun(
                ax,
                values,
                #x_list,
                #y_list,
                labels,
                subplot_ids,
                value_names,
                label_names,
                subplot_names,
                ):

            x_list = values['x']
            y_list = values['y']
            x_name = value_names['x']
            y_name = value_names['y']


            for x, y, l in zip(x_list, y_list, labels):
                ax.plot(x, y, label=f'{label_names}={l}')

            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)

            ax_title = ' '.join([f'{n}={v}' for n, v  in zip(subplot_names, subplot_ids)])

            ax.set_title(ax_title)
            ax.legend(loc='best')


        plot_gen = group_and_plot_dataframe(
            plot_test_df,
            plot_fun,
            values={'x':'x', 'y':'y'},
            inplot_groups='function',
            subplot_groups=['row', 'col'],
            figure_groups='figure',
            )

        for _ in plot_gen:
            pyplot.show()


    def test_enum_multigroupby_gen(self):
        test_df = pandas.DataFrame(data={
            'a': ['x', 'y', 'z', 'x', 'y', 'x'],
            'b': [ 1,   1,   1,   2,   2,   1],
            'c': [ 1,   2,   3,   4,   5,   6],
            })
        inds_ref = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1))
        vals_ref = ((1, 'x'), (1, 'y'), (1, 'z'), (2, 'x'), (2, 'y')) 
        grp_data_ref = ([1, 6], [2], [3], [4], [5])

        inds, vals, grp_data = zip(*[(i, v, d['c'].to_numpy())
                for i, v, d in enum_multigroupby_gen(test_df, ['b', 'a'])])
        self.assertEqual(inds_ref, inds)
        self.assertEqual(vals_ref, vals)

        for n, (d_ref, d) in enumerate(zip(grp_data_ref, grp_data)):
            np.testing.assert_array_equal(d_ref, d,
                    err_msg=f'differing element {n}')
        '''***'''

if __name__ == '__main__':
    unittest.main(verbosity=2)
