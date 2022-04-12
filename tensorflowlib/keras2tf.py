'''***'''
import tensorflow as tf
import tensorflow.keras as keras
import tensorflowlib.spatialsoftmax as spatialsoftmax
from tensorflowlib.heatmaplayer import GaussianHeatmap, HeatmapErr
from tensorflow.keras import backend as K
import os.path as osp

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

def convert_keras_to_const_graph(model, model_dir, model_name='model'):
    '''
    model - keras.Model 
        model to export
    model_dir - str
        path to export to, should exists
    model_name - str, (default: 'model')
        name of model
    '''
    output_pref = 'output'

    inputs = []
    outputs = []

    input_names = []
    input_shapes = []
    output_names = []
    output_shapes = []

    #input_names = [in_tens.name for in_tens in model.inputs]
    #output_names = [out_tens.name for out_tens in model.outputs]

    for i, input_tens in enumerate(model.inputs):
        input_names.append(input_tens.name)
        input_shapes.append(input_tens.shape)

    for i, output_tens in enumerate(model.outputs):
        name = f'{output_pref}_{i+1}'
        output_alias = tf.identity(output_tens, name=name)
        output_names.append(name)
        output_shapes.append(output_alias.shape)

    print(f'inputs:')
    for name, shape in zip(input_names, input_shapes):
        print(f'\t{name} - {shape}')
    print(f'outputs:')
    for name, shape in zip(output_names, output_shapes):
        print(f'\t{name} - {shape}')

    sess = tf.keras.backend.get_session()
    constant_graph = graph_util.convert_variables_to_constants(sess,
            sess.graph.as_graph_def(),
            output_names,
            variable_names_blacklist=output_names)

    tf.train.write_graph(constant_graph, model_dir, f'{model_name}.pb.ascii', as_text=True)
    graph_io.write_graph(constant_graph, model_dir, f'{model_name}.pb', as_text=False)

