
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.platform import gfile

def pbtxt_to_graphdef(filename):
  with open(filename, 'r') as f:
    graph_def = tf.GraphDef()
    file_content = f.read()
    text_format.Merge(file_content, graph_def)
    tf.import_graph_def(graph_def, name='')
    tf.train.write_graph(graph_def, 'pbtxt/', 'protobuf.pb', as_text=False)

def graphdef_to_pbtxt(filename): 
  with gfile.FastGFile(filename,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    tf.train.write_graph(graph_def, 'pbtxt/', 'protobuf.pbtxt', as_text=True)
  return

  #===================================================
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

def graphdef_to_pbtxt22(filename): 
    with tf.Session() as sess:
        with gfile.FastGFile(filename,'rb') as f:
            data = compat.as_bytes(f.read())
            sm = saved_model_pb2.SavedModel()
            sm.ParseFromString(data)  
            g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)    
            tf.train.write_graph(sess.graph, 'pbtxt/', 'protobuf.pbtxt', as_text=True)
    return
  
  
# graphdef_to_pbtxt22('saved_model.pb')  # here you can write the name of the file to be converted
# and then a new file will be made in pbtxt directory.

graphdef_to_pbtxt("models/Frozen_model_ssd_train3_10000_step/frozen_inference_graph.pb")