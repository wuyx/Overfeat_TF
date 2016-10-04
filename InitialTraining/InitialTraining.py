from tensorflow.core.framework import graph_pb2
from google.protobuf import text_format
import os

model_paths = '../Models/ModelFiles'

model_name = 'OverFeatAccurate.pbtxt'

graph_def = graph_pb2.GraphDef()

with open(os.path.join(model_paths,model_name), "rb") as f:
	text_format.Merge(f.read(), graph_def)

	