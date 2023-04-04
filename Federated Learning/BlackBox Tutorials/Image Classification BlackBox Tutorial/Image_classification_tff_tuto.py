import silence_tensorflow.auto
import nest_asyncio
nest_asyncio.apply()

import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

print("TensorFlow version:", tf.__version__)
print("TensorFlow Federated version:", tff.__version__)

#%% Preparing the input data

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
len(emnist_train.client_ids)