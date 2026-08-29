 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import keras
import PyQt5
import sklearn
import pandas
import numpy
import psutil
import scipy

print('='*40)
print('All packages OK!')
print('Python  :', __import__('sys').version[:6])
print('TF      :', tf.__version__)
print('Keras   :', keras.__version__)
print('NumPy   :', numpy.__version__)
print('Sklearn :', sklearn.__version__)
print('Pandas  :', pandas.__version__)
print('='*40)