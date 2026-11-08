import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import sklearn

print("TensorFlow :", tf.__version__)
print("OpenCV     :", cv2.__version__)
print("NumPy      :", np.__version__)
print("Pandas     :", pd.__version__)
print("Scikit-learn:", sklearn.__version__)

print("\nPerangkat yang tersedia:")

print(tf.config.list_physical_devices())