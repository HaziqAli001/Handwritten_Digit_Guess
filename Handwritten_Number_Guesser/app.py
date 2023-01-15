from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import load_model
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import load_img
model = load_model('MNIST_Handwritting.h5')
st.title('Handwritten Number Guess')
file = st.file_uploader("Choose an image of number",type=["png","jpg"])

if file is not None:
    st.button("Guess")
    if st.button:
        file = Image.open(file)
        data = np.array(file)
        data.resize(28,28)
        res = model.predict([np.array(data).astype('float32').reshape((1,784))])
        st.write(res.argmax())
        st.balloons()
