
import tensorflow as tf
from tensorflow import keras
import streamlit as st

import numpy as np
import matplotlib.pyplot as plt
import glob, os
import re
import PIL
from PIL import Image
st.title("Healthy Meat for a Healthy You!")
st.markdown("<span style=“background-color:#121922”>",unsafe_allow_html=True)
st.write('Choose your meat type')
from PIL import Image
image = Image.open('chick.jpg')

st.image(image,width=200)

if st.button('Chicken'):
     st.write('Start Capturing or Upload Image')
else:
     st.write('Choose an Image')
def jpeg_to_8_bit_greyscale(path, maxsize):
        img = Image.open(path).convert('L')   
        WIDTH, HEIGHT = img.size
        if WIDTH != HEIGHT:
                m_min_d = min(WIDTH, HEIGHT)
                img = img.crop((0, 0, m_min_d, m_min_d))
        img.thumbnail(maxsize, PIL.Image.ANTIALIAS)
        return np.asarray(img)
    
def load_image_dataset(path_dir, maxsize):
        images = []
        labels = []
        os.chdir(path_dir)
        for file in glob.glob("*.jpg"):
                img = jpeg_to_8_bit_greyscale(file, maxsize)
                if re.match('ad.*', file):
                        images.append(img)
                        labels.append(0)
                elif re.match('unad.*', file):
                        images.append(img)
                        labels.append(1)
        return (np.asarray(images) , np.asarray(labels) )
def single(img, maxsize):
        images = []
        labels = []
        img = jpeg_to_8_bit_greyscale(img, maxsize)
        if re.match('ad.*', img):
            images.append(img)
            labels.append(0)
        elif re.match('unad.*', img):
            images.append(img)
            labels.append(1)
        return (np.asarray(images) , np.asarray(labels) )
maxsize = 100, 100
import streamlit as st
from streamlit_webrtc import webrtc_streamer

webrtc_streamer(key="example")
ii = st.file_uploader("Choose an image...", type=".jpg")
if ii is not None:
    (train_images, train_labels) = load_image_dataset('beef1/train1', maxsize)

    (test_images, test_labels) = load_image_dataset('beef1/train1', maxsize)

    class_names = ['ad', 'unad']



    def display_images(images, labels):
            plt.figure(figsize=(10,10))
            grid_size = min(25, len(images))
            for i in range(grid_size):
                    plt.subplot(5, 5, i+1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.grid(False)
                    plt.imshow(images[i], cmap=plt.cm.binary)
                    plt.xlabel(class_names[labels[i]])

    display_images(train_images, train_labels)

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(100, 100)),
            keras.layers.Dense(128, activation=tf.nn.sigmoid),
            keras.layers.Dense(16, activation=tf.nn.sigmoid),
        keras.layers.Dense(2, activation=tf.nn.softmax),
        keras.layers.Flatten()
    ])

    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.7, nesterov=True)


    model.compile(optimizer=sgd,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    predictions = model.predict(test_images)

    model.fit(train_images, train_labels, epochs=100)

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    predictions = model.predict(test_images)
    print(predictions)
    st.title("Freshness Report")
    st.write("Chosen Meat: Chicken")
    image = Image.open('chick.jpg')
    st.image(image,width=200)
    st.write('Color')
    st.write('Texture')
    st.write('Freshness')
    st.write("Decision ")
    if ((predictions[0][0])>(predictions[0][1])):
        st.write("Fresh Meat")
        im = Image.open('wb.jpg')
        st.image(im,width=200)
    elif ((predictions[0][0])<(predictions[0][1])):
        st.write("Old Meat")
        im = Image.open('harm.jpg')
        st.image(im,width=200)
        
    st.write('Thanks for choosing our application, Find your complementary Recepie to Healthify You!')
    im = Image.open('r1.jpeg')
    st.image(im)
    display_images(test_images, np.argmax(predictions, axis = 1))

    plt.show()
