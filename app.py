import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

# LOAD EVERYTHING
model = tf.keras.models.load_model("model/model.keras")
tokenizer = pickle.load(open("model/tokenizer.pkl", "rb"))
max_len = pickle.load(open("model/max_len.pkl", "rb"))

def build_cnn():
    inputs = Input(shape=(128,128,3))
    x = Conv2D(32,(3,3),activation='relu')(inputs)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64,(3,3),activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(128,(3,3),activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    return Model(inputs, x)

def word_for_id(i):
    for word, index in tokenizer.word_index.items():
        if index == i:
            return word
    return None

def generate_caption(feature):
    text = "startseq"
    for _ in range(max_len):
        seq = tokenizer.texts_to_sequences([text])[0]
        seq = pad_sequences([seq], maxlen=max_len)
        yhat = model.predict([feature, seq], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat)
        if word is None:
            break
        text += " " + word
        if word == "endseq":
            break
    return text

st.title("Image Caption Generator 🔥")

uploaded = st.file_uploader("Upload Image", type=["jpg","png"])

if uploaded:
    st.image(uploaded)

    img = load_img(uploaded, target_size=(128,128))
    img = img_to_array(img)/255.0
    img = np.expand_dims(img, axis=0)

    cnn = build_cnn()
    feature = cnn.predict(img, verbose=0)

    caption = generate_caption(feature)
    st.success(caption)
