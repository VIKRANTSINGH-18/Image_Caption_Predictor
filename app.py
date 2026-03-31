import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("model/image_caption_model.h5")
tokenizer = pickle.load(open("model/tokenizer.pkl", "rb"))
max_len = 20

def word_for_id(i, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == i:
            return word
    return None

def generate_caption(model, tokenizer, feature, max_len):
    text = "startseq"
    for i in range(max_len):
        seq = tokenizer.texts_to_sequences([text])[0]
        seq = pad_sequences([seq], maxlen=max_len)

        yhat = model.predict([feature, seq], verbose=0)
        yhat = np.argmax(yhat)

        word = word_for_id(yhat, tokenizer)
        if word is None:
            break

        text += " " + word
        if word == "endseq":
            break

    return text

st.title("🔥 Image Caption Generator")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png"])

if uploaded_file is not None:
    st.image(uploaded_file)

    img = load_img(uploaded_file, target_size=(128,128))
    img = img_to_array(img)/255.0
    img = np.expand_dims(img, axis=0)

    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
    from tensorflow.keras.models import Model

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

    cnn = build_cnn()
    feature = cnn.predict(img, verbose=0)

    caption = generate_caption(model, tokenizer, feature, max_len)

    st.success("Caption:")
    st.write(caption)
