# train.py

import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, add, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model

def load_captions(file):
    mapping = {}
    with open(file, 'r') as f:
        for line in f:
            img, normal, social = line.strip().split('|')
            img_id = img.split('.')[0]

            mapping.setdefault(img_id, [])
            mapping[img_id].append(normal)
            mapping[img_id].append(social)
    return mapping

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

def main():

    captions = load_captions("dataset/captions.txt")
    cnn = build_cnn()

    features = {}
    for img_name in os.listdir("dataset/images"):
        path = os.path.join("dataset/images", img_name)

        img = load_img(path, target_size=(128,128))
        img = img_to_array(img)/255.0
        img = np.expand_dims(img, axis=0)

        feature = cnn.predict(img, verbose=0)
        features[img_name.split('.')[0]] = feature[0]

    tokenizer = Tokenizer()
    lines = [c for key in captions for c in captions[key]]
    tokenizer.fit_on_texts(lines)

    max_len = max(len(c.split()) for c in lines)
    vocab_size = len(tokenizer.word_index) + 1

    X1, X2, y = [], [], []

    for key in captions:
        for cap in captions[key]:
            seq = tokenizer.texts_to_sequences([cap])[0]

            for i in range(1, len(seq)):
                in_seq = seq[:i]
                out_seq = seq[i]

                in_seq = pad_sequences([in_seq], maxlen=max_len)[0]

                X1.append(features[key])
                X2.append(in_seq)
                y.append(out_seq)

    X1, X2, y = np.array(X1), np.array(X2), np.array(y)

    input1 = Input(shape=(256,))
    fe = Dense(256, activation='relu')(input1)

    input2 = Input(shape=(max_len,))
    se = Embedding(vocab_size,256,mask_zero=True)(input2)
    se = LSTM(256)(se)

    decoder = add([fe, se])
    decoder = Dense(256, activation='relu')(decoder)

    output = Dense(vocab_size, activation='softmax')(decoder)

    model = Model([input1, input2], output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    model.fit([X1, X2], y, epochs=10, batch_size=32)

    os.makedirs("model", exist_ok=True)
    model.save("model/image_caption_model.h5")
    pickle.dump(tokenizer, open("model/tokenizer.pkl", "wb"))

    print("Training Done & Model Saved!")

if __name__ == "__main__":
    main()
