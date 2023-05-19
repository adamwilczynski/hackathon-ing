import pickle

import tensorflow as tf

import pandas as pd
import numpy as np
from py3langid.langid import LanguageIdentifier, MODEL_FILE
from langdetect import detect, detect_langs

from tensorflow.keras.preprocessing.sequence import pad_sequences


def pad(sequences):
    return pad_sequences(sequences, maxlen=400, padding='post', truncating='post')


def get_model_prediction(text, model, tokenizer):
    text_representation = np.array(
        pad(
            tokenizer.texts_to_sequences(
                [str(text)]
            )
        )
    )
    prediction = list(model.predict(text_representation)[0])
    column = prediction.index(max(prediction))

    print(model.predict(text_representation)[0], column)
    return column


model_eng = tf.keras.models.load_model('model_eng')
tokenizer_eng = pickle.load(open('tokenizer_eng.pkl', 'rb'))

model_pl = tf.keras.models.load_model('model_pl')
tokenizer_pl = pickle.load(open('tokenizer_pl.pkl', 'rb'))

test_df = pd.read_csv("train_input_data_pl_and_eng.csv").sample(500)
identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE)
identifier.set_languages(['pl', 'en'])

test_df["lang"] = test_df["text"].apply(lambda text: identifier.classify(str(text))[0])

pl_filter = test_df["lang"] == "pl"
test_df.loc[pl_filter, "predict"] = test_df.loc[pl_filter, "text"].apply(get_model_prediction,
                                                                         args=[model_pl, tokenizer_pl])

eng_filter = test_df["lang"] == "en"
test_df.loc[eng_filter, "predict"] = test_df.loc[eng_filter, "text"].apply(get_model_prediction,
                                                                           args=[model_eng, tokenizer_eng])
print(test_df.head())

test_df.to_csv("test_model_after_predictions.csv")
