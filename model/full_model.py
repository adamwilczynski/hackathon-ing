import json
import pickle

import tensorflow as tf

import pandas as pd
import numpy as np
from py3langid.langid import LanguageIdentifier, MODEL_FILE
from langdetect import detect, detect_langs

from tensorflow.keras.preprocessing.sequence import pad_sequences


def read_json(filename):
    with open(filename, encoding="UTF-8") as f:
        return list(json.load(f).items())


def get_filename_hash(filename):
    return filename[filename.rindex("/") + 1:] if "/" in filename else filename


testing_output_df = pd.DataFrame(read_json("output_data/test_ocr_clean.json"), columns="filename text".split())
testing_output_df["filename"] = testing_output_df["filename"].apply(get_filename_hash)
print(testing_output_df.head())


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

identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE)
identifier.set_languages(['pl', 'en'])

testing_output_df["lang"] = testing_output_df["text"].apply(lambda text: identifier.classify(str(text))[0])

pl_filter = testing_output_df["lang"] == "pl"
testing_output_df.loc[pl_filter, "predict"] = testing_output_df.loc[pl_filter, "text"].apply(get_model_prediction,
                                                                                             args=[model_pl,
                                                                                                     tokenizer_pl])

eng_filter = testing_output_df["lang"] == "en"
testing_output_df.loc[eng_filter, "predict"] = testing_output_df.loc[eng_filter, "text"].apply(get_model_prediction,
                                                                                               args=[model_eng,
                                                                                                       tokenizer_eng])
print(testing_output_df.head())

testing_output_df["predict"] = testing_output_df["predict"].astype(int)
testing_output_df = testing_output_df.sort_values(by="filename")
testing_output_df["filename predict".split()].to_csv("Gryfki_test_ids_final.csv", index=False)
