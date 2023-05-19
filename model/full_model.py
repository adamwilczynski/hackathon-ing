import pickle

import pandas as pd
from py3langid.langid import LanguageIdentifier, MODEL_FILE
from langdetect import detect, detect_langs


def get_model_prediction(text):
    pass


if __name__ == "__main__":
    model_and_reader_eng = pickle.load(open('model_eng.pkl', 'rb'))
    # model_eng, model_reader_eng = model_and_reader_eng[0], model_and_reader_eng[1]
    # print(type(model_eng), model_eng)

    model_and_reader_pl = pickle.load(open("model_pl.pkl", 'rb'))
    # model_pl, model_reader_pl = model_and_reader_pl[0], model_and_reader_pl[1]

    test_df = pd.read_csv("train_input_data_pl_and_eng.csv")
    identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE)
    identifier.set_languages(['pl', 'en'])

    test_df["lang"] = test_df["text"].apply(lambda text: identifier.classify(str(text))[0])

    pl_filter = test_df["lang"] == "pl"
    test_df.loc[pl_filter, "predict"] = test_df["text"].apply(model_and_reader_pl)

    eng_filter = test_df["lang"] == "eng"
    test_df.loc[eng_filter, "predict"] = test_df["text"].apply(model_and_reader_eng)

    test_df.to_csv("test_model_after_predictions.csv")
