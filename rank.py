from pathlib import Path
from joblib import load
import pickle
import time

import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

from pipe import Pipe, DataPreparing


def search_accuracy(predict_proba, k):
    '''
    Calculate accuracy of search, formula given by mentor
    '''
    top_comp_index = predict_proba.argsort()
    top_comp_index = top_comp_index[-k:]
    top_comp_index = top_comp_index[::-1]

    if predict_proba[predict_proba > 0.75].size >= 1:
        weights_1 = [(-0.2*x + 1) for x in range(len(top_comp_index))]
        answer = min(((weights_1 * predict_proba[top_comp_index]).sum()), 1)
    
    elif predict_proba[predict_proba > 0.75].size == 0:
        weights_2 = [0.2*x for x in range(len(top_comp_index))]
        answer = max((1 - (weights_2 * predict_proba[top_comp_index]).sum()), 0)

    return answer


def rank(comp_name: str, k: int, full_df, data, model, clear_via_pipe):
    
    comp_name_clear = clear_via_pipe(comp_name, inline=True).strip()
    full_df['search_comp'] = full_df['name_1'] + ' ' + comp_name
    corpus = full_df['search_comp'].values.astype('U')
    count_tf_idf = pickle.load(open("data/tfidf.pickle", "rb"))
    features_train = count_tf_idf.transform(corpus)
    predict_proba = model.predict_proba(features_train)[:,1]
    if predict_proba.max() > 0.5:
        top_comp_index = predict_proba.argsort()
        top_comp_index = top_comp_index[-k:]
        top_comp_index = list(top_comp_index[::-1])
        ans = data.iloc[top_comp_index]['name_1'].values.tolist()
    else:
        ans = 0
    
    return ans, predict_proba


def main():
    
    # load model and data vectors
    logit = load("data/logit.joblib")
    full_df = pd.read_hdf("data/full_df.h5")
    data = pd.read_csv(Path(r"data/train.csv"))

    clear_via_pipe = Pipe(
        DataPreparing.remove_countries(),
        DataPreparing.remove_abbreviation()
        )

    while True:
        comp_name = input("\nВведите Название компании для поиска или exit для выхода:\n\n").rstrip()
        if comp_name == "exit":
            break
        k = 5
        
        start_time = time.time()
        top_comp, predict_proba = rank(comp_name, k, full_df, data, logit, clear_via_pipe)
        time_1 = time.time() - start_time
        acc_search = search_accuracy(predict_proba, k)
        
        if top_comp == 0:
            print("Похожих компаний нет в списке \n")
            print(f"Время обработки запроса: {round(time_1, 4)} секунды,\nКачество поиска: {round(acc_search, 3)}")
            continue
        
        print(f"\nТоп {k} похожих компаний:\n")
        
        for i, comp in enumerate(top_comp):
            print(f"{i + 1}: {comp}")
        print("\n")
        print(f"Время обработки запроса: {round(time_1, 4)} секунды,\nКачество поиска: {round(acc_search, 3)}")


if __name__ == "__main__":
    main()