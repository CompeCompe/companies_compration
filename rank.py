from pathlib import Path
from joblib import load

import numpy as np
import pandas as pd
import re

from pipe import Pipe, DataPreparing

 
def rank(comp_name: str, k: int, full_df, model, clear_via_pipe):
    comp_name_clear = clear_via_pipe(comp_name, inline=True).strip()
    comp_name_df = np.array([comp_name_clear for i in range(full_df.shape[0])])
    df_search = pd.DataFrame(np.hstack([comp_name_df, full_df[range(full_df.shape[1] - 1)]]))
    df_search['preds'] = model.predict_proba(df_search)[:,1]
    df_search['names'] = full_df['names'].values
    ans = df_search.sort_values(by='preds', ascending=False)[['names', 'preds']][0:k].values.tolist()
    return ans


def main():
    
    # Загрзука датасета и моделей
    logit = load("data/logit.joblib")
    full_df = pd.read_hdf("data/full_df.h5")

    clear_via_pipe = Pipe(
        DataPreparing.remove_countries(),
        DataPreparing.remove_abbreviation()
        )

    while True:
        comp_name = input("Введите Название компании для поиска или exit для \
                          выхода:\n\n").rstrip()
        if comp_name == "exit":
            break
        k = 5
        try:
            top_comp = rank(comp_name, k, full_df, logit, clear_via_pipe) 
        except ValueError:
            print("Похожих компаний нет в списке \n\n")
            continue
        print(f"Топ {k} похожих компаний:\n\n")
        for i, comp in enumerate(top_comp):
            print(f"{i + 1}: {comp[0]}; вероятность дубля: {round(comp[1],2)}")
        print("\n")


if __name__ == "__main__":
    main()