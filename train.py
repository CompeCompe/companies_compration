from pathlib import Path

import numpy as np
import pandas as pd
import pickle

from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from pipe import Pipe, DataPreparing

data = pd.read_csv(Path(r"data/train.csv"))

clear_via_pipe = Pipe(
    DataPreparing.remove_countries(),
    DataPreparing.remove_abbreviation()
)

# clearing data
data['name_1'] = clear_via_pipe(data['name_1'].astype('str'))
data['name_2'] = clear_via_pipe(data['name_2'].astype('str'))
data['is_duplicate'] = data['is_duplicate'].fillna(0).astype('int')
data['full_name'] = data['name_1'] + ' ' + data['name_2']

# data to vector
count_tf_idf = TfidfVectorizer(ngram_range=(1,5), analyzer='char_wb', max_features=10000)
corpus = data['full_name'].values.astype('U')
features_train = count_tf_idf.fit_transform(corpus)
target_train = data['is_duplicate'].values

# model train
logreg = LogisticRegression(solver='sag', max_iter=300, random_state=101)
tuned_parameters = {"C": np.logspace(-2, 2, 10)}
best_logreg = GridSearchCV(logreg, param_grid=tuned_parameters, scoring='roc_auc',
                           cv=5, n_jobs=-1).fit(features_train, target_train)

# save model and data
full_df = data.copy()
pickle.dump(count_tf_idf, open("data/tfidf.pickle", "wb"))
full_df.to_hdf(Path(r"data/full_df.h5"), key="df", mode="w", index=False)
dump(best_logreg, Path(r"data/logit.joblib", sep=";"))