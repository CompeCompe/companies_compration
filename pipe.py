from typing import Callable, List

import numpy as np
import pandas as pd
import pycountry
import re


class DataPreparing:

    def for_pipeline(func):
        def wrapper(*args, **kwargs):
            return func, args, kwargs

        return wrapper

    @for_pipeline
    def remove_countries(s: str) -> str:
        continents_names = ['Asia', 'Africa', 'North America', 'South America', 'Antarctica', 'Europe', 'Australia']

        continents_names_regex = '|'.join([continent for continent in continents_names])

        countries_names_regex = '|'.join([country.name for country in pycountry.countries])
        countries_names_regex = re.sub(' ', '|', countries_names_regex)
        countries_names_regex = re.sub("[^A-Za-z-|]", "", countries_names_regex)

        full_regex = continents_names_regex + countries_names_regex

        pattern = re.compile(full_regex, re.UNICODE | re.IGNORECASE)
        return pattern.sub("", s).strip()

    @for_pipeline
    def remove_abbreviation(s: str) -> str:
        # delete short words comp
        companies = ['ltd', 'ltda', 'saic', 's', 'a', 'i', 'c', 'co', 'ag', 'r', 'o',
                     'p', 'group', 'inc', 'sp', 'ооо', 'зао', 'ncr', 'cv', 'limited',
                     'sa', 'spa', 'pte', 'pvt', 'gmbh', 'nv', 'imp', 'corp', 'pt',
                     'mfg', 'do', 'l', 't', 'd', 'corporation', 'corp', 'doo', 'do',
                     'bv', 'de', 'llc', 'sti', 'c', 'inds', 'industriesusa', 'holdings',
                     'sas', 'ad', 'kg', 'srl', 'sociedad', 'anoni', 'private', 'bhd',
                     'alo', 'asrc', 'lc', 'holding', 'rl', 'ca', 'na', 'llp', 'b', 'on',
                     'tic', 've', 'san', 'sl', 'roppongi', 'zoo', 'lt', 'z', 'oo', 'gmb',
                     'h', 'apm', 'cd', 'as', 'pty', 'kft', 'fm', 'sdn', 'vat', 'id', 'nl',
                     'the', 'tse', 'tax', 'be', 'lp', 'cokg', 'v', 'филиал', 'компании',
                     'компани', 'ccp', 'research', 'and', 'development', 'center', 'tld',
                     'f', 'u', 'lda', 'plc', 'm', 'общество', 'с', 'ограниченной',
                     'ответственностью', 'ieo', 'cie', 'k', 'e', 'ind', 'рус', 'ао',
                     'лтд', 'cs', 'industrial', 'se', 'через', 'терминал', 'в']
        s = s.lower()
        s = re.sub(r'[\W\d]', ' ', s)
        s = s.split()
        for i in range(len(s)):
            if s[i] in companies:
                s[i] = ''
            else:
                continue
        s = ' '.join(s).strip()
        return s


class Pipe:
    sequence: List[DataPreparing]

    def ndarray_map(data: np.ndarray, func: Callable, args, kwargs):
        return np.vectorize(func)(data, *args, **kwargs)

    def series_map(data: pd.Series, func: Callable, args, kwargs):
        return data.apply(func, args=args, **kwargs)

    def dataframe_map(data: pd.DataFrame, func: Callable, args, kwargs):
        for c in data.columns:
            data[c] = data[c].apply(func, args=args, **kwargs)
        return data

    def __init__(self, *sequence):
        self.sequence = sequence

    def __call__(self, data: pd.Series, inline: bool = False) -> pd.Series:
        if type(data) != str:
            d = data if inline else data.copy()
        else:
            d = data
        if type(d) == np.ndarray:
            for s in self.sequence:
                d = Pipe.ndarray_map(d, s[0], s[1], s[2])
        elif type(d) == pd.Series:
            for s in self.sequence:
                d = Pipe.series_map(d, s[0], s[1], s[2])
        elif type(d) == str:
            for s in self.sequence:
                d = s[0](d, *s[1], **s[2])
        elif type(d) == pd.DataFrame:
            for s in self.sequence:
                d = Pipe.dataframe_map(d, s[0], s[1], s[2])
        return d

    def __repr__(self) -> str:
        r = [s[0].__name__ for s in self.sequence]
        return " -> ".join(r)

    def add2pipe(self, func: Callable, *args, **kwargs):
        return self.sequence.append(func, args, kwargs)
