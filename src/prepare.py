from dvc import api
import pandas as pd
from io import StringIO
import sys
import logging
import numpy as np

from pandas.core.tools import numeric

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

logging.info('Fetching data...')

movie_data_path = api.get_url('data/movies.csv', remote='datafile')
finantial_data_path = api.get_url('data/finantials.csv', remote='datafile')
opening_data_path = api.get_url('data/opening_gross.csv', remote='datafile')

fin_data = pd.read_csv(finantial_data_path)
movie_data = pd.read_csv(movie_data_path)
opening_data = pd.read_csv(opening_data_path)

column_numeric = movie_data.select_dtypes(include=np.number)
movie_data = pd.concat([column_numeric, movie_data['movie_title']], axis=1)

fin_data = fin_data[['movie_title' , 'production_budget','worldwide_gross']]

fin_movie_data = pd.merge(fin_data, movie_data, on='movie_title',how= 'left')
full_movie_data = pd.merge(opening_data, fin_movie_data, on = 'movie_title',how='left')

full_movie_data = full_movie_data.drop(['gross','movie_title'],axis=1)

full_movie_data.to_csv('data/full_data.csv',index=False)

logger.info('data Fetched and prepared...')