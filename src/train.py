from sklearn.model_selection import train_test_split , cross_validate , GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

import logging
import sys
import numpy as np
import pandas as pd
from utils import update_model , save_simple_metrics_report , get_model_perfomance_test_set

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

logger.info('Loading data...')
data = pd.read_csv('../data/full_data.csv')

logger.info('Loading model...')
model = Pipeline([
    ('inputer',SimpleImputer(strategy='mean',missing_values=np.nan)),
    ('core_model' , GradientBoostingRegressor())
])

logger.info('Seraparating dataset into train and test')

X = data.drop(['worldwide_gross'],axis=1)
Y = data['worldwide_gross']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,shuffle=True,test_size=0.35,random_state=42)

logging.info('Setting Hyperparameter to tune')
pram_tuning = {'core_model__n_estimators':range(20,301,20)}

grid_search = GridSearchCV(model,param_grid=pram_tuning,scoring='r2',cv=5)

logger.info('Starting grid search... ')
grid_search.fit(X_train,Y_train)

logger.info('Cross validating with best model... ')
final_result = cross_validate(grid_search.best_estimator_, X_train,Y_train,return_train_score=True,cv=5)

train_score = np.mean(final_result['train_score'])
test_score1 = np.mean(final_result['test_score'])

assert train_score > 0.7
assert test_score1 > 0.05

logger.info(f'Train Score: {train_score}')
logger.info(f'Test Score: {test_score1}')

logger.info('Updating model....')
update_model(grid_search.best_estimator_)

logger.info('Generating model report...')

test_score = grid_search.best_estimator_.score(X_test,Y_test)
save_simple_metrics_report(train_score,test_score1,test_score,grid_search.best_estimator_)

Y_test_pred = grid_search.best_estimator_.predict(X_test)
get_model_perfomance_test_set(Y_test,Y_test_pred)

logger.info('Training Finished')