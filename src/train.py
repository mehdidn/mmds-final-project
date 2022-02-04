import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier
def categorical_to_numerical(X):
    if len(X.shape) == 1:
        _, XX = np.unique(X, return_inverse=True)
    elif len(X.shape) == 2:
        XX = np.zeros(X.shape)
        for i in np.arange(X.shape[1]):
            _, XX[:,i] = np.unique(X[:,i], return_inverse=True)
    else:
        raise(ValueError('Do not support array of shape > 2'))
    return XX

def func_compute_AUC(labels, scores):
    assert len(labels) == len(scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    return(roc_auc)

data_path = '../Data/'
filePath_csv = '../result/'

print('Load features')
train_feat_file = data_path + 'train_shopper.csv'
dfTrain = pd.read_csv(train_feat_file)
dfTrain = dfTrain.fillna(0.0)
numTrain = dfTrain.shape[0]

test_feat_file = data_path + 'test_shopper.csv'
dfTest = pd.read_csv(test_feat_file)
dfTest = dfTest.fillna(0.0)
ID_test = dfTest['id'].values

# convert categorical variables
categorical_names = [
'offer', 'market', 'company', 'category', 'brand',
'dept', 'offer_mday', 'offer_mweek', 'offer_weekday',
]
df = pd.concat([dfTrain[categorical_names], dfTest[categorical_names]])
df_c = categorical_to_numerical(df.values)
dfTrain[categorical_names] = df_c[:numTrain,:]
dfTest[categorical_names] = df_c[numTrain:,:]

# get the names of predictors
all_vars = dfTrain.columns
unused = [
'id', 'repeater', 'repeattrips', 'offer_date',
'offer_mday', 'offer_days', 'offer_weekday'
]
predictors = []
for i in all_vars:
    if i not in unused:
        predictors.append(i)

print('Perform training-validation')
offer_date = dfTrain['offer_date'].values

train_fraction = 0.6

order = offer_date.argsort()
RankOrder = order.argsort()
top = np.int32(np.floor(train_fraction * len(offer_date)))
thresh = offer_date[RankOrder==top][0]
thresh = '2013-04-01'

train_idx = np.where(offer_date < thresh)[0]
valid_idx = np.where(offer_date >= thresh)[0]

# train GBM
# params for GBM
GBM_ntree = 50
GBM_subsample = 0.5
GBM_lr = 0.01
random_seed = 2014
clf = GradientBoostingClassifier(n_estimators=GBM_ntree,
                                 subsample=GBM_subsample,
                                 learning_rate=GBM_lr,
                                 random_state=random_seed,
                                 verbose=3)

clf.fit(dfTrain[predictors].values[train_idx,:], dfTrain['repeater'].values[train_idx])
y_valid_pred = clf.predict_proba(dfTrain[predictors].values[valid_idx,:])[:,1]
auc_valid = func_compute_AUC(dfTrain['repeater'].values[valid_idx], y_valid_pred)
print("Valid AUC: {}".format(auc_valid))

print('Train the final model')
clf.fit(dfTrain[predictors].values, dfTrain['repeater'].values)
print('Make prediction')
p = clf.predict_proba(dfTrain[predictors].values)[:,1]

print('Create submission')
test_history_file = data_path + 'testHistory.csv'
dfTestHistory = pd.read_csv(test_history_file)
allID = np.asarray(dfTestHistory.values[:,0], dtype=np.int64)

y_test_prob = np.zeros((len(allID),))
for i in range(len(allID)):
    y_test_prob[i] = p[i]

sub = dict()
sub['id'] = allID
sub['repeatProbability'] = y_test_prob
sub = pd.DataFrame(sub)
fileName = filePath_csv + 'final_result.csv'
sub.to_csv(fileName, index = False)
