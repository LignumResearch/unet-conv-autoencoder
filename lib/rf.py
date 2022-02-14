# random forest classifier

from time import localtime, strftime
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from lib import metrics

def fit(latent, labels):
    
    latent_dim = latent.shape[1]
    # train_test_split params
    seed = 199
    np.random.seed(seed)
    split = 0.1

    # latent = np.load('latent.npy')
    # labels = np.load('labels_train.npy')

    # print(type(latent), latent.shape)
    # print(type(labels), labels.shape)

    x_train, x_test, y_train, y_test = train_test_split(latent, labels, test_size = split, 
        random_state = seed, stratify = labels)

    # print(y_test.shape)

    param_grid = {
        'n_estimators':     [x for x in range(80, 400, 25)],
        # 'max_leaf_nodes':   [x for x in range(10,25)],
        # 'max_depth':        [x for x in range(20, 31)],
        'min_samples_leaf': [x for x in range(2,5)],
        'max_samples':      [x / 100.0 for x in range(13, 19)],
        'min_samples_split': [x for x in range(2,11)],
    }

    clf = GridSearchCV(
        RandomForestClassifier(
            n_jobs=-1, 
            # verbose=True, 
            random_state=seed,
            # Hyperparameters
            max_leaf_nodes=16,
            max_depth=30,
        ), 
        param_grid,
        # n_candidates='exhaust', # use in HalvingRandomSearchCV
        # aggressive_elimination=True, # may result in suboptimal estimator
        verbose=True,
        )

    # print(y_train.argmax(1))
    # targets = ['bfr','dfr', 'erc', 'icr', 'poc', 'rwd', 'sgp', 'spr', 'syp', 'wrc']
        
    clf.fit(x_train, y_train.argmax(1))
    # print("Best estimator found by random havling search:")
    print(clf.best_estimator_)

    pred = clf.predict(x_test)
    # print(pred)

    # score = clf.score(x_test, y_test.argmax(1))
    # print(score)

    # cm = confusion_matrix(pred, y_test.argmax(1))
    # print(cm)

    # #targets = ['am', 'bl', 'hk', 'hl', 'oo', 'rm', 'ro', 'ss', 'wa', 'wo']
    # #targets = ['awc', 'bfr', 'rwd', 'syp']
    # #targets = ['dfr', 'erc', 'pdp', 'poc', 'rwd', 'sgp', 'spr', 'syp']
    # #targets = ['awc', 'bfr','dfr', 'erc', 'icr', 'pdp', 'poc', 'rwd', 'sgp', 'spr', 'syp', 'wrc'] #FULL DATASET
    # # targets = ['bfr','dfr', 'erc', 'icr', 'poc', 'rwd', 'sgp', 'spr', 'syp', 'wrc']
    # if labels.shape[1] == 11: # softwoods
    #     targets = ['bfr','dfr', 'erc', 'icr', 'poc', 'rwd', 'sgp', 'spr', 'syp', 'wrc']
    # else: # hardwoods
    #     tagets = ['am', 'bl', 'hk', 'hl', 'oo', 'rm', 'ro', 'ss', 'wa', 'wo']


    # print(classification_report(y_test.argmax(1), pred, target_names = targets))

    # save metrics
    metrics.save(y_test.argmax(1), pred, "rf", latent_dim)
    
    # return clf.score(x_test, y_test.argmax(1))
