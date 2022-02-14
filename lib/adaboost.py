# Adaboost classifier using decision stumps
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from lib import metrics

def fit(latent, labels):

    latent_dim = latent.shape[1]
    
    # train_test_split params
    seed = 199
    np.random.seed(seed)
    split = 0.1

    # latent = np.load('latent.npy')
    # labels = np.load('labels_train.npy')

    x_train, x_test, y_train, y_test = train_test_split(latent, labels, test_size = split, 
        random_state = seed, stratify = labels)

    param_grid = {
        'n_estimators':     [100, 500, 600, 700, 1000],
        'learning_rate':    [.01, .001, .0001],
    }

    clf = GridSearchCV(
        AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=1), # decision stump
        ),
        param_grid,
        n_jobs = 1,
        verbose=True,
    )

    # clf = AdaBoostClassifier(
    #     DecisionTreeClassifier(max_depth=1), # decision_stump
    #     n_estimators=600,
    #     learning_rate=.01,
    # )

    clf.fit(x_train, y_train.argmax(1))
    print(clf.best_estimator_)
    
    pred = clf.predict(x_test)

    # print(clf.score(x_test, y_test.argmax(1)))

    # print(y_test.argmax(1), y_test.argmax(1).shape, pred, pred.shape)

    metrics.save(y_test.argmax(1), pred, "adaboost", latent_dim)
    
    return clf
