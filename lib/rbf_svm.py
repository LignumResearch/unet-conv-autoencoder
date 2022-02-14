# rbf svm classifier
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from lib import metrics

def fit(latent, labels):
    
    latent_dim = latent.shape[1]
    
    seed = 199
    np.random.seed(seed)
    split = 0.1


    latent = np.load('latent.npy')
    labels = np.load('labels_train.npy')

    print(type(latent), latent.shape)
    print(type(labels), labels.shape)


    x_train, x_test, y_train, y_test = train_test_split(latent, labels, test_size = split, 
        random_state = seed, stratify = labels)

    print(y_test.shape)

    # SVC(C=50000.0, class_weight='balanced', decision_function_shape='ovo', degree=1, gamma=0.005, kernel='poly', verbose=True)
    param_grid = {'C': [1e1, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
                'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], 
                # 'kernel': ['poly', 'rbf', 'linear'],
                'degree': [1, 2, 3, 4, 5, 6, 7]
                }

    clf = GridSearchCV(
        SVC(
            kernel='rbf',
            decision_function_shape = 'ovo', 
            verbose = False, 
            class_weight = 'balanced',
        ),
        param_grid,
        verbose=False,
    )

    print(y_train.argmax(1))

    clf.fit(x_train, y_train.argmax(1))
    # print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    pred = clf.predict(x_test)
    print(pred)

    # score = clf.score(x_test, y_test.argmax(1))
    # print(score)

    # cm = confusion_matrix(pred, y_test.argmax(1))
    # print(cm)

    # #targets = ['am', 'bl', 'hk', 'hl', 'oo', 'rm', 'ro', 'ss', 'wa', 'wo']
    # #targets = ['awc', 'bfr', 'rwd', 'syp']
    # #targets = ['dfr', 'erc', 'pdp', 'poc', 'rwd', 'sgp', 'spr', 'syp']
    # #targets = ['awc', 'bfr','dfr', 'erc', 'icr', 'pdp', 'poc', 'rwd', 'sgp', 'spr', 'syp', 'wrc'] #FULL DATASET

    # if labels.shape[1] == 11: # softwoods
    #     targets = ['bfr','dfr', 'erc', 'icr', 'poc', 'rwd', 'sgp', 'spr', 'syp', 'wrc']
    # else: # hardwoods
    #     tagets = ['am', 'bl', 'hk', 'hl', 'oo', 'rm', 'ro', 'ss', 'wa', 'wo']
        
    # class_report = classification_report(y_test.argmax(1), pred, target_names = targets)
    # print(class_report)

    # save metrics
    metrics.save(y_test.argmax(1), pred, "rbf_svm", latent_dim)