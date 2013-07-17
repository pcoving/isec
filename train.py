import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score

from nnet import NeuralNetworkClassifier
import time

if __name__ == '__main__':
    
    data = np.loadtxt('train.csv', skiprows=1, delimiter=',')
    
    np.random.seed(1)
    np.random.shuffle(data)
    
    #data = data[:5000,:]
    
    labels = data[:,0]
    features = data[:,1:]
    
    enc = OneHotEncoder().fit(features)
    
    features = enc.transform(features)#.toarray()
    #print features.shape
    
    #classifier = GradientBoostingClassifier()
    #classifier = RandomForestClassifier(n_estimators=100, random_state=1, verbose=2)
    #classifier = LogisticRegression(C=2) #0.861367908825 0.00737749594104
    #classifier = KNeighborsClassifier(n_neighbors=10) #KNeighborsClassifier
    #classifier = SGDClassifier(loss='log', penalty='l2', alpha=3e-5, random_state=1) #0.840771244407 0.00683565339505

    #classifier = LogisticRegression(C=2) #0.861367908825 0.00737749594104
                                         #0.748410641523 0.0181605746841
    for learning_rate in [0.1, 0.01, 0.001]:
        for n_epochs in [100, 500, 1000]:
            for n_hidden in [50, 100, 200]:
                for reg in [0.1, 0.01, 0.001]:
                    classifier = NeuralNetworkClassifier(learning_rate=learning_rate,
                                                         momentum=0.9,
                                                         n_epochs=n_epochs,
                                                         batch_size=50,
                                                         n_hidden=n_hidden,
                                                         reg=reg,
                                                         random_state=1)
                    # 0.744104492462 0.0127844421145

                    start = time.time()
                    cv = KFold(len(labels), n_folds=5, shuffle=True, random_state=1)
                    score = cross_val_score(classifier, features, labels, cv=cv, scoring='roc_auc', n_jobs=-1)
                    #score = cross_val_score(classifier, features, labels, cv=cv,  n_jobs=1)
                    print (time.time() - start)/60
                    score = np.asarray(score)
            
                    print learning_rate, n_epochs, n_hidden, reg, score.mean(), score.std()

        
        
