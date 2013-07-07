import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score

if __name__ == '__main__':
    
    #classifier = GradientBoostingClassifier()
    classifier = RandomForestClassifier(n_estimators=100, random_state=1, verbose=2)
    #classifier = LogisticRegression(C=3)
    
    data = np.loadtxt('train.csv', skiprows=1, delimiter=',')
    
    #data = data[:2000,:]
    labels = data[:,0]
    features = data[:,1:]
    
    enc = OneHotEncoder().fit(features)
    
    #features = enc.transform(features).toarray()
    print features.shape

    cv = KFold(len(labels), n_folds=5, shuffle=True, random_state=1)
    score = cross_val_score(classifier, features, labels, cv=cv, scoring='roc_auc', n_jobs=1)

    score = np.asarray(score)
    
    print score.mean(), score.std()

        
        
