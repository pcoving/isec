import numpy as np
from sklearn.base import BaseEstimator
from sklearn import preprocessing

class NeuralNetworkClassifier(BaseEstimator):
    
    def __init__(self, learning_rate=0.01,
                 momentum=0.9,
                 n_epochs=100,
                 batch_size=500,
                 n_hidden=100,
                 cost='entropy',
                 reg=0.0,
                 random_state=1):

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.cost=cost
        self.reg = reg

        if cost not in ['entropy', 'mse']:
            raise ValueError;
        self.random_state = random_state

        self.weights_1 = None
        self.bias_1 = None
        self.weights_2 = None
        self.bias_2 = None

    def fit(self, X, y):
        n_batches = X.shape[0]/self.batch_size

        n_input = X.shape[1]
    
        np.random.seed(self.random_state)

        # initialize weights
        self.weights_1 = 4.*np.sqrt(6./(n_input+self.n_hidden))*(1. - 2*np.random.randn(n_input, self.n_hidden)) #dim_in ** -0.5 * np.random.randn(dim_in, self.n_hidden)
        self.bias_1 = np.zeros((self.n_hidden, 1))
        self.weights_2 = 4.*np.sqrt(6./(self.n_hidden+1))*(1. - 2*np.random.randn(self.n_hidden, 1)) #self.n_hidden ** -0.5 * np.random.randn(self.n_hidden, 1)
        self.bias_2 = np.zeros((1, 1))
        
        # initialize weight update matrices
        weights_update_1 = np.zeros(self.weights_1.shape)
        bias_update_1 = np.zeros(self.bias_1.shape)
        weights_update_2 = np.zeros(self.weights_2.shape)
        bias_update_2 = np.zeros(self.bias_2.shape)
        
        # Train neural network.
        for epoch in range(self.n_epochs):
            err = []
            for batch in range(n_batches):
                # get current minibatch
                feature_batch = X[batch*self.batch_size:(batch + 1)*self.batch_size, :].toarray()
                label_batch = y[batch*self.batch_size:(batch + 1)*self.batch_size]
                
                # apply self.momentum
                weights_update_1 *= self.momentum
                bias_update_1 *= self.momentum
                weights_update_2 *= self.momentum
                bias_update_2 *= self.momentum
            
                ''' forward pass '''
                # compute activation function
                hidden = 1. / (1 + np.exp(-(feature_batch.dot(self.weights_1).T + self.bias_1)))
            
                # compute sigmoidal output
                output = 1./ (1 + np.exp(-(np.dot(self.weights_2.T, hidden) + self.bias_2)))
                
                err.append(np.sum((label_batch-output)**2)/float(self.batch_size))
                #err.append(np.sum(target != np.round(out))/float(target.shape[1]))
                
                ''' back propogation '''
                # back propogate errors
                error = (output - label_batch)

                # gradients for weights_2 and bias_2
                weights_update_2 += np.dot(hidden, error.T)
                #import pdb; pdb.set_trace()
                bias_update_2 += error.sum(1)[:, np.newaxis]
                
                # compute delta
                if self.cost == 'entropy':
                    delta = np.dot(self.weights_2, error)
                elif self.cost == 'mse':
                    delta = np.dot(self.weights_2, error)*hidden*(1.-hidden)
                
                weights_update_1 += feature_batch.T.dot(delta.T)
                bias_update_1 += delta.sum(1)[:, np.newaxis]
                
                # regularization
                weights_update_1 += self.reg*np.sign(self.weights_1)
                weights_update_2 += self.reg*np.sign(self.weights_2)
                #weights_update_1 += self.reg*self.weights_1
                #weights_update_2 += self.reg*self.weights_2
                
                # update weights
                self.weights_1 -= weights_update_1*self.learning_rate/self.batch_size
                self.bias_1 -= bias_update_1*self.learning_rate/self.batch_size
                self.weights_2 -= weights_update_2*self.learning_rate/self.batch_size
                self.bias_2 -= bias_update_2*self.learning_rate/self.batch_size
            
            #print np.mean(err)

    def predict_proba(self, X):
        
        h = 1. / (1 + np.exp(-(X.dot(self.weights_1).T + self.bias_1)))
        
        out = 1./ (1 + np.exp(-(np.dot(self.weights_2.T, h) + self.bias_2)))
        #print np.min(np.abs(self.weights_1)), np.min(np.abs(self.weights_2))
        return np.vstack((1.-out, out)).T
