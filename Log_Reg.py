
### Data should be an array

class LogisticRegression(object):
    def __init__(self,lr,n_iters,weight,x,y):
        self.lr = lr
        self.n_iters = n_iters
        self.weight = weight
        self.x = x
        self.y = y
        
    def Sigmoid(self,z):
        import numpy as np
        sign = 1 / (1+np.exp(-z))
        return sign
    
    def Cost(self):
        import numpy as np
        yhat= np.dot(self.x,self.weight)
        cost = np.dot(-self.y.T,np.log2(self.Sigmoid(yhat))) - np.dot((1 - self.y).T,np.log2(1 - self.Sigmoid(yhat)))
        #print(yhat.shape)
        return cost
    
    def Gradient(self):
        import numpy as np
        yhat= np.dot(self.x,self.weight)
        return np.dot(self.x.T , ((self.y) - self.Sigmoid(yhat)))
    
    def fit(self):
        import numpy as np
        loss = np.zeros((self.n_iters))
        for i in range(self.n_iters):
            self.weight = self.weight + (self.lr * self.Gradient())
            
            loss[i] = self.Cost()
            
        return self.weight #, loss
    
    
    def Good_pred(self,z):
        import numpy as np
        
        yhat=np.dot(z,self.fit())
        
        return np.where(self.Sigmoid(yhat)>0.5,1,0)
    
    def accuracy(self,z,w):
        import numpy as np
        acc = w==self.Good_pred(z)
        return np.sum(acc)/len(w)
        
        
