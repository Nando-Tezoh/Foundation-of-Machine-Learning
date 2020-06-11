### Softmax

class softmax_Regression(object):
    def __init__(self,x,y,theta,n_iters,lr):
        self.theta = theta
        self.lr= lr
        self.x = x
        self.y = y
        self.n_iters = n_iters
    def softmax(self,z):
        import numpy as np
        return(((np.exp(z.T))/ np.sum(np.exp(z), axis=1).T)).T
    
    def cost_function(self):
        import numpy as np
        classes = np.unique(self.y)
        a= np.dot(self.x,self.theta)
        P  = self.softmax(a)
        count = 0
        for i in range(len(self.x)):
            for j in range(len(classes)):
                if self.y[i] == classes[j]:
                    count+= np.log(P[i,j])
        return -count
    def indicator(self,a,b):
        if a==b:
            return 1
        else:
            return 0
    
    def Gradient(self):
        import numpy as np
        classes = np.unique(self.y)
        a=np.dot(self.x,self.theta)
        P  = self.softmax(a)
        Grad  = np.zeros((len(classes),self.x.shape[1]))
        for c in range(len(classes)):
            diff = 0
            for i in range(len(self.x)):
                p1=self.indicator(self.y[i],c)-P[i,c]
                diff  +=self.x[i]*p1 
            #ipdb.set_trace()
            Grad[c]=diff
        return -Grad/len(self.x)
    
    def fit(self):
    
        import numpy as np
        #loss = np.zeros((self.n_iters))
        for i in range(self.theta.shape[1]):
            for j in range(self.n_iters):
                #Grad=Gradient(x,y,theta)
                self.theta[:,i] = self.theta[:,i] -(self.lr *self.Gradient()[i])

        return self.theta #, loss
    def prediction (self,z):
        import numpy as np
        a=np.dot(z,self.fit())   
        b = self.softmax(a)
        return np.argmax(b,axis=1)
    
    def accuracy(self,z,w):
        import numpy as np
        return (np.sum(self.prediction(z).reshape(-1,1)==w)*1.0/len(w))*100.0
    
        
    
                
