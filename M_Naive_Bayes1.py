class MNB(object):
    def __init__(self,x,y):
        self.x= x
        self.y=y
    def phi_y(self):
        import numpy as np
        classes  = np.unique(self.y)
        phi=np.zeros((len(classes),1))
        for i in classes:
            phi[i-1] = (np.sum(self.y==i)/len(self.y))
        
        return phi
    
    def phi_x_y(self):
        import numpy as np
        classes = np.unique(self.y)
        phi  = np.zeros((len(classes),self.x.shape[1]))
        x = self.x.values
        y=self.y.values
        for k in classes:
            for i in range(x.shape[1]):
                count = 0
                for j in range(x.shape[0]):
                    if y[j]==k and x[j,i]==1:
                        count+=x[j,i]
                phi[k-1,i] = count/np.sum(y==k)
            
        return phi
    
    def bernouilli(self,z,ph):
    
        return ph**(z)*(1-ph)**(1-z)
    
    def prediction1(self,z):
        import numpy as np
        p_x_y= self.phi_x_y()
        p_y = self.phi_y()
        predt= np.zeros(len(z))
        for i in range(z.shape[0]):
        
            prob = np.ones((1,len(p_x_y)))
            for j in range(z.shape[1]):
            
                for k in range(prob.shape[1]):
                
                    prob[:,k]= prob[:,k]*self.bernouilli(z.values[i,j],p_x_y[k,j])
                
            for k in range(p_y.shape[0]):
                prob[:,k]  = prob[:,k]*p_y[k,:]    
    
            predt[i] = np.argmax(prob)
        return predt #, prob_0,prob_1
    
    def accuracy(self,z,w):
        
        import numpy as np
        return np.sum(self.prediction1(z)==w)*1.0/len(w)
    
    
    
