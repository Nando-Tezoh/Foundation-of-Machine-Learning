#### Naive Bayes Code



class NB(object):
    def __init__(self,x,y):
        self.x= x
        self.y=y
        
    def phi_y(self):
        import numpy as np
        return np.sum(self.y==1)/len(self.y)
        
    def phi_x_y_0(self):
        import numpy as np
        x  = self.x.values
        y=self.y.values
        P= np.zeros(x.shape[1])
        for i in range(x.shape[1]):
            count = 0
            for j in range(x.shape[0]):
                if y[j] == 0 and x[j,i]==1:
                    count += x[j,i]
            P[i] = count
        return (P)*(1/(sum(y==0)))
    def phi_x_y_1(self):
        import numpy as np
        x  = self.x.values
        y=self.y.values
        P= np.zeros(x.shape[1])
        for i in range(x.shape[1]):
            count = 0
            for j in range(x.shape[0]):
                if y[j] == 1 and x[j,i]==1:
                    count += x[j,i]
            P[i] = count
    
        
        return (P)*(1/(sum(y==1)))
    

    def bernouilli(self,z,phi):
        import numpy as np
    
        return phi**(z)*(1-phi)**(1-z)
    
    def prediction1(self,z):
        import numpy as np
        p = self.phi_y()
        p1= self.phi_x_y_1()
        p0  = self.phi_x_y_0()
        predt= np.zeros(len(z))
        prob_0 = np.zeros((z.shape[0],z.shape[1]))
        prob_1 = np.zeros((z.shape[0],z.shape[1]))
        for i in range(z.shape[0]):
            prob0 = 1
            prob1 = 1
            for j in range(z.shape[1]):
                prob0=prob0*self.bernouilli(z.values[i,j],p0[j])
                prob1=prob1*self.bernouilli(z.values[i,j],p1[j])
            
                prob_0[i,j] = prob0
                prob_1[i,j] = prob1
            prob0= prob0*(1-p)
            prob1= prob1*(p)
        #print('probabibilty 0',prob0)
        #print(prob1)
        
    
            predt[i] = np.argmax([prob0,prob1])
        return predt
        
    def accuracy(self,z,w):
        import numpy as np
        return np.sum(self.prediction1(z)==w)*1.0/len(w)

    
    
    
    
