#### Multinomial GDA


class MGDA(object):
    def __init__(self,x,y):
        self.x= x
        self.y=y
    def phi_y(self):
        import numpy as np
        classes  = np.unique(self.y)
        phi  = np.zeros((len(classes),1))
        for i in range(len(classes)):
            phi[i]  = (np.sum(self.y==i)/len(self.y))
        return phi
    
    def compute_mu(self):
        import numpy as np
        classes  = np.unique(self.y)
        mu= np.zeros((len(classes),self.x.shape[1]))
        for i in range(len(classes)):
        #count = 0
            for j in range(self.x.shape[0]): 
                if self.y[j]==i:
                    mu[i,:]+=self.x[j]
            mu[i,:] = mu[i,:]/np.sum(self.y==i)
        return mu
    
    def compute_sigma(self):
        import numpy as np
        mu = self.compute_mu()
        classes = np.unique(self.y)
        sigma = 0
        for i in range (len(self.x)):
            for c in range(len(classes)):
                if self.y[i]==c:
                    m = mu[c]
                    sigma+=np.dot((self.x[i].reshape(-1,1)-m.reshape(-1,1)),(self.x[i].reshape(-1,1)-m.reshape(-1,1)).T)
                #print(sigma.shape)
        return sigma/len(self.x)
    
    def probability_pxpy(self,z,mu,sigm):
        import numpy as np
        dim = len(mu)
        c = (1./np.sqrt((2*np.pi)**(dim) *np.abs( np.linalg.det(sigm))))
        exp = np.dot((z.reshape(-1,1)-mu.reshape(-1,1)).T,np.dot(np.linalg.inv(sigm),(z.reshape(-1,1)-mu.reshape(-1,1))))
        return c * np.exp(-0.5 * exp)[0][0]
    
    def prediction(self,z):
        
        import numpy as np
        mu = self.compute_mu()
        sigma = self.compute_sigma()
        phi =self.phi_y()
        predt= np.zeros(len(z))
        prob = np.zeros((1,len(phi)))
    
        for i in range(len(z)):
            for c in range(len(phi)):
                prob[:,c]= self.probability_pxpy(z[i],mu[c],sigma)*phi[c,:]
            #print(prob[:,c])
            predt[i] = np.argmax(prob)
        return predt   
    
    def accuracy(self,z,w):
        import numpy as np
        Acc= np.sum(self.prediction(z)==w)*100.0/len(w)
        return Acc
    
    
    
