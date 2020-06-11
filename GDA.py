
class GDA(object):
    def __init__(self,x,y):
        self.x= x
        self.y=y

        
    def phi(self):
        a= self.y==1
        return (1./len(self.y))*sum(a)[0]

    def values_mu_0(self):
        a= self.y==0
        T= sum(a)[0]
        count = 0
        for i in range(len(self.x)):
            if self.y[i]==0:
                count += self.x[i]
        return (count/T).reshape(-1,1)

    def values_mu_1(self):
        a= self.y==1
        T= sum(a)[0]
        count = 0
        for i in range(len(self.x)):
            if self.y[i]==1:
                count += self.x[i]
        return (count/T).reshape(-1,1)
    
    def compute_sigma(self):
        import numpy as np
        count = 0
        mu0 = self.values_mu_0()
        mu1 = self.values_mu_1()
        for i in range(len(self.x)):
            if self.y[i]==0:
                count = count + np.dot(self.x[i].reshape(-1,1)-mu0,(self.x[i].reshape(-1,1)-mu0).T)
            else:
                count = count + np.dot(self.x[i].reshape(-1,1)-mu1,(self.x[i].reshape(-1,1)-mu1).T)
            
        return count/len(self.x)
    
    def probability_pxpy(self,z,mu,sigm):
        import numpy as np
    	
        dim = len(mu)
        c = (1./np.sqrt((2*np.pi)**(dim) *np.abs( np.linalg.det(sigm))))
        exp = np.dot((z.reshape(-1,1)-mu).T,np.dot(np.linalg.inv(sigm),(z.reshape(-1,1)-mu)))
        return c * np.exp(-0.5 * exp)[0][0]
    
    def pred(self,z):
        import numpy as np
        mu0=self.values_mu_0()
        mu1=self.values_mu_1()
        p= self.phi()
        sigma=self.compute_sigma()
        predt= np.zeros(len(z))

        for i in range(len(z)):
            p0=self.probability_pxpy(z[i],mu0,sigma)*(1-p)
            p1=self.probability_pxpy(z[i],mu1,sigma)*p
            predt [i] = np.argmax([p0,p1]) 
        
        return predt.reshape(-1,1)
    
    def accuracy(self,z,w):
        import numpy as np
        Acc= np.sum(self.pred(z)==w)*100.0/len(w)
        return Acc
        
    
    
