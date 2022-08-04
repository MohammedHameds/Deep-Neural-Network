from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt 
import numpy as np
np.random.seed(3)

X,y = make_blobs(10,centers= 2)
# plt.scatter(X[:,0],X[:,1] ,c=y,s=50, cmap = 'vlag_r')
# plt.grid()
# plt.show()



class NeuralNetwork:
    
    def __init__(self):
        
        self.NNSizes = []
        self.NNActivationFunctions = []
        # self.NNInput = []
        # self.NNOutput = []
        self.NNCostFunction = []
        self.NNCache = {}
        self.NNParameters = {}
        self.NNDerivatived = {}
        
    def Initialize(self) :
       layer_dims = self.NNSizes
       L = len(layer_dims)
       
       for i in range (1,L) :
           self.NNParameters['W' + str(i)]  = np.random.randn(layer_dims[i],layer_dims[i-1]) * 0.01
           self.NNParameters['B' + str(i)]  = np.random.randn(layer_dims[i],1)
           

    def Add (self, size = 5 , activation_fun = 'sigmoid' ):
       self.NNSizes.append(size)
       self.NNActivationFunctions.append(activation_fun)
    
    
    def sigmoid(self,Z):
      A = 1/(1+np.exp(-Z))
      return A
  
    
    def relu(self,Z):
      A = np.maximum(0,Z)
      return A

    def tanh(self,Z):
      A = (np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z)) 
      return A
  
    def sigmoid_backward(self , dA, cache):
      Z = cache
      s = 1/(1+np.exp(-Z))
      dZ = dA * s * (1-s)
      
      return dZ          
        
    def LinearForward(self,A_prev, W , b) :
        Z = np.dot(W,A_prev) + b
        return Z       

    def LinearActivationForward(self , A_prev, W , b , activation) :
        if activation == 'sigmoid' :
          Z  = self.LinearForward(A_prev, W , b) 
          A  = self.sigmoid(Z)

        elif activation == 'relu' :
          Z  = self.LinearForward(A_prev, W , b) 
          A  = self.relu(Z)

        elif activation == 'tanh' :
          Z  = self.LinearForward(A_prev, W , b) 
          A  = self.tanh(Z) 
        
        return A  , Z      

    def FeedForward (self , x ) :
       cache = {}
       layer_dims = self.NNSizes
       L = len(layer_dims)    
       A0 = x.T
       cache['A0'] = A0
       Parameters = self.NNParameters

    
       for i in range (1,L) :
         a = cache.get(f'A{i-1}')
         w = Parameters.get(f'W{i}')
         b = Parameters.get(f'B{i}')
         activation_fun = self.NNActivationFunctions[i]
    
         A , Z = self.LinearActivationForward(a,w,b,activation_fun)
         
         cache[f'A{i}'] = A 
         cache[f'Z{i}'] = Z 
               
       self.NNCache = cache
       

    def CostFunction(self,A,Y) :
        m = len(Y)
        left = np.multiply(Y,np.log(A))
        right = np.multiply((1-Y),np.log(1-A))
        cost = -1/m * np.sum( left + right )
        cost = np.squeeze(cost) 
        return cost

    def CostFunctionDerivatived(self,A,Y) :
        dL = A - Y
        return dL        



    def LinearBackward (self) :
        pass
        
    def LinearActivationBackward (self) :
        pass
    
    def FeedBackward (self,output) :
        n = int(len(self.NNParameters)/2)
        hx = self.NNCache.get('A2')  # A2
        dA2 = self.CostFunctionDerivatived(hx,output) # dA3  
        self.NNDerivatived['dA2'] = dA2



        A1 = self.NNCache.get('A1')  # A1
        A0 = self.NNCache.get('A0')  # A0
        
        W2 = self.NNParameters.get('W2')
        W1 = self.NNParameters.get('W1')
            
        Z2 = self.NNCache.get('Z2')
        Z1 = self.NNCache.get('Z1')
        
        m = len(A1[0])
       


        
        dZ2 = self.sigmoid_backward(dA2,Z2)
        dW2 =  1/m * np.dot(dZ2, A1.T)
        db2 =  1/m * np.sum(dZ2, axis=1, keepdims=True)
        dA1 = np.dot(W2.T, dZ2) 

        
        dZ1 = self.sigmoid_backward(dA1,Z1)
        dW1 =  1/m * np.dot(dZ1, A0.T)
        db1 =  1/m * np.sum(dZ1, axis=1, keepdims=True)


        

        
  
        
        self.NNDerivatived['dZ2'] = dZ2
        self.NNDerivatived['dW2'] = dW2
        self.NNDerivatived['db2'] = db2
        self.NNDerivatived['dA1'] = dA1
        
        self.NNDerivatived['dZ1'] = dZ1
        self.NNDerivatived['dW1'] = dW1
        self.NNDerivatived['db1'] = db1
        
    def UpdateParameters(self, learning_rate = 0.001):
  
        L = len(self.NNParameters) // 2 # number of layers in the neural network

        for l in range(L):
            W = self.NNParameters.get("W" + str(l+1))
            B = self.NNParameters.get("B" + str(l+1))
            
            W -= learning_rate * self.NNDerivatived.get("dW" + str(l+1))
            # B -= learning_rate * self.NNDerivatived.get("db" + str(l+1))
            self.NNParameters["W" + str(l+1)] = W
            # self.NNParameters["b" + str(l+1)] = B
            
    def Fit (self,input,output) :
        self.NNInput = input
        self.NNOutput = output
        
        if self.NNSizes != [] :
            self.NNSizes.insert(0, len(self.NNInput[0])) 
            self.NNActivationFunctions.insert(0, 'sigmoid') 
            
        self.Initialize()    
        
        
        # m = int(len(self.NNParameters)/2)
        # hx = self.NNCache.get(f'A{m}')
        # cost_fun = self.CostFunction(hx,output)
        # self.NNCostFunction.append(cost_fun)
        
        W = self.NNParameters.get("W" + str(2))
        B = self.NNParameters.get("B" + str(2))        
        print(W)
    
        for i in range (1) :
                    
          self.FeedForward(input)  
          self.FeedBackward(output)
          self.UpdateParameters()
          
        print('-'*20)
        W = self.NNParameters.get("W" + str(2))
        B = self.NNParameters.get("B" + str(2))        
        print(W)
#--------------------------------------------    
nn = NeuralNetwork()
nn.Add(3,activation_fun='sigmoid')
nn.Add(1,activation_fun='sigmoid')
nn.Fit(X,y)


params = nn.NNParameters
size = nn.NNSizes
activation_fun = nn.NNActivationFunctions
cache = nn.NNCache
cost = nn.NNCostFunction
dd = nn.NNDerivatived







    