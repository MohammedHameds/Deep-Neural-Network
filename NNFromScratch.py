import numpy as np
import matplotlib.pyplot as plt 


class NeuralNetwork:
    
    def __init__(self):
        
        self.NNSizes = []
        self.NNActivationFunctions = []
        self.NNCostFunction = []
        self.NNCache = {}
        self.NNParameters = {}
        self.NNDerivatived = {}
        np.random.seed(1)
        
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
  

    def relu_backward(self , dA, Z):
      dZ = np.array(dA, copy=True) 
      dZ[Z <= 0] = 0
      return dZ


    def tanh_backward(self , dA, Z):
      a = self.tanh(Z)
      dZ = dA * (1-a**2)
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
       

    def CostFunction(self,Y) :
        m = int(len(self.NNParameters)/2)
        A = self.NNCache.get(f'A{m}')
        
        m = len(Y)
        left = np.multiply(Y,np.log(A))
        right = np.multiply((1-Y),np.log(1-A))
        cost = -1/m * np.sum( left + right )
        cost = np.squeeze(cost) 
        return cost

    def CostFunctionDerivatived(self,A,Y) :
        dL = A - Y
        return dL        

    
        
    def LinearActivationBackward (self , dA , Z , actifation_fun) :
        if actifation_fun == 'sigmoid' :
           dZ = self.sigmoid_backward(dA,Z)
           
        elif actifation_fun == 'relu' :
           dZ = self.relu_backward(dA,Z)
           
        elif actifation_fun == 'tanh' :
           dZ = self.tanh_backward(dA,Z)     
           
        return dZ
    
    
    def FeedBackward (self,output) :
        n = int(len(self.NNParameters)/2)
        hx = self.NNCache.get('A' + str(n))
        dA2 = self.CostFunctionDerivatived(hx,output) # dA2
        self.NNDerivatived['dA'+str(n)] = dA2



        for i in range (n, 0, -1)  :
            
            A_prev = self.NNCache.get('A' + str(i-1))
            W = self.NNParameters.get('W' + str(i))
            activ_fun = self.NNActivationFunctions[i]
            m = len(A_prev)
            
            dA = self.NNDerivatived.get('dA' + str(i))
            
            Z = self.NNCache.get('Z' + str(i))
            dZ = self.LinearActivationBackward (dA , Z , activ_fun)       

            dW =  1/m * np.dot(dZ, A_prev.T)
            db =  1/m * np.sum(dZ, axis=1, keepdims=True)   
            
            dA_prev = np.dot(W.T, dZ) 
            
            self.NNDerivatived['dZ' +str(i)] = dZ                      
            self.NNDerivatived['dW' +str(i) ] = dW       
            self.NNDerivatived['db' +str(i) ] = db       
            self.NNDerivatived['dA' +str(i-1)] = dA_prev

    def predict(self, x):
         self.FeedForward(x)
         
         n = int(len(self.NNParameters)/2)
         predicted_values = self.NNCache.get('A' + str(n))
         
         return predicted_values

        
    def UpdateParameters(self, learning_rate ):
  
        L = len(self.NNParameters) // 2 # number of layers in the neural network

        for l in range(L):

            W = self.NNParameters.get("W" + str(l+1))
            B = self.NNParameters.get("B" + str(l+1))
            
            W -= learning_rate * self.NNDerivatived.get("dW" + str(l+1))
            B -= learning_rate * self.NNDerivatived.get("db" + str(l+1))
            
            self.NNParameters["W" + str(l+1)] = W
            self.NNParameters["B" + str(l+1)] = B
            
    def Fit (self,input,output , learning_rate = 0.001 , iterations = 1000) :
        self.NNInput = input
        self.NNOutput = output
        
        if self.NNSizes != [] :
            self.NNSizes.insert(0, len(self.NNInput[0])) 
            self.NNActivationFunctions.insert(0, 'sigmoid') 
            
        self.Initialize()    
        
        for i in range (iterations) :
                    
          self.FeedForward(input)  
          self.FeedBackward(output)
          self.UpdateParameters(learning_rate)
          self.NNCostFunction.append(self.CostFunction(output))
          
     
    def History (self) :
        cost = self.NNCostFunction
        print('Training loss :' , cost[-1])
        plt.plot( cost, color = 'red')
        plt.title('Training loss function')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.show()





    