# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 12:24:17 2017

@author: ThaNyneTray
another attempt at a Neural Network
"""
import random
import math

class Perceptron: #Perceptron needs only track weights. That's it.
    def __init__(self,inputN):
        self.n = inputN+1
        self.inputs=[]
        self.weights = self.random_weights()
#        print("len of weights in init: ", len(self.weights))
        self.result = 0
#        self.learning = 0.9
        self.error=0
        
    def sigmoid(self,x):
         activation = 1/(1+math.exp(-x))
         return activation
    
    def sig_derivative(self,x):
        return x*(1-x)
    
    def random_weights(self):
        weights=[]
        for i in range(self.n):
            weights.append(random.randrange(-1,1))
#        print("len of weights: ", len(weights))
        return weights
        
    def get_result(self, inputs):        
        self.inputs=inputs+[1]
        for i in range(self.n):
            self.result = self.weights[i]*self.inputs[i]
        self.result = self.sigmoid(self.result)
        return self.result

class Layer:
    def __init__(self, inputN, outputN):# n is number of Perceptrons
        self.perceptrons = []
        self.inputN = inputN
        self.outputN = outputN
        self.output = []
        
    def makeLayer(self, inputPrev): #input of previous layer
        for idx in range(self.inputN):
            self.perceptrons.append(Perceptron(inputPrev)) 
        return self.perceptrons
    
    def forward(self, inputs):
        output=[]
#        print("percep: ",self.perceptrons[0])
        for perc in self.perceptrons:
            result = perc.get_result(inputs)
            output.append(result)
        self.output=output
            
    def setFirstLayer(self,firstL_input):
        self.output = firstL_input            
    

class Network:
    def __init__(self,layers):
        self.layers=layers #list containing layer objects
        self.firstL_inputs=[]
        self.learningR = 0.6
        self.errors=0
        
    def connectL(self): #connect layers. basically checks if output number equals input number of next layer are made correctly
        for i in range(1, len(self.layers)):
            if self.layers[i-1].outputN != self.layers[i].inputN:
                return
        for i in range (len(self.layers)):
            if i ==0:
                self.layers[i].makeLayer(0)
            else:    
                self.layers[i].makeLayer(self.layers[i-1].inputN)
                
    def forward_prop(self, firstL_input):
    
        for layerID in range(len(self.layers)): # start at index 1 since index 0 is the input layer
            layer=self.layers[layerID]
            if layerID==0:
                layer.setFirstLayer(firstL_input)
            else:    
                inputSource=self.layers[layerID-1] # the layer that comes before the current one.
                layer.forward(inputSource.output)
        
    
    def backprop(self, expected): #expected is a list 
        self.errors=0
        for layerID in range(len(self.layers)-1, -1, -1):
#            error_trace=0
            currLayer=self.layers[layerID]
            if layerID==(len(self.layers)-1):
                for i in range(len(currLayer.perceptrons)):
                    perceptron = currLayer.perceptrons[i]
                    error = (expected[i]-perceptron.result)*perceptron.sig_derivative(perceptron.result)
                    perceptron.error=error
#                    error_trace+=error
                    self.errors+=error

            elif layerID == 0:
                break
            else:
                for i in range(len(currLayer.perceptrons)):
                    error=0
                    nextLayer = self.layers[layerID+1]
                    
                    for perceptron in nextLayer.perceptrons:
                        error += (perceptron.weights[i]*perceptron.error)*currLayer.perceptrons[i].sig_derivative(currLayer.output[i])  
                        currLayer.perceptrons[i].error = error
                        self.errors+=error
#            self.errors.append(error_trace)
            
            
    def update_weights(self):
        for layerID in range(1,len(self.layers)):
            currLayer = self.layers[layerID]
            for perceptron in currLayer.perceptrons:
                inputs=perceptron.inputs
                weights=perceptron.weights
                error=perceptron.error
#                print("perceptron inputs ", perceptron.inputs)
#                print("perceptron weights\n", len(perceptron.weights),"\ninputs \n",len(perceptron.inputs))
                for i in range(len(inputs)):
                    perceptron.weights[i]=weights[i] + self.learningR * error * inputs[i]
#                print(perceptron.weights)
#            print("perceptron weights\n", len(perceptron.weights),"\ninputs \n",len(perceptron.inputs))
            
 
class Brain:
    def __init__(self, nn,train_input, expected_output, cyclesN):
        self.cyclesN=cyclesN
        self.nn = nn
        self.expected_output=expected_output
        self.train_input = train_input
#        self.learningR = 0.6 # learning rate
#        self.layers=layers
        
        return

    def train(self):
#        sum_error=0
        for cycle in range(self.cyclesN):
            self.nn.forward_prop(self.train_input)
            self.nn.backprop(self.expected_output)
#            print("train inp ",self.train_input, "expected ", self.expected_output)
#            print("cycle ",cycle," error ",self.nn.errors)
            self.nn.update_weights()
            
    def predict(self,predict_data):
        self.nn.forward_prop(predict_data)
#        print(self.nn.layers[-1].output)
#        predicted=map(round,self.nn.layers[-1].output)
        print(self.nn.layers[-1].output)
           
l=Layer(4, 5)
l2=Layer(5,2)
l3=Layer(2,1)
la=[l,l2,l3]
n=Network(la)
n.connectL()
b=Brain(n, [1,1,0,0], [1,0], 10000)
b.predict([1,1,0,0])
b.train()
b.predict([1,1,0,0])

#n.forward_prop([1,1,0,0])
#n.backprop([1,0])
#n.update_weights()

#l.makeLayer()
#l.forward([1,2,3,4])
    
#P = Perceptron(3)
#res = P.get_result([1,2,4])
#print(res)