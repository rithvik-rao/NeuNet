import numpy as np
import json

#Cost Functions

def MSE(vec : np.ndarray, goal : np.ndarray, deriv = False):

    p = goal - vec

    if deriv:
        return -2*p/len(p)
    
    else:
        return np.sum(p**2) / len(p)
    
def CrossEntroyLoss(vec : np.ndarray, goal : np.ndarray, deriv = False, pfunc = None):

    epsilon = 1e-15
    output = np.clip(vec, epsilon, 1 - epsilon)

    if deriv:

        if pfunc.__name__ == 'Softmax':

            return output - goal
        
        else:

            quit('Derivative not defined yet .!..!...!....')

    else:
        return -np.sum(goal * np.log(output))


#defining activation functions

def ReLU(vec, deriv = False):
    
    # p = np.sign(np.sign(vec+1))

    if deriv:
        return (vec > 0).astype(float)
    else:
        return np.maximum(0,vec)

def Sigmoid(vec, deriv = False):

    p = 1/(1+np.exp(-vec))

    if deriv:
        return p*(1-p)
    else:
        return p
    
def tanh(vec, deriv = False):

    if deriv:
        return  np.cosh(vec)**(-2)
    
    else:
        return np.tanh(vec)
    
def linear(vec, deriv = False):

    if deriv:
        return np.ones_like(vec)
    else:
        return vec

def Softmax(vec):

    vec_shifted = vec - np.max(vec)
    exp_vec = np.exp(vec_shifted)
    return exp_vec / np.sum(exp_vec)

#Classes

class Data():

    '''

    --Things TO DO--
    
        - Single matrix form of epoch data has to be summed along a specific axis (1)
    
        - the data Array must contain the expected output within it to stay with the corresponding data after shuffling
    
    --S U MM A R Y--
    
        - Handle the data and labels carefully
    '''

    def __init__(self, dataArray, trainDataSize = None, shuffle = True):
        
        self.dataArray = np.random.shuffle(dataArray)

        if trainDataSize < len(dataArray)/2:

            print('CAUTION!!! --- train data is less than test data')

            check = input('Proceed (Y/N) : ')

            if check.capitalize() == 'Y':
                pass
            else:
                quit('User Exit')

        self.trainDataArray = self.dataArray[0:trainDataSize]
        self.testDataArray = self.dataArray[trainDataSize:]

        self.trainBatches = None
        self.testBatches = None

    def CreateBatches(self, batchSize = 1, singleMatrix = True):

        if singleMatrix:
            self.trainBatches = np.array([self.trainDataArray[i:batchSize*i].T for i in len(self.trainDataArray)/batchSize])
            self.testBatches = np.array([self.testDataArray[i:batchSize*i].T for i in len(self.testDataArray)/batchSize])
        else:
            self.trainBatches = np.array([self.trainDataArray[i:batchSize*i] for i in len(self.trainDataArray)/batchSize])
            self.testBatches = np.array([self.testDataArray[i:batchSize*i] for i in len(self.testDataArray)/batchSize])


class InputLayer():

    def __init__(self, size : int):

        self.size = size
        self.output = None
        self.aGradient = None

    def GiveInput(self, inputs : np.ndarray):

        self.output = inputs  

class Layer():

    def __init__(self, size : int, biases : np.ndarray, weights : np.ndarray, prevLayer : object, activationFunction = Sigmoid): #All inputs must be in their datatype
        self.size = size
        self.biases = biases
        self.weights = weights
        self.activationFunction = activationFunction
        self.prevLayer = prevLayer
        self.aGradient = np.zeros((size,1))
        self.input = None
        self.evals = None
        self.output = None
        self.biasGradient = np.zeros_like(self.biases)
        self.weightGradient = np.zeros_like(self.weights)
        self.epochcount = 0

    def Evaluate(self):

        self.input = self.prevLayer.output
        self.evals = self.weights @ self.input + self.biases
        self.output = self.activationFunction(self.evals)

    def BackPropLayer(self):

        #back propagation must run only after evaluation

        self.epochcount += 1

        if self.activationFunction.__name__ == 'Softmax':

            self.biasGradient += self.aGradient

        else:

            self.biasGradient += self.activationFunction(self.evals,deriv = True) * self.aGradient
                    
        #updating prev layer parameters
        self.prevLayer.aGradient = self.weights.T @ self.biasGradient
        
        self.weightGradient += (self.biasGradient @ self.input.T)

    def UpdateParameters(self, mutationFactor : float):

        self.weights -= mutationFactor * self.weightGradient/self.epochcount
        self.biases -= mutationFactor * self.biasGradient/self.epochcount

        self.epochcount = 0

class Network():

    def __init__(self, inputSize, hiddenLayerSizes, activationFunctions, filename, randomize = False ): #parameters should be a json file name of the appropriate format
        
        self.inputLayer = InputLayer(inputSize)
        self.fileName = filename
        self.layers = []
        prevLayer = self.inputLayer
        self.activationFuncs = activationFunctions
        self.hiddenLayerSizes = hiddenLayerSizes

        for i, size in enumerate(hiddenLayerSizes):
            if randomize:
                biases = np.zeros(shape=(size, 1))
                limit = np.sqrt(6.0 / (prevLayer.size + size))
                weights = np.random.uniform(-limit, limit, (size, prevLayer.size))
                # weights = np.random.uniform(-0.5,0.5,(size, prevLayer.size))

                self.File = open(self.fileName, 'w+')

            else:

                self.File = open(self.fileName, 'r+')

                loadedData = json.load(self.File)
                biases = np.array(loadedData[str(i)]["biases"])
                weights = np.array(loadedData[str(i)]["weights"])

            activationFunction = self.activationFuncs[i]
            layer = Layer(size, biases, weights, prevLayer, activationFunction)
            self.layers.append(layer)
            prevLayer = layer

    @classmethod
    def FromJSON(classname, filename):

        with open(filename, 'r') as f:

            Hyperparameters = json.load(f)
            activationFuncs = [globals()[func] for func in Hyperparameters['Hyperparameters']['activationFunctions']]

        return classname(Hyperparameters['Hyperparameters']['inputLayerSize'], Hyperparameters['Hyperparameters']['hiddenLayerSizes'], activationFuncs, filename)

    def ForProp(self, inputs):
        try:
            self.inputLayer.GiveInput(inputs)
            for layer in self.layers:

                layer.Evaluate()

            return self.layers[-1].output
        except ValueError:

            quit('---------------------------------\nDimension mismatch !!! \nPlease check the input array size\n---------------------------------')
    
    def FeedEpochs(self):
        
        pass
    
    def BackProp(self, output, goal, costFunction = MSE):

        if costFunction == CrossEntroyLoss:

            self.layers[-1].aGradient = costFunction(output , goal, deriv = True, pfunc = self.activationFuncs[-1])

        else:

            self.layers[-1].aGradient = costFunction(output , goal, deriv = True)
        
        for layer in reversed(self.layers):

            layer.BackPropLayer()

        return costFunction(output , goal)

    def UpdateAllParameters(self, mutationFactor : float):

        for layer in self.layers:

            layer.UpdateParameters(mutationFactor)

    def SaveToJSON(self):

        param_dict = {}
        for i, layer in enumerate(self.layers):
            param_dict[str(i)] = {
                "biases": layer.biases.tolist(),
                "weights": layer.weights.tolist()
            }

        activationFunctions = [activationFunction.__name__ for activationFunction in self.activationFuncs]

        param_dict['Hyperparameters'] = {
            "inputLayerSize": self.inputLayer.size,
            "hiddenLayerSizes": self.hiddenLayerSizes,
            "activationFunctions": activationFunctions
            }
        
        del activationFunctions

        json.dump(param_dict, self.File)

if __name__ == '__main__': # Just a DUMMY code to check if the above functions are working (Rough code)
    '''
    the below code takes a 2x2 normalised monochrome image vector as input and predicts if there is a slant line or a straight line
    if the input image looks like 
    |a|b|
    |c|d| where a,b,c,d are values between 0 and 1

    the output is [1,0] if the max two elements of the image are a,d or b,c i.e. slant

    else it outputs [0,1] i.e. straight

    i feel that this method is more interesting to check the functions compared to the basic XOR problem
    '''
    # def f(vec):

    #     vec2 = np.sort(vec)

    #     p = [np.where(vec == vec2[-1])[0][0],np.where(vec == vec2[-2])[0][0]]

    #     if set(p) == {0,2} or set(p) == {1,4}:

    #         return np.array([[1],[0]])
        
    #     else:

    #         return np.array([[0],[1]])

    def f(vec):

        return vec**2
    
    # def u(vec , deriv = False):

    #     if deriv:
    #         return vec/vec
    #     else:
    #         return vec

    # np.random.seed(0)

    N = 100000
    inputSize = 1
    outputSize = 1

    trainingdata = np.random.randn(N, inputSize)

    expectedoutputs = [f(v) for v in trainingdata]

    filename = 'testData.json'

    Neurals = Network(inputSize, [10,10,outputSize],[tanh, tanh, linear], filename)

    for i,t in enumerate(trainingdata):

        # Neurals.layers[1].

        input_vec = t.reshape(-1, 1)
        output = Neurals.ForProp(input_vec)
        Neurals.BackProp(output, expectedoutputs[i])
        Neurals.UpdateAllParameters(0.01)

    Neurals.SaveToJSON()

    z = 0

    for i,t in enumerate(trainingdata):

        input_vec = t.reshape(-1, 1)
        output = Neurals.ForProp(input_vec)

        if abs(output[0][0] - expectedoutputs[i][0]) < 0.01:
            z += 1

    print(100*z/N , '%', 'accuracy')