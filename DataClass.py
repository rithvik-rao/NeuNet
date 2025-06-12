import numpy as np

class SizeException(Exception):

    def __init__(self, message, trainSize):
        super().__init__(message)
        self.message = message
        self.trainSize = trainSize

    def __str__(self):

        if self.trainSize > 1 or self.trainSize < 0:

            return f"{self.message} :\n----Keep the trainDataSize between 0 and 1 (included)"

        return f"{self.message} :\n----your Training Data Size ({100*self.trainSize}%) is less than Test Data Size ({100 - 100*self.trainSize}%)"

class Data():

    '''

    --Things TO DO--
    
        - Single matrix form of epoch data has to be summed along a specific axis (1)
    
        - the data Array must contain the expected output within it to stay with the corresponding data after shuffling
    
    --S U MM A R Y--
    
        - Handle the data and labels carefully
    '''

    def __init__(self, dataArray, labelEndsAt : int, trainDataSize = 0.8, shuffle = True):

        if len(dataArray) == 0:

            raise ValueError('length of dataArray must be greater than zero')

        if trainDataSize < 0 or trainDataSize > 1:

            raise SizeException('Value out of bounds', trainDataSize)
        
        self.dataArray = np.array(dataArray, copy=True)
        
        if shuffle:
            np.random.shuffle(self.dataArray)

        self.labelEndsAt = labelEndsAt

        if trainDataSize < 0.5:

            print('CAUTION!!! --- train data is less than test data')

            check = input('Proceed? (Y/N) : ')

            if check.capitalize() == 'Y':
                pass
            else:
                raise SizeException('Size Error, User Aborted',trainDataSize)
            
        trainsize = int(trainDataSize*len(dataArray))
        
        self.trainDataArray = self.dataArray[0:trainsize]
        self.testDataArray = self.dataArray[trainsize:]

        del trainsize

        self.trainBatches = None
        self.testBatches = None
        self.currentTrainBatchId = 0
        self.currentTestBatchId = 0

    def CreateBatches(self, batchSize = 1):

        self.trainBatches = np.array([self.trainDataArray[i * batchSize : (i + 1) * batchSize] for i in range(int(len(self.trainDataArray)/batchSize))] , dtype=object)
        self.testBatches = np.array([self.testDataArray[i * batchSize : (i + 1) * batchSize] for i in range(int(len(self.testDataArray)/batchSize))] , dtype=object)

    def GiveNextBatch(self, Test = False):
        if Test:
            if self.currentTestBatchId >= len(self.testBatches):

                raise StopIteration("End of test batches")

            batch = self.testBatches[self.currentTestBatchId]
            self.currentTestBatchId += 1
        else:
            if self.currentTrainBatchId >= len(self.trainBatches):
                
                raise StopIteration("End of training batches")

            batch = self.trainBatches[self.currentTrainBatchId]
            self.currentTrainBatchId += 1

        features = batch[:, :self.labelEndsAt + 1]
        labels = batch[:, self.labelEndsAt + 1:]
        
        return features, labels
    
    def ResetOrder(self):

        self.currentTrainBatchId = 0
        self.currentTestBatchId = 0

if __name__ == "__main__":

    '''the follwing dataArray has no meaning'''
    dataArray = np.random.uniform(size = (10,5))
    dataArray[:,0] = [1,6,7,5,8,4,5,5,9,0]

    data = Data(dataArray=dataArray,labelEndsAt=0,trainDataSize=0.6,shuffle=True) #trainDataSize must be between 0 and 1

    print('Train Data Generated \n',data.trainDataArray, end ='\n\n')
    print('Test Data Generated \n',data.testDataArray, end ='\n\n')

    data.CreateBatches(batchSize=2)

    print('Train Batches \n', data.trainBatches , end = '\n\n')
    print('Test Batches \n', data.testBatches , end = '\n\n')

    print('first batch :')
    print('labels :\n' ,data.GiveNextBatch()[0] , end = '\n\n')
    print('data :\n' ,data.GiveNextBatch()[1] , end = '\n\n')

    # data.ResetOrder() #can be used to reset the GiveNextBatch function
