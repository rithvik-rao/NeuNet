"""

NeuNet Module for Data Class
============

Features :

- divide train and test data
- seperate label and feature vectors
- create batches
- Shuffle Order
- get next datapoint from the dataset via next()
- Generates Epochs

Todo : 

- Auto generate epoch on exhaution of previous one untill epoch count is satisfied (Done)
- The output feactures vectors is not compatible with NNF in shape and dtype (Done)

"""

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

    def __init__(self, dataArray, labelEndsAt : int, trainDataSize = 0.8, shuffle = True):

        """
        
        Data Class compatible with the Neural Network Functions

        Parameters
        ----------

        dataArray : Array or list of data/samples along with labels

            structure: 
            - the dataArray must be a nested list or an array
            - it should be a concatenation of label vector and data vector (input vector)
            - a template of element of dataArray is given below
                
                `[label_1 label_2 ... label_m data_1 data_2 data_3 ... data_n]`

                label vector = `[label_1 label_2 ... label_m]`

                input vector = `[data_1 data_2 data_3 ... data_n]`

            where `k` is the number of data points, `n` is the dimension of data space
            `m` is the dimension of the label space

        labelEndsAt : The index of last label column i.e. `m-1`

            type : Integer

        trainDataSize : The fraction of the entire data which will be used to create Train data

            Value : between 0 and 1 (Default = 0.8)

        shuffle : Shuffles the data (if set to True) before dividing into train and test arrays

            (Default = false)
        ----------
        """

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
        self.currentTrainBatchId = 0
        """
        
        Current Train Batch Index in Current Epoch

            Set value to zero to **restart** the current epoch

        """
        self.currentTestBatchId = 0

        """
        
        Current Test Batch Index

            Set value to zero to **restart** the Test Set

        """

        self.test = False
        self.batchsize : int
        self.epochcount = 0

    def __str__(self):
        
        return 'Data Array :'+'\n'+str(self.dataArray) +'\n'+'Label ends at :'+'\n'+ str(self.labelEndsAt)

    def NextEpoch(self):

        """
        
        Generate a new Epoch of the Dataset

        NOTE : works only if `self.test == False`

        """

        if not self.test:

            self.epochcount += 1

            np.random.shuffle(self.trainDataArray)
            self.currentTrainBatchId = 0

            self.CreateBatches(self.batchsize)

        else:

            raise Exception('mode is set to test')

    def CreateBatches(self, batchSize :int = 1):

        """
        
        Creates Batches which are stored in `self.trainBatches` and `self.testbatches`

        Parameters
        ----------

        batchSize : Determines the size of Batches (Default = 1)

        ----------

        Generates Batches only for train data when self.test = False

        """

        self.batchsize = batchSize

        if not self.test:
            _l = int(len(self.trainDataArray)/self.batchsize)
            self.trainBatches = np.array([self.trainDataArray[i * self.batchsize : (i + 1) * self.batchsize] for i in range(_l)])
            


    def __next__(self): #GiveNextBatch(self, Test = False)

        """

        Gives next ( `features` , `labels` ) from the current division ( train or test depending upon `self.test` )

        NOTE : raises `StopIteration` when the Epoch/Array is exhausted

        """
        
        if self.test:

            if self.currentTestBatchId >= len(self.testDataArray):

                raise StopIteration('Test Data Exhausted')

            batch = np.array([self.testDataArray[self.currentTestBatchId]])
            self.currentTestBatchId += 1
        else:
            if self.currentTrainBatchId >= len(self.trainBatches):
                
                self.NextEpoch()

            batch = self.trainBatches[self.currentTrainBatchId]
            self.currentTrainBatchId += 1

        labels = batch[:, :self.labelEndsAt + 1]
        features = batch[:, self.labelEndsAt + 1:]
        
        return features, labels
    
    def __iter__(self):

        return self
    
    def ResetOrder(self):

        """
        To restart
        """

        self.currentTrainBatchId = 0
        self.currentTestBatchId = 0

if __name__ == "__main__":

    '''the follwing dataArray has no meaning'''
    dataArray = np.random.uniform(size = (10,5))
    dataArray[:,0] = [1,6,7,5,8,4,5,5,9,0]

    data = Data(dataArray=dataArray,labelEndsAt=0,trainDataSize=0.6,shuffle=True) #trainDataSize must be between 0 and 1
    
    print(data)
    data.test = False

    print('Train Data Generated \n',data.trainDataArray, end ='\n\n')
    print('Test Data Generated \n',data.testDataArray, end ='\n\n')

    data.CreateBatches(batchSize=2)

    print('Train Batches \n', data.trainBatches , end = '\n\n')
    print('Test Batches \n', data.testDataArray , end = '\n\n')

    print('first batch :')
    print(next(data) , end = '\n\n')

    data.test = True

    print('data :\n' ,next(data) , end = '\n\n')

    data.t = False

    data.NextEpoch()

    print('Next Epoch Train Batches \n', data.trainBatches , end = '\n\n')


    # data.ResetOrder() #can be used to reset the GiveNextBatch function
