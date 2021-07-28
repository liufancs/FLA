'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np
import pandas as pd
from time import time

ITEM_CLIP = 300 

class Dataset(object):
    '''
    Loading the data file
        trainMatrix: load rating records as sparse matrix for class Data
        trianList: load rating records as list to speed up user's feature retrieval
        testRatings: load leave-one-out rating test for class Evaluate
        testNegatives: sample the items not rated by user
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix, self.trainList = self.load_training_file(path + "/train.csv")
        self.testDict = self.load_test_file(path + "/test.csv")
        self.num_users, self.num_items = self.trainMatrix.shape

    def load_training_file(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        train = pd.read_csv(filename, ',', names=['u', 'i', 'r'], engine='python')
        num_users = train['u'].max()
        num_items = train['i'].max()
        lists = []
        for u in range(num_users+1):
            items = train[train['u']==u].values[:,1].tolist()[:ITEM_CLIP]
            lists.append(items)
        train = [[int(x[0]), int(x[1]), 1.0] for x in train.values]
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        for i in range(len(train)):
            user, item, _ = train[i]
            mat[user, item] = 1.0
        print("already load the trainMatrix...")
        return mat, lists

    def load_test_file(self, filename):
        test = pd.read_csv(filename, ',', names=['u', 'i', 'r'], engine='python')
        test = [[int(x[0]), int(x[1])] for x in test.values]
        testDict = {}
        for [u, i] in test:
            if testDict.get(u):
                testDict[u].append(i)
            else:
                testDict[u] = [i]
        return testDict
