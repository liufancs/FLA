'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)
@author: hexiangnan
'''
import math
import heapq # for retrieval topK
import numpy as np
import multiprocessing

from time import time
#from numba import jit, autojit

def eval(model, sess, testDict, trainList, n_user, n_item):

    global _model
    global _testDict
    global _trainList
    global _DictList
    global _K
    global _sess
    global _n_user
    global _n_item
    global _items

    _model = model
    _testDict = testDict
    _trainList = trainList
    _sess = sess
    _n_user = n_user
    _n_item = n_item
    _K = 10
    num_thread = 1
    hits, ndcgs, losses = [],[],[]
    _items = [i for i in range(_n_item)]
    if num_thread > 1:
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(_eval_one_rating, range(_n_user))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        losses = [r[2] for r in res]
    else:
        for idx in range(_n_user):
            (hr,ndcg, loss) = _eval_one_rating(idx)
            hits.append(hr)
            ndcgs.append(ndcg)
            losses.append(loss)

    return (hits, ndcgs, losses)

def load_test_as_list():
    DictList = []
    for idx in range(_n_user):

        user = _trainList[idx]
        items = [i for i in range(_n_item)]
        num_idx_ = len(user)
        # Get prediction scores
        num_idx = np.full(len(items),num_idx_, dtype=np.int32 )[:,None]
        print(num_idx.shape)
        user_input = []
        for i in range(len(items)):
            user_input.append(user)
        user_input = np.array(user_input)
        item_input = np.array(items)[:,None]
        feed_dict = {_model.user_input: user_input, _model.num_idx: num_idx, _model.item_input: item_input}
        DictList.append(feed_dict)
    print("already load the evaluate model...")
    return DictList

def _eval_one_rating(idx):

    user = _trainList[idx]
    num_idx_ = _n_user
    num_idx = np.full(_n_item, num_idx_, dtype=np.int32)[:, None]
    user_input = [user]*_n_item
    user_input = np.array(user_input)
    item_input = np.array(_items)[:, None]

    feed_dict = {_model.user_input: user_input, _model.num_idx: num_idx, _model.item_input: item_input}
    labels = np.zeros(_n_item)[:, None]
    labels[_testDict[idx]] = 1
    feed_dict[_model.labels] = labels

    predictions,loss = _sess.run([_model.output,_model.loss], feed_dict = feed_dict)
    predictions = predictions[:,0]
    predictions[_trainList[idx]] = -(1<<10)
    rk = np.argpartition(-predictions,_K)[:_K]
    r = np.zeros(_K, dtype=int)
    for i in range(_K):
        if rk[i] in _testDict[idx]:
            r[i] = 1
    hr = hit_at_k(r)
    ndcg = ndcg_at_k(r, _K, len(_testDict[idx]))

    return (hr, ndcg, loss)


def hit_at_k(r):
    if np.sum(r) > 0:
        return 1
    else:
        return 0

def ndcg_at_k(r, k, N):
    idcg, dcg = 0, 0
    for i in range(min(N, k)):
        idcg += 1 / (np.log(i + 2) / np.log(2))
    for i in range(k):
        if r[i] != 0:
            dcg += 1 / (np.log(i*r[i] + 2) / np.log(2))
    result = dcg / idcg
    return result

