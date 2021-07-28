from __future__ import absolute_import
from __future__ import division

import os

import cProfile
import tensorflow as tf
import numpy as np
import logging

from time import time
from time import strftime
from time import localtime

from Dataset import Dataset
import Batch_gen as data
import Evaluate as evaluate

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run FISM.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='Music',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--batch_choice', nargs='?', default='user',
                        help='user: generate batches by user, fixed:batch_size: generate batches by batch size')    
    parser.add_argument('--embed_size', type=int, default=8,
                        help='Embedding size.')
    parser.add_argument('--gpu', nargs='?', default='0',
                        help='gpu.')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--alpha', type=float, default=0,
                        help='Index of coefficient of embedding vector')
    parser.add_argument('--train_loss', type=float, default=0,
                        help='Caculate training loss or not')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with updating FISM variables, 2:Pretrain with fixed FISM variables.')
    parser.add_argument('--save', type=int, default=1,
                        help='save the parameters of FISM or not')
    return parser.parse_args()

class FISM:

    def __init__(self, num_items, args):
        self.num_items = num_items
        self.dataset_name = args.dataset
        self.learning_rate = args.lr
        self.embedding_size = args.embed_size
        self.alpha = args.alpha
        self.verbose = args.verbose
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.batch_choice = args.batch_choice
        self.train_loss = args.train_loss

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, None])	#the index of users
            self.num_idx = tf.placeholder(tf.float32, shape=[None, 1])	#the number of items rated by users
            self.item_input = tf.placeholder(tf.int32, shape=[None, 1])	  #the index of items
            self.labels = tf.placeholder(tf.float32, shape=[None,1])	#the ground truth

    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            self.c1 = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01), #why [0, 3707)?
                                                 name='c1', dtype=tf.float32)
            self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2' )
            self.embedding_Q_ = tf.concat([self.c1,self.c2], 0, name='embedding_Q_')
            self.embedding_Q = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='embedding_Q', dtype=tf.float32)
            self.bias = tf.Variable(tf.zeros(self.num_items),name='bias')

    def _create_inference(self):
        with tf.name_scope("inference"):
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q_, self.user_input), 1)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, self.item_input), 1)
            self.bias_i = tf.nn.embedding_lookup(self.bias, self.item_input)
            self.coeff = tf.pow(self.num_idx, -tf.constant(self.alpha, tf.float32, [1]))
            self.output = tf.sigmoid(self.coeff * tf.expand_dims(tf.reduce_sum(self.embedding_p*self.embedding_q, 1),1) + self.bias_i)

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.loss = tf.losses.log_loss(self.labels, self.output) + \
                        self.lambda_bilinear*tf.reduce_sum(tf.square(self.embedding_Q)) + self.gamma_bilinear*tf.reduce_sum(tf.square(self.embedding_Q_))

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

def training(flag, model, dataset,  epochs, num_negatives, save):
    
    saver = tf.train.Saver({'c1':model.c1,'embedding_Q':model.embedding_Q, 'bias':model.bias})
    weight_path = 'Pretraining/%s/%s/%s/%s/f/' % (model.dataset_name,model.batch_choice,model.learning_rate,model.embedding_size)
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # pretrain nor not
        if flag != 0:
            ckpt = tf.train.get_checkpoint_state(weight_path)
            if ckpt and ckpt.model_checkpoint_path:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, ckpt.model_checkpoint_path)
                logging.info("using pretrained variables")
                print("using pretrained variables")
        else:
            sess.run(tf.global_variables_initializer())
            logging.info("initialized")
            print("initialized")
        
        #initialize for training batches
        batch_begin = time()
        batches = data.shuffle(dataset, model.batch_choice, num_negatives)

        batch_time = time() - batch_begin
        num_batch = len(batches[1])
        batch_index = range(num_batch)  

        besthr = 0
        ep_begin = time()
        ndcg_loger, hit_loger = [], []
        cur_best_pre_0 = 0.
        stopping_step = 0

        #train by epoch
        for epoch_count in range(epochs):
            train_begin = time()
            training_batch(batch_index, model, sess, batches)
            train_time = time() - train_begin

            if epoch_count % model.verbose == 0:
                
                if model.train_loss:
                    loss_begin = time()
                    train_loss = training_loss(model, sess, batches)
                    loss_time = time() - loss_begin
                else:
                    loss_time, train_loss = 0, 0
                eval_begin = time()
                (hits, ndcgs, losses) = evaluate.eval(model, sess, dataset.testDict, dataset.trainList,dataset.num_users, dataset.num_items)
                hr, ndcg, test_loss = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(losses).mean()
                eval_time = time() - eval_begin
                ep_time = time() - ep_begin
                ndcg_loger.append(ndcg)
                hit_loger.append(hr)
                logging.info(
                    "Epoch %d [%.1fs][%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1fs] train_loss = %.4f [%.1fs]" % (
                    epoch_count, ep_time, batch_time, train_time, hr, ndcg, test_loss, eval_time, train_loss, loss_time))
                print("Epoch %d [%.1fs][%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1fs] train_loss = %.4f [%.1fs]" % (
                    epoch_count, ep_time, batch_time, train_time, hr, ndcg, test_loss, eval_time, train_loss, loss_time))

                cur_best_pre_0, stopping_step, should_stop = early_stopping(
                    hr, cur_best_pre_0,
                    stopping_step, expected_order='acc', flag_step=10)
                if should_stop == True:
                    break

            if hr >= besthr:
                besthr = hr
                if save:
                    saver.save(sess, weight_path+'modelf.ckpt', global_step=epoch_count)
                    logging.info('save model...')
                    print('save model...')

        ndcgs = np.array(ndcg_loger)
        hit = np.array(hit_loger)

        best_rec_0 = max(hit)
        idx = list(hit).index(best_rec_0)
        final_perf = "Best Iter = hr:%.5f, ndcg:%.5f" % (
        hit[idx], ndcgs[idx])
        print(final_perf)
            

def training_batch(batch_index, model, sess, batches):
    for index in batch_index:
        user_input, num_idx, item_input, labels = data.batch_gen(batches, index)
        feed_dict = {model.user_input: user_input, model.num_idx: num_idx[:, None], model.item_input: item_input[:, None],
                    model.labels: labels[:, None]}
        sess.run(model.optimizer, feed_dict)

def training_loss(model, sess, batches):
    train_loss = 0.0
    num_batch = len(batches[1])
    for index in range(num_batch):
        user_input, num_idx, item_input, labels = data.batch_gen(batches, index)
        feed_dict = {model.user_input: user_input, model.num_idx: num_idx[:, None], model.item_input: item_input[:, None],model.labels: labels[:, None]}
        train_loss += sess.run(model.loss, feed_dict)
    return train_loss / num_batch

if __name__=='__main__':

    args = parse_args()
    regs = eval(args.regs)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    log_dir = "Log/%s/" % args.dataset
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, "%s_%s_f_%s" %(args.dataset, args.embed_size, strftime('%Y-%m-%d-%H:%M:%S', localtime()))), level = logging.INFO)
    print(args)

    logging.info(args)

    dataset = Dataset(args.path + args.dataset)
    model = FISM(dataset.num_items,args)
    model.build_graph()
    training(args.pretrain, model, dataset, args.epochs, args.num_neg, args.save)
