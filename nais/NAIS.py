from __future__ import absolute_import
from __future__ import division

import argparse
import Evaluate as evaluate
import Batch_gen as data
from Dataset import Dataset
from time import localtime
from time import strftime
from time import time
import logging
import numpy as np
import tensorflow as tf
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Run NAIS.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='delicious',
                        help='Choose a dataset.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with updating FISM variables, 2:Pretrain with fixed FISM variables.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--batch_choice', nargs='?', default='user',
                        help='user: generate batches by user, fixed:batch_size: generate batches by batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--weight_size', type=int, default=16,
                        help='weight size.')
    parser.add_argument('--embed_size', type=int, default=16,
                        help='Embedding size.')
    parser.add_argument('--gpu', nargs='?', default='0',
                        help='gpu.')

    parser.add_argument('--data_alpha', type=float, default=0,
                        help='Index of coefficient of embedding vector')
    parser.add_argument('--regs', nargs='?', default='[0,0,0]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--alpha', type=float, default=0,
                        help='Index of coefficient of embedding vector')
    parser.add_argument('--train_loss', type=float, default=0,
                        help='Caculate training loss or nor')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='Index of coefficient of sum of exp(A)')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--prelr', type=float, default=0.05,
                        help='pretrain Learning rate.')
    parser.add_argument('--activation', type=int, default=0,
                        help='Activation for ReLU, sigmoid, tanh.')
    parser.add_argument('--algorithm', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')
    return parser.parse_args()


class NAIS:

    def __init__(self, num_items, args):
        self.pretrain = args.pretrain
        self.num_items = num_items
        self.dataset_name = args.dataset
        self.learning_rate = args.lr
        self.embedding_size = args.embed_size
        self.weight_size = args.weight_size
        self.alpha = args.alpha
        self.beta = args.beta
        self.data_alpha = args.data_alpha
        self.verbose = args.verbose
        self.activation = args.activation
        self.algorithm = args.algorithm
        self.batch_choice = args.batch_choice
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.eta_bilinear = regs[2]
        self.train_loss = args.train_loss

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(
                tf.int32, shape=[None, None])  # the index of users
            # the number of items rated by users
            self.num_idx = tf.placeholder(tf.float32, shape=[None, 1])
            self.item_input = tf.placeholder(
                tf.int32, shape=[None, 1])  # the index of items
            self.labels = tf.placeholder(
                tf.float32, shape=[None, 1])  # the ground truth

    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            trainable_flag = (self.pretrain != 2)
            self.c1 = tf.Variable(tf.truncated_normal(shape=[
                                  self.num_items, self.embedding_size], mean=0.0, stddev=0.01), name='c1', dtype=tf.float32, trainable=trainable_flag)
            self.c2 = tf.constant(
                0.0, tf.float32, [1, self.embedding_size], name='c2')
            self.embedding_Q_ = tf.concat(
                [self.c1, self.c2], 0, name='embedding_Q_')
            self.embedding_Q = tf.Variable(tf.truncated_normal(
                shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01), name='embedding_Q', dtype=tf.float32, trainable=trainable_flag)
            self.bias = tf.Variable(
                tf.zeros(self.num_items), name='bias', trainable=trainable_flag)

            # Variables for attention
            if self.algorithm == 0:
                self.W = tf.Variable(tf.truncated_normal(shape=[self.embedding_size, self.weight_size], mean=0.0, stddev=tf.sqrt(
                    tf.div(2.0, self.weight_size + self.embedding_size))), name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            else:
                self.W = tf.Variable(tf.truncated_normal(shape=[2*self.embedding_size, self.weight_size], mean=0.0, stddev=tf.sqrt(
                    tf.div(2.0, self.weight_size + (2*self.embedding_size)))), name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            self.b = tf.Variable(tf.truncated_normal(shape=[1, self.weight_size], mean=0.0, stddev=tf.sqrt(tf.div(
                2.0, self.weight_size + self.embedding_size))), name='Bias_for_MLP', dtype=tf.float32, trainable=True)
            self.h = tf.Variable(
                tf.ones([self.weight_size, 1]), name='H_for_MLP', dtype=tf.float32)

    def _attention_MLP(self, q_):
        with tf.name_scope("attention_MLP"):
            b = tf.shape(q_)[0]
            n = tf.shape(q_)[1]
            r = (self.algorithm + 1)*self.embedding_size

            # (b*n, e or 2*e) * (e or 2*e, w) + (1, w)
            MLP_output = tf.matmul(tf.reshape(q_, [-1, r]), self.W) + self.b
            if self.activation == 0:
                MLP_output = tf.nn.relu(MLP_output)
            elif self.activation == 1:
                MLP_output = tf.nn.sigmoid(MLP_output)
            elif self.activation == 2:
                MLP_output = tf.nn.tanh(MLP_output)

            # (b*n, w) * (w, 1) => (None, 1) => (b, n)
            A_ = tf.reshape(tf.matmul(MLP_output, self.h), [b, n])

            # softmax for not mask features
            exp_A_ = tf.exp(A_)
            num_idx = tf.reduce_sum(self.num_idx, 1)
            mask_mat = tf.sequence_mask(
                num_idx, maxlen=n, dtype=tf.float32)  # (b, n)
            exp_A_ = mask_mat * exp_A_
            exp_sum = tf.reduce_sum(exp_A_, 1, keep_dims=True)  # (b, 1)
            exp_sum = tf.pow(exp_sum, tf.constant(self.beta, tf.float32, [1]))

            A = tf.expand_dims(tf.div(exp_A_, exp_sum), 2)  # (b, n, 1)

            return tf.reduce_sum(A * self.embedding_q_, 1)

    def _create_inference(self):
        with tf.name_scope("inference"):
            self.embedding_q_ = tf.nn.embedding_lookup(
                self.embedding_Q_, self.user_input)  # (b, n, e)
            self.embedding_q = tf.nn.embedding_lookup(
                self.embedding_Q, self.item_input)  # (b, 1, e)

            if self.algorithm == 0:
                self.embedding_p = self._attention_MLP(
                    self.embedding_q_ * self.embedding_q)
            else:
                n = tf.shape(self.user_input)[1]
                self.embedding_p = self._attention_MLP(tf.concat(
                    [self.embedding_q_, tf.tile(self.embedding_q, tf.stack([1, n, 1]))], 2))

            self.embedding_q = tf.reduce_sum(self.embedding_q, 1)
            self.bias_i = tf.nn.embedding_lookup(self.bias, self.item_input)
            self.coeff = tf.pow(
                self.num_idx, -tf.constant(self.alpha, tf.float32, [1]))
            self.output = tf.sigmoid(self.coeff * tf.expand_dims(
                tf.reduce_sum(self.embedding_p*self.embedding_q, 1), 1) + self.bias_i)

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.loss = tf.losses.log_loss(self.labels, self.output) + \
                self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_Q)) + \
                self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q_)) + \
                self.eta_bilinear * tf.reduce_sum(tf.square(self.W))

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdagradOptimizer(
                learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)

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



def training(flag, model, dataset,  epochs, num_negatives):

    saver = tf.train.Saver(
        {'c1': model.c1, 'embedding_Q': model.embedding_Q, 'bias': model.bias})
    args = parse_args()
    weight_path = 'Pretraining/%s/%s/%s/%s/f/' % (
    model.dataset_name, model.batch_choice, args.prelr, model.embedding_size)
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

        # initialize for training batches
        batch_begin = time()
        batches = data.shuffle(dataset, model.batch_choice, num_negatives)
        batch_time = time() - batch_begin

        num_batch = len(batches[1])
        batch_index = list(range(num_batch))

        ep_begin = time()
        ndcg_loger, hit_loger = [], []
        cur_best_pre_0 = 0.
        stopping_step = 0

        # train by epoch
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
                hr, ndcg, test_loss = np.array(hits).mean(), np.array(
                    ndcgs).mean(), np.array(losses).mean()
                eval_time = time() - eval_begin

                ep_time = time() - ep_begin
                ndcg_loger.append(ndcg)
                hit_loger.append(hr)
                logging.info(
                    "Epoch %d [%.1fs][%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1fs] train_loss = %.4f [%.1fs]" % (
                        epoch_count, ep_time, batch_time, train_time, hr, ndcg, test_loss, eval_time, train_loss,
                        loss_time))
                print(
                    "Epoch %d [%.1fs][%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1fs] train_loss = %.4f [%.1fs]" % (
                        epoch_count, ep_time, batch_time, train_time, hr, ndcg, test_loss, eval_time, train_loss,
                        loss_time))

                cur_best_pre_0, stopping_step, should_stop = early_stopping(
                    hr, cur_best_pre_0,
                    stopping_step, expected_order='acc', flag_step=10)
                if should_stop == True:
                    break

        ndcgs = np.array(ndcg_loger)
        hit = np.array(hit_loger)

        best_rec_0 = max(hit)
        idx = list(hit).index(best_rec_0)
        final_perf = "Best Iter = hr:%.5f, ndcg:%.5f" % (
            hit[idx], ndcgs[idx])
        print(final_perf)

def training_batch(batch_index, model, sess, batches):
    for index in batch_index:
        user_input, num_idx, item_input, labels = data.batch_gen(
            batches, index)
        feed_dict = {model.user_input: user_input, model.num_idx: num_idx[:, None], model.item_input: item_input[:, None],
                     model.labels: labels[:, None]}
        sess.run([model.loss, model.optimizer], feed_dict)


def training_loss(model, sess, batches):
    train_loss = 0.0
    num_batch = len(batches[1])
    for index in range(num_batch):
        user_input, num_idx, item_input, labels = data.batch_gen(
            batches, index)
        feed_dict = {model.user_input: user_input,
                     model.num_idx: num_idx[:, None], model.item_input: item_input[:, None], model.labels: labels[:, None]}
        train_loss += sess.run(model.loss, feed_dict)
    return train_loss / num_batch

if __name__ == '__main__':

    args = parse_args()
    regs = eval(args.regs)

    pret = "w" if args.pretrain else "o"

    log_dir = "Log/%s/" % args.dataset
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, "%s_%s_n:%s_%s" % (
        args.dataset, args.embed_size, pret, strftime('%Y-%m-%d-%H:%M:%S', localtime()))), level=logging.INFO)
    print(args)

    logging.info(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    dataset = Dataset(args.path + args.dataset)
    model = NAIS(dataset.num_items, args)
    model.build_graph()
    training(args.pretrain, model, dataset, args.epochs, args.num_neg)
