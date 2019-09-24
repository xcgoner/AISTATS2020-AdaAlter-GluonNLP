"""
Large Word Language Model
===================

This example shows how to build a word-level language model on Google Billion Words dataset
with Gluon NLP Toolkit.
By using the existing data pipeline tools and building blocks, the process is greatly simplified.

We implement the LSTM 2048-512 language model proposed in the following work.

@article{jozefowicz2016exploring,
 title={Exploring the Limits of Language Modeling},
 author={Jozefowicz, Rafal and Vinyals, Oriol and Schuster, Mike and Shazeer, Noam and Wu, Yonghui},
 journal={arXiv preprint arXiv:1602.02410},
 year={2016}
}

"""

# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import time
import math
import logging
import os
import random
import sys
import argparse
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
import gluonnlp as nlp
from gluonnlp.utils import Parallel, Parallelizable
from sampler import LogUniformSampler

from distributed_local_sgd_v2 import DistributedHierLocalHVDTrainer

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '..', '..'))

###############################################################################
# Arg parser
###############################################################################
parser = argparse.ArgumentParser(description=
                                 'Gluon-NLP Big LSTM 2048-512 Language Model on GBW')
parser.add_argument('--save', type=str, default='model.params',
                    help='path to save the final model.')
parser.add_argument('--emsize', type=int, default=512,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=2048,
                    help='number of hidden units per layer')
parser.add_argument('--nproj', type=int, default=512,
                    help='number of projection units per layer. Could be different from embsize')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--from-epoch', type=int, default=None,
                    help='start training or testing from the provided epoch')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epoch for training')
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size per gpu')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--eps', type=float, default=1,
                    help='initial history accumulation for adagrad')
parser.add_argument('--bptt', type=int, default=20,
                    help='sequence length')
parser.add_argument('--k', type=int, default=8192,
                    help='number of noise samples for estimation')
parser.add_argument('--log-interval', type=int, default=1000,
                    help='report interval')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--optimizer', type=str, default='adagrad',
                    help='optimizer')
parser.add_argument('--lr', type=float, default=0.2,
                    help='initial learning rate')
parser.add_argument('--local-sgd-interval', type=int, default=10,
                    help='interval of local SGD')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clipping by global norm.')
parser.add_argument('--test-mode', action='store_true',
                    help='Whether to run through the script with few examples')
parser.add_argument('--eval-only', action='store_true',
                    help='Whether to only run evaluation for the trained model')
parser.add_argument('--warmup-steps', type=int, default=1,
                    help='number of steps for warmup')
args = parser.parse_args()

segments = ['train', 'test']
max_nbatch_eval = None

if args.test_mode:
    args.emsize = 200
    args.log_interval = 1
    args.nhid = 200
    args.nlayers = 1
    args.epochs = 20
    max_nbatch_eval = 3
    segments = ['test', 'test']

# logging.info(args)

# logging
logging.getLogger().setLevel(logging.INFO)
logging.info(args)

np.random.seed(args.seed)
random.seed(args.seed)
mx.random.seed(args.seed)


os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
os.environ['MXNET_CPU_PARALLEL_RAND_COPY'] = str(1)
os.environ['MXNET_CPU_WORKER_NTHREADS'] = str(1)

# init hvd
try:
    import horovod.mxnet as hvd
except ImportError:
    logging.info('horovod must be installed.')
    exit()
hvd.init()
store = None
num_workers = hvd.size()
rank = hvd.rank()
local_rank = hvd.local_rank()
is_master_node = rank == local_rank

ctx = mx.gpu(local_rank)

###############################################################################
# Data stream
###############################################################################
num_files = 99
split_sampler = nlp.data.SplitSamplerV2(num_files, num_parts=num_workers, part_index=rank)
train_data_stream, test_data_stream = \
    [nlp.data.GBWStream(segment=segment, skip_empty=True, bos=None, eos='<eos>', file_sampler=split_sampler)
     for segment in segments]
vocab = train_data_stream.vocab
ntokens = len(vocab)
num_batches_per_epoch = 80000 * 4 * 128 // (num_workers * args.batch_size)

# Sampler for generating negative classes during training with importance sampling
sampler = LogUniformSampler(ntokens, args.k)

def _load(x):
    if isinstance(x, tuple):
        return [y.as_in_context(ctx) for y in x]
    else:
        return x.as_in_context(ctx)

# Transformation for a data batch for training.
# First, load the data, target and mask to target contexts.
# Second, the LSTM-2048-512 model performs importance sampling for decoding
# during training, we need to sample negative candidate classes by invoking the
# log uniform sampler.
def _load_sample(x, y):
    m = x != vocab[vocab.padding_token]  # mask padding
    xs = _load(x)
    ys = _load(y)
    ms = _load(m)
    ss = sampler(y)
    ss = _load(ss)
    return xs, ys, ms, ss

train_batch_size = args.batch_size
train_batchify = nlp.data.batchify.StreamBPTTBatchify(vocab, args.bptt, train_batch_size)
train_data = train_batchify(train_data_stream)
train_data = train_data.transform(_load_sample)

test_batch_size = args.batch_size
test_batchify = nlp.data.batchify.StreamBPTTBatchify(vocab, args.bptt, test_batch_size)
test_data = test_batchify(test_data_stream)
test_data = nlp.data.PrefetchingStream(test_data)

###############################################################################
# Build the model
###############################################################################

eval_model = nlp.model.language_model.BigRNN(ntokens, args.emsize, args.nhid,
                                             args.nlayers, args.nproj,
                                             embed_dropout=args.dropout,
                                             encode_dropout=args.dropout)
model = nlp.model.language_model.train.BigRNN(ntokens, args.emsize, args.nhid,
                                              args.nlayers, args.nproj, args.k,
                                              embed_dropout=args.dropout,
                                              encode_dropout=args.dropout, 
                                             sparse_weight=False)
loss = gluon.loss.SoftmaxCrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def train():
    """Training loop for language model.
    """
    # logging.info(model)
    from_epoch = 0
    model.initialize(mx.init.Xavier(factor_type='out'), ctx=ctx)
    trainer_params = {'learning_rate': args.lr, 'wd': 0, 'eps': args.eps}
    # trainer = gluon.Trainer(model.collect_params(), args.optimizer, trainer_params)
    # fully sync at the beginning
    trainer = DistributedHierLocalHVDTrainer(model.collect_params(), args.optimizer, trainer_params, local_sgd_interval = 0)
    trainer._optimizer._full_sync = True


    if args.from_epoch:
        from_epoch = args.from_epoch
        checkpoint_name = '%s.%s'%(args.save, format(from_epoch - 1, '02d'))
        model.load_parameters(checkpoint_name)
        trainer.load_states('%s.state'%args.save)
        logging.info('Loaded parameters from checkpoint %s'%(checkpoint_name))

    hvd.broadcast_parameters(model.collect_params(), root_rank=0)

    model.hybridize(static_alloc=True, static_shape=True)
    encoder_params = model.encoder.collect_params().values()
    embedding_params = list(model.embedding.collect_params().values())

    step_num = 0
    lr = args.lr
    current_lr = lr

    epoch = from_epoch
    start_epoch_time = time.time()
    start_log_interval_time = time.time()
    nbatch = 0

    while epoch < args.epochs:
        sys.stdout.flush()
        total_L = 0.0
        hidden = model.begin_state(batch_size=args.batch_size,
                                     func=mx.nd.zeros, ctx=ctx)
        has_next = True
        train_data_iter = iter(train_data)
        data, target, mask, sample = next(train_data_iter)

        while has_next:
            nbatch += 1

            step_num += 1
            if step_num <= args.warmup_steps:
                new_lr = lr * step_num / args.warmup_steps
                trainer.set_learning_rate(new_lr)
                current_lr = new_lr
            
            if step_num == args.warmup_steps + 1:
                # trainer._local_sgd_interval = args.local_sgd_interval
                # trainer._optimizer._full_sync = False
                trainer.init_states()

            hidden = detach(hidden)

            with autograd.record():
                output, hidden, new_target = model(data, target, hidden, sample)
                output = output.reshape((-3, -1))
                new_target = new_target.reshape((-1,))
                ls = loss(output, new_target) * mask.reshape((-1,))
                ls = ls / args.batch_size
                ls.backward()

            # prefetch the next batch of data
            try:
                data, target, mask, sample = next(train_data_iter)
            except StopIteration:
                has_next = False

            # rescale embedding grad
            x = embedding_params[0].grad(ctx)
            x[:] *= args.batch_size
            encoder_grad = [p.grad(ctx) for p in encoder_params]
            # perform gradient clipping per ctx
            gluon.utils.clip_global_norm(encoder_grad, args.clip)

            trainer.step(1)

            ls_sum = mx.nd.sum(ls)

            total_L += ls_sum / args.bptt

            # total_L += mx.nd.sum(ls).asscalar() / args.bptt

            if nbatch % args.log_interval == 0:

                hvd.allreduce_(total_L, average=True, name='ls', priority=-9999)

                cur_L = total_L.asscalar() / args.log_interval
                ppl = math.exp(cur_L) if cur_L < 100 else float('inf')
                if rank == 0:
                    logging.info('[Epoch %d Batch %d] loss %.2f, ppl %.2f, '
                        'throughput %.2f samples/s, lr %.4f'
                        %(epoch, nbatch, cur_L, ppl,
                            train_batch_size*num_workers*args.log_interval/(time.time()-start_log_interval_time), current_lr))
                total_L = 0.0
                start_log_interval_time = time.time()
                sys.stdout.flush()

            if nbatch == num_batches_per_epoch:
                end_epoch_time = time.time()
                logging.info('Epoch %d took %.2f seconds.'%(epoch, end_epoch_time - start_epoch_time))
                mx.nd.waitall()
                checkpoint_name = '%s.%s'%(args.save, format(epoch, '02d'))
                if rank == 0:
                    model.save_parameters(checkpoint_name)
                if rank == 1:
                    trainer.save_states('%s.state'%args.save)
                nbatch = 0
                start_epoch_time = time.time()
                epoch += 1
                if epoch == args.epochs:
                    break

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [detach(h) for h in hidden]
    else:
        hidden = hidden.detach()
    return hidden

def test(data_stream, batch_size, ctx=None):
    """Evaluate the model on the dataset.

    Parameters
    ----------
    data_stream : DataStream
        The dataset is tested on.
    batch_size : int
        The size of the mini-batch.
    ctx : mx.cpu() or mx.gpu()
        The context of the computation.

    Returns
    -------
    loss: float
        The loss on the dataset
    """
    total_L = 0.0
    ntotal = 0
    nbatch = 0
    hidden = eval_model.begin_state(batch_size=batch_size, func=mx.nd.zeros, ctx=ctx)
    start_time = time.time()
    for data, target in data_stream:
        data = data.as_in_context(ctx)
        target = target.as_in_context(ctx)
        mask = data != vocab[vocab.padding_token]
        output, hidden = eval_model(data, hidden)
        hidden = detach(hidden)
        output = output.reshape((-3, -1))
        L = loss(output, target.reshape(-1,)) * mask.reshape((-1,))
        total_L += L.mean()
        ntotal += mask.mean()
        nbatch += 1
        avg = total_L / ntotal
        if nbatch % args.log_interval == 0:
            avg_scalar = float(avg.asscalar())
            ppl = math.exp(avg_scalar)
            throughput = batch_size*args.log_interval/(time.time()-start_time)
            logging.info('Evaluation batch %d: test loss %.2f, test ppl %.2f, '
                  'throughput = %.2f samples/s'%(nbatch, avg_scalar, ppl, throughput))
            start_time = time.time()
        if max_nbatch_eval and nbatch > max_nbatch_eval:
            logging.info('Quit evaluation early at batch %d'%nbatch)
            break
    return float(avg.asscalar())

def evaluate():
    """ Evaluate loop for the trained model """
    logging.info(eval_model)
    eval_model.initialize(mx.init.Xavier(), ctx=ctx)
    eval_model.hybridize(static_alloc=True, static_shape=True)
    epoch = args.from_epoch if args.from_epoch else 0
    while epoch < args.epochs:
        checkpoint_name = '%s.%s'%(args.save, format(epoch, '02d'))
        if not os.path.exists(checkpoint_name):
            logging.info('Wait for a new checkpoint...')
            # check again after 600 seconds
            time.sleep(600)
            continue
        eval_model.load_parameters(checkpoint_name)
        logging.info('Loaded parameters from checkpoint %s'%(checkpoint_name))
        start_epoch_time = time.time()
        final_test_L = test(test_data, test_batch_size, ctx=ctx)
        end_epoch_time = time.time()
        logging.info('[Epoch %d] test loss %.2f, test ppl %.2f'%
              (epoch, final_test_L, math.exp(final_test_L)))
        logging.info('Epoch %d took %.2f seconds.'%(epoch, end_epoch_time - start_epoch_time))
        sys.stdout.flush()
        epoch += 1

if __name__ == '__main__':
    if args.eval_only:
        evaluate()
    else:
        train()
