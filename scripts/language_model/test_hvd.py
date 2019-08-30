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
 journal={arXiv prelogging.info arXiv:1602.02410},
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
import os
import sys
import argparse
import logging
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
import gluonnlp as nlp

# logging
logging.getLogger().setLevel(logging.INFO)

# os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
# os.environ['MXNET_CPU_PARALLEL_RAND_COPY'] = str(len(context))
# os.environ['MXNET_CPU_WORKER_NTHREADS'] = str(len(context))

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
context = [ctx]

x = mx.nd.zeros((1, num_workers), ctx)
x[0, rank] = 2
x_row_sparse = x.tostype('row_sparse')
mx.nd.waitall()

logging.info(x)

y = hvd.allgather(x)
mx.nd.waitall()
logging.info(y)