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

# coding: utf-8
# pylint: disable=line-too-long
"""Parameter optimizer."""

from mxnet import optimizer as opt
from mxnet.model import _create_kvstore, _create_sparse_kvstore
from mxnet.gluon.parameter import ParameterDict, Parameter

import mxnet as mx
import types
import warnings
import math

import horovod.mxnet as hvd
from horovod.mxnet.mpi_ops import allreduce, allreduce_

class DistributedHierKVHVDTrainer(mx.gluon.Trainer):
    def __init__(self, params, optimizer, optimizer_params=None, sdtype='float32'):

        super(DistributedHierKVHVDTrainer, self).__init__(
            params, optimizer, optimizer_params=optimizer_params, update_on_kvstore=False)

        self._hvd_param_buf = {}
        self._sdtype = sdtype
        if sdtype == 'float32':
            self._dtype_mismatch = False
        else:
            self._dtype_mismatch = True

    def _allreduce_grads(self):
        # super(DistributedHierKVHVDTrainer, self)._allreduce_grads()

        print(self._update_on_kvstore)

        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                reduce_ind = i % len(param.list_grad())
                self._kvstore.push(i, param.list_grad(), priority=-i)
                # TODO(xcong) allreduce the buffer, avoid the extra copy in kvstore.pull
                self._kvstore.pull(i, [param.list_grad()[reduce_ind]], priority=-i)

                # allreduce between processes
                if param.list_grad()[reduce_ind].stype == 'default':
                    allreduce_(param.list_grad()[reduce_ind], average=True,
                               name=str(i), priority=-i)
                else:
                    if i not in self._hvd_param_buf:
                        self._hvd_param_buf[i] = mx.nd.zeros(param.list_grad()[reduce_ind].shape, param.list_grad()[reduce_ind].context, dtype=self._sdtype)
                    param_dense = self._hvd_param_buf[i]
                    if self._dtype_mismatch:
                        param_dense[:] = param.list_grad()[reduce_ind].tostype('default')
                    else:
                        mx.nd.sparse.cast_storage(param.list_grad()[reduce_ind], 'default', out=param_dense)
                    allreduce_(param_dense, average=True,
                                name=str(i), priority=-i)

                    if self._dtype_mismatch:
                        param.list_grad()[reduce_ind][:] = param_dense.tostype('row_sparse')
                    else:
                        mx.nd.sparse.cast_storage(param_dense, 'row_sparse', out=param.list_grad()[reduce_ind])

                # local broadcast
                for j in range(len(param.list_grad())):
                    if j != reduce_ind:
                        param.list_grad()[reduce_ind].copyto(param.list_grad()[j])