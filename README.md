# Zeno

### This is part of the python implementation of the paper "Local AdaAlter: Communication-Efficient Stochastic Gradient Descent with Adaptive Learning Rates", by anonymous authors.

This repository is a modified version of GluonNLP
-----------------

### Requirements

The following python packages needs to be installed:

1. MXNet (a modified version with anonymous link: https://anonymous.4open.science/r/f176a9b2-41cd-445d-84cc-8737bbe1646b/)
2. Numpy
3. Horovod

**To install MXNet, please follow the instructions in README.md in https://anonymous.4open.science/r/f176a9b2-41cd-445d-84cc-8737bbe1646b/**

The following files/folders are modified for the implementation of AdaAlter optimizer:
  * src/gluonnlp/optimizer/
  * src/gluonnlp/data/sampler.py
  * scripts/language_model/

Since anonymous.4open.science does not support downloading the entire project as a zip package, to reproduce the experiments, please install this repository by following the instructions below:
  1. Fetch the latest public version of GluonNLP ```git clone https://github.com/dmlc/gluon-nlp.git```
  2. Replace the files/folders above by the ones of this repository
  3. Install the python packages: ```python3 setup.py install --user```

### Run the demo

* Train with 8 GPU workers, local AdaAlter, H=4, batch size is 256 for each GPU:
```bash
cd scripts/language_model
horovodrun -np 8 -H localhost:8 python3 large_word_language_model_local_hvd.py --clip=10 --optimizer localadaalter --warmup-steps 600 --lr 0.5 --local-sgd-interval 4 --batch-size 256
```

* Train with 8 GPU workers, fully synchronous AdaGrad, batch size is 256 for each GPU:
```bash
cd scripts/language_model
horovodrun -np 8 -H localhost:8 python3 large_word_language_model_hvd.py --clip=10 --optimizer adagrad --warmup-steps 1 --lr 0.5 --batch-size 256
```

* Evaluate the results on test dataset:
```bash
cd scripts/language_model
python3 large_word_language_model.py --gpus 4 --eval-only --batch-size=1
```



