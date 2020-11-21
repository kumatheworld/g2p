# g2p
This repository lets you train and test neural networks for pronunciation prediction of American English. Hopefully, you will know how people would sensibly pronounce brand new words.

## Setup
* Python ≥ 3.6.
* ```pip install -r requirements.txt```.
* ```export PYTHONPATH=/path/to/g2p:$PYTHONPATH```.

## Train
* ```python train.py```.
* Watch training progress by ```tensorboard --logdir runs```.

## Test
* ```python test.py```.
