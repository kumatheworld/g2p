# g2p
This repository lets you train and test neural networks for pronunciation prediction of American English. Hopefully, you will know how people would sensibly pronounce brand new words.

![demo](https://drive.google.com/uc?export=view&id=1w7dpxUDPoaMHwoLueIeDD5CREtqV-vY_)![IPA embedding](https://drive.google.com/uc?export=view&id=1fdj14oyWNvKgfYTBQbaPDc-iBrrBpzon)

## Setup
* Python ≥ 3.6.
* ```pip install -r requirements.txt```.
* ```export PYTHONPATH=/path/to/g2p:$PYTHONPATH```.
* Download .pth files from [here](https://drive.google.com/drive/folders/1fyj9mBHauAuXW33mcol3J2RORFEMdHzr?usp=sharing) and put them under ```g2p/checkpoints```.

## Test
* ```python main/test.py```.

## Train
* ```python main/train.py```.
* Watch training progress by ```tensorboard --logdir runs```.

## Validate
* ```python main/val.py```.
