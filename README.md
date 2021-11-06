# g2p
This repository lets you train and test neural networks for pronunciation prediction. Once you feed an English word, whether it be in your dictionary or not, you will get a sequence of IPA symbols that represents its pronunciation.

![demo](https://drive.google.com/uc?export=view&id=1w7dpxUDPoaMHwoLueIeDD5CREtqV-vY_)![IPA embedding](https://drive.google.com/uc?export=view&id=1DQB4jgNnIwn4VXPhv0KPEN6CBmkiq3Wl)

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
