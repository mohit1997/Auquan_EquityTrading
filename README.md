# Auquan Capstone Project
## Requirements
1. Python (2/3)
2. Tensorflow, Numpy, Pandas, os, glob, time

## How to train
1. Put the stock data files in parsedData folder.
2. Change 'files', python list in train_multi.py to select which files to be used for training.
3. Run train_multi.py to train and save the model(time wise validation split 70/30)
4. Use predict.py with the same steps to test the saved model on new csv files.
