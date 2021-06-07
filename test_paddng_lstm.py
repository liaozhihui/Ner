import numpy as np
from lstm_model import BieberLSTM
import torch
from torch.autograd import Variable
sent_1_x = ['is', 'it', 'too', 'late', 'now', 'say', 'sorry']
sent_1_y = ['VB', 'PRP', 'RB', 'RB', 'RB', 'VB', 'JJ']
sent_2_x = ['ooh', 'ooh']
sent_2_y = ['NNP', 'NNP']
sent_3_x = ['sorry', 'yeah']
sent_3_y = ['JJ', 'NNP']
X = [sent_1_x, sent_2_x, sent_3_x]
Y = [sent_1_y, sent_2_y, sent_3_y]

# map sentences to vocab
vocab = {'<PAD>': 0, 'is': 1, 'it': 2, 'too': 3, 'late': 4, 'now': 5, 'say': 6, 'sorry': 7, 'ooh': 8, 'yeah': 9}
# fancy nested list comprehension
X =  [[vocab[word] for word in sentence] for sentence in X]
# X now looks like:
# [[1, 2, 3, 4, 5, 6, 7], [8, 8], [7, 9]]

tags = {'<PAD>': 0, 'VB': 1, 'PRP': 2, 'RB': 3, 'JJ': 4, 'NNP': 5}
# fancy nested list comprehension
Y = [[tags[tag] for tag in sentence] for sentence in Y]
# Y now looks like:
# [[1, 2, 3, 3, 3, 1, 4], [5, 5], [4, 5]]

X_lengths = [len(sentence) for sentence in X]
# create an empty matrix with padding tokens
pad_token = vocab['<PAD>']
longest_sent = max(X_lengths)
batch_size = len(X)
padded_X = np.ones((batch_size, longest_sent)) * pad_token
# copy over the actual sequences
for i, x_len in enumerate(X_lengths):
 sequence = X[i]
 padded_X[i, 0:x_len] = sequence[:x_len]

 Y_lengths = [len(sentence) for sentence in Y]
 # create an empty matrix with padding tokens
 pad_token = tags['<PAD>']
 longest_sent = max(Y_lengths)
 batch_size = len(Y)
 padded_Y = np.ones((batch_size, longest_sent)) * pad_token
 # copy over the actual sequences
 for i, y_len in enumerate(Y_lengths):
     sequence = Y[i]
     padded_Y[i, 0:y_len] = sequence[:y_len]


if __name__ == '__main__':
    model = BieberLSTM(1)
    y_hat = model(torch.tensor(padded_X,dtype=torch.long),X_lengths)
    model.loss(y_hat,torch.Tensor(padded_Y),X_lengths)

