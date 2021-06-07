import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

"""
Blog post:
Taming LSTMs: Variable-sized mini-batches and why PyTorch is good for your health:
https://medium.com/@_willfalcon/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
"""


class BieberLSTM(nn.Module):
   def __init__(self, nb_layers, nb_lstm_units=100, embedding_dim=3, batch_size=3):
       super(BieberLSTM,self).__init__()
       self.vocab = {'<PAD>': 0, 'is': 1, 'it': 2, 'too': 3, 'late': 4, 'now': 5, 'say': 6, 'sorry': 7, 'ooh': 8,
                     'yeah': 9}
       self.tags = {'<PAD>': 0, 'VB': 1, 'PRP': 2, 'RB': 3, 'JJ': 4, 'NNP': 5}

       self.nb_layers = nb_layers
       self.nb_lstm_units = nb_lstm_units
       self.embedding_dim = embedding_dim
       self.batch_size = batch_size

       # don't count the padding tag for the classifier output
       self.nb_tags = len(self.tags) - 1

       # when the model is bidirectional we double the output dimension
       self.nb_lstm_layers=nb_layers

       # build actual NN
       self.__build_model()

   def __build_model(self):
       # build embedding layer first
       nb_vocab_words = len(self.vocab)

       # whenever the embedding sees the padding index it'll make the whole vector zeros
       padding_idx = self.vocab['<PAD>']
       self.word_embedding = nn.Embedding(
           num_embeddings=nb_vocab_words,
           embedding_dim=self.embedding_dim,
           padding_idx=padding_idx
       )

       # design LSTM
       self.lstm = nn.LSTM(
           input_size=self.embedding_dim,
           hidden_size=self.nb_lstm_units,
           num_layers=self.nb_lstm_layers,
           batch_first=True,
       )

       # output layer which projects back to tag space
       self.hidden_to_tag = nn.Linear(self.nb_lstm_units, self.nb_tags)

   def init_hidden(self):
       # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
       hidden_a = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)
       hidden_b = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)



       hidden_a = Variable(hidden_a)
       hidden_b = Variable(hidden_b)

       return (hidden_a, hidden_b)

   def forward(self, X, X_lengths):
       # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
       # a new batch as a continuation of a sequence

       print(X.shape)
       self.hidden = self.init_hidden()

       batch_size, seq_len = X.size()

       # ---------------------
       # 1. embed the input
       # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
       X = self.word_embedding(X)

       # ---------------------
       # 2. Run through RNN
       # TRICK 2 ********************************
       # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

       # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
       X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

       # now run through LSTM
       X, self.hidden = self.lstm(X, self.hidden)

       # undo the packing operation
       X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

       # ---------------------
       # 3. Project to tag space
       # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

       # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
       X = X.contiguous()
       X = X.view(-1, X.shape[2])

       # run through actual linear layer
       X = self.hidden_to_tag(X)

       # ---------------------
       # 4. Create softmax activations bc we're doing classification
       # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
       X = F.log_softmax(X, dim=1)

       # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
       X = X.view(batch_size, seq_len, self.nb_tags)

       Y_hat = X
       return Y_hat

   def loss(self, Y_hat, Y, X_lengths):
       # TRICK 3 ********************************
       # before we calculate the negative log likelihood, we need to mask out the activations
       # this means we don't want to take into account padded items in the output vector
       # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
       # and calculate the loss on that.

       # flatten all the labels
       Y = Y.view(-1)

       # flatten all predictions
       Y_hat = Y_hat.view(-1, self.nb_tags)

       # create a mask by filtering out all tokens that ARE NOT the padding token
       tag_pad_token = self.tags['<PAD>']
       mask = (Y > tag_pad_token).float()

       # count how many tokens we have
       nb_tokens = int(torch.sum(mask).data[0])

       # pick the values for the label and zero out the rest with the mask
       t1 = range(Y_hat.shape[0])
       temp = Y_hat[range(Y_hat.shape[0]), Y.long()]
       Y_hat = Y_hat[range(Y_hat.shape[0]), Y.long()] * mask

       # compute cross entropy loss which ignores all <PAD> tokens
       ce_loss = -torch.sum(Y_hat) / nb_tokens

       return ce_loss