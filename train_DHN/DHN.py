#==========================================================================
# This file is under License LGPL-3.0 (see details in the license file).
# This file is a part of implementation for paper:
# How To Train Your Deep Multi-Object Tracker.
# This contribution is headed by Perception research team, INRIA.
# Contributor(s) : Yihong Xu
# INRIA contact  : yihong.xu@inria.fr
# created on 16th April 2020.
#==========================================================================

import torch
import torch.nn as nn

# approximation of Munkrs


class Munkrs(nn.Module):
    def __init__(self, element_dim, hidden_dim, target_size, bidirectional, minibatch, is_cuda, is_train=True,
                 sigmoid=True, trainable_delta=False):
        super(Munkrs, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirect = bidirectional
        self.minibatch = minibatch
        self.is_cuda = is_cuda
        self.sigmoid = sigmoid
        if trainable_delta:
            if self.is_cuda:
                self.delta = torch.nn.Parameter(torch.FloatTensor([10]).cuda())
            else:
                self.delta = torch.nn.Parameter(torch.FloatTensor([10]))

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        ## The number of expected features in the input is element_dim = 1
        ## The number of features in the hidden state h=256
        ## stacking two GRUs together to form a stacked GRU, with the second GRU taking in outputs of the first GRU and computing the final results
        self.lstm_row = nn.GRU(element_dim, hidden_dim, bidirectional=self.bidirect, num_layers=2, dropout=0.2)
        self.lstm_col = nn.GRU(hidden_dim*2, hidden_dim, bidirectional=self.bidirect, num_layers=2, dropout=0.2)

        # The linear layer that maps from hidden state space to tag space
        if self.bidirect:
            ## Go through 3 fully connected layers to be used after RNN
            # *2 directions * 2 ways concat
            self.hidden2tag_1 = nn.Linear(hidden_dim * 2, 256)
            self.hidden2tag_2 = nn.Linear(256, 64)
            self.hidden2tag_3 = nn.Linear(64, target_size)
        else:
            # * 2 ways concat
            self.hidden2tag_1 = nn.Linear(hidden_dim, target_size)

        self.hidden_row = self.init_hidden(1)
        self.hidden_col = self.init_hidden(1)

        # init layers
        if is_train:
            for m in self.modules():
                if isinstance(m, nn.GRU):
                    print("weight initialization")
                    torch.nn.init.orthogonal_(m.weight_ih_l0.data)
                    torch.nn.init.orthogonal_(m.weight_hh_l0.data)
                    torch.nn.init.orthogonal_(m.weight_ih_l0_reverse.data)
                    torch.nn.init.orthogonal_(m.weight_hh_l0_reverse.data)

                    # initial gate bias as -1
                    m.bias_ih_l0.data[0:self.hidden_dim].fill_(-1)
                    m.bias_hh_l0.data[0:self.hidden_dim].fill_(-1)
                    m.bias_ih_l0_reverse.data[0:self.hidden_dim].fill_(-1)
                    m.bias_hh_l0_reverse.data[0:self.hidden_dim].fill_(-1)

                    torch.nn.init.orthogonal_(m.weight_ih_l1.data)
                    torch.nn.init.orthogonal_(m.weight_hh_l1.data)
                    torch.nn.init.orthogonal_(m.weight_ih_l1_reverse.data)
                    torch.nn.init.orthogonal_(m.weight_hh_l1_reverse.data)

                    # initial gate bias as one
                    m.bias_ih_l1.data[0:self.hidden_dim].fill_(-1)
                    m.bias_hh_l1.data[0:self.hidden_dim].fill_(-1)
                    m.bias_ih_l1_reverse.data[0:self.hidden_dim].fill_(-1)
                    m.bias_hh_l1_reverse.data[0:self.hidden_dim].fill_(-1)



    def init_hidden(self, batch):
        # The axes semantics are (num_layers * num_directions, minibatch_size, hidden_dim),
        # one for hidden, others for memory cell

        if self.bidirect:
            if self.is_cuda:
                hidden = torch.zeros(2*2, batch, self.hidden_dim).cuda()
            else:
                hidden = torch.zeros(2*2, batch, self.hidden_dim)

        else:
            if self.is_cuda:
                hidden = torch.zeros(2, batch, self.hidden_dim).cuda()
            else:
                hidden = torch.zeros(2, batch, self.hidden_dim)
        return hidden

    def forward(self, Dt):
        ## hidden size [2*2(bi-directional and two GRU stacked together), batch, 256]
        self.hidden_row = self.init_hidden(Dt.size(0))
        self.hidden_col = self.init_hidden(Dt.size(0))

        # Dt is of shape [batch, h, w]
        # input_row is of shape [h*w, batch, 1], [time steps, mini batch, element dimension]
        # row lstm #
        ## first step "view" changes Dt to dim (batch, ?, 1), then ? is calculated to be h*w
        ## second step "permute" changes Dt to (h*w, batch, 1)
        ## then in contiguous memory format
        ## Row-wise flatten
        input_row = Dt.view(Dt.size(0), -1, 1).permute(1, 0, 2).contiguous()
        ## flattened Dt as input_row, hidden_row inited as zeros given as argument
        ## hidden_row of the last block is returned, but not further used
        ## lstm_R_out has the size [seq_len, batch, num_directions*hidden_size] since the output at every sequential 
        ## step is copied from the hidden state to be transfered into the next sequential step
        lstm_R_out, self.hidden_row = self.lstm_row(input_row, self.hidden_row)

        # column lstm #
        # lstm_R_out is of shape [seq_len=h*w, batch, hidden_size * num_directions]

        # [h * w*batch, hidden_size * num_directions]
        lstm_R_out = lstm_R_out.view(-1, lstm_R_out.size(2))


        # [h * w*batch, 1]
        ## the following line is commented out
        # lstm_R_out = self.hidden2tag_1(lstm_R_out).view(-1, Dt.size(0))

        # [h,  w, batch, hidden_size * num_directions]
        ## this is exactly the shape of "first-stage hidden representation" in Figure 2 of the paper
        ## without considering batch = 1
        lstm_R_out = lstm_R_out.view(Dt.size(1), Dt.size(2), Dt.size(0), -1)

        # col wise vector
        # [w,  h, batch, hidden_size * num_directions]
        input_col = lstm_R_out.permute(1, 0, 2, 3).contiguous()

        ## Column-wise flatten
        # [w*h, batch, hidden_size * num_directions]
        ## therefore, the input size of the second RNN is hidden_dim*2, this is different from the first RNN (= 1)
        ## The input structures of the two RNNs are different
        input_col = input_col.view(-1, input_col.size(2), input_col.size(3)).contiguous()
        lstm_C_out, self.hidden_col = self.lstm_col(input_col, self.hidden_col)
        ## lstm_C_out is now of size [seq_len (that is w*h), batch, hidden_size * num_directions]
        # undo col wise vector
        # lstm_out is of shape [seq_len=time steps=w*h, batch, hidden_size * num_directions]

        # [h, w, batch, hidden_size * num_directions]
        # "second-stage hidden representation"
        lstm_C_out = lstm_C_out.view(Dt.size(2), Dt.size(1), Dt.size(0), -1).permute(1, 0, 2, 3).contiguous()

        ## here lstm_C_out is also flattened
        # [h*w*batch, hidden_size * num_directions]
        lstm_C_out = lstm_C_out.view(-1, lstm_C_out.size(3))

        ## Go through 3 fully connected layers
        # [h*w, batch, 1]
        tag_space = self.hidden2tag_1(lstm_C_out)
        tag_space = self.hidden2tag_2(tag_space)
        tag_space = self.hidden2tag_3(tag_space).view(-1, Dt.size(0))

        ## sigmoid function is after the 3 fully connected layers, this is different from Figure 2
        if self.sigmoid:
            tag_scores = torch.sigmoid(tag_space)
        else:
            tag_scores = tag_space
        # tag_scores is of shape [batch, h, w] as Dt
        ## this is the soft assignment matrix
        return tag_scores.view(Dt.size(1), Dt.size(2), -1).permute(2, 0, 1).contiguous()