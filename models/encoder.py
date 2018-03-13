from torch import nn
from torch.autograd import Variable
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

class Encoder(nn.Module):
    def __init__(self, hidden_size, height, width, bgru, step, flip):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.height = height
        self.width = width
        self.step = step
        self.flip = flip
        self.n_layers = 1
        self.dropout = 0.5

        self.layer0 = nn.Sequential(
                nn.Conv2d(1, 48, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(48),
                nn.MaxPool2d(2))
        self.layer1 = nn.Sequential(
                nn.Conv2d(48, 128, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                nn.Conv2d(128, 192, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(192),
                )
        self.layer3 = nn.Sequential(
                nn.Conv2d(192, 192, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(192),
                )
        self.layer4 = nn.Sequential(
                nn.Conv2d(192, 128, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2))

        if self.step is not None:
            #self.output_proj = nn.Linear((((((self.height-2)//2)-2)//2-2-2-2)//2)*128*self.step, self.hidden_size)
            self.output_proj = nn.Linear(self.height//8*128*self.step, self.height//8*128)
        if bgru: #8: 3 MaxPool->2**3    128: last hidden_size of layer4
            self.gru = nn.GRU(self.height//8*128, self.hidden_size, self.n_layers, dropout=self.dropout, bidirectional=True)
        else:
            self.gru = nn.GRU(self.height//8*128, self.hidden_size, self.n_layers, dropout=self.dropout, bidirectional=False)

    # (32, 1, 80, 1400)
    def forward(self, in_data, in_data_len, hidden=None):
        batch_size = in_data.shape[0]
        out = self.layer0(in_data) #(32, 32, 39, 699)
        out = self.layer1(out) # (32, 64, 18, 348)
        out = self.layer2(out) # (32, 128, 8, 173)
        out = self.layer3(out)
        out = self.layer4(out) # [128, 128, 4, 122]
        #out.register_hook(print)
        out = out.permute(3, 0, 2, 1) # (width, batch, height, channels)
        out.contiguous()
        #out = out.view(-1, batch_size, (((((self.height-2)//2)-2)//2-2-2-2)//2)*128) # (t, b, f) (173, 32, 1024)
        out = out.view(-1, batch_size, self.height//8*128)
        if self.step is not None:
            time_step, batch_size, n_feature = out.shape[0], out.shape[1], out.shape[2]
            out_short = Variable(torch.zeros(time_step//self.step, batch_size, n_feature*self.step)).cuda() # t//STEP, b, f*STEP
            for i in range(0, time_step//self.step):
                part_out = [out[j] for j in range(i*self.step, (i+1)*self.step)]
                # reverse the image feature map
                out_short[i] = torch.cat(part_out, 1) # b, f*STEP

            out = self.output_proj(out_short) # t//STEP, b, hidden_size
        width = out.shape[0]
        src_len = in_data_len.numpy()*(width/self.width)
        src_len = src_len + 0.999 # in case of 0 length value from float to int
        src_len = src_len.astype('int')
        out = pack_padded_sequence(out, src_len.tolist(), batch_first=False)
        output, hidden = self.gru(out, hidden)
        output, output_len = pad_packed_sequence(output, batch_first=False)
       # # output: t, b, f*2     hidden: 2, b, f
        if self.flip:
            hidden = hidden.permute(1, 0, 2) # b, 2, f
            hidden = hidden.contiguous().view(batch_size, -1) # b, f*2
        else:
            hidden = output[0] # b, f*2
        return output, hidden.squeeze(0) # t, b, f*2    b, f*2

    # matrix: b, c, h, w    lens: list size of batch_size
    def conv_mask(self, matrix, lens):
        lens = np.array(lens)
        width = matrix.shape[-1]
        lens2 = lens * (width / self.width)
        lens2 = lens2 + 0.999 # in case le == 0
        lens2 = lens2.astype('int')
        matrix_new = matrix.permute(0, 3, 1, 2) # b, w, c, h
        matrix_out = Variable(torch.zeros(matrix_new.shape)).cuda()
        for i, le in enumerate(lens2):
            if self.flip:
                matrix_out[i, -le:] = matrix_new[i, -le:]
            else:
                matrix_out[i, :le] = matrix_new[i, :le]
        matrix_out = matrix_out.permute(0, 2, 3, 1) # b, c, h, w
        return matrix_out
