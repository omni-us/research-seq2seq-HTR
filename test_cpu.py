import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#import loadData2 as loadData
import loadData
import numpy as np
import time
import os
import cv2
import argparse

print(time.ctime())
parser = argparse.ArgumentParser(description='seq2seq net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('layers', type=int, help='decoder layer numbers')
parser.add_argument('model', type=str, help='the best model')
args = parser.parse_args()

DECODER_LAYER = args.layers
Bi_GRU = True
print_shape_flag = True
CON_STEP = False
VISUALIZE_TRAIN = False
EARLY_STOPPING = True
TF_LOG = False
FREEZE = True

if CON_STEP:
    STEP = 4 # encoder output squeeze step
#BATCH_SIZE = 256
BATCH_SIZE = 70

MODEL_FILE = args.model

HEIGHT = loadData.IMG_HEIGHT
WIDTH = loadData.IMG_WIDTH
output_max_len = loadData.OUTPUT_MAX_LEN
tokens = loadData.tokens
num_tokens = loadData.num_tokens
vocab_size = loadData.num_classes + num_tokens
index2letter = loadData.index2letter


grad_clip = 10.0
HIDDEN_SIZE = 512
#EMBEDDING_SIZE = 10 # MNIST_seq
EMBEDDING_SIZE = 60 # IAM
TRADEOFF_EMBED_CONTEXT = False # tradeoff between embedding and context vector

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.hidden_size = HIDDEN_SIZE
        self.n_layers = 1
        self.dropout = 0.5

        self.layer0 = nn.Sequential(
                nn.Conv2d(1, 48, 3),
                nn.ReLU(),
                nn.BatchNorm2d(48),
                nn.MaxPool2d(2))
        self.layer1 = nn.Sequential(
                nn.Conv2d(48, 128, 3),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                nn.Conv2d(128, 192, 3),
                nn.ReLU(),
                nn.BatchNorm2d(192),
                )
        self.layer3 = nn.Sequential(
                nn.Conv2d(192, 192, 3),
                nn.ReLU(),
                nn.BatchNorm2d(192),
                )
        self.layer4 = nn.Sequential(
                nn.Conv2d(192, 128, 3),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2))

        if CON_STEP:
            self.output_proj = nn.Linear((((((HEIGHT-2)//2)-2)//2-2-2-2)//2)*128*STEP, self.hidden_size)
        if Bi_GRU:
            self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers, dropout=self.dropout, bidirectional=True)
            #self.output_comb = nn.Linear(self.hidden_size*2, self.hidden_size)
            #self.hidden_proj = nn.Linear(self.hidden_size*2, self.hidden_size)
        else:
            self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers, dropout=self.dropout, bidirectional=False)

    # (32, 1, 80, 1400)
    def forward(self, in_data, in_data_len, hidden=None):
        batch_size = in_data.shape[0]
        out = self.layer0(in_data) #(32, 32, 39, 699)
        #out.register_hook(print)
        #out = self.conv_mask(out, in_data_len)
        out = self.layer1(out) # (32, 64, 18, 348)
        #out = self.conv_mask(out, in_data_len)
        out = self.layer2(out) # (32, 128, 8, 173)
        #out = self.conv_mask(out, in_data_len)
        out = self.layer3(out)
        #out = self.conv_mask(out, in_data_len)
        out = self.layer4(out) # [128, 128, 4, 122]
        #out.register_hook(print)
        #out = self.conv_mask(out, in_data_len)
        out = out.permute(3, 0, 2, 1) # (width, batch, height, channels)
        out.contiguous()
        out = out.view(-1, batch_size, (((((HEIGHT-2)//2)-2)//2-2-2-2)//2)*128) # (t, b, f) (173, 32, 1024)
        if CON_STEP:
            time_step, batch_size, n_feature = out.shape[0], out.shape[1], out.shape[2]
            out_short = Variable(torch.zeros(time_step//STEP, batch_size, n_feature*STEP)) # t//STEP, b, f*STEP
            for i in range(0, time_step//STEP):
                part_out = [out[j] for j in range(i*STEP, (i+1)*STEP)]
                # reverse the image feature map
                #out_short[time_step//STEP - i - 1] = torch.cat(part_out, 1) # b, f*STEP
                out_short[i] = torch.cat(part_out, 1) # b, f*STEP

            out = self.output_proj(out_short) # t//STEP, b, hidden_size
        width = out.shape[0]
        src_len = in_data_len.numpy()*(width/WIDTH)
        src_len = src_len + 0.999 # in case of 0 length value from float to int
        src_len = src_len.astype('int')
        out = pack_padded_sequence(out, src_len.tolist(), batch_first=False)
        output, hidden = self.gru(out, hidden)
        output, output_len = pad_packed_sequence(output, batch_first=False)
       # # output: t, b, f*2     hidden: 2, b, f
        if loadData.FLIP:
            hidden = hidden.permute(1, 0, 2) # b, 2, f
            hidden = hidden.contiguous().view(batch_size, -1) # b, f*2
        else:
            hidden = output[0] # b, f*2
        #if Bi_GRU:
            #output = self.output_comb(output) # t, b, f
            #hidden = self.hidden_proj(hidden) # b, f

        return output, hidden.squeeze(0) # t, b, f*2    b, f*2
        #hidden_new = (output[0] + output[1])/2
        #return output, hidden_new # (t,b,f)  (b,f)

    # matrix: b, c, h, w    lens: list size of batch_size
    def conv_mask(self, matrix, lens):
        lens = np.array(lens)
        width = matrix.shape[-1]
        lens2 = lens * (width / WIDTH)
        lens2 = lens2 + 0.999 # in case le == 0
        lens2 = lens2.astype('int')
        matrix_new = matrix.permute(0, 3, 1, 2) # b, w, c, h
        matrix_out = Variable(torch.zeros(matrix_new.shape))
        for i, le in enumerate(lens2):
            if loadData.FLIP:
                matrix_out[i, -le:] = matrix_new[i, -le:]
            else:
                matrix_out[i, :le] = matrix_new[i, :le]
        matrix_out = matrix_out.permute(0, 2, 3, 1) # b, c, h, w
        return matrix_out

#def create_mask(length_list, max_length):
#    batch_size = length_list.size
#    mask = np.zeros([batch_size, max_length], dtype='float32')
#    for i, le in enumerate(length_list):
#        mask[i, :le] = 1
#    return mask

# Standard Bahdanau Attention
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(dim=0)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)
        #self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.hidden_proj = nn.Linear(self.hidden_size, self.hidden_size)
        #self.encoder_output_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    # hidden: b, f  encoder_output: t, b, f  enc_len: numpy
    def forward(self, hidden, encoder_output, enc_len):
        encoder_output = encoder_output.transpose(0, 1) # b, t, f
        attn_energy = self.score(hidden, encoder_output) # b, t

        attn_weight = Variable(torch.zeros(attn_energy.shape))
        for i, le in enumerate(enc_len):
            attn_weight[i, :le] = self.softmax(attn_energy[i, :le])
        return attn_weight.unsqueeze(2)
        #return self.softmax(attn_energy).unsqueeze(2) # b, t, 1

    # hidden: 1, batch, features
    # encoder_output: batch, time_step, features
    def score(self, hidden, encoder_output):
        hidden = hidden.permute(1, 2, 0) # batch, features, layers
        addMask = torch.FloatTensor([1/DECODER_LAYER] * DECODER_LAYER).view(1, DECODER_LAYER, 1)
        addMask = torch.cat([addMask] * hidden.shape[0], dim=0)
        addMask = Variable(addMask) # batch, layers, 1
        hidden = torch.bmm(hidden, addMask) # batch, feature, 1
        hidden = hidden.permute(0, 2, 1) # batch, 1, features
        hidden_attn = self.hidden_proj(hidden) # b, 1, f
        #hidden_attn = hidden_attn.permute(1, 0, 2) # batch, 1, features
        #encoder_output_attn = self.encoder_output_proj(encoder_output)
        #res_attn = self.tanh(encoder_output_attn + hidden_attn) # b, t, f
        res_attn = self.tanh(encoder_output + hidden_attn) # b, t, f
        out_attn = self.out(res_attn) # b, t, 1
        out_attn = out_attn.squeeze(2) # b, t
        return out_attn

#class Attention(nn.Module):
#    def __init__(self, hidden_size):
#        super(Attention, self).__init__()
#        self.hidden_size = hidden_size
#        self.softmax = nn.Softmax(dim=1)
#        self.proj = nn.Linear(self.hidden_size, self.hidden_size)
#        self.relu = nn.ReLU()
#        self.hidden_proj = nn.Linear(self.hidden_size, self.hidden_size)
#        self.out = nn.Linear(hidden_size, 1)
#
#    # hidden: b, f  encoder_output: t, b, f
#    def forward(self, hidden, encoder_output):
#        encoder_output = encoder_output.transpose(0, 1) # b, t, f
#        attn_energy = self.score(hidden, encoder_output) # b, t
#        return self.softmax(attn_energy).unsqueeze(2) # b, t, 1
#
#    # hidden: 1, batch, features
#    # encoder_output: batch, time_step, features
#    def score(self, hidden, encoder_output):
#        hidden_attn = self.hidden_proj(hidden) # 1, b, f
#        hidden_attn = hidden_attn.permute(1, 2, 0) # batch, features, 1
#        out_attn = torch.bmm(encoder_output, hidden_attn) # b, t, 1
#        out_attn = out_attn.squeeze(2) # b, t
#        return out_attn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        if Bi_GRU:
            self.hidden_size = HIDDEN_SIZE * 2
        else:
            self.hidden_size = HIDDEN_SIZE
        self.embed_size = EMBEDDING_SIZE
        self.n_layers = DECODER_LAYER

        self.embedding = nn.Embedding(vocab_size, self.embed_size)
        self.dropout = 0.5
        self.attention = Attention(self.hidden_size)
        if TRADEOFF_EMBED_CONTEXT:
            self.context_shrink = nn.Linear(self.hidden_size, self.embed_size*5) # !! trade-off between embedding and context
            self.gru = nn.GRU(self.embed_size*5 + self.embed_size, self.hidden_size, self.n_layers, dropout=self.dropout)
        else:
            self.gru = nn.GRU(self.embed_size + self.hidden_size, self.hidden_size, self.n_layers, dropout=self.dropout)
        self.out = nn.Linear(self.hidden_size, vocab_size)
        #self.context_proj = nn.Linear((((width//2-1)//2-1)//2-1)//STEP*self.hidden_size, self.hidden_size)

    # hidden: (32, 256)  encoder_output: (55, 32, 256)
    def forward(self, in_char, hidden, encoder_output, src_len):
        #embed_char = self.dropout(embed_char)

        width = encoder_output.shape[0]
        enc_len = src_len.numpy() * (width/src_len[0])
        enc_len = enc_len + 0.999
        enc_len = enc_len.astype('int')
        attn_weights = self.attention(hidden, encoder_output, enc_len) # b, t, 1
        #context = attn_weights.bmm(encoder_output.transpose(0, 1))
        # (32, 1, 55) * (32, 55, 256) -> (32, 1, 256)
        #context = context.transpose(0, 1) # (1, 32, 256)
        #context = context.squeeze(0) # (32, 256)
        #encoder_output_b = encoder_output.permute(1, 0, 2) # b, t, f
        #pre_context = encoder_output_b * attn_weights # b, t, f
        #batch_size = pre_context.shape[0]
        #cat_context = pre_context.view(batch_size, -1) # b, t*f
        #context = self.context_proj(cat_context) # b, f

        encoder_output_b = encoder_output.permute(1, 2, 0) # b, f, t
        context = torch.bmm(encoder_output_b, attn_weights) # b, f, 1
        context = context.squeeze(2)
        if TRADEOFF_EMBED_CONTEXT:
            context = self.context_shrink(context)

        #top1_random_list = []
        #for i in in_char:
        #    top1_random = np.random.choice(np.arange(vocab_size), p=F.softmax(i, dim=0).cpu().data.numpy())
        #    top1_random_list.append(top1_random.tolist())
        #top1_random_np = np.array(top1_random_list)

        top1 = in_char.topk(1)[1] # batch, 1
        embed_char = self.embedding(top1) # batch,1,embed
        embed_char = embed_char.squeeze(1)

        in_dec = torch.cat((embed_char, context), 1) # 16, 557
        in_dec = in_dec.unsqueeze(0)
        #self.gru.flatten_parameters() ## Multi-GPU add this @@@
        output, latest_hidden = self.gru(in_dec, hidden) # 1,16,512   3,16,512  nn.GRU
        #output.register_hook(print)
        output = output.squeeze(0)
        #out_dec = torch.cat((output, context), 1)

        output = F.softmax(self.out(output), dim=1) #(32,62)
        return output, latest_hidden, attn_weights.squeeze(2) # (32,62), (32,256), (32,55)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    # src: Variable
    # tar: Variable
    def forward(self, src, tar, src_len, teacher_forcing, train=True):
        tar = tar.permute(1, 0) # time_s, batch
        batch_size = src.size(0)
        #max_len = tar.size(0) # <go> true_value <end>
        outputs = Variable(torch.zeros(output_max_len-1, batch_size, vocab_size), requires_grad=False) # (14, 32, 62) not save the first <GO>
        outputs = outputs
        #src = Variable(src)
        out_enc, hidden_enc = self.encoder(src, src_len)
        # t,b,f    b,f
        global print_shape_flag
        if print_shape_flag:
            print('First batch shape: (The shape of batches are not same)')
            print(out_enc.shape, output_max_len)
            print_shape_flag = False

        output = one_hot(tar[0].data.cpu())
        output = Variable(output)
        attns = []

        hidden = hidden_enc.unsqueeze(0) # 1, batch, hidden_size

        init_hidden_dec = [hidden] * DECODER_LAYER
        hidden = torch.cat(init_hidden_dec, dim=0)

        for t in range(0, output_max_len-1): # max_len: groundtruth + <END>
            output, hidden, attn_weights = self.decoder(
                    output, hidden, out_enc, src_len)
            outputs[t] = output
            #top1 = output.data.topk(1)[1].squeeze()
            output = Variable(one_hot(tar[t+1].data.cpu()) if train and teacher_forcing else output.data)
            attns.append(attn_weights.data.cpu()) # [(32, 55), ...]
        return outputs, attns

def to_onehot(src, vocab_size): # src: Variable
    batch_size = src.size(0)
    onehot = torch.FloatTensor(batch_size, vocab_size).zero_()
    onehot.scatter_(1, src.data.cpu().unsqueeze(1), 1)
    return onehot

def one_hot(src): # src: torch.LongTensor
    ones = torch.sparse.torch.eye(vocab_size)
    return ones.index_select(0, src)

def visualizeAttn(img, first_img_real_len, attn, epoch, batches, name):
    img *= 255./img.max()
    img = img[:, :first_img_real_len]
    img = img.cpu().numpy().astype(np.uint8)
    weights = [img] # (80, 460)
    for m in attn:
        mask_img = np.vstack([m]*10) # (10, 55)
        mask_img *= 255./mask_img.max()
        mask_img = mask_img.astype(np.uint8)
        mask_img = cv2.resize(mask_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        weights.append(mask_img)
    output = np.vstack(weights)
    if loadData.FLIP:
        output = np.flip(output, 1)
    #logger.add_image('attention', torch.from_numpy(output.copy()).float(), 'valid', batches)
    cv2.imwrite('imgs_layer_'+str(DECODER_LAYER)+'/'+name+'_'+str(epoch)+'.jpg', output)

def writePredict(epoch, index, pred, flag): # [batch_size, vocab_size] * max_output_len
    folder_name = 'pred_logs_layer_'+str(DECODER_LAYER)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if flag == 'train':
        file_prefix = folder_name+'/train_predict_seq.'
    elif flag == 'valid':
        file_prefix = folder_name+'/valid_predict_seq.'
    elif flag == 'test':
        file_prefix = folder_name+'/test_predict_seq.'

    pred = pred.data
    pred2 = pred.topk(1)[1].squeeze(2) # (15, 32)
    pred2 = pred2.transpose(0, 1) # (32, 15)
    pred2 = pred2.cpu().numpy()

    with open(file_prefix+str(epoch)+'.log', 'a') as f:
        for n, seq in zip(index, pred2):
            f.write(n+' ')
            for i in seq:
                if i ==tokens['END_TOKEN']:
                    #f.write('<END>')
                    break
                else:
                    if i ==tokens['GO_TOKEN']:
                        f.write('<GO>')
                    elif i ==tokens['PAD_TOKEN']:
                        f.write('<PAD>')
                    else:
                        f.write(index2letter[i-num_tokens])
            f.write('\n')

def preProcess(data_in, data_out):
    data_in = np.array(data_in)
    data_in = np.expand_dims(data_in, axis=1) # (batch, 1, height, width)
    #data_in = torch.from_numpy(data_in) # (32, 1, 80, 460)
    data_in = torch.FloatTensor(data_in)

    data_out = np.array(data_out)
    #data_out = np.transpose(data_out , (1, 0)) # (15, 32)
    #data_out = torch.from_numpy(data_out)
    data_out = torch.LongTensor(data_out)
    return data_in, data_out

def sort_batch(data):
        train_index, train_in, train_out, train_out_mask, train_in_len = data['index_sa'], data['input_sa'], data['output_sa'], data['out_len_sa'], data['in_len_sa']
        train_out_mask = train_out_mask # useless

        train_index = np.array(train_index)
        train_in = np.array(train_in, dtype='float32')
        train_out = np.array(train_out, dtype='int64')
        train_in_len = np.array(train_in_len, dtype='int64')

        train_in = torch.from_numpy(train_in)
        train_out = torch.from_numpy(train_out)
        train_in_len = torch.from_numpy(train_in_len)

        train_in_len, idx = train_in_len.sort(0, descending=True)
        train_in = train_in[idx]
        train_out = train_out[idx]
        train_index = train_index[idx]
        return train_index, train_in, train_in_len, train_out

encoder = Encoder()
decoder = Decoder()
seq2seq = Seq2Seq(encoder, decoder)
seq2seq.load_state_dict(torch.load(MODEL_FILE, map_location=lambda storage, loc: storage))
seq2seq.eval()
total_loss_t = 0
start_t = time.time()
data_train, data_valid, data_test = loadData.loadData()

for i in range(len(data_test)//BATCH_SIZE + 1):
    start_t_batch = time.time()
    data_t = data_test[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
    test_index, test_in, test_in_len, test_out = sort_batch(data_t)
    #test_index, test_in, test_out, test_out_len, test_in_len = data_t['index_sa'], data_t['input_sa'], data_t['output_sa'], data_t['out_len_sa'], data_t['in_len_sa']
    test_in = test_in.unsqueeze(1)
    test_in, test_out = Variable(test_in, volatile=True), Variable(test_out, volatile=True)
    output_t, attn_weights_t = seq2seq(test_in, test_out, test_in_len, teacher_forcing=False, train=False)
    writePredict(0, test_index, output_t, 'test')
    test_label = test_out.permute(1, 0)[1:].contiguous().view(-1)
    loss_t = F.cross_entropy(output_t.view(-1, vocab_size),
                             test_label)
    total_loss_t += loss_t.data[0]

    if i == 0:
        # (32,1,80,460)->(80,460)  [(32,55),...]->[(55),...]
        visualizeAttn(test_in.data[0,0], test_in_len[0], [j[0] for j in attn_weights_t], 0, len(data_test)//BATCH_SIZE, 'test-first')
        visualizeAttn(test_in.data[2,0], test_in_len[0], [j[2] for j in attn_weights_t], 0, len(data_test)//BATCH_SIZE, 'test-second')
        visualizeAttn(test_in.data[36,0], test_in_len[0], [j[36] for j in attn_weights_t], 0, len(data_test)//BATCH_SIZE, 'test-third')
        visualizeAttn(test_in.data[41,0], test_in_len[0], [j[41] for j in attn_weights_t], 0, len(data_test)//BATCH_SIZE, 'test-forth')

    print('%d / %d -- %.2fs' % (i, len(data_test)//BATCH_SIZE + 1, time.time() - start_t_batch))

total_loss_t /= len(data_test)//BATCH_SIZE
print('Test loss=%.3f, time=%.3f' % (total_loss_t, time.time()-start_t))

