import torch
import random
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
#from torch.nn.utils import clip_grad_norm
#import processData as prod
import loadData
import numpy as np
import time
import os
import cv2
from LogMetric import Logger

torch.cuda.set_device(1)
Bi_GRU = False
print_shape_flag = True

STEP = 1 # encoder output squeeze step
BATCH_SIZE = 128

CurriculumLearning = False

MODEL_FILE = 'models/seq2seq-25.model'
#dataModel = prod.preProcess()
#dataModel.createGT(True)
#dataModel.createGT(False)
#n_per_epoch = dataModel.n_per_epoch
#n_per_epoch_t = dataModel.n_per_epoch_t
#batch_size = dataModel.batch_size
#height = dataModel.height
#width = dataModel.width
#output_max_len = dataModel.output_max_len #groundtruth + <END>
#tokens = dataModel.tokens
#vocab_size = dataModel.vocab_size
#index2letter = dataModel.index2letter
#num_tokens = dataModel.num_tokens

height = loadData.IMG_HEIGHT
width = loadData.IMG_WIDTH
output_max_len = loadData.OUTPUT_MAX_LEN
tokens = loadData.tokens
num_tokens = loadData.num_tokens
vocab_size = loadData.num_classes + num_tokens
index2letter = loadData.index2letter


grad_clip = 10.0
HIDDEN_SIZE = 512
#EMBEDDING_SIZE = 10 # MNIST_seq
EMBEDDING_SIZE = 50 # IAM

## variable_in: torch.autograd.Variable  dim: the dimension to flip
#def flip_variable(variable_in, dim):
#    idx = [i for i in range(variable_in.shape[dim]-1, -1, -1)]
#    idx = Variable(torch.LongTensor(idx)).cuda()
#    reversed_variable = variable_in.index_select(dim, idx)
#    return reversed_variable

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.hidden_size = HIDDEN_SIZE
        self.n_layers = 1
        self.dropout = 0.5

        self.layer0 = nn.Sequential(
                nn.Conv2d(1, 32, 3),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(2))
        self.layer1 = nn.Sequential(
                nn.Conv2d(32, 64, 3),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                nn.Conv2d(64, 128, 3),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2))

        self.output_proj = nn.Linear((((height//2-1)//2-1)//2-1)*128*STEP, self.hidden_size)
        if Bi_GRU:
            self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers, dropout=self.dropout, bidirectional=True)
            self.output_comb = nn.Linear(self.hidden_size*2, self.hidden_size)
        else:
            self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers, dropout=self.dropout, bidirectional=False)
        #self.hidden_proj = nn.Linear(self.hidden_size*2, self.hidden_size)

    # (32, 1, 80, 1400)
    def forward(self, in_data, hidden=None):
        #if type(in_data.data) != torch.cuda.FloatTensor:
        #    in_data = in_data.float()
        #    print('Occur torch.DoubleTensor, it has been changed to FloatTensor automatically')
        batch_size = in_data.shape[0]
        out = self.layer0(in_data) #(32, 32, 39, 699)
        out = self.layer1(out) # (32, 64, 18, 348)
        out = self.layer2(out) # (32, 128, 8, 173)
        out = out.permute(3, 0, 2, 1) # (width, batch, height, channels)
        out.contiguous()
        out = out.view(-1, batch_size, (((height//2-1)//2-1)//2-1)*128) # (t, b, f) (173, 32, 1024)
        time_step, batch_size, n_feature = out.shape[0], out.shape[1], out.shape[2]
        out_short = Variable(torch.zeros(time_step//STEP, batch_size, n_feature*STEP)).cuda() # t//STEP, b, f*STEP
        for i in range(0, time_step//STEP):
            part_out = [out[j] for j in range(i*STEP, (i+1)*STEP)]
            # reverse the image feature map
            #out_short[time_step//STEP - i - 1] = torch.cat(part_out, 1) # b, f*STEP
            out_short[i] = torch.cat(part_out, 1) # b, f*STEP

        out = self.output_proj(out_short) # t//STEP, b, hidden_size
        #last_hidden = Variable(torch.zeros(batch_size, self.hidden_size)).cuda()
        #return out, last_hidden

        #self.gru.flatten_parameters() ## multi-GPU add this @@@
        output, hidden = self.gru(out, hidden)
       # # output: t, b, f*2     hidden: 2, b, f
        if Bi_GRU:
            output = self.output_comb(output) # t, b, f
            #hidden = hidden.permute(1, 0, 2) # b, 2, f
            #hidden = hidden.contiguous().view(batch_size, -1) # b, f*2
            #hidden = self.hidden_proj(hidden) # b, f

        return output, hidden.squeeze(0)
        #hidden_new = (output[0] + output[1])/2
        #return output, hidden_new # (t,b,f)  (b,f)

# Standard Bahdanau Attention
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(dim=1)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)
        #self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.hidden_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.encoder_output_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    # hidden: b, f  encoder_output: t, b, f
    def forward(self, hidden, encoder_output):
        encoder_output = encoder_output.transpose(0, 1) # b, t, f
        attn_energy = self.score(hidden, encoder_output) # b, t
        return self.softmax(attn_energy).unsqueeze(2) # b, t, 1

    # hidden: 1, batch, features
    # encoder_output: batch, time_step, features
    def score(self, hidden, encoder_output):
        hidden_attn = self.hidden_proj(hidden) # 1, b, f
        hidden_attn = hidden_attn.permute(1, 0, 2) # batch, 1, features
        encoder_output_attn = self.encoder_output_proj(encoder_output)
        res_attn = self.tanh(encoder_output_attn + hidden_attn) # b, t, f
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
        self.hidden_size = HIDDEN_SIZE
        self.embed_size = EMBEDDING_SIZE
        self.n_layers = 1

        self.embedding = nn.Embedding(vocab_size, self.embed_size)
        self.dropout = 0.1
        self.attention = Attention(self.hidden_size)
        self.gru = nn.GRU(self.embed_size*5 + self.embed_size, self.hidden_size, self.n_layers, dropout=self.dropout)
        self.out = nn.Linear(self.hidden_size, vocab_size)
        #self.context_proj = nn.Linear((((width//2-1)//2-1)//2-1)//STEP*self.hidden_size, self.hidden_size)
        self.context_shrink = nn.Linear(self.hidden_size, self.embed_size*5) # !! trade-off between embedding and context

    # hidden: (32, 256)  encoder_output: (55, 32, 256)
    def forward(self, in_char, hidden, encoder_output):
        #embed_char = self.dropout(embed_char)

        attn_weights = self.attention(hidden, encoder_output) # b, t, 1
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
        context = self.context_shrink(context)

        #top1_random_list = []
        #for i in in_char:
        #    top1_random = np.random.choice(np.arange(vocab_size), p=F.softmax(i, dim=0).cpu().data.numpy())
        #    top1_random_list.append(top1_random.tolist())
        #top1_random_np = np.array(top1_random_list)

        top1 = in_char.topk(1)[1] # batch, 1
        #in_char_oh = Variable(one_hot(top1.cpu())).cuda()
        #in_char_oh = Variable(one_hot(torch.LongTensor([top1_random.tolist()]))).cuda()
        embed_char = self.embedding(top1) # batch,1,embed
        embed_char = embed_char.squeeze(1)

        in_dec = torch.cat((embed_char, context), 1) # 16, 557
        in_dec = in_dec.unsqueeze(0)
        #self.gru.flatten_parameters() ## Multi-GPU add this @@@
        output, latest_hidden = self.gru(in_dec, hidden) # 1,16,512   3,16,512  nn.GRU
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
    def forward(self, src, tar, teacher_forcing, train=True):
        tar = tar.permute(1, 0) # time_s, batch
        batch_size = src.size(0)
        #max_len = tar.size(0) # <go> true_value <end>
        outputs = Variable(torch.zeros(output_max_len-1, batch_size, vocab_size), requires_grad=True) # (14, 32, 62) not save the first <GO>
        outputs = outputs.cuda()
        #src = Variable(src)
        out_enc, hidden_enc = self.encoder(src)
        # t,b,f    b,f
        global print_shape_flag
        if print_shape_flag:
            print(out_enc.shape, output_max_len)
            print_shape_flag = False

        output = one_hot(tar[0].data.cpu())
        output = Variable(output.cuda())
        attns = []

        hidden = hidden_enc.unsqueeze(0) # 1, batch, hidden_size
        for t in range(0, output_max_len-1): # max_len: groundtruth + <END>
            output, hidden, attn_weights = self.decoder(
                    output, hidden, out_enc)
            outputs[t] = output
            #top1 = output.data.topk(1)[1].squeeze()
            output = Variable(one_hot(tar[t+1].data.cpu()).cuda() if train and teacher_forcing else output.data)
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

def visualizeAttn(img, attn, epoch, batches):
    img *= 255./img.max()
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
    cv2.imwrite('imgs/'+str(epoch)+'.jpg', output)

def writePredict(epoch, index, pred, flag): # [batch_size, vocab_size] * max_output_len
    if not os.path.exists('pred_logs'):
        os.makedirs('pred_logs')
    if flag == 'train':
        file_prefix = 'pred_logs/train_predict_seq.'
    elif flag == 'valid':
        file_prefix = 'pred_logs/valid_predict_seq.'
    elif flag == 'test':
        file_prefix = 'pred_logs/test_predict_seq.'

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

def writeLoss(loss_value, flag):
    if not os.path.exists('pred_logs'):
        os.makedirs('pred_logs')
    if flag == 'train':
        file_name = 'pred_logs/loss_train.log'
    elif flag == 'valid':
        file_name = 'pred_logs/loss_valid.log'
    elif flag == 'test':
        file_name = 'pred_logs/loss_test.log'
    with open(file_name, 'a') as f:
        f.write(str(loss_value))
        f.write(' ')

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

learning_rate = 1e-3
encoder = Encoder().cuda()
decoder = Decoder().cuda()
seq2seq = Seq2Seq(encoder, decoder).cuda()
#seq2seq = torch.nn.DataParallel(seq2seq_pre, device_ids=[0, 1])
if CurriculumLearning:
    seq2seq.load_state_dict(torch.load(MODEL_FILE)) #load
opt = optim.Adam(seq2seq.parameters(), lr=learning_rate)
#opt = optim.SGD(seq2seq.parameters(), lr=learning_rate, momentum=0.9)
#opt = optim.RMSprop(seq2seq.parameters(), lr=learning_rate, momentum=0.9)

#scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=1)
scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[20, 400], gamma=0.1)
epochs = 500
logger = Logger('logs_tensorboard')
for epoch in range(epochs):
    data_train, data_valid, data_test = loadData.loadData() # reload to shuffle train data
    #data_train, data_valid, data_test = loadData.loadData_sample() # reload to shuffle train data
    total_loss = 0
    start = time.time()
    scheduler.step()
    lr = scheduler.get_lr()[0]
    count_update = 0 # opt

    for i in range(len(data_train)//BATCH_SIZE):
        logger.add_scalar('learning_rate', lr, 'train')
        data = data_train[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
        train_index, train_in, train_out, train_out_mask, train_in_len = data['index_sa'], data['input_sa'], data['output_sa'], data['out_len_sa'], data['in_len_sa']
        # train_in: batch, height, width  train_out: batch, time_s
        train_in, train_out = torch.from_numpy(np.array(train_in, dtype='float32')), torch.from_numpy(np.array(train_out, dtype='int64'))
        train_in = train_in.unsqueeze(1)
        train_in, train_out = Variable(train_in).cuda(), Variable(train_out).cuda()
        #if epoch < epochs//3:
        #    ratio = 0.9
        #    is_teacher = random.random() < ratio
        #elif epoch > (epochs - epochs//3):
        #    ratio = 0.1
        #    is_teacher = random.random() < ratio
        #else:
        #    ratio = 0.5
        #    is_teacher = random.random() < ratio
        ##is_teacher = False # remove the teacher forcing method
        is_teacher = random.random() < 0.5

        logger.add_scalar('is_teacher', float(is_teacher), 'train')

        output, attn_weights = seq2seq(train_in, train_out, teacher_forcing=is_teacher, train=True) # (100-1, 32, 62+1)
        writePredict(epoch, train_index, output, 'train')
        #train_label = Variable(train_out[1:-1].contiguous().view(-1))#remove<GO> and <END>
        #output_l = output[:-1].view(-1, vocab_size) # remove last <EOS>
        train_label = train_out.permute(1, 0)[1:].contiguous().view(-1)#remove<GO>
        output_l = output.view(-1, vocab_size) # remove last <EOS>

        loss = F.cross_entropy(output_l.view(-1, vocab_size),
                               train_label, ignore_index=tokens['PAD_TOKEN'])
        #loss = F.cross_entropy(output_l.view(-1, vocab_size), train_label)
        #if count_update == 0:
        #    opt.zero_grad()
        #loss.backward()
        #count_update += 1
        ##clip_grad_norm(seq2seq.parameters(), grad_clip)
        #if count_update == 64:
        #    opt.step()
        #    count_update = 0
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.data[0]
        logger.add_scalar('train_loss', loss.data[0], 'train')
        logger.step_train()

    if epoch%10 == 0:
        torch.save(seq2seq.state_dict(), 'models/seq2seq-%d.model'%epoch)
    total_loss /= (len(data_train)//BATCH_SIZE)
    writeLoss(total_loss, 'train')
    print('epoch %d/%d, loss=%.3f, lr=%.15f, time=%.3f' % (epoch, epochs, total_loss, lr, time.time()-start))

    total_loss_t = 0
    start_t = time.time()
    for i in range(len(data_valid)//BATCH_SIZE):
        data_t = data_valid[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
        test_index, test_in, test_out, test_out_len, test_in_len = data_t['index_sa'], data_t['input_sa'], data_t['output_sa'], data_t['out_len_sa'], data_t['in_len_sa']
        test_in, test_out = preProcess(test_in, test_out)
        test_in, test_out = Variable(test_in).cuda(), Variable(test_out).cuda()
        output_t, attn_weights_t = seq2seq(test_in, test_out, teacher_forcing=False, train=False)
        writePredict(epoch, test_index, output_t, 'valid')
        test_label = test_out.permute(1, 0)[1:].contiguous().view(-1)
        loss_t = F.cross_entropy(output_t.view(-1, vocab_size),
                                 test_label)
        total_loss_t += loss_t.data[0]

        if i == 0:
            # (32,1,80,460)->(80,460)  [(32,55),...]->[(55),...]
            visualizeAttn(test_in.data[4,0], [j[4] for j in attn_weights_t], epoch, len(data_valid)//BATCH_SIZE)
        logger.add_scalar('valid_loss', loss_t.data[0], 'valid')
        logger.step_valid()
    total_loss_t /= len(data_valid)//BATCH_SIZE
    writeLoss(total_loss_t, 'valid')
    print('  Valid loss=%.3f, time=%.3f' % (total_loss_t, time.time()-start_t))

    total_loss_t = 0
    start_t = time.time()
    for i in range(len(data_test)//BATCH_SIZE):
        data_t = data_test[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
        test_index, test_in, test_out, test_out_len, test_in_len = data_t['index_sa'], data_t['input_sa'], data_t['output_sa'], data_t['out_len_sa'], data_t['in_len_sa']
        test_in, test_out = preProcess(test_in, test_out)
        test_in, test_out = Variable(test_in).cuda(), Variable(test_out).cuda()
        output_t, attn_weights_t = seq2seq(test_in, test_out, teacher_forcing=False, train=False)
        writePredict(epoch, test_index, output_t, 'test')
        test_label = test_out.permute(1, 0)[1:].contiguous().view(-1)
        loss_t = F.cross_entropy(output_t.view(-1, vocab_size),
                                 test_label)
        total_loss_t += loss_t.data[0]
        logger.add_scalar('test_loss', loss_t.data[0], 'test')
        logger.step_test()
        #if i == 0:
            # (32,1,80,460)->(80,460)  [(32,55),...]->[(55),...]
            #visualizeAttn(test_in.data[0,0], [j[0] for j in attn_weights_t], epoch)
    total_loss_t /= len(data_test)//BATCH_SIZE
    writeLoss(total_loss_t, 'test')
    print('  Test  loss=%.3f, time=%.3f' % (total_loss_t, time.time()-start_t))

