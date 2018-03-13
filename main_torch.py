import torch
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
#import loadData2 as loadData
import loadData
import numpy as np
import time
import os
import cv2
from LogMetric import Logger
#import argparse
from models.encoder import Encoder
from models.decoder import Decoder
from models.attention import locationAttention as Attention
from models.seq2seq import Seq2Seq

print(time.ctime())
#parser = argparse.ArgumentParser(description='seq2seq net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('layers', type=int, help='decoder layer numbers')
#args = parser.parse_args()

torch.cuda.set_device(1)
Bi_GRU = True
VISUALIZE_TRAIN = False
TF_LOG = False

BATCH_SIZE = 80
EARLY_STOP_EPOCH = 20 # None: no early stopping
DECODER_LAYER = 1
HIDDEN_SIZE_ENC = 512
HIDDEN_SIZE_DEC = 512
CON_STEP = None # CON_STEP = 4 # encoder output squeeze step
CurriculumLearning = False
MODEL_FILE = 'save_weights/seq2seq-170.model.backup'
EMBEDDING_SIZE = 60 # IAM
TRADEOFF_CONTEXT_EMBED = None # = 5 tradeoff between embedding:context vector = 1:5
learning_rate = 1e-4
lr_milestone = [60, 100]
lr_gamma = 0.5
FREEZE = False
freeze_milestone = [65, 90]
TEACHER_FORCING = True
MODEL_SAVE_EPOCH = 5
TEST = False # True: train, valid, test  False: train, valid

HEIGHT = loadData.IMG_HEIGHT
WIDTH = loadData.IMG_WIDTH
output_max_len = loadData.OUTPUT_MAX_LEN
tokens = loadData.tokens
num_tokens = loadData.num_tokens
vocab_size = loadData.num_classes + num_tokens
index2letter = loadData.index2letter
FLIP = loadData.FLIP

def visualizeAttn(img, first_img_real_len, attn, epoch, batches, name):
    folder_name = 'imgs'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
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
    cv2.imwrite(folder_name+'/'+name+'_'+str(epoch)+'.jpg', output)

def writePredict(epoch, index, pred, flag): # [batch_size, vocab_size] * max_output_len
    folder_name = 'pred_logs'
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

def writeLoss(loss_value, flag):
    folder_name = 'pred_logs'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if flag == 'train':
        file_name = folder_name + '/loss_train.log'
    elif flag == 'valid':
        file_name = folder_name + '/loss_valid.log'
    elif flag == 'test':
        file_name = folder_name + '/loss_test.log'
    with open(file_name, 'a') as f:
        f.write(str(loss_value))
        f.write(' ')

def preProcess(data_in, data_out):
    data_in = np.array(data_in)
    data_in = np.expand_dims(data_in, axis=1) # (batch, 1, height, width)
    data_in = torch.FloatTensor(data_in)

    data_out = np.array(data_out)
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

def train():
    encoder = Encoder(HIDDEN_SIZE_ENC, HEIGHT, WIDTH, Bi_GRU, CON_STEP, FLIP).cuda()
    decoder = Decoder(HIDDEN_SIZE_DEC, EMBEDDING_SIZE, DECODER_LAYER, vocab_size, Bi_GRU, Attention, TRADEOFF_CONTEXT_EMBED).cuda()
    seq2seq = Seq2Seq(encoder, decoder, output_max_len, vocab_size).cuda()
    if CurriculumLearning:
        seq2seq.load_state_dict(torch.load(MODEL_FILE)) #load
    opt = optim.Adam(seq2seq.parameters(), lr=learning_rate)
    #opt = optim.SGD(seq2seq.parameters(), lr=learning_rate, momentum=0.9)
    #opt = optim.RMSprop(seq2seq.parameters(), lr=learning_rate, momentum=0.9)

    #scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=1)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=lr_milestone, gamma=lr_gamma)
    epochs = 5000000
    if TF_LOG:
        logger = Logger('logs_tensorboard')
    if EARLY_STOP_EPOCH is not None:
        min_loss = 1e3
        min_loss_index = 0
        min_loss_count = 0

    for epoch in range(epochs):
        data_train, data_valid, data_test = loadData.loadData() # reload to shuffle train data
        #data_train, data_valid, data_test = loadData.loadData_sample() # reload to shuffle train data
        total_loss = 0
        start = time.time()
        scheduler.step()
        lr = scheduler.get_lr()[0]
        #count_update = 0 # opt

        if FREEZE:
            if epoch > freeze_milestone[0] and epoch < freeze_milestone[1]:
                for param in decoder.parameters():
                    param.requires_grad = False
            else:
                for param in decoder.parameters():
                    param.requires_grad = True

        if TEACHER_FORCING:
            if epoch < 50:
                teacher_rate = 0.5
            elif epoch < 150:
                teacher_rate = (50 - (epoch-50)//2) / 100.
            else:
                teacher_rate = 0.

        seq2seq.train()
        for i in range(len(data_train)//BATCH_SIZE + 1):
            #print('batch %d / %d' % (i, len(data_train)//BATCH_SIZE))
            if TF_LOG:
                logger.add_scalar('learning_rate', lr, 'train')
            data = data_train[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
            train_index, train_in, train_in_len, train_out = sort_batch(data)
            # train_in: batch, height, width  train_out: batch, time_s
            #train_in, train_out = torch.from_numpy(np.array(train_in, dtype='float32')), torch.from_numpy(np.array(train_out, dtype='int64'))
            train_in = train_in.unsqueeze(1)
            train_in, train_out = Variable(train_in).cuda(), Variable(train_out).cuda()
            output, attn_weights = seq2seq(train_in, train_out, train_in_len, teacher_rate=teacher_rate, train=True) # (100-1, 32, 62+1)
            writePredict(epoch, train_index, output, 'train')
            train_label = train_out.permute(1, 0)[1:].contiguous().view(-1)#remove<GO>
            output_l = output.view(-1, vocab_size) # remove last <EOS>

            if VISUALIZE_TRAIN:
                if i == 0:
                    visualizeAttn(train_in.data[0,0], train_in_len[0], [j[0] for j in attn_weights], epoch, len(data_train)//BATCH_SIZE, 'first')
                    visualizeAttn(train_in.data[4,0], train_in_len[0], [j[4] for j in attn_weights], epoch, len(data_train)//BATCH_SIZE, 'second')
                    visualizeAttn(train_in.data[6,0], train_in_len[0], [j[6] for j in attn_weights], epoch, len(data_train)//BATCH_SIZE, 'third')

            loss = F.cross_entropy(output_l.view(-1, vocab_size),
                                   train_label, ignore_index=tokens['PAD_TOKEN'])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.data[0]
            if TF_LOG:
                logger.add_scalar('train_loss', loss.data[0], 'train')
                logger.step_train()

        if epoch%MODEL_SAVE_EPOCH == 0:
            folder_weights = 'save_weights'
            if not os.path.exists(folder_weights):
                os.makedirs(folder_weights)
            torch.save(seq2seq.state_dict(), folder_weights+'/seq2seq-%d.model'%epoch)
        total_loss /= (len(data_train)//BATCH_SIZE)
        writeLoss(total_loss, 'train')
        print('epoch %d/%d, loss=%.3f, lr=%.8f, teacher_rate=%.3f, time=%.3f' % (epoch, epochs, total_loss, lr, teacher_rate, time.time()-start))

        seq2seq.eval()
        total_loss_t = 0
        start_t = time.time()
        for i in range(len(data_valid)//BATCH_SIZE + 1):
            data_t = data_valid[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
            test_index, test_in, test_in_len, test_out = sort_batch(data_t)
            test_in = test_in.unsqueeze(1)
            test_in, test_out = Variable(test_in, volatile=True).cuda(), Variable(test_out, volatile=True).cuda()
            output_t, attn_weights_t = seq2seq(test_in, test_out, test_in_len, teacher_rate=False, train=False)
            writePredict(epoch, test_index, output_t, 'valid')
            test_label = test_out.permute(1, 0)[1:].contiguous().view(-1)
            loss_t = F.cross_entropy(output_t.view(-1, vocab_size),
                                     test_label)
            total_loss_t += loss_t.data[0]

            if i == 0:
                # (32,1,80,460)->(80,460)  [(32,55),...]->[(55),...]
                visualizeAttn(test_in.data[0,0], test_in_len[0], [j[0] for j in attn_weights_t], epoch, len(data_valid)//BATCH_SIZE, 'first')
                visualizeAttn(test_in.data[2,0], test_in_len[0], [j[2] for j in attn_weights_t], epoch, len(data_valid)//BATCH_SIZE, 'second')
                visualizeAttn(test_in.data[36,0], test_in_len[0], [j[36] for j in attn_weights_t], epoch, len(data_valid)//BATCH_SIZE, 'third')
                visualizeAttn(test_in.data[41,0], test_in_len[0], [j[41] for j in attn_weights_t], epoch, len(data_valid)//BATCH_SIZE, 'forth')
            if TF_LOG:
                logger.add_scalar('valid_loss', loss_t.data[0], 'valid')
                logger.step_valid()
        total_loss_t /= len(data_valid)//BATCH_SIZE
        writeLoss(total_loss_t, 'valid')
        print('  Valid loss=%.3f, time=%.3f' % (total_loss_t, time.time()-start_t))

        if EARLY_STOP_EPOCH is not None:
            if total_loss_t < min_loss:
                min_loss = total_loss_t
                min_loss_index = epoch
                min_loss_count = 0
            else:
                min_loss_count += 1
            if min_loss_count >= EARLY_STOP_EPOCH:
                print('Early Stopping at: %d. Best epoch is: %d' % (epoch, min_loss_index))
                break

        if TEST:
            seq2seq.eval()
            total_loss_t = 0
            start_t = time.time()
            for i in range(len(data_test)//BATCH_SIZE + 1):
                data_t = data_test[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
                test_index, test_in, test_in_len, test_out = sort_batch(data_t)
                test_in = test_in.unsqueeze(1)
                test_in, test_out = Variable(test_in, volatile=True).cuda(), Variable(test_out, volatile=True).cuda()
                output_t, attn_weights_t = seq2seq(test_in, test_out, test_in_len, teacher_rate=False, train=False)
                writePredict(epoch, test_index, output_t, 'valid')
                test_label = test_out.permute(1, 0)[1:].contiguous().view(-1)
                loss_t = F.cross_entropy(output_t.view(-1, vocab_size),
                                         test_label)
                total_loss_t += loss_t.data[0]

                if i == 0:
                    # (32,1,80,460)->(80,460)  [(32,55),...]->[(55),...]
                    visualizeAttn(test_in.data[0,0], test_in_len[0], [j[0] for j in attn_weights_t], epoch, len(data_test)//BATCH_SIZE, 'first')
                    visualizeAttn(test_in.data[2,0], test_in_len[0], [j[2] for j in attn_weights_t], epoch, len(data_test)//BATCH_SIZE, 'second')
                    visualizeAttn(test_in.data[36,0], test_in_len[0], [j[36] for j in attn_weights_t], epoch, len(data_test)//BATCH_SIZE, 'third')
                    visualizeAttn(test_in.data[41,0], test_in_len[0], [j[41] for j in attn_weights_t], epoch, len(data_test)//BATCH_SIZE, 'forth')
                if TF_LOG:
                    logger.add_scalar('valid_loss', loss_t.data[0], 'valid')
                    logger.step_valid()
            total_loss_t /= len(data_test)//BATCH_SIZE
            writeLoss(total_loss_t, 'test')
            print('    TEST loss=%.3f, time=%.3f' % (total_loss_t, time.time()-start_t))

if __name__ == '__main__':
    train()
