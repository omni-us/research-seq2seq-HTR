import torch
from torch.autograd import Variable
import torch.nn.functional as F
#import loadData2 as loadData
import loadData
import numpy as np
import time
import os
import cv2
#import argparse
from models.encoder import Encoder
from models.decoder import Decoder
from models.attention import TroAttention
from models.seq2seq import Seq2Seq

print(time.ctime())
#parser = argparse.ArgumentParser(description='seq2seq net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#parser.add_argument('layers', type=int, help='decoder layer numbers')
#args = parser.parse_args()

torch.cuda.set_device(1)
BATCH_SIZE = 88
DECODER_LAYER = 1
Bi_GRU = True

CON_STEP = None # CON_STEP = 4 # encoder output squeeze step
HIDDEN_SIZE_ENC = 512
HIDDEN_SIZE_DEC = 512
EMBEDDING_SIZE = 60 # IAM
TRADEOFF_CONTEXT_EMBED = None # = 5 tradeoff between embedding:context vector = 1:5

HEIGHT = loadData.IMG_HEIGHT
WIDTH = loadData.IMG_WIDTH
output_max_len = loadData.OUTPUT_MAX_LEN
tokens = loadData.tokens
num_tokens = loadData.num_tokens
vocab_size = loadData.num_classes + num_tokens
index2letter = loadData.index2letter
FLIP = loadData.FLIP

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

def writeLoss(loss_value, flag):
    folder_name = 'pred_logs_layer_'+str(DECODER_LAYER)
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

def test(model_file):
    epoch = int(model_file.split('.')[0].split('-')[1])
    encoder = Encoder(HIDDEN_SIZE_ENC, HEIGHT, WIDTH, Bi_GRU, CON_STEP, FLIP).cuda()
    decoder = Decoder(HIDDEN_SIZE_DEC, EMBEDDING_SIZE, DECODER_LAYER, vocab_size, Bi_GRU, TroAttention, TRADEOFF_CONTEXT_EMBED).cuda()
    seq2seq = Seq2Seq(encoder, decoder, output_max_len, vocab_size).cuda()
    seq2seq.load_state_dict(torch.load(model_file))

    _, _, data_test = loadData.loadData() # reload to shuffle train data
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
    total_loss_t /= len(data_test)//BATCH_SIZE
    writeLoss(total_loss_t, 'test')
    print('TEST loss=%.3f, time=%.3f' % (total_loss_t, time.time()-start_t))


if __name__ == '__main__':
    test('save_weights/seq2seq-30.model.backup')
