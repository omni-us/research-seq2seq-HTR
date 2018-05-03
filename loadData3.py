import torch.utils.data as D
import cv2
import numpy as np
import random
import glob
import generate_text_Adria

'''
training data: 12729, validation data: 1415
'''
WORD_LEVEL = True

RM_BACKGROUND = True
FLIP = False # flip the image
#BATCH_SIZE = 64
IMG_HEIGHT = 64
if WORD_LEVEL:
    OUTPUT_MAX_LEN = 23 # max-word length is 21  This value should be larger than 21+2 (<GO>+groundtruth+<END>)
    IMG_WIDTH = 1011 # m01-084-07-00 max_length
    baseDir = '/home/lkang/datasets/synthetic_Pau/texts/raw_text_index.gt'
else:
    pass

ttfs = glob.glob('/home/lkang/datasets/synthetic_Pau/fonts/*.ttf')

def labelDictionary():
    labels = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    letter2index = {label: n for n, label in enumerate(labels)}
    index2letter = {v: k for k, v in letter2index.items()}
    return len(labels), letter2index, index2letter

num_classes, letter2index, index2letter = labelDictionary()
tokens = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'PAD_TOKEN': 2}
num_tokens = len(tokens.keys())

class IAM_words(D.Dataset):
    def __init__(self, file_label):
        self.file_label = file_label
        self.output_max_len = OUTPUT_MAX_LEN

    def __getitem__(self, index):
        file_name = [] # as index
        in_data = []
        in_len = []
        out_data = []
        out_data_mask = []
        sub_file_label = self.file_label[index]
        for i in sub_file_label:
            img, img_width = self.readImage_keepRatio(i[1], flip=FLIP)
            label, label_mask = self.label_padding(i[1], num_tokens)
            file_name.append(i[0])
            in_data.append(img)
            in_len.append(img_width)
            out_data.append(label)
            out_data_mask.append(label_mask)
        return {'index_sa': file_name, 'input_sa': in_data, 'output_sa': out_data, 'in_len_sa': in_len, 'out_len_sa': out_data_mask}

    def __len__(self):
        return len(self.file_label)

    def readImage_keepRatio(self, transcription, flip):
        count_ttf = 0
        while True:
            count_ttf += 1
            font = random.choice(ttfs)
            img = generate_text_Adria.generate_text(transcription, font, 75)
            if img.shape[0] == 0 or img.shape[1] == 0:
                continue
            else:
                break
            if count_ttf > 50:
                print('All 50 random fonts make error on: ', transcription)
                break

        rate = float(IMG_HEIGHT) / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1]*rate)+1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC) # INTER_AREA con error
        img = 255 - img
        img_width = img.shape[-1]

        if flip: # because of using pack_padded_sequence, first flip, then pad it
            img = np.flip(img, 1)

        if img_width > IMG_WIDTH:
            outImg = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            #outImg = img[:, :IMG_WIDTH]
            img_width = IMG_WIDTH
        else:
            outImg = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='uint8')
            outImg[:, :img_width] = img
        outImg = outImg/255. #float64
        outImg = outImg.astype('float32')
        return outImg, img_width

    def label_padding(self, labels, num_tokens):
        new_label_len = []
        ll = [letter2index[i] for i in labels]
        num = self.output_max_len - len(ll) - 2
        new_label_len.append(len(ll)+2)
        ll = np.array(ll) + num_tokens
        ll = list(ll)
        ll = [tokens['GO_TOKEN']] + ll + [tokens['END_TOKEN']]
        if not num == 0:
            ll.extend([tokens['PAD_TOKEN']] * num) # replace PAD_TOKEN

        def make_weights(seq_lens, output_max_len):
            new_out = []
            for i in seq_lens:
                ele = [1]*i + [0]*(output_max_len -i)
                new_out.append(ele)
            return new_out
        return ll, make_weights(new_label_len, self.output_max_len)

def loadData():
    with open(baseDir, 'r') as f_todo:
        data = f_todo.readlines()
        total_size = len(data)
        train_size = int(0.9 * total_size)
        #valid_size = total_size - train_size
        #print('training data: %d, validation data: %d' % (train_size, valid_size))

        data = [(i.split(' ')[0], i.split(' ')[1][:-1]) for i in data]
        idx_train = np.random.choice(total_size, train_size, replace=False).tolist()
        idx_valid = [i for i in range(total_size) if i not in idx_train]
        list_train = [data[i] for i in idx_train]
        list_valid = [data[i] for i in idx_valid]

    np.random.shuffle(list_train)
    data_train = IAM_words(list_train)
    data_valid = IAM_words(list_valid)
    data_test = data_valid
    return data_train, data_valid, data_test

if __name__ == '__main__':
    import time
    start = time.time()
    data_train, data_valid, data_test = loadData()
    maxLen = max(data_train[:]['in_len_sa'])
    num = data_train[:]['in_len_sa'].index(maxLen)
    idx = data_train[:]['index_sa'][num]
    print('[Train] Max length: ', maxLen, idx)

    maxLen = max(data_valid[:]['in_len_sa'])
    num = data_valid[:]['in_len_sa'].index(maxLen)
    idx = data_valid[:]['index_sa'][num]
    print('[Valid] Max length: ', maxLen, idx)

    #maxLen = max(data_test[:]['in_len_sa'])
    #num = data_test[:]['in_len_sa'].index(maxLen)
    #idx = data_test[:]['index_sa'][num]
    #print('[Test] Max length: ', maxLen, idx)
    #print('tiempo de uso: %.3f' % (time.time()-start))

    #print(global_filename)
    #print(global_length)

