import torch.utils.data as D
import cv2
import numpy as np

FLIP = True # flip the image
#BATCH_SIZE = 64
OUTPUT_MAX_LEN = 25 # max-word length is 21  This value should be larger than 21+2 (<GO>+groundtruth+<END>)
IMG_HEIGHT = 64
IMG_WIDTH = 400
baseDir = '/home/lkang/datasets/iam_laia_words/'

def labelDictionary():
    labels = ['3', 't', 'E', '"', '5', 'G', 'Y', 'H', 'B', '(', 'O', '?', '/', 'F', '9', 'v', 'J', 'T', "'", '1', 'Z', 's', 'r', 'U', 'I', 'o', ':', 'Q', 'q', 'R', 'm', 'A', 'k', ';', 'W', '-', 'y', '2', 'w', ')', 'P', '+', '8', 'M', 'i', '7', '!', 'C', ' ', '0', 'K', 'p', 'z', '*', 'x', 'j', '6', 'b', '#', '.', 'X', '&', 'l', 'h', 'd', 'n', 'V', 'a', 'S', 'N', 'e', 'c', 'g', 'u', '4', 'f', ',', 'L', 'D']
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
            img, img_width = self.readImage(i[0], flip=FLIP)
            label, label_mask = self.label_padding(i[1], num_tokens)
            file_name.append(i[0])
            in_data.append(img)
            in_len.append(img_width)
            out_data.append(label)
            out_data_mask.append(label_mask)
        return {'index_sa': file_name, 'input_sa': in_data, 'output_sa': out_data, 'in_len_sa': in_len, 'out_len_sa': out_data_mask}

    def __len__(self):
        return len(self.file_label)

    def readImage(self, file_name, flip):
        url = baseDir + 'lines_h128/' + file_name + '.jpg'
        img = cv2.imread(url, 0)
        rate = float(IMG_HEIGHT) / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1]*rate), IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        ret, mask = cv2.threshold(img, 175, 255, cv2.THRESH_BINARY_INV)
        binary = mask/255.
        img_width = binary.shape[-1]
        outImg = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='float32')
        if img_width > IMG_WIDTH:
            outImg = binary[:, :IMG_WIDTH]
            img_width = IMG_WIDTH
        else:
            outImg[:, :img_width] = binary
        if flip:
            outImg = np.flip(outImg, 1)
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
    with open(baseDir+'label_all_rm_err.txt', 'r') as f:
        data = f.readlines()
        #print('---Read %d words' % len(data))
        file_label = [i[:-1].split(' ') for i in data]

    total_num = len(file_label)
    file_label_train = file_label[:int(total_num*0.8)]
    np.random.shuffle(file_label_train)
    data_train = IAM_words(file_label_train)
    data_valid = IAM_words(file_label[int(total_num*0.8):int(total_num*0.9)])
    data_test = IAM_words(file_label[int(total_num*0.9):])
    return data_train, data_valid, data_test

def loadData_sample():
    with open(baseDir+'label_all_rm_err.txt', 'r') as f:
        data = f.readlines()
        #print('---Read %d words' % len(data))
        file_label = [i[:-1].split(' ') for i in data]

    file_label_train = file_label[:512]
    np.random.shuffle(file_label_train)
    data_train = IAM_words(file_label_train)
    data_valid = IAM_words(file_label[512:1024])
    data_test = IAM_words(file_label[1024:1280])
    return data_train, data_valid, data_test

#train_loader = D.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
#valid_loader = D.DataLoader(data_valid, batch_size=BATCH_SIZE, pin_memory=True)
#test_loader = D.DataLoader(data_test, batch_size=BATCH_SIZE, pin_memory=True)

if __name__ == '__main__':
    data_train, data_valid, data_test = loadData()
    print(data_train[0:2])
