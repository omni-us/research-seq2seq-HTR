import torch.utils.data as D
import cv2
import numpy as np
#from torchvision import transforms
import marcalAugmentor
#import Augmentor
#from torchsample.transforms import RangeNormalize
#import torch

# train data: 46945
# valid data: 6445
# test data: 13752

RM_BACKGROUND = True
FLIP = False # flip the image
#BATCH_SIZE = 64
OUTPUT_MAX_LEN = 25 # max-word length is 21  This value should be larger than 21+2 (<GO>+groundtruth+<END>)
IMG_HEIGHT = 64
KEEP_RATIO = True
if KEEP_RATIO:
    IMG_WIDTH = 1011 # m01-084-07-00 max_length
else:
    IMG_WIDTH = 256 # img_width < 256: padding   img_width > 256: resize to 256
baseDir = '/home/lkang/datasets/iam_final_words/'

#global_filename = []
#global_length = []
def labelDictionary():
    #labels = ['_', '3', 't', 'E', '"', '5', 'G', 'Y', 'H', 'B', '(', 'O', '?', '/', 'F', '9', 'v', 'J', 'T', "'", '1', 'Z', 's', 'r', 'U', 'I', 'o', ':', 'Q', 'q', 'R', 'm', 'A', 'k', ';', 'W', '-', 'y', '2', 'w', ')', 'P', '+', '8', 'M', 'i', '7', '!', 'C', ' ', '0', 'K', 'p', 'z', '*', 'x', 'j', '6', 'b', '#', '.', 'X', '&', 'l', 'h', 'd', 'n', 'V', 'a', 'S', 'N', 'e', 'c', 'g', 'u', '4', 'f', ',', 'L', 'D']
    labels = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    letter2index = {label: n for n, label in enumerate(labels)}
    index2letter = {v: k for k, v in letter2index.items()}
    return len(labels), letter2index, index2letter

num_classes, letter2index, index2letter = labelDictionary()
tokens = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'PAD_TOKEN': 2}
num_tokens = len(tokens.keys())

class IAM_words(D.Dataset):
    def __init__(self, file_label, augmentation=True):
        self.file_label = file_label
        self.output_max_len = OUTPUT_MAX_LEN
        self.augmentation = augmentation

        self.transformer = marcalAugmentor.augmentor
        ## image augmentation for training
        #pipeline = Augmentor.Pipeline()
        #pipeline.zoom(1.0, 0.75, 1.0)
        #pipeline.rotate(1.0, 0.2, 0.2)
        #pipeline.shear(1.0, 2, 2)
        #pipeline.skew(1.0, 0.1)
        ##pipeline.random_distortion(1.0, 3, 3, 3) #b04-081-02-12 64*4
        ## if grid_width > 4, then it will occur ZeroDivisionError: float division by zero
        #self.transform = transforms.Compose([
        #                    transforms.ToPILImage(),
        #                    pipeline.torch_transform(),
        #                    #transforms.ToTensor(),
        #                    ])
        ##self.norm = RangeNormalize(0, 1)

    def __getitem__(self, index):
        file_name = [] # as index
        in_data = []
        in_len = []
        out_data = []
        out_data_mask = []
        sub_file_label = self.file_label[index]
        for i in sub_file_label:
            if KEEP_RATIO:
                img, img_width = self.readImage_keepRatio(i[0], flip=FLIP)
            else:
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

    def readImage_keepRatio(self, file_name, flip):
        if RM_BACKGROUND:
            file_name, thresh = file_name.split(',')
            thresh = int(thresh)
        url = baseDir + 'words/' + file_name + '.png'
        img = cv2.imread(url, 0)
        size = img.shape[0] * img.shape[1]
        # c04-066-01-08.png 4*3, for too small images do not augment
        if self.augmentation and size > 100: # augmentation for training data
            img_new = self.transformer(img)
            if img_new.shape[0] ==0 or img_new.shape[1] == 0:
                print(file_name, img_new.shape)
                if RM_BACKGROUND:
                    img[img>thresh] = 255
                img = 255 - img
            else:
                img = img_new
        else: # evaluation
            if RM_BACKGROUND:
                img[img>thresh] = 255
            img = 255 - img
        img = img/255. #float64
        img = img.astype('float32')
        rate = float(IMG_HEIGHT) / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1]*rate), IMG_HEIGHT), interpolation=cv2.INTER_AREA)

        img_width = img.shape[-1]

        if flip: # because of using pack_padded_sequence, first flip, then pad it
            img = np.flip(img, 1)
        outImg = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='float32')
        if img_width > IMG_WIDTH:
            #global global_filename, global_length ######
            #global_filename.append(file_name)
            #global_length.append(img_width)
            outImg = img[:, :IMG_WIDTH]
            img_width = IMG_WIDTH
        else:
            outImg[:, :img_width] = img
        return outImg, img_width

    def readImage(self, file_name, flip):
        if RM_BACKGROUND:
            file_name, thresh = file_name.split(',')
            thresh = int(thresh)
        url = baseDir + 'words/' + file_name + '.png'
        img = cv2.imread(url, 0)
        size = img.shape[0] * img.shape[1]
        if self.augmentation and size > 100: # augmentation for training data
            img = self.transformer(img)
        else: # evaluation
            if RM_BACKGROUND:
                img[img>thresh] = 255
                img = 255 - img
        img = img/255. #float64
        img = img.astype('float32')
        rate = float(IMG_HEIGHT) / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1]*rate), IMG_HEIGHT), interpolation=cv2.INTER_AREA)

        img_width = img.shape[-1]

        if flip:
            img = np.flip(img, 1)

        if img_width > IMG_WIDTH:
            outImg = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            img_width = IMG_WIDTH
        else:
            outImg = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='float32')
            outImg[:, :img_width] = img

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
    if RM_BACKGROUND:
        gt_tr = 'iam_word_gt_final.train.thresh'
        gt_va = 'iam_word_gt_final.valid.thresh'
        gt_te = 'iam_word_gt_final.test.thresh'
    else:
        gt_tr = 'iam_word_gt_final.train'
        gt_va = 'iam_word_gt_final.valid'
        gt_te = 'iam_word_gt_final.test'

    with open(baseDir+gt_tr, 'r') as f_tr:
        data_tr = f_tr.readlines()
        file_label_tr = [i[:-1].split(' ') for i in data_tr]

    with open(baseDir+gt_va, 'r') as f_va:
        data_va = f_va.readlines()
        file_label_va = [i[:-1].split(' ') for i in data_va]

    with open(baseDir+gt_te, 'r') as f_te:
        data_te = f_te.readlines()
        file_label_te = [i[:-1].split(' ') for i in data_te]

    #total_num_tr = len(file_label_tr)
    #total_num_va = len(file_label_va)
    #total_num_te = len(file_label_te)
    #print('Loading training data ', total_num_tr)
    #print('Loading validation data ', total_num_va)
    #print('Loading testing data ', total_num_te)

    np.random.shuffle(file_label_tr)
    data_train = IAM_words(file_label_tr, augmentation=True)
    data_valid = IAM_words(file_label_va, augmentation=False)
    data_test = IAM_words(file_label_te, augmentation=False)
    return data_train, data_valid, data_test

def loadData_sample():
    with open(baseDir+'iam_word_gt_final.train', 'r') as f_tr:
        data_tr = f_tr.readlines()[:512]
        file_label_tr = [i[:-1].split(' ') for i in data_tr]

    with open(baseDir+'iam_word_gt_final.valid', 'r') as f_va:
        data_va = f_va.readlines()[:128]
        file_label_va = [i[:-1].split(' ') for i in data_va]

    with open(baseDir+'iam_word_gt_final.test', 'r') as f_te:
        data_te = f_te.readlines()[:256]
        file_label_te = [i[:-1].split(' ') for i in data_te]

    total_num_tr = len(file_label_tr)
    total_num_va = len(file_label_va)
    total_num_te = len(file_label_te)
    print('Loading training data ', total_num_tr)
    print('Loading validation data ', total_num_va)
    print('Loading testing data ', total_num_te)

    np.random.shuffle(file_label_tr)
    data_train = IAM_words(file_label_tr)
    data_valid = IAM_words(file_label_va)
    data_test = IAM_words(file_label_te)
    return data_train, data_valid, data_test

#train_loader = D.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
#valid_loader = D.DataLoader(data_valid, batch_size=BATCH_SIZE, pin_memory=True)
#test_loader = D.DataLoader(data_test, batch_size=BATCH_SIZE, pin_memory=True)

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

    maxLen = max(data_test[:]['in_len_sa'])
    num = data_test[:]['in_len_sa'].index(maxLen)
    idx = data_test[:]['index_sa'][num]
    print('[Test] Max length: ', maxLen, idx)
    print('tiempo de uso: %.3f' % (time.time()-start))

    #print(global_filename)
    #print(global_length)

