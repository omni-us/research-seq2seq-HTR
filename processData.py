MNIST = False
if MNIST:
    import mnist as IAM
else:
    #import IAM_data as IAM
    import IAM_data_words as IAM
import numpy as np
import os

if MNIST:
    OUTPUT_MAX_LEN_TEXTLINE = IAM.MAX_DIGITS_TOTAL
else:
    # OUTPUT_MAX_LEN_TEXTLINE = 80 # textline
    OUTPUT_MAX_LEN_TEXTLINE = 20 # word
BATCH_SIZE = 32
MAX_POOL_NUM = 3

class preProcess():
    def __init__(self):
        #n_classes, num_train, num_valid, num_test, self.datasets = IAM.getData(64, 64, 64)
        n_classes, num_train, num_valid, num_test, self.datasets = IAM.getData(None, None, None)
        self.output_max_len = OUTPUT_MAX_LEN_TEXTLINE
        max_pool_num = MAX_POOL_NUM
        self.tokens = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'PAD_TOKEN': 2}
        self.num_tokens = len(self.tokens.keys())
        self.height = IAM.IMG_HEIGHT
        self.width = IAM.IMG_WIDTH
        self.batch_size = BATCH_SIZE
        self.vocab_size = n_classes + self.num_tokens
        self.num_train = num_train
        self.num_valid = num_valid
        _, _, self.index2letter = IAM.labelDictionary()
        self.n_per_epoch = self.num_train // self.batch_size
        self.n_per_epoch_t = self.num_valid // self.batch_size
        self.total_data_train, self.total_data_valid = self.processData(max_pool_num)

    def label_padding(self, labels, num_tokens):
        new_labels = []
        new_label_len = []
        for l in labels:
            num = self.output_max_len - len(l) - 1
            new_label_len.append(len(l)+2)
            l = np.array(l) + num_tokens
            l = list(l)
            l = [self.tokens['GO_TOKEN']] + l + [self.tokens['END_TOKEN']]
            if not num == 0:
                l.extend([self.tokens['PAD_TOKEN']] * num) # replace PAD_TOKEN
            new_labels.append(l)

        def make_weights(seq_lens, output_max_len):
            new_out = []
            for i in seq_lens:
                ele = [1]*i + [0]*(output_max_len + 1 -i)
                new_out.append(ele)
            return new_out
        return new_labels, make_weights(new_label_len, self.output_max_len)

    def processData(self, max_pool_num):
        trainImg, seqLen_train, trainLabel, validationImg, seqLen_validation, validationLabel, testImg, seqLen_test, testLabel = self.datasets
        seqLen_train = self.proper_seq_len(seqLen_train, 2**max_pool_num)
        seqLen_validation = self.proper_seq_len(seqLen_validation, 2**max_pool_num)
        seqLen_test = self.proper_seq_len(seqLen_test, 2**max_pool_num)
        trainLabel, trainLabel_mask = self.label_padding(trainLabel, self.num_tokens)
        validationLabel, validationLabel_mask = self.label_padding(validationLabel, self.num_tokens)
        total_data_train = []
        for i in range(self.num_train):
            data_train = dict()
            data_train['index'] = i
            data_train['img'] = trainImg[i]
            data_train['img_len'] = seqLen_train[i]
            data_train['label'] = trainLabel[i]
            data_train['label_mask'] = trainLabel_mask[i]
            total_data_train.append(data_train)

        total_data_valid = []
        for i in range(self.num_valid):
            data_valid = dict()
            data_valid['index'] = i
            data_valid['img'] = validationImg[i]
            data_valid['img_len'] = seqLen_validation[i]
            data_valid['label'] = validationLabel[i]
            data_valid['label_mask'] = validationLabel_mask[i]
            total_data_valid.append(data_valid)
        return total_data_train, total_data_valid

    def shuffle(self):
        np.random.shuffle(self.total_data_train)

    # data: [{'index':, 'img':, 'img_len':, 'label':, 'label_mask':}]
    def createGT(self, train=True):
        if not os.path.exists('pred_logs'):
            os.makedirs('pred_logs')
        if train:
            file_name = 'pred_logs/train_groundtruth.dat'
            #total_num = self.num_train
            data = self.total_data_train
        else:
            file_name = 'pred_logs/test_groundtruth.dat'
            #total_num = self.num_valid
            data = self.total_data_valid
        with open(file_name, 'w') as f:
            #num = total_num - (total_num%self.batch_size)
            #for i in range(num):
            #    element = data[i]
            for element in data:
                f.write(str(element['index'])+' ')
                for i in element['label'][1:]: # remove the first <GO>
                    if i == self.tokens['END_TOKEN']:
                        break
                    else:
                        if i == self.tokens['GO_TOKEN']:
                            f.write('<GO>')
                        elif i == self.tokens['PAD_TOKEN']:
                            f.write('<PAD>')
                        else:
                            f.write(self.index2letter[i-self.num_tokens])
                f.write('\n')

    def sampler(self): # should be shuffled before call this func
        data_train = self.total_data_train
        batches = self.num_train // self.batch_size
        while True:
            for i in range(batches):
                data_slice = data_train[i*self.batch_size: (i+1)*self.batch_size]
                index = []
                in_data = []
                out_data = []
                out_data_mask = []
                in_len = []
                for d in data_slice:
                    index.append(d['index'])
                    in_data.append(d['img'])
                    out_data.append(d['label'])
                    out_data_mask.append(d['label_mask'])
                    in_len.append(d['img_len'])
                yield {'index_sa': index, 'input_sa': in_data, 'output_sa': out_data, 'out_len_sa': out_data_mask, 'in_len_sa': in_len}


    def sampler_t(self):
        data_valid = self.total_data_valid
        batches = self.num_valid // self.batch_size
        while True:
            for i in range(batches):
                data_slice = data_valid[i*self.batch_size: (i+1)*self.batch_size]
                index = []
                in_data = []
                out_data = []
                out_data_mask = []
                in_len = []
                for d in data_slice:
                    index.append(d['index'])
                    in_data.append(d['img'])
                    out_data.append(d['label'])
                    out_data_mask.append(d['label_mask'])
                    in_len.append(d['img_len'])
                yield {'index_sa_t': index, 'input_sa_t': in_data, 'output_sa_t': out_data, 'out_len_sa_t': out_data_mask, 'in_len_sa_t': in_len}


    def proper_seq_len(self, seqLen, timeRatio):
        return [int(l/timeRatio) for l in seqLen]

if __name__ == '__main__':
    dataModel = preProcess()
    dataModel.createGT(True)
    dataModel.createGT(False)
    sample = dataModel.sampler()
    data = sample.__next__()
    print(len(data['output_sa']))
    print(data['output_sa'])

