import cv2
import time
import numpy as np

# height: 128
#>>> max(trainWidth)
#5031
#>>> max(valWidth)
#4379
#>>> max(testWidth)
#4347

#IMG_HEIGHT = 128
#IMG_WIDTH = 5000

IMG_HEIGHT = 64
IMG_WIDTH = 2500
baseDir = '/home/lkang/datasets/iam_laia/'

def labelDictionary():
    labels = ['3', 't', 'E', '"', '5', 'G', 'Y', 'H', 'B', '(', 'O', '?', '/', 'F', '9', 'v', 'J', 'T', "'", '1', 'Z', 's', 'r', 'U', 'I', 'o', ':', 'Q', 'q', 'R', 'm', 'A', 'k', ';', 'W', '-', 'y', '2', 'w', ')', 'P', '+', '8', 'M', 'i', '7', '!', 'C', ' ', '0', 'K', 'p', 'z', '*', 'x', 'j', '6', 'b', '#', '.', 'X', '&', 'l', 'h', 'd', 'n', 'V', 'a', 'S', 'N', 'e', 'c', 'g', 'u', '4', 'f', ',', 'L', 'D']
    letter2index = {label: n for n, label in enumerate(labels)}
    index2letter = {v: k for k, v in letter2index.items()}
    return len(labels), letter2index, index2letter

def readImage(url):
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
    return outImg, img_width

# a01-000u-00 A MOVE to stop Mr. Gaitskell from
def indexLabel(line):
    data = line.split(' ')
    index = data[0]
    return index, ' '.join(data[1:])

# home/lkang/datasets/iam_laia/lines_h128/a01-000u-00.jpg
def indexImage(line):
    data = line.split('/')
    index = data[-1].split('.')[0]
    return index, line

def getData(train_size=None, val_size=None, test_size=None):
    start = time.time()
    print('Load IAM dataset...')
    trainIx = []
    valIx = []
    testIx = []

    trainImage = []
    valImage = []
    testImage = []

    trainWidth = []
    valWidth = []
    testWidth = []

    trainLabel = [] # characters
    valLabel = []
    testLabel = []

    trainLabel2 = [] # index numbers
    valLabel2 = []
    testLabel2 = []

    with open(baseDir+'labels/tr.txt', 'r') as gt_train:
        count = 0
        for line in gt_train:
            index, label = indexLabel(line[:-1])
            if index == 'a05-116-09': #remove 1526th image
                continue              #
            count += 1
            trainIx.append(index)
            trainLabel.append(label)
            if train_size and count >= train_size:
                break

    with open(baseDir+'lists/tr_h128.lst', 'r') as img_train:
        count = 0
        for line in img_train:
            index, imgUrl = indexImage(line[:-1])
            if index == 'a05-116-09': #remove 1526th image
                continue              #
            if index != trainIx[count]:
                print('training image and label do not match!!!')
                exit()
            count += 1
            img, img_width = readImage(imgUrl)
            trainImage.append(img)
            trainWidth.append(img_width)
            if train_size and count >= train_size:
                break

    with open(baseDir+'labels/va.txt', 'r') as gt_val:
        count = 0
        for line in gt_val:
            index, label = indexLabel(line[:-1])
            count += 1
            valIx.append(index)
            valLabel.append(label)
            if val_size and count >= val_size:
                break

    with open(baseDir+'lists/va_h128.lst', 'r') as img_val:
        count = 0
        for line in img_val:
            index, imgUrl = indexImage(line[:-1])
            if index != valIx[count]:
                print('validation image and label do not match!!!')
                exit()
            count += 1
            img, img_width = readImage(imgUrl)
            valImage.append(img)
            valWidth.append(img_width)
            if val_size and count >= val_size:
                break

    with open(baseDir+'labels/te.txt', 'r') as gt_test:
        count = 0
        for line in gt_test:
            index, label = indexLabel(line[:-1])
            count += 1
            testIx.append(index)
            testLabel.append(label)
            if test_size and count >= test_size:
                break

    with open(baseDir+'lists/te_h128.lst', 'r') as img_test:
        count = 0
        for line in img_test:
            index, imgUrl = indexImage(line[:-1])
            if index != testIx[count]:
                print('testing image and label do not match!!!')
                exit()
            count += 1
            img, img_width = readImage(imgUrl)
            testImage.append(img)
            testWidth.append(img_width)
            if test_size and count >= test_size:
                break

    labelNum, letter2index, index2letter = labelDictionary()

    for label in trainLabel:
        idx = []
        for i in list(label):
            idx.append(letter2index[i])
        trainLabel2.append(idx)

    for label in valLabel:
        idx = []
        for i in list(label):
            idx.append(letter2index[i])
        valLabel2.append(idx)

    for label in testLabel:
        idx = []
        for i in list(label):
            idx.append(letter2index[i])
        testLabel2.append(idx)

    print('Done! time: %.3f' % (time.time()-start))

    return labelNum, len(trainImage), len(valImage), len(testImage), (trainImage, trainWidth, trainLabel2, valImage, valWidth, valLabel2, testImage, testWidth, testLabel2)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    labelNum, num_train, num_val, num_test, (trainImg, trainWidth, trainLabel, valImg, valWidth, valLabel, testImg, testWidth, testLabel) = getData(32, 8, 16)
    #labelNum, num_train, num_val, num_test, (trainImg, trainWidth, trainLabel, valImg, valWidth, valLabel, testImg, testWidth, testLabel) = getData()
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.title.set_text(str(trainLabel[4]))
    ax1.imshow(trainImg[4], cmap='gray')
    ax2.title.set_text(str(valLabel[4]))
    ax2.imshow(valImg[4], cmap='gray')
    ax3.title.set_text(str(testLabel[4]))
    ax3.imshow(testImg[4], cmap='gray')
    plt.show()

