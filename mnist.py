from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import random
import time

#MIN_DIGITS = 5
#MAX_DIGITS = 10
#MIN_DIGITS_T = 5
#MAX_DIGITS_T = 10
MIN_DIGITS = 5
MAX_DIGITS = 25
MIN_DIGITS_T = 5
MAX_DIGITS_T = 25
IMG_HEIGHT = 28
MAX_DIGITS_TOTAL = max(MAX_DIGITS, MAX_DIGITS_T) + 1 #<END>
IMG_WIDTH = IMG_HEIGHT*MAX_DIGITS_TOTAL

def labelDictionary():
    labels = list('0123456789')
    letter2index = {label: n for n, label in enumerate(labels)}
    index2letter = {v: k for k, v in letter2index.items()}
    return len(labels), letter2index, index2letter

def getData(train_num=None, val_num=None, test_num=None):
    start = time.time()
    print('Loading MNIST-seq dataset...')
    mnist = input_data.read_data_sets('.', one_hot=False)
    trainImg = mnist.train.images.reshape(-1, 28, 28)
    trainLabel = mnist.train.labels
    testImg = mnist.test.images.reshape(-1, 28, 28)
    testLabel = mnist.test.labels

    trainLines = []
    trainLabels = []
    trainWidth = []
    for i in range(10000):
        imgs = []
        labels = []
        zero_img = np.zeros((IMG_HEIGHT, IMG_HEIGHT), dtype='float32')
        num = int(random.random()*(MAX_DIGITS-MIN_DIGITS))+MIN_DIGITS
        trainWidth.append(num)
        for j in range(num): # 2-10 characters
            index = int(random.random()*55000)
            imgs.append(trainImg[index])
            labels.append(trainLabel[index])
        for x in range(MAX_DIGITS_TOTAL-num):
            imgs.append(zero_img)
        line = np.hstack(imgs)
        trainLines.append(line)
        trainLabels.append(labels)

    testLines = []
    testLabels = []
    testWidth = []
    for i in range(500):
        imgs = []
        labels = []
        num = int(random.random()*(MAX_DIGITS_T-MIN_DIGITS_T))+MIN_DIGITS_T
        testWidth.append(num)
        for j in range(num): # 2-10 characters
            index = int(random.random()*10000)
            imgs.append(testImg[index])
            labels.append(testLabel[index])
        for x in range(MAX_DIGITS_TOTAL-num):
            imgs.append(zero_img)
        line = np.hstack(imgs)
        testLines.append(line)
        testLabels.append(labels)

    print('Done! time: %.3f' % (time.time()-start))

    return 10, len(trainLines), len(testLines), len(testLines), (trainLines, trainWidth, trainLabels, testLines, testWidth, testLabels, testLines, testWidth, testLabels)

if __name__ == '__main__':
    labelNum, num_tr, num_te, num_te, (trainLines, trainWidth, trainLabels, testLines, testWidth, testLabels, testLines, testWidth, testLabels) = getData()
    print(len(trainLabels), len(testLabels))
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.title.set_text(str(trainLabels[0]))
    ax1.imshow(trainLines[0], cmap='gray')
    ax2.title.set_text(str(testLabels[0]))
    ax2.imshow(testLines[0], cmap='gray')
    plt.show()
