import pylab
import cv2

epochs = 61

img_names = ['c04-110-03-08,168', 'd01-016-00-10,189', 'd04-101-01-10,181', 'd06-060-08-06,158', 'e06-015-02-01,178']
gts = []
for i in img_names:
    with open('RWTH.iam_word_gt_final.test.thresh', 'r') as f:
        gt = f.readlines()
        for j in gt:
            ss = j.strip().split(' ')
            if i == ss[0]:
                gts.append(''.join(ss[1:]))

def drawSuper(epoch):
    fig = pylab.figure(figsize=(9.6, 5.4), dpi=80)
    with open('pred_logs/loss_train.log', 'r') as l_tr:
        d_tr = l_tr.read().split(' ')[:epoch+1]
        dd_tr = [float(i) for i in d_tr]
    with open('pred_logs/loss_valid.log', 'r') as l_va:
        d_va = l_va.read().split(' ')[:epoch+1]
        dd_va = [float(i) for i in d_va]
    with open('pred_logs/loss_test.log', 'r') as l_te:
        d_te = l_te.read().split(' ')[:epoch+1]
        dd_te = [float(i) for i in d_te]
    axplot = fig.add_axes([0.04, 0.6, .45, .34])
    axplot.plot(dd_tr, 'r-', label='Train Loss')
    axplot.plot(dd_va, 'b-', label='Valid Loss')
    axplot.plot(dd_te, 'g-', label='Test Loss')
    axplot.legend()

    with open('pred_logs/cer_train.log', 'r') as l_tr:
        d_tr = l_tr.read().split(' ')[:epoch+1]
        dd_tr = [float(i) for i in d_tr]
    with open('pred_logs/cer_valid.log', 'r') as l_va:
        d_va = l_va.read().split(' ')[:epoch+1]
        dd_va = [float(i) for i in d_va]
    with open('pred_logs/cer_test.log', 'r') as l_te:
        d_te = l_te.read().split(' ')[:epoch+1]
        dd_te = [float(i) for i in d_te]
    axplot = fig.add_axes([0.54, 0.6, .45, .34])
    axplot.plot(dd_tr, 'r-', label='Train CER')
    axplot.plot(dd_va, 'b-', label='Valid CER')
    axplot.plot(dd_te, 'g-', label='Test CER')
    axplot.legend()

    imgs = []
    preds = []
    for i in img_names:
        #print('imgs/test_samples/test_'+i+'_'+str(epoch)+'.jpg')
        img = cv2.imread('imgs/test_samples/test_'+i.split(',')[0]+'_'+str(epoch)+'.jpg', 0)
        with open('pred_logs/test_predict_seq.'+str(epoch)+'.log', 'r') as f:
            pred = f.readlines()
            for j in pred:
                ss = j.strip().split(' ')
                if i == ss[0]:
                    preds.append(''.join(ss[1:]))
        imgs.append(img)

    n = 0.16
    fig.text(0.05, 0.45, 'Groundtruth:', horizontalalignment='left', verticalalignment='center')
    fig.text(0.05, 0.5, 'Prediction:', horizontalalignment='left', verticalalignment='center')
    fig.text(0.5, 0.97, 'Convolve, Attend and Spell: An Attention-based Sequence-to-Sequence Model for Handwritten Word Recognition', horizontalalignment='center', verticalalignment='center', fontsize=12)
    fig.text(0.03, 0.05, 'Epoch: '+str(epoch+1), horizontalalignment='left', verticalalignment='center', fontsize=16)
    for c, i in enumerate(imgs):
        axicon = fig.add_axes([c*n+0.15, 0.02, 0.2, 0.4])
        axicon.imshow(i, interpolation='nearest', cmap='gray')
        axicon.set_xticks([])
        axicon.set_yticks([])
        fig.text(c*n+0.25, 0.45, gts[c], horizontalalignment='center', verticalalignment='center')
        fig.text(c*n+0.25, 0.5, preds[c], horizontalalignment='center', verticalalignment='center')

    #fig.show()
    #pylab.show()
    fig.savefig('super_img/'+str(epoch+1)+'.png')
    pylab.close(fig)


for i in range(epochs):
    drawSuper(i)
