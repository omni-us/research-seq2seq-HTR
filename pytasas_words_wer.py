import subprocess as sub
#import sys
import argparse

parser = argparse.ArgumentParser('Tasas for WER')
parser.add_argument('epochs', type=int, help='how many epochs')
parser.add_argument('flag', type=str, help='si/no with/without testing')
#parser.add_argument('folder', type=str, help='pred_logs_layer_1 or others')
args = parser.parse_args()

epochs = args.epochs
flag = args.flag
#base = args.folder + '/'
base = 'pred_logs/'
#if len(sys.argv) != 3:
#    print('USAGE: python3 pytasas.py <epochs> <flag: with text or not, si: with, no: not>')
#    exit()
#base = 'pred_logs/'

f_cer = open(base+'wer_train.log', 'w')
f_cer_v = open(base+'wer_valid.log', 'w')
if flag == 'si':
    f_cer_t = open(base+'wer_test.log', 'w')

for i in range(epochs):
    gt_tr = 'RWTH.iam_word_gt_final.train.thresh'
    gt_va = 'RWTH.iam_word_gt_final.valid.thresh'
    gt_te = 'RWTH.iam_word_gt_final.test.thresh'
    decoded = base+'train_predict_seq.'+str(i)+'.log'
    decoded_v = base+'valid_predict_seq.'+str(i)+'.log'
    if flag == 'si':
        decoded_t = base+'test_predict_seq.'+str(i)+'.log'
    res_cer = sub.Popen(['./tasas_wer.sh', gt_tr, decoded], stdout=sub.PIPE)
    res_cer_v = sub.Popen(['./tasas_wer.sh', gt_va, decoded_v], stdout=sub.PIPE)
    if flag == 'si':
        res_cer_t = sub.Popen(['./tasas_wer.sh', gt_te, decoded_t], stdout=sub.PIPE)
    res_cer = res_cer.stdout.read().decode('utf8')
    res_cer_v = res_cer_v.stdout.read().decode('utf8')
    if flag == 'si':
        res_cer_t = res_cer_t.stdout.read().decode('utf8')
    res_cer = float(res_cer)/100
    res_cer_v = float(res_cer_v)/100
    if flag == 'si':
        res_cer_t = float(res_cer_t)/100
    f_cer.write(str(res_cer))
    f_cer.write(' ')
    f_cer_v.write(str(res_cer_v))
    f_cer_v.write(' ')
    if flag == 'si':
        f_cer_t.write(str(res_cer_t))
        f_cer_t.write(' ')
    print(i)

f_cer.close()
f_cer_v.close()
if flag == 'si':
    f_cer_t.close()
