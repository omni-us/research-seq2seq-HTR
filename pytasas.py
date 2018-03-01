import subprocess as sub
import sys

if len(sys.argv) != 3:
    print('USAGE: python3 pytasas.py <epochs> <flag: with text or not, si: with, no: not>')
    exit()
base = 'pred_logs/'

f_cer = open(base+'cer_train.log', 'w')
f_cer_v = open(base+'cer_valid.log', 'w')
if sys.argv[2] == 'si':
    f_cer_t = open(base+'cer_test.log', 'w')

for i in range(int(sys.argv[1])):
    gt_tr = 'iam_word_gt_final.train.thresh'
    gt_va = 'iam_word_gt_final.valid.thresh'
    gt_te = 'iam_word_gt_final.test.thresh'
    decoded = base+'train_predict_seq.'+str(i)+'.log'
    decoded_v = base+'valid_predict_seq.'+str(i)+'.log'
    if sys.argv[2] == 'si':
        decoded_t = base+'test_predict_seq.'+str(i)+'.log'
    res_cer = sub.Popen(['./tasas_cer.sh', gt_tr, decoded], stdout=sub.PIPE)
    res_cer_v = sub.Popen(['./tasas_cer.sh', gt_va, decoded_v], stdout=sub.PIPE)
    if sys.argv[2] == 'si':
        res_cer_t = sub.Popen(['./tasas_cer.sh', gt_te, decoded_t], stdout=sub.PIPE)
    res_cer = res_cer.stdout.read().decode('utf8')
    res_cer_v = res_cer_v.stdout.read().decode('utf8')
    if sys.argv[2] == 'si':
        res_cer_t = res_cer_t.stdout.read().decode('utf8')
    res_cer = float(res_cer)/100
    res_cer_v = float(res_cer_v)/100
    if sys.argv[2] == 'si':
        res_cer_t = float(res_cer_t)/100
    f_cer.write(str(res_cer))
    f_cer.write(' ')
    f_cer_v.write(str(res_cer_v))
    f_cer_v.write(' ')
    if sys.argv[2] == 'si':
        f_cer_t.write(str(res_cer_t))
        f_cer_t.write(' ')
    print(i)

f_cer.close()
f_cer_v.close()
if sys.argv[2] == 'si':
    f_cer_t.close()
