import subprocess as sub
import sys

if len(sys.argv) != 2:
    print('USAGE: python3 pytasas.py <epochs>')
    exit()
base = 'pred_logs/'

f_cer = open(base+'cer_train.log', 'w')
f_cer_v = open(base+'cer_valid.log', 'w')
f_cer_t = open(base+'cer_test.log', 'w')

for i in range(int(sys.argv[1])):
    gt = 'label_all_rm_err.txt'
    decoded = base+'train_predict_seq.'+str(i)+'.log'
    decoded_v = base+'valid_predict_seq.'+str(i)+'.log'
    decoded_t = base+'test_predict_seq.'+str(i)+'.log'
    res_cer = sub.Popen(['./tasas_cer.sh', gt, decoded], stdout=sub.PIPE)
    res_cer_v = sub.Popen(['./tasas_cer.sh', gt, decoded_v], stdout=sub.PIPE)
    res_cer_t = sub.Popen(['./tasas_cer.sh', gt, decoded_t], stdout=sub.PIPE)
    res_cer = res_cer.stdout.read().decode('utf8')
    res_cer_v = res_cer_v.stdout.read().decode('utf8')
    res_cer_t = res_cer_t.stdout.read().decode('utf8')
    res_cer = float(res_cer)/100
    res_cer_v = float(res_cer_v)/100
    res_cer_t = float(res_cer_t)/100
    f_cer.write(str(res_cer))
    f_cer.write(' ')
    f_cer_v.write(str(res_cer_v))
    f_cer_v.write(' ')
    f_cer_t.write(str(res_cer_t))
    f_cer_t.write(' ')
    print(i)

f_cer.close()
f_cer_v.close()
f_cer_t.close()
