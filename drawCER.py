import matplotlib.pyplot as plt

base = 'pred_logs/'

cer = open(base+'cer_train.log', 'r')
cer_data = cer.read().split(' ')[:-1]
cerr = [float(i) for i in cer_data]

cer_v = open(base+'cer_valid.log', 'r')
cer_data_v = cer_v.read().split(' ')[:-1]
cerr_v = [float(i) for i in cer_data_v]

cer_t = open(base+'cer_test.log', 'r')
cer_data_t = cer_t.read().split(' ')[:-1]
cerr_t = [float(i) for i in cer_data_t]

plt.plot(cerr, 'r-')
cer_spot, = plt.plot(cerr, 'ro')

plt.plot(cerr_v, 'b-')
cer_spot_v, = plt.plot(cerr_v, 'bo')

plt.plot(cerr_t, 'g-')
cer_spot_t, = plt.plot(cerr_t, 'go')
plt.legend([cer_spot, cer_spot_v, cer_spot_t], ['CER train', 'CER valid', 'CER test'])
plt.xlabel('epoch')
plt.ylim(0, 1)
plt.title('character error rate')
plt.show()

cer.close()
cer_v.close()
cer_t.close()
