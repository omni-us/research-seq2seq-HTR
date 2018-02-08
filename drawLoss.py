import matplotlib.pyplot as plt
import sys

if len(sys.argv) == 1:
    n = -1
elif len(sys.argv) == 2:
    n = int(sys.argv[1])
else:
    print('Usage: python3 drawLoss.py <epoch>')
    exit()
base = 'pred_logs/'

loss = open(base+'loss_train.log', 'r')
loss_v = open(base+'loss_valid.log', 'r')
loss_t = open(base+'loss_test.log', 'r')

loss_data = loss.read().split(' ')[:n]
loss_data = [float(i) for i in loss_data]

loss_data_v = loss_v.read().split(' ')[:n]
loss_data_v = [float(i) for i in loss_data_v]

loss_data_t = loss_t.read().split(' ')[:n]
loss_data_t = [float(i) for i in loss_data_t]

plt.plot(loss_data, 'r-')
loss_train, = plt.plot(loss_data, 'ro')

plt.plot(loss_data_v, 'b-')
loss_valid, = plt.plot(loss_data_v, 'bo')

plt.plot(loss_data_t, 'g-')
loss_test, = plt.plot(loss_data_t, 'go')
plt.legend([loss_train, loss_valid, loss_test], ['training loss', 'validation loss', 'testing loss'])

plt.xlabel('epoch')
#plt.ylim(0, 1)
plt.title('loss')
plt.show()

loss.close()
loss_v.close()
loss_t.close()
