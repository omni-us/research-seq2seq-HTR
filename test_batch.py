from main_torch_latest import test_data_loader_batch, test
import argparse
import os

parser = argparse.ArgumentParser(description='test')
parser.add_argument('epoch', type=int, help='epoch that you want to evaluate')
args = parser.parse_args()

epoch = args.epoch

for b in range(1, 64):
    os.system('rm pred_logs/test_predict_seq.'+str(epoch)+'.log')
    test_loader = test_data_loader_batch(b)
    test(test_loader, epoch, showAttn=False)
    print('---------batch size: '+str(b)+'---------')
    os.system('./test.sh '+str(args.epoch))

