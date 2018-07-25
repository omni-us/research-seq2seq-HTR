from main_torch_latest import all_data_loader, test, test_sample, test_sample_no_model
import argparse
#import os

parser = argparse.ArgumentParser(description='test')
parser.add_argument('epoch', type=int, help='epoch that you want to evaluate')
args = parser.parse_args()

_, _, test_loader = all_data_loader()
#test(test_loader, args.epoch, showAttn=True)
#test_sample(test_loader, args.epoch, showAttn=True)
test_sample_no_model(test_loader, args.epoch, showAttn=True)
#os.system('./test.sh '+str(args.epoch))

#for i in range(86, 87):
#    os.system('rm pred_logs/test_predict_seq.'+str(i)+'.log')
#    print('@@@', i, '@@@')
#    _, _, test_loader = all_data_loader()
#    test(test_loader, i)
#    os.system('./test.sh '+str(i))
#    print('<END>', i, '</END>')
