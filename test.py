from main_torch_latest import all_data_loader, test
import argparse
#import os

parser = argparse.ArgumentParser(description='test')
parser.add_argument('epoch', type=int, help='epoch that you want to evaluate')
args = parser.parse_args()

_, _, test_loader = all_data_loader()
test(test_loader, args.epoch, showAttn=True)
#os.system('./test.sh '+str(args.epoch))

