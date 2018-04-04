from main_torch import test
import argparse
import os

parser = argparse.ArgumentParser(description='test')
parser.add_argument('epoch', type=int, help='epoch that you want to evaluate')
args = parser.parse_args()

test(args.epoch)
os.system('./test.sh '+str(args.epoch))

#for i in range(135, 155):
#    print('@@@', i, '@@@')
#    test(i)
#    os.system('./test.sh '+str(i))
#    print('<END>', i, '</END>')
