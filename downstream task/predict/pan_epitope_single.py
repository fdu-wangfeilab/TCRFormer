import os,sys
b_directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'model'))
sys.path.insert(0, b_directory)
from pan_epitope_single import *

b_directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'utils'))
sys.path.insert(0, b_directory)
from pan_epitope_single_util import predict

import argparse

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epitope', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--cuda', type=str, default='cuda:0')
    parser.add_argument('--datatype', type=str, default='mixpredtcr',
                       help='mixpredtcr or singleindouble')

    args = parser.parse_args()

    if args.datatype=='mixpredtcr':
        predict(args.epitope,args.path,args.cuda)
    else:
        predict(args.epitope,args.path,args.cuda,args.datatype)
   
    
main()