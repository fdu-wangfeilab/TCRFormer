import os,sys
b_directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'model'))
sys.path.insert(0, b_directory)
from pan_epitope_double import *

b_directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'utils'))
sys.path.insert(0, b_directory)
from pan_epitope_double_util import predict


import argparse

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epitope', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--cuda', type=str, default='cuda:0')

    args = parser.parse_args()

    predict(args.epitope,args.path,args.cuda)

   
    
main()