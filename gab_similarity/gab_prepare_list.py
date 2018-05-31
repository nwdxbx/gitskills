#coding=utf-8

import argparse
import os
import sys
import json
import random
import subprocess
import numpy as np
from timeit import default_timer as timer

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

_BASE_DIR_ = os.path.dirname(__file__)

sys.path.append(os.path.join(os.path.dirname(__file__),'..', '..'  ,'utils_felix'))
import file_io





# os.environ["CUDA_VISIBLE_DEVICES"]="3,4,5,6,7"
import tensorflow as tf
import multiprocessing as mp

from tensorflow.python.client import device_lib
import csv

import glob
COL_BS = 100000

version = "Test1"
# version = "Test2"

imageListDir = "/home/user/beijing_face_new/split_files/%s"%(version)
noface_list = "/home/user/beijing_face_new/noface_%s.txt"%(version)
IMAGE_ROOT = "/home/user/beijing_face_new/data_val/%s/"%(version)
FEA_ROOT = "/home/user/beijing_face_new/features_%s/"%(version)
FEA_SUFFIX = "_112x112.bin"


targetQueryFeatureList = "/home/user/beijing_face_new/split_files/%s_fea/query_list"%(version)
targetDbFeatureList = "/home/user/beijing_face_new/split_files/%s_fea/db_list"%(version)

if not os.path.exists(os.path.dirname(targetQueryFeatureList)):
    os.makedirs(os.path.dirname(targetQueryFeatureList))

def check_feature_completed(imageListDir, noface_list, fea_root, image_root):

    imageListFiles = glob.glob(os.path.join(imageListDir,"*"))
    imageList = []
    for iFile in imageListFiles:
        # print iFile
        with open(iFile,'r') as f:
            imageList.extend(f.readlines())

    with open(noface_list,'r') as f:
        noface_list = f.readlines()
            
    fea_list = glob.glob(os.path.join(fea_root,"*/*"))
    image_file_list = glob.glob(os.path.join(image_root,"*/*"))

    print "imageList: ",len(imageList)
    print "image_file_list: ",len(image_file_list)

    print "noface_list: ", len(noface_list)
    print "fea_list: ",len(fea_list)

    assert(len(fea_list) + len(noface_list) == len(imageList))
    assert(len(fea_list) + len(noface_list) == len(image_file_list))

    pass
def main(args):
    
    import glob2
    fea_list = glob2.glob(os.path.join(FEA_ROOT,"*/**/*.bin"), recursive=True)
    dbList = [p.strip() for p in fea_list if "db/" in p]
    queryList = [p.strip() for p in fea_list if "query/" in p]
    
    # print dbList
    print len(dbList) 
    print len(queryList) 
    print len(fea_list)
    # random.shuffle(fea_list)
    print "\n".join(fea_list[0:5])
    print "\n".join(fea_list[-5:])
    assert(len(dbList) + len(queryList) == len(fea_list))

    

    # dbList = [p.replace(IMAGE_ROOT,FEA_ROOT).strip()+FEA_SUFFIX for p in imageList if "db/d" in p]
    # queryList = [p.replace(IMAGE_ROOT,FEA_ROOT).strip()+FEA_SUFFIX for p in imageList if "query/q" in p]
    # print len(dbList)
    # print dbList[0]
    # print len(queryList)
    # print queryList[0]
    # assert(len(dbList) + len(queryList) == len(imageList))

    # check_feature_completed(imageListDir, noface_list,  FEA_ROOT, IMAGE_ROOT)

    with open(targetQueryFeatureList,'w') as f:
        f.write(json.dumps(
            {"path":queryList},indent=2
            ))

    print len(queryList)
    print "saved to : ", targetQueryFeatureList

    with open(targetDbFeatureList,'w') as f:
        f.write(json.dumps(
            {"path":dbList},indent=2
            ))
    print len(dbList)
    print "saved to : ", targetDbFeatureList
 

def parse_arguments(argv):    
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # # parser.add_argument('model', help='not used place holder')
    # # parser.add_argument('path', help='not used place holder')
    # parser.add_argument('list1',help='')
    # parser.add_argument('list2', help='')
    # parser.add_argument('output', help='')
    # parser.add_argument('--gpus', default = "7", help='')
    
    return parser.parse_args(argv)
if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
