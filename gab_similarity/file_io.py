#coding=utf-8
import os,re,glob,sys,json
import struct
import numpy as np
def read_bin(path):
  feature = None

  with open(path,'rb') as f:
    iBytes = f.read()

  feaLen,_,_,_ = struct.unpack('4i',iBytes[0:16])
  # print feaLen
  feature = struct.unpack("%df"%feaLen,iBytes[16:])
  return feature

def write_bin(path, feature):
  feature = list(feature)
  with open(path, 'wb') as f:
    f.write(struct.pack('4i', len(feature),1,4,5))
    f.write(struct.pack("%df"%len(feature), *feature))


def write_feature_bin(path, feat,nRow,nCol):
  assert (feat.shape[0] == nRow)
  assert (feat.shape[1] == nCol)
  feat = list(feat.flatten())
  with open(path, 'wb') as f:
    f.write(struct.pack('4i', nRow,nCol,4,5))
    f.write(struct.pack("%df"%len(feat), *feat))



def read_feature_bin(path,iRow=None):
  feat = None

  
  if iRow is None:
    with open(path,'rb') as f:
      iBytes = f.read()
    numRow,numCol,_,_ = struct.unpack('4i',iBytes[0:4*4])
    iRow = numRow
    # print numRow,numCol
    feat = struct.unpack("%df"%numRow*numCol,iBytes[4*4:])
    feat = np.asarray(feat).reshape((numRow,numCol))
    # print feat.shape
    return numRow,numCol,feat
  else:
    with open(path,'rb') as f:
      iBytes = f.read(4*4)
    numRow,numCol,_,_ = struct.unpack('4i',iBytes[0:4*4])

    fromIndex = 4*(4+iRow*numCol)
    toIndex = 4*(4+(iRow+1)*numCol)

    with open(path,'rb') as f:
      f.seek(fromIndex)
      iBytes = f.read(toIndex - fromIndex)

    feat = struct.unpack("%df"%1*numCol,iBytes)
    feat = np.asarray(feat).reshape((1,numCol)).flatten()

    return numRow,numCol,feat