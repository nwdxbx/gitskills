import os,re,glob,sys,json
import numpy as np
import binascii
import argparse
# dist = np.linalg.norm(vec1 - vec2) 
import struct
_BASE_DIR_ = os.path.dirname(__file__)
insightFaceSrcDir = '../../insightFace/src/common'
sys.path.append(os.path.join(_BASE_DIR_, insightFaceSrcDir))
# import face_preprocess
# from sklearn import metrics
# import matplotlib.pyplot as plt
# _RESULT_DIR_ = '/home/guoxufeng/work/faceRec/devkit/felix/special_feature/Arc_cm_results_felix/otherFiles/'
# _RESULT_DIR_ = os.path.join(_BASE_DIR_,'tmp')
# import random
import time
_BASE_DIR_ = os.path.dirname(__file__)
# _INSIGHTFACE_DIR_ = os.path.join(_BASE_DIR_,'../insightFace')
_MEGAFACE_DIR_ = os.path.join(_BASE_DIR_,'../../devkit')
sys.path.append(_MEGAFACE_DIR_)
sys.path.append(os.path.join(_MEGAFACE_DIR_, 'experiments'))
# import run_experiment

# sizeData = 1000000
# scoreMatPathPP = os.path.join(_RESULT_DIR_,"facescrub_facescrub_arcface_112x112.bin")
# scoreMatPathPG = os.path.join(_RESULT_DIR_, "facescrub_megaface_arcface_112x112_%d_1.bin"%(sizeData))

# _SPECIAL_FEATURE_ROOT_ = "/home/guoxufeng/work/faceRec/devkit/felix/special_feature/"
_SPECIAL_FEATURE_ROOT_ = "/home/guoxufeng/work/data/special_feature/"


def write_bin(path, feature):
  feature = list(feature)
  with open(path, 'wb') as f:
    f.write(struct.pack('4i', len(feature),1,4,5))
    f.write(struct.pack("%df"%len(feature), *feature))

def read_feature_bin(path):
    feature = None

    with open(path,'rb') as f:
        iBytes = f.read()

    feaLen,_,_,_ = struct.unpack('4i',iBytes[0:16])
    print feaLen
    feature = struct.unpack("%df"%feaLen,iBytes[16:])
    return feature

# This function will create an random float array arr, 
#  the mod of the array is equal to mod
def get_array_with_fix_mod(arrLen, mod):
    arr = np.random.uniform(size=arrLen)
    norm = np.linalg.norm(arr)
    arr = arr/norm*mod
    # norm = np.linalg.norm(arr)
    return arr

# print get_array_with_fix_mod(10,2)

num_distractor = 100000000
num_probe = 100000
emb_dim = 128
probe_fea_matrix = np.zeros((emb_dim,num_probe))
target_fea_matrix = np.zeros((emb_dim,num_probe))
for i in range(num_probe):
    probe_fea_matrix[:,i] = np.random.uniform(size=emb_dim)
    target_fea_matrix[:,i] = probe_fea_matrix[:,i] + \
         get_array_with_fix_mod(emb_dim, np.sqrt(np.random.choice([1.0,2.0,3.0], 1)))
    

# print probe_fea_matrix
# exit(0)
# featurePath = "/home/guoxufeng/work/faceRec/devkit/felix/special_feature/FaceScrub_Features/Joan_Collins/Joan_Collins_14907.png_mobilenet_112x112.bin"

# feature = read_feature_bin(featurePath)
# # print feature

def read_score_bin_old(path,iRow=None):
    score = None

    with open(path,'rb') as f:
        iBytes = f.read()
    numRow,numCol,_,_ = struct.unpack('4i',iBytes[0:4*4])
    if iRow is None:
        iRow = numRow
        # print numRow,numCol
        score = struct.unpack("%df"%numRow*numCol,iBytes[4*4:])
        score = np.asarray(score).reshape((numRow,numCol))
        return numRow,numCol,score
    else:
        fromIndex = 4*(4+iRow*numCol)
        toIndex = 4*(4+(iRow+1)*numCol)
        # print "row, fromIndex, toIndex : ",iRow, fromIndex,toIndex
        # print "\tdetail: ",4+iRow*numCol,4+(iRow+1)*numCol
        score = struct.unpack("%df"%1*numCol,iBytes[fromIndex:toIndex])
        score = np.asarray(score).reshape((1,numCol)).flatten()
        return 1,numCol,score

def write_score_bin(path, score,nRow,nCol):
  score = list(score.flatten())
  with open(path, 'wb') as f:
    f.write(struct.pack('4i', nRow,nCol,4,5))
    f.write(struct.pack("%df"%len(score), *score))

def read_score_bin(path,iRow=None):
    score = None

    
    if iRow is None:
        with open(path,'rb') as f:
            iBytes = f.read()
        numRow,numCol,_,_ = struct.unpack('4i',iBytes[0:4*4])
        iRow = numRow
        # print numRow,numCol
        score = struct.unpack("%df"%numRow*numCol,iBytes[4*4:])
        score = np.asarray(score).reshape((numRow,numCol))
        return numRow,numCol,score
    else:
        with open(path,'rb') as f:
            iBytes = f.read(4*4)
        numRow,numCol,_,_ = struct.unpack('4i',iBytes[0:4*4])

        fromIndex = 4*(4+iRow*numCol)
        toIndex = 4*(4+(iRow+1)*numCol)

        with open(path,'rb') as f:
            f.seek(fromIndex)
            iBytes = f.read(toIndex - fromIndex)

        score = struct.unpack("%df"%1*numCol,iBytes)
        score = np.asarray(score).reshape((1,numCol)).flatten()

        return numRow,numCol,score

def make_special_lists():
    PATH_BASE = os.path.join(_SPECIAL_FEATURE_ROOT_, "list")

    if not os.path.exists(PATH_BASE):
        os.makedirs(PATH_BASE)
    probeListFilePath = os.path.join(PATH_BASE,'probe_list.json')
    distractorListFilePath = os.path.join(PATH_BASE,'megaface_features_list.json_%d_1'%(num_distractor))

    probePathDict = {}
    probePathDict['path'] = []
    probePathDict['id'] = []

    for i in range(num_probe):
        probePathDict['path'].append('probe_%03d.png'%(i))
        probePathDict['id'].append('%03d'%(i/5))
    print probePathDict
    with open(probeListFilePath,'w') as f:
        f.write(json.dumps(probePathDict,indent=2))

    distractorPathDict = {}
    distractorPathDict['path'] = []
    for i in range(num_distractor):
        distractorPathDict['path'].append('dis_%03d.png'%(i))

    with open(distractorListFilePath,'w') as f:
        f.write(json.dumps(distractorPathDict,indent=2))

    return probePathDict,distractorPathDict,probeListFilePath,distractorListFilePath

def make_special_probe_feature(probePathDict):
    featureDir = os.path.join(_SPECIAL_FEATURE_ROOT_, "probe_special_feature")
    # featureDir = "/home/guoxufeng/work/faceRec/devkit/felix/special_feature/probe_special_feature/"
    if not os.path.exists(featureDir):
        os.makedirs(featureDir)
    postfix = "_arcface_112x112.bin"

    pathList = probePathDict['path']
    idList = probePathDict['id']
    for indexPath, iPath in enumerate(pathList):
        iFullPath = os.path.join(featureDir,iPath+postfix)

        iFeature = probe_fea_matrix[:,indexPath]
        # iFeature = np.ones((emb_dim,1)).flatten()*indexPath*1.0
        # iFeature = np.ones((1,2)).flatten()*indexPath

        # if indexPath == 1 or indexPath == 2:
        #   iFeature = np.ones((1,1)).flatten()*1
        # else:
        #   iFeature = np.ones((1,1)).flatten()*2
        if indexPath < 100:
            print indexPath,iFeature,idList[indexPath]
        write_bin(iFullPath,iFeature)
        # print iFullPath

def make_special_distractor_feature(distractorPathDict):
    featureDir = os.path.join(_SPECIAL_FEATURE_ROOT_, "distractor_special_feature")
    # featureDir = "/home/guoxufeng/work/faceRec/devkit/felix/special_feature/distractor_special_feature/"
    postfix = "_arcface_112x112.bin"
    if not os.path.exists(featureDir):
        os.makedirs(featureDir)
    pathList = distractorPathDict['path']
    selectLocationForTarget = [e*e*10+1 for e in range(num_probe)]
    MAX_RANDOM_SAMPLES = 100000
    REFRESH_IFEATHURE = 1000000
    selectLocationForTarget = selectLocationForTarget[0:1000]
    # probeIDForTarget = range(num_probe)

    iFeature = np.random.uniform(size=emb_dim)

    for indexPath, iPath in enumerate(pathList): 
        iFullPath = os.path.join(featureDir,iPath+postfix)

        if indexPath < 3e6:
            continue
        if indexPath %10000 == 0:
            print "generating distractor %.2E out of %.2E"%(indexPath, num_distractor)

        if os.path.exists(iFullPath):
            continue

        if indexPath in selectLocationForTarget:
            theProbeIndex = selectLocationForTarget.index(indexPath)
            iFeature = target_fea_matrix[:, theProbeIndex]
        else:

            if indexPath > MAX_RANDOM_SAMPLES:
                if indexPath % REFRESH_IFEATHURE == 0:
                    iFeature = np.random.uniform(size=emb_dim)
            else:
                iFeature = probe_fea_matrix[:, np.random.choice(range(num_probe), 1)] + np.expand_dims(get_array_with_fix_mod(emb_dim, np.sqrt(np.random.choice([5.0,10.0,20.0,30.0], 1))),axis=1)


            # iFeature = np.ones((emb_dim,1)).flatten()*1.0+indexPath*10

        # iFeature[1] = 1.1
        # print indexPath,iFeature.shape
        write_bin(iFullPath,iFeature)
        # print iFullPath

def read_sample_feature(i):
    # featurePath = "/home/guoxufeng/work/faceRec/devkit/felix/special_feature/FaceScrub_arcface_Features_felix_cm/Chris_Noth/Chris_Noth_11649.png_arcface_112x112.bin"
    # featurePathMine = "/home/guoxufeng/work/faceRec/devkit/felix/special_feature/probe_special_feature/probe_000.png_arcface_112x112.bin"
    featurePathMinei = os.path.join(_SPECIAL_FEATURE_ROOT_, "probe_special_feature", "probe_%03d.png_arcface_112x112.bin"%(i))
    featureMinei = read_feature_bin(featurePathMinei)

    # featurePathMine1 = os.path.join(_SPECIAL_FEATURE_ROOT_, "probe_special_feature", "probe_%3d.png_arcface_112x112.bin"%(i))
    # featureMine1 = read_feature_bin(featurePathMine1)
    # print np.array(featureMinei).shape
    return np.array(featureMinei)

def read_from_special_matrix():
    specialScoreMatrixPathPG = os.path.join(_SPECIAL_FEATURE_ROOT_, "results", "otherFiles", "probe_megaface_arcface_112x112_%d_1.bin"%(num_distractor))
    # specialScoreMatrixPathPG = "/home/guoxufeng/work/faceRec/devkit/felix/special_feature/output_special_feature/otherFiles/probe_megaface_arcface_112x112_100_1.bin"
    _,_,scoreMatrixPG = read_score_bin(specialScoreMatrixPathPG)
    print scoreMatrixPG

    specialScoreMatrixPathPP = os.path.join(_SPECIAL_FEATURE_ROOT_, "results", "otherFiles", "probe_probe_arcface_112x112.bin")
    # specialScoreMatrixPathPP = "/home/guoxufeng/work/faceRec/devkit/felix/special_feature/output_special_feature/otherFiles/probe_probe_arcface_112x112.bin"
    _,_,scoreMatrixPP = read_score_bin(specialScoreMatrixPathPP)
    print scoreMatrixPP

    return scoreMatrixPP, scoreMatrixPG

def run_exp(probeListFilePath, distractorListFilePath):
    # PATH_BASE = os.path.join(_SPECIAL_FEATURE_ROOT_, "distractor_special_feature")
    #Create feature lists for megaface for all sets and sizes and verifies all features exist
    size = num_distractor
    index = 1
    missing = False
    distractor_feature_path=os.path.join(_SPECIAL_FEATURE_ROOT_, "distractor_special_feature")
    probe_feature_path=os.path.join(_SPECIAL_FEATURE_ROOT_, "probe_special_feature")
    file_ending='_arcface_112x112.bin'
    out_root=os.path.join(_SPECIAL_FEATURE_ROOT_, "results")
    other_out_root = os.path.join(out_root, "otherFiles")
    if not os.path.exists(other_out_root):
        os.makedirs(other_out_root)
    distractor_name = os.path.basename(distractorListFilePath).split('_')[0]
    alg_name = file_ending.split('.')[0].strip('_')
    # for index in set_indices:
        # for size in sizes:
            # print('Creating feature list of {} photos for set {}'.format(size,str(index)))
    # cur_list_name = megaface_list_basename + "_{}_{}".format(str(size), str(index))
    with open(distractorListFilePath) as fp:
        featureFile = json.load(fp)
        path_list = featureFile["path"]
        for i in range(len(path_list)):
            path_list[i] = os.path.join(distractor_feature_path,path_list[i] + file_ending)
            if(not os.path.isfile(path_list[i])):
                print path_list[i] + " is missing"
                missing = True
            if (i % 100000 == 0 and i > 0):
                print str(i) + " / " + str(len(path_list))
        featureFile["path"] = path_list
        lstPathOutputDistractor = os.path.join(other_out_root, '{}_features_{}_{}_{}'.format(distractor_name,alg_name,size,index))
        json.dump(featureFile, open(lstPathOutputDistractor, 'w'), sort_keys=True, indent=4)
    if(missing):
        sys.exit("Features are missing...")
    
    print "--gallery-list ",lstPathOutputDistractor
    probe_name = os.path.basename(probeListFilePath).split('_')[0]
    #Create feature list for probe set
    print ("opening: ",probeListFilePath)
    with open(probeListFilePath) as fp:
        featureFile = json.load(fp)
        path_list = featureFile["path"]
        for i in range(len(path_list)):
            path_list[i] = os.path.join(probe_feature_path,path_list[i] + file_ending)
            if(not os.path.isfile(path_list[i])):
                print path_list[i] + " is missing"
                missing = True
        featureFile["path"] = path_list
        json.dump(featureFile, open(os.path.join(
            other_out_root, '{}_features_{}'.format(probe_name,alg_name)), 'w'), sort_keys=True, indent=4)
        probe_feature_list = os.path.join(other_out_root, '{}_features_{}'.format(probe_name,alg_name))
    if(missing):
        sys.exit("Features are missing...")

    print "--probe-list ", probe_feature_list

    # args_run_exp = argparse.Namespace(
 #    delete_matrices=False, 
 #    distractor_feature_path=os.path.join(_SPECIAL_FEATURE_ROOT_, "distractor_special_feature"), 
 #    distractor_list_path=  distractorListFilePath,
 #    file_ending='_arcface_112x112.bin', 
 #    model=_MEGAFACE_DIR_ + '/models/jb_identity.bin', 
 #    num_sets=1, 
 #    out_root=os.path.join(_SPECIAL_FEATURE_ROOT_, "results"),
 #    probe_feature_path=os.path.join(_SPECIAL_FEATURE_ROOT_, "probe_special_feature"),
 #    probe_list=probeListFilePath, 
 #    sizes=[num_distractor]
 #    )
 #      run_experiment.main(args_run_exp)

def verify_special_score_matrix(scoreMatrixPP):
    print "now start to verify score matrix: "

    n_samples = scoreMatrixPP.shape[0]
    featList = []
    for i in range(n_samples):
        featList.append(read_sample_feature(i))

    def score_method(vec1, vec2):
        # score = 0
        score = -1 * np.sum(np.square(vec1 - vec2))
        # score = np.linalg.norm(arr1 - arr2) 
        return score

    myScoreMat = np.zeros(scoreMatrixPP.shape)

    for r in range(n_samples):
        for c in range(n_samples):
            myScoreMat[r][c] = score_method(featList[r],featList[c])

    print myScoreMat
    print scoreMatrixPP

    print "method estimation difference: ", myScoreMat - scoreMatrixPP
    print "method estimation difference: ", np.sum(myScoreMat - scoreMatrixPP)

def main_make_special_feature():

    probePathDict,distractorPathDict,probeListFilePath,distractorListFilePath = make_special_lists()

    
    # make_special_probe_feature(probePathDict)
    make_special_distractor_feature(distractorPathDict)
    print probeListFilePath
    print distractorListFilePath
    run_exp(probeListFilePath, distractorListFilePath)
    # run_exp(probeListFilePath, os.path.dirname(distractorListFilePath))

    # read_sample_feature(1)
    # scoreMatrixPP, scoreMatrixPG = read_from_special_matrix()

    # verify_special_score_matrix(scoreMatrixPP)
    pass

main_make_special_feature()