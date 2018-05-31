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

COL_BS = 1000


def gen_scores_for_batchs(batchList,row_batch_size,featPathList1,featPathList2,galleryIndices,gpu_id,output_temp_dir):
    for iRowBatch in batchList:
        indexStart, indexEnd = get_batch_index(len(featPathList1),iRowBatch,row_batch_size)
        print "=============================="
        print "iRowBatch: ", iRowBatch
        print "batch list: ",batchList
        print "indexStart, indexEnd",indexStart, indexEnd
        print "=============================="
        # run script with indexStart, indexEnd
        subFeatPathList1 = featPathList1[indexStart:indexEnd]
        savePathScore, savePathIndices = get_batch_path(output_temp_dir,iRowBatch)
        
        if not os.path.exists(savePathScore):
            scoreMat,indices = score_with_path(
                subFeatPathList1,
                featPathList2,
                galleryIndices,
                gpu_id=gpu_id,
                col_batch_size=COL_BS)
            assert(scoreMat.shape[0] == indices.shape[0])
            assert(scoreMat.shape[1] == indices.shape[1])
            print savePathScore
            print savePathIndices
            file_io.write_feature_bin(savePathScore,scoreMat,scoreMat.shape[0],scoreMat.shape[1])
            file_io.write_feature_bin(savePathIndices,indices,indices.shape[0],indices.shape[1])
    pass

def get_output_csv_path(queryImageName, output_dir):
    filebase,ext = os.path.splitext(queryImageName)
    return os.path.join(output_dir, "test2%s.csv"%(filebase))

def write_csv(file_path, similarity,   queryImageName, dbImages):
    with open (file_path, 'w') as csvfile:
        simwriter = csv.writer(csvfile)
        simwriter.writerow([queryImageName]+['FF']*18)
        for i, sim in enumerate(similarity):
            simwriter.writerow([dbImages[i]]+[sim])
def fea_fullpath_to_image_name(feaPath):
    return os.path.basename(feaPath).replace('_112x112.bin',"").strip()
def main(args):
    
    ''' 
    do the filename staff
    '''
    start = timer()
    
    with open (args.probe_list, 'r') as f:
        featPathList1 = json.loads(f.read())['path']

    with open (args.gallery_list, 'r') as f:
        featPathList2 = json.loads(f.read())['path']
    
    # print "copying..."
    # featPathList3 = list(np.repeat(featPathList2,10))
    # print "copy done"
    # with open (args.gallery_list.replace('1000000','10m'), 'w') as f:
    #     f.write(json.dumps({"path":featPathList3}))
    # print "wrting done"

    output_temp_dir = os.path.join(args.output_dir,'tmp')
    output_dir = args.output_dir

    if not os.path.exists(output_temp_dir):
        os.makedirs(output_temp_dir)



    # featPathList2 = featPathList2[:99000]
    galleryIndices = range(len(featPathList2))

    # featPathList1 = featPathList1[:1000]
    print "will run score with path"
    row_batch_size = 1500
    # 1388 will make each of 8 jobs maximize 
    row_batch_number = get_total_batch_number(len(featPathList1),row_batch_size)

    batchList = range(row_batch_number)
    gpuList = [int(g) for g in args.gpus.split(',')]
    jobNumbers = len(gpuList)
    print "gpuList: ",gpuList
    print "jobNumbers: ",jobNumbers
    cutSize = int(np.ceil(row_batch_number*1.0/jobNumbers))
    print "cutSize: ",cutSize

    cutJobs = [batchList[i:i+cutSize] for i in range(0,row_batch_number,cutSize)]
    print "cutJobs: ",cutJobs

    assert(len(cutJobs) == len(gpuList))
    processes = []
    ind = range(len(featPathList2))
    for i,iBatchList in enumerate(cutJobs):
        random.shuffle(ind)
        processes.append(mp.Process(
            target=gen_scores_for_batchs, 
            args=(
                iBatchList,
                row_batch_size,
                featPathList1,
                [featPathList2[iidx] for iidx in ind],
                [galleryIndices[iidx] for iidx in ind],
                gpuList[i],
                output_temp_dir
                )
            ))

    # Run processes
    for p in processes:
      p.start()
      pass

    # # Exit the completed processes
    for p in processes:
      p.join()

    # check completeness
    scoreMat = np.zeros((len(featPathList1), COL_BS))

    for i,iRowBatch in enumerate(batchList):
        tmpBatchPathScore, tmpBatchPathIndices = get_batch_path(output_temp_dir,iRowBatch)
        indexStart, indexEnd = get_batch_index(len(featPathList1),iRowBatch,row_batch_size)
        print "checking: ",tmpBatchPathScore
        if not os.path.exists(tmpBatchPathScore) or not os.path.exists(tmpBatchPathIndices):
            print "not existed!! exit"
            exit(0)
        _,_,scoreTemp = file_io.read_feature_bin(tmpBatchPathScore)
        _,_,indicesTemp = file_io.read_feature_bin(tmpBatchPathIndices)
        # scoreMat[indexStart:indexEnd,:] = scoreTemp
        for ii, query_id in enumerate(range(indexStart, indexEnd)):
            queryImageName = fea_fullpath_to_image_name(featPathList1[query_id])
            csvPath = get_output_csv_path(queryImageName, output_dir)
            write_csv(
                csvPath, 
                scoreTemp[ii,:], 
                queryImageName=queryImageName,
                dbImages = [fea_fullpath_to_image_name(featPathList2[int(ind)]) for ind in indicesTemp[ii,:]],
                )
            # if query_id == 1363:
            #     print queryImageName
            if query_id %10 ==0:
                print "\twriting csv for : ",query_id
                # print scoreMat[query_id,:]



    
    run_time = timer() - start  
    print "finish running score with paths, total time %f seconds"%(run_time)
    # file_io.write_feature_bin(args.output, scoreMat, scoreMat.shape[0], scoreMat.shape[1])
   

def get_batch_path(tmp,iRowBatch):
    savePathScore = os.path.join(tmp,'tmp_%05d_score.bin'%(iRowBatch))
    savePathIndices = os.path.join(tmp,'tmp_%05d_indices.bin'%(iRowBatch))
    return savePathScore, savePathIndices
def get_total_batch_number(maxLen,batch_size):
    batch_number = int(np.ceil(maxLen * 1.0 / batch_size))
    return batch_number
def get_batch_index(maxLen,batch, batch_size):
    indexStart = batch*batch_size
    indexEnd = (batch+1)*batch_size 
    indexEnd = maxLen if indexEnd >= maxLen else indexEnd

    return indexStart, indexEnd
def get_emb_cpu(featPathList,batch, batch_size):
    print "prepare to load the second list, size: ",len(featPathList)
    emb_dim,_,_ = file_io.read_feature_bin(featPathList[0])
    indexStart, indexEnd = get_batch_index(len(featPathList),batch, batch_size)
    emb_cpu = np.empty((emb_dim,indexEnd - indexStart))
    # print "get_emb_cpu: get_batch_index: ",indexStart, indexEnd, 
    counter = 0
    for i in range(len(featPathList)):

        if i< indexStart or i >= indexEnd:
            continue
        if i%10000 == 0:
            print "finish %.2E out of %.1E"%(i,len(featPathList))
        # if i> N2:
        #     break
        # emb_b[:,i] = file_io.read_feature_bin(featPathList[i])[2][:,0].astype(np.float32)
        emb_cpu[:,counter] = file_io.read_feature_bin(featPathList[i])[2][:,0].astype(np.float32)
        counter += 1
    return emb_cpu 
def get_new_indices(galleryIndices, batch, batch_size, num_probe):
    indexStart, indexEnd = get_batch_index(len(galleryIndices),batch, batch_size)
    indices_1d = np.array(galleryIndices[indexStart:indexEnd],dtype=np.int32)
    indices_2d = np.tile(indices_1d,[num_probe,1])
    return indices_2d
def score_with_path(featPathList1, featPathList2,galleryIndices, col_batch_size=100000,gpu_id=0):
    # N = 1024 * 1024 * 90   # float: 4M = 1024 * 102
    
    # featPathList2 = featPathList2[:300000]
    # N = np.int32(len(featPathList1) + len(featPathList2))
    N1 = np.int32(len(featPathList1))
    N2 = np.int32(len(featPathList2))

    # col_batch_size = 100000
    # N1=row_batch_size
    # N1 = col_batch_size if N1 > col_batch_size else N1
    N2 = col_batch_size if N2 > col_batch_size else N2

    # in MB
    # expect_mem_usage = col_batch_size * N1 * 1.0 / 1000 / 1000 * 25
    # expect_mem_percent = expect_mem_usage / 15000.0
    # print "expect_mem_usage (MB): ", expect_mem_usage
    # print "expect_mem_percent(0-1.0): ", expect_mem_percent

    batch_number = get_total_batch_number(len(featPathList2), col_batch_size)
    print "batch_number: ",batch_number

    # print get_batch_index(featPathList2, batch_number-1, col_batch_size)

    # N2=500
    print "prepare to load the first list"
    emb_dim,_,_ = file_io.read_feature_bin(featPathList1[0])
    emb_a_cpu = np.empty((emb_dim,N1))
    # emb_a = tf.zeros((emb_dim,len(featPathList1)))
    for i in range(len(featPathList1)):
        if i%10000 == 0:
            print "finish %.2E out of %.1E"%(i,len(featPathList1))
        if i >= N1 :
        	break
        emb_a_cpu[:,i] = file_io.read_feature_bin(featPathList1[i])[2][:,0].astype(np.float32)

    # emb_a = tf.convert_to_tensor(emb_a_cpu)

    # emb_a = tf.placeholder(tf.float32,shape=(emb_dim,None))

    if not gpu_id == -1:
        os.environ["CUDA_VISIBLE_DEVICES"]="%d"%(gpu_id)
    # with tf.device('/gpu:0'):
    with tf.Graph().as_default() as g:
      # Define operations and tensors in `g`.
      # c = tf.constant(30.0)
      # assert c.graph is g
        emb_a = tf.placeholder(tf.float32,shape=(emb_dim,N1),name="emb_a")


        # extract emb for b
        # print "prepare to load the second list"
        emb_dim,_,_ = file_io.read_feature_bin(featPathList2[0])
        # emb_b = tf.placeholder(tf.float32,shape=(emb_dim,None))
        feed_size = tf.placeholder(tf.int32,name="feed_size")
        emb_b = tf.placeholder(tf.float32,shape=(emb_dim,None),name="emb_b")
        new_indices = tf.placeholder(tf.int32,shape=(N1,None),name="new_indices")


        start = timer()
        # convert a and b to 3D volumn for the convenient of calculation
        emb_a_sum_square = tf.reduce_sum(
            tf.square(emb_a,name="square-emb_a"),
            0,
            name="reduce-emb_a_sum_square") 

        emb_b_sum_square = tf.reduce_sum(
            tf.square(emb_b,name="square-emb_b"),
            0,
            name="reduce-emb_b_sum_square") 

        print "emb_a_sum_square.shape: ",emb_a_sum_square.shape
        print "emb_b_sum_square.shape: ",emb_b_sum_square.shape

        emb_a_sum_square = tf.expand_dims(
            emb_a_sum_square,
            1,
            name="expand_dims-emb_a_sum_square") #from N1, to N1,1
        emb_b_sum_square = tf.expand_dims(
            emb_b_sum_square,
            0,
            name="expand_dims-emb_b_sum_square") #from N2 to 1,N2



        # emb_score = tf.Variable(tf.zeros((N1,col_batch_size),dtype=tf.float32))
        emb_score = tf.zeros((N1,feed_size),dtype=tf.float32,name="emb_score")
        # emb_score is N1*N2 shape
        emb_score = -2 * tf.einsum('ki,kj->ij',emb_a,emb_b,name="einsum-emb_score")

        # emb_score = tf.zeros((N1, N2))

        # this  will broadcast along axis
        emb_score = emb_score + emb_a_sum_square
        emb_score = emb_score + emb_b_sum_square

        emb_score = -1 * emb_score

        # emb_score = emb_score - tf.tile(emb_a_sum_square,[1,N2])
        # emb_score = emb_score - tf.tile(emb_b_sum_square,[N1,1])

        print "emb_score.shape: ",emb_score.shape

        

        # scoreGPU = np.zeros((N1,len(featPathList2)),dtype=np.float32)

        oldScoreGPU = tf.Variable(
            -9999*tf.ones((N1,col_batch_size),dtype=tf.float32),
            name="oldScoreGPU")

        oldIndices = tf.Variable(
            get_new_indices(galleryIndices, 0, col_batch_size,N1), 
            name="oldIndices")
  
        emb_long_score = tf.concat(
            [emb_score,oldScoreGPU],
            1,
            name="emb_long_score")

        long_indices = tf.concat(
            [new_indices,oldIndices],
            1,
            name="emb_long_indice")




        # indices = tf.contrib.framework.argsort(emb_long_score,direction ='DESCENDING')
        values, indices = tf.nn.top_k(emb_long_score,k=N2,name="top_k")
        print "indices.shape:", indices.shape
        print "long_indices.shape:", long_indices.shape
        # long_indices_sort = tf.zeros_like(long_indices)
        indices_good = tf.zeros_like(indices)

        r = tf.expand_dims(tf.range(N1),1)
        # print "r: ",r.shape
        rowIndices = tf.tile(r,[1,N2])
        print "rowIndices shape: ",rowIndices.shape
        # X, Y = tf.meshgrid(tf.reshape(rowIndices,[-1]), tf.reshape(indices,[-1]))
        X, Y = tf.reshape(rowIndices,[-1]), tf.reshape(indices,[-1])
        xy = tf.stack([X,Y],axis=1)
        print "x, y shape: ", X.shape, Y.shape
        print "xy shape: ",xy.shape
        # packIndices = zip(tf.reshape(rowIndices,[-1]),tf.reshape(indices,[-1]))
        # print "packIndices: ", packIndices.shape, packIndices[0:5]
        # long_indices = tf.Print(long_indices,[
        #     tf.shape(long_indices), 
        #     xy.shape, 
        #     feed_size,
        #     tf.shape(emb_score),
        #     tf.shape(new_indices),
        #     "long_indices and xy runtime shape<------------"])

        res = tf.gather_nd(long_indices, xy)
        print "res: ",res.shape
        indices_good = tf.reshape(res,[N1,col_batch_size])
        


        print "emb_score shape: ",emb_score.shape
        print "emb_long_score shape: ",emb_long_score.shape
        print "values shape: ",values.shape
        print "indices shape: ",indices.shape
        # indices_good = long_indices[indices]

        updateValue = tf.assign(oldScoreGPU,values,name="update")
        updateIndices = tf.assign(oldIndices,indices_good,name="update")
        # indices = indices


        # scoreGPU2 = np.zeros((N1,len(featPathList2)),dtype=np.float32)
        config = tf.ConfigProto(
            # device_count = {'GPU': gpu_id},
            # log_device_placement=True
            )  
        config.gpu_options.allow_growth=True  
        # config.gpu_options.per_process_gpu_memory_fraction=expect_mem_percent  
        # config.gpu_options.per_process_gpu_memory_fraction=0.5  

        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        writer = tf.summary.FileWriter(os.path.join(_BASE_DIR_ , "log"),tf.get_default_graph())
        print "trying to use gpu: ",gpu_id
        with tf.Session(config=config) as sess:
            sess.run(init_g)
            sess.run(init_l)
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            for ibatch in range(batch_number):
                print ibatch

                new_emb_b_cpu = get_emb_cpu(featPathList2, ibatch, col_batch_size)
                print new_emb_b_cpu.shape
                iScoreGPU,indices = sess.run([updateValue, updateIndices], feed_dict={
                    emb_a: emb_a_cpu,
                    emb_b: new_emb_b_cpu,
                    new_indices: get_new_indices(galleryIndices, ibatch, col_batch_size,N1),
                    feed_size:new_emb_b_cpu.shape[1],
                    },
                    options=run_options,
                    run_metadata=run_metadata)
                iScoreGPU = np.array(iScoreGPU)
                if len(np.where(iScoreGPU >1e-5)[0])>0:
                    axxx = np.where(iScoreGPU >1e-5)
                    print "WARNING: .......~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",len(axxx[0])
                    print iScoreGPU[axxx][0]
                    print axxx
                    print axxx[0][0]
                    print featPathList1[axxx[0][0]]
                    print indices[axxx][0]
                    # print galleryIndices[indices[axxx][0]]
                    print featPathList2[indices[axxx][0]]
                    print featPathList2[86249]
                    found = [p for p in featPathList2 if "5011260955" in p][0]
                    found_ind  = featPathList2.index(found)
                    print found, found_ind
                    print indices[axxx[0],found_ind]
                    # print featPathList2[galleryIndices[indices[axxx][0]]]

                # print iScoreGPU, indices
               
                writer.add_run_metadata(run_metadata, 'batch: %d' % ibatch)


        

        run_time = timer() - start  
        print("GPU time %f seconds " % run_time)    
        writer.close()


    iScoreGPU = 1.0/(-1.0*iScoreGPU  + 1 )
    return iScoreGPU, indices

def parse_arguments(argv):    
    parser = argparse.ArgumentParser(
        description='get similarity score and save',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('model', help='not used place holder')
    # parser.add_argument('path', help='not used place holder')
    parser.add_argument('--probe-list',help='')
    parser.add_argument('--gallery-list', help='')
    parser.add_argument('--output-dir', help='')
    parser.add_argument('--gpus', default = "7", help='')
    
    return parser.parse_args(argv)
if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)

    #  https://wiki.tiker.net/PyCuda/Installation/Linux/Ubuntu
    # wget https://files.pythonhosted.org/packages/b3/30/9e1c0a4c10e90b4c59ca7aa3c518e96f37aabcac73ffe6b5d9658f6ef843/pycuda-2017.1.1.tar.gz


    # python gab_similarity.py --probe-list /home/user/beijing_face_new/split_files/Test1_fea/query_list --gallery-list /home/user/beijing_face_new/split_files/Test1_fea/db_list --output-dir /home/user/beijing_face_new/scores_Test1/


    # python my_similarity_1.py ../models/jb_identity.bin path /home/guoxufeng/work/data/bulkMega/tmp_just_a_exp_test/otherFiles/facescrub_features_arcface_112x112 /home/guoxufeng/work/data/bulkMega/tmp_just_a_exp_test/otherFiles/facescrub_features_arcface_112x112 /home/guoxufeng/work/data/bulkMega/tmp_just_a_exp_test/otherFiles/facescrub_facescrub_arcface_112x112.bin

    # python my_similarity_2.py ../models/jb_identity.bin path /home/guoxufeng/work/faceRec/devkit/felix/special_feature/results/otherFiles/probe_features_arcface_112x112  /home/guoxufeng/work/faceRec/devkit/felix/special_feature/results/otherFiles/megaface_features_arcface_112x112_10_1  /home/guoxufeng/work/faceRec/devkit/felix/special_feature/results/otherFiles/facescrub_facescrub_arcface_112x112.bin

    

    

    # python gab_similarity.py --probe-list /home/guoxufeng/work/data/special_feature/results/otherFiles/probe_features_arcface_112x112 --gallery-list /home/guoxufeng/work/data/special_feature/results/otherFiles/megaface_features_arcface_112x112_10000_1 --output-dir /home/guoxufeng/work/data/bulkMega/test-GPU-dist/otherFiles/

    # python my_similarity_4.py  /home/guoxufeng/work/data/bulkMega/result_r100-vgg-9991/otherFiles/facescrub_features_arcface_112x112 /home/guoxufeng/work/data/bulkMega/result_r100-vgg-9991/otherFiles/megaface_features_arcface_112x112_1000000_1 /home/guoxufeng/work/data/bulkMega/test-GPU-dist/otherFiles/facescrub_megaface_arcface_112x112_1000000_1.bin

    # CUDA_VISIBLE_DEVICES="4,5,6,7"  python my_similarity_4.py  /home/guoxufeng/work/data/special_feature/results/otherFiles/probe_features_arcface_112x112  /home/guoxufeng/work/data/special_feature/results/otherFiles/megaface_features_arcface_112x112_10_1  /home/guoxufeng/work/data/special_feature/results/otherFiles/facescrub_facescrub_arcface_112x112.bin

    # python gab_similarity.py  --probe-list /home/guoxufeng/work/data/bulkMega/result_r100-vgg-9991/otherFiles/megaface_features_arcface_112x112_100000_1 --gallery-list /home/guoxufeng/work/data/bulkMega/result_r100-vgg-9991/otherFiles/megaface_features_arcface_112x112_1000000_1 --output-dir /home/guoxufeng/work/data/bulkMega/test-GPU-dist/otherFiles/

	# /home/guoxufeng/work/data/bulkMega/mega_fea_r100-vgg-9991_cm/876/87667287@N00/4440872808_0.jpg_arcface_112x112.bin

	# '../bin/Identification', 
	# '../models/jb_identity.bin', 
	# 'path', 
	# '/home/guoxufeng/work/data/bulkMega/tmp_just_a_exp_test/otherFiles/facescrub_features_arcface_112x112', 
	# '/home/guoxufeng/work/data/bulkMega/tmp_just_a_exp_test/otherFiles/facescrub_features_arcface_112x112', 
	# '/home/guoxufeng/work/data/bulkMega/tmp_just_a_exp_test/otherFiles/facescrub_facescrub_arcface_112x112.bin'
	# /home/guoxufeng/work/data/bulkMega/mega_fea_r100-vgg-9991_cm/876/87667287@N00/4440872808_0.jpg_arcface_112x112.bin
	# {
	#     "id": [
	#         "Michael_Landes", 
	#         "Michael_Landes", 
