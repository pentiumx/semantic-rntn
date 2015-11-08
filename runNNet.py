import optparse
import cPickle as pickle

import sgd as optimizer
import rntn as nnet
import tree_rte as tr
import rnn_entailment as nnet_rte
import process_data as process_data
import time
import numpy as np


def start(opts):
    print "Loading data..."
    # load training data
    trees, vocab = tr.loadTrees(opts.dataset,opts.data) # sick, train_parsed
    opts.numWords = len(tr.loadWordMap(opts.dataset))

    print "Loading word2vec vectors..."
    # Load pre-built word matrix using cPickle
    #w2v_file = "/Users/pentiumx/Projects/word2vec/GoogleNews-vectors-negative300.bin"
    #word_vecs = process_data.load_bin_vec(w2v_file, vocab)
    #revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]

    x = pickle.load(open("mr_%s.p" % opts.dataset, "rb"))
    W = x[0]
    W2 = 0.01*np.random.randn(opts.wvecDim,opts.numWords)

    #rnn = nnet.RNN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
    # embeddingDim=200 for now


    if opts.use_denotation == 0:
        rnn = nnet_rte.RNNRTE(opts.wvecDim,opts.outputDim,200,opts.numWords,opts.minibatch)
        rnn.initParams(W)   # Use W2 for experiments with randomly initialized vectors
        sgd = optimizer.SGD(rnn,alpha=opts.step,minibatch=opts.minibatch,
            optimizer=opts.optimizer)
    else:
        """with open('models/denotation_sample.bin','r') as fid:
            _ = pickle.load(fid)# skip opts data
            __ = pickle.load(fid)

            x = pickle.load(open("mr_%s.p" % opts.dg_dataset, "rb"))
            W_dg = x[0]
            rnn = nnet_rte.RNNRTE(opts.wvecDim,opts.outputDim,200,opts.numWords,opts.minibatch)
            rnn.initParams(W, W_dg)
            rnn.from_file_denotation(fid)

            sgd = optimizer.SGD(rnn,alpha=opts.step,minibatch=opts.minibatch,
                optimizer=opts.optimizer)"""
        rnn = nnet_rte.RNNRTE(opts.wvecDim,opts.outputDim,200,opts.numWords,opts.minibatch)
        rnn.initParams(W)

        x = pickle.load(open("mr_%s.p" % opts.dg_dataset, "rb"))
        W_dg = x[0]
        rnn_dg = nnet_rte.RNNRTE(opts.wvecDim,2,200,opts.numWords,opts.minibatch)
        rnn_dg.initParams(W_dg)

        sgd = optimizer.SGD(rnn,alpha=opts.step,minibatch=opts.minibatch,
            optimizer=opts.optimizer, model_dg=rnn_dg)


    for e in range(opts.epochs):
        start = time.time()
        print "Running epoch %d" % e
        if opts.use_denotation == 0:
            sgd.run(trees)
        else:
            lines = tr.get_lines(opts.dg_dataset, opts.data)
            sgd.run_using_denotation(trees, lines)
        end = time.time()
        print "Time per epoch : %f"%(end-start)

        with open(opts.outFile,'w') as fid:
            pickle.dump(opts,fid)
            pickle.dump(sgd.costt,fid)

            # debug
            if e == opts.epochs-1:
                rnn.toFile(fid, True)
            else:
                rnn.toFile(fid)

def start_denotation(opts):
    print "Loading data..."
    # load training data
    #trees, vocab = tr.loadTrees(opts.dataset,opts.data)
    #chunked_lines = tr.get_chunked_lines(opts.dataset, opts.data)
    opts.numWords = len(tr.loadWordMap(opts.dataset))
    print 'vocab size: %d' % opts.numWords

    print "Loading word2vec vectors..."
    # Load pre-built word matrix using cPickle
    x = pickle.load(open("mr_%s.p" % opts.dataset, "rb"))
    W = x[0]
    rnn = nnet_rte.RNNRTE(opts.wvecDim,opts.outputDim,200,opts.numWords,opts.minibatch)
    rnn.initParams(W)
    sgd = optimizer.SGD(rnn,alpha=opts.step,minibatch=opts.minibatch,
        optimizer=opts.optimizer)

    for e in range(opts.epochs):
        start = time.time()
        print "Running epoch %d" % e
        sgd.run_denotation(tr.get_lines(opts.dataset, opts.data))
        end = time.time()
        print "Time per epoch : %f"%(end-start)

        with open(opts.outFile,'w') as fid:
            pickle.dump(opts,fid)
            pickle.dump(sgd.costt,fid)

            # debug
            if e == opts.epochs-1:
                rnn.toFile(fid, True)
            else:
                rnn.toFile(fid)


def run(args=None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--test",action="store_true",dest="test",default=False)

    # Optimizer
    parser.add_option("--minibatch",dest="minibatch",type="int",default=30)#30
    parser.add_option("--optimizer",dest="optimizer",type="string",
        default="adagrad")
    parser.add_option("--epochs",dest="epochs",type="int",default=500)#50
    parser.add_option("--step",dest="step",type="float",default=1e-2)

    #parser.add_option("--outputDim",dest="outputDim",type="int",default=5)
    parser.add_option("--outputDim",dest="outputDim",type="int",default=3)  # 3: ENTAILMENT, NEUTRAL, CONTRADICTION for SICK dataset
    parser.add_option("--wvecDim",dest="wvecDim",type="int",default=30)#30
    parser.add_option("--outFile",dest="outFile",type="string",
        default="models/test.bin")
    parser.add_option("--inFile",dest="inFile",type="string",
        default="models/test.bin")
    parser.add_option("--data",dest="data",type="string",default="train")
    parser.add_option("--dataset",dest="dataset",type="string",default="sick")
    parser.add_option("--use_denotation",dest="use_denotation",type="int",default=0)
    parser.add_option("--dg_dataset",dest="dg_dataset",type="string",default='denotation_sample')

    (opts,args)=parser.parse_args(args)

    # Testing
    if opts.test:
        if opts.dataset == 'denotation' or opts.dataset=='denotation_sample':
            test_denotation(opts.inFile, opts.data, opts.dataset)
        else:
            test(opts.inFile,opts.data,opts.dataset)
        return

    if opts.dataset == 'denotation' or opts.dataset=='denotation_sample':
        start_denotation(opts)
    else:
        start(opts)

def test(netFile,data, dataset):
    #trees = tr.loadTrees(dataSet)
    trees, vocab = tr.loadTrees(dataset,data)
    assert netFile is not None, "Must give model to test"
    with open(netFile,'r') as fid:
        opts = pickle.load(fid)
        _ = pickle.load(fid)
        #rnn = nnet.RNN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)

        x = pickle.load(open("mr_%s.p" % dataset,"rb"))
        W = x[0]
        W2 = 0.01*np.random.randn(opts.wvecDim,opts.numWords)
        rnn = nnet_rte.RNNRTE(opts.wvecDim,opts.outputDim,200,opts.numWords,opts.minibatch)
        rnn.initParams(W)
        rnn.fromFile(fid)
    print "Testing..."
    cost,correct,total = rnn.costAndGrad(trees,test=True)
    print "Cost %f, Correct %d/%d, Acc %f"%(cost,correct,total,correct/float(total))


def test_denotation(netFile, data, dataset):
    #trees, vocab = tr.loadTrees(dataset,data)
    assert netFile is not None, "Must give model to test"
    with open(netFile,'r') as fid:
        opts = pickle.load(fid)
        _ = pickle.load(fid)


        x = pickle.load(open("mr_%s.p" % dataset,"rb"))
        W = x[0]
        W2 = 0.01*np.random.randn(opts.wvecDim,opts.numWords)
        rnn = nnet_rte.RNNRTE(opts.wvecDim,opts.outputDim,200,opts.numWords,opts.minibatch)
        rnn.initParams(W)
        rnn.fromFile(fid)

        lines = tr.get_lines(opts.dataset, opts.data)
        m = len(lines)
        CHUNK_SIZE=10000
        minibatch=opts.minibatch
        print "Testing..."

        cost=correct=total=0
        for i in xrange(0, m-CHUNK_SIZE+1, CHUNK_SIZE):
            # Get data as much as it's specified by minibatch num
            # Load&parse trees beforehand
            # Get data as much as it's specified by minibatch num
            # Parse&load tree data beforehand
            trees = list(tr.inputarray(lines[i:i+CHUNK_SIZE], dataset))

            # map word indices to loaded trees
            tr.map_words_to_trees(trees, dataset)

            c,cor,tot = rnn.costAndGrad(trees,test=True)
            cost+=c
            correct+=cor
            total+=tot
            if i % CHUNK_SIZE == 0:
                print 'tested: %d' % i

    print "Cost %f, Correct %d/%d, Acc %f"%(cost,correct,total,correct/float(total))



if __name__=='__main__':
    np.seterr(all='warn')   # Avoid Underflow error, and just output warning.
    run()


