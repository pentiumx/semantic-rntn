import optparse
import cPickle as pickle

import sgd as optimizer
import rntn as nnet
import tree_rte as tr
import rnn_entailment as nnet_rte
import process_data as process_data
import time

def run(args=None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--test",action="store_true",dest="test",default=False)

    # Optimizer
    parser.add_option("--minibatch",dest="minibatch",type="int",default=10)#30
    parser.add_option("--optimizer",dest="optimizer",type="string",
        default="adagrad")
    parser.add_option("--epochs",dest="epochs",type="int",default=50)#50
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


    (opts,args)=parser.parse_args(args)

    # Testing
    if opts.test:
        test(opts.inFile,opts.data,opts.dataset)
        return

    print "Loading data..."
    # load training data
    trees, vocab = tr.loadTrees(opts.dataset,opts.data)
    opts.numWords = len(tr.loadWordMap())


    print "Loading word2vec vectors..."
    # Load pre-built word matrix using cPickle
    #w2v_file = "/Users/pentiumx/Projects/word2vec/GoogleNews-vectors-negative300.bin"
    #word_vecs = process_data.load_bin_vec(w2v_file, vocab)
    x = pickle.load(open("mr.p","rb"))
    #revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    W = x[0]

    #rnn = nnet.RNN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
    rnn = nnet_rte.RNNRTE(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
    rnn.initParams(W)

    sgd = optimizer.SGD(rnn,alpha=opts.step,minibatch=opts.minibatch,
        optimizer=opts.optimizer)

    for e in range(opts.epochs):
        start = time.time()
        print "Running epoch %d" % e
        sgd.run(trees)
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

def test(netFile,data, dataset):
    #trees = tr.loadTrees(dataSet)
    trees, vocab = tr.loadTrees(dataset,data)
    assert netFile is not None, "Must give model to test"
    with open(netFile,'r') as fid:
        opts = pickle.load(fid)
        _ = pickle.load(fid)
        #rnn = nnet.RNN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)

        rnn = nnet_rte.RNNRTE(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
        rnn.initParams()
        rnn.fromFile(fid)
    print "Testing..."
    cost,correct,total = rnn.costAndGrad(trees,test=True)
    print "Cost %f, Correct %d/%d, Acc %f"%(cost,correct,total,correct/float(total))


if __name__=='__main__':
    run()


