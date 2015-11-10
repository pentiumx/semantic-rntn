import numpy as np
import collections
import rnn as rnn
import cPickle as pickle
#from pympler import tracker

def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2

# RNN for textual entailment task.
# Compute the word vectors for two sentences, and merge them at the comparison NN layer.
class RNNRTE:

    def __init__(self,wvecDim,outputDim,embeddingDim,numWords,mbSize=30,rho=1e-4):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.embeddingDim = embeddingDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho


    # Init params for the comparison layer
    def initParams(self, W, dropout=False):
        # Word vectors
        #self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)# why not [numWords, wvecDim]?
        self.L = W
        self.dropout = dropout
        random_seed = 1234
        self.rng = np.random.RandomState(random_seed)

        # embedding transformation layer weights
        # embeddingDim = word2vec dim, wvecDim = composition layer's word vec dim.
        self.We = 0.01*np.random.rand(self.wvecDim, self.embeddingDim)
        self.be = np.zeros((self.wvecDim))

        # Comparison layer weights
        self.Wc = 0.01*np.random.randn(self.wvecDim,2*self.wvecDim)
        self.bc = np.zeros((self.wvecDim))

        # Hidden activation weights
        self.W = 0.01*np.random.randn(self.wvecDim,2*self.wvecDim)
        self.b = np.zeros((self.wvecDim))

        # Softmax weights
        self.Ws = 0.01*np.random.randn(self.outputDim,self.wvecDim)
        self.bs = np.zeros((self.outputDim))

        self.stack = [self.L, self.W, self.b, self.Ws, self.bs, self.Wc, self.bc, self.We, self.be]

        # Gradients
        self.dW = np.empty(self.W.shape)
        self.db = np.empty((self.wvecDim))
        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty((self.outputDim))
        self.dWc = np.empty(self.W.shape)
        self.dbc = np.empty((self.wvecDim))
        self.dWe = np.empty(self.We.shape)
        self.dbe = np.empty((self.wvecDim))

    def init_softmax_params(self):
        # Softmax weights
        self.Ws = 0.01*np.random.randn(self.outputDim,self.wvecDim)
        self.bs = np.zeros((self.outputDim))
        # Update stack
        self.stack = [self.L, self.W, self.b, self.Ws, self.bs, self.Wc, self.bc, self.We, self.be]

    # Update params other than softmax weights and word vectors.
    def transfer_params(self, stack):
        self.stack = [self.L, stack[1], stack[2], self.Ws, self.bs, stack[5], stack[6], stack[7], stack[8]]


    def costAndGrad(self,mbdata,test=False):
        """
        mbdata => [TreePair]

        Each datum in the minibatch is a set of two trees (two sentences).
        Forward prop each tree.
        Backprop each tree.
        Returns
           cost
           Gradient w.r.t. W, Ws, b, bs
           Gradient w.r.t. L in sparse form.
        """
        cost = 0.0
        correct = 0.0
        total = 0.0

        #if not test:
        self.L,self.W,self.b,self.Ws,self.bs,self.Wc,self.bc,self.We,self.be = self.stack
        #else:
        #    # Other than word2vec vectors.
        #    self.W,self.b,self.Ws,self.bs,self.Wc,self.bc,self.We,self.be = self.stack[1:]
        # Zero gradients
        self.dW[:] = 0
        self.db[:] = 0
        self.dWs[:] = 0
        self.dbs[:] = 0
        self.dWc[:] = 0
        self.dbc[:] = 0
        self.dWe[:] = 0
        self.dbe[:] = 0
        self.dL = collections.defaultdict(self.defaultVec)

        # Initialize dL and other params of each rnn
        #self.rnn_left.init_param_grads()
        #self.rnn_right.init_param_grads()

        # Forward prop each tree in minibatch
        for i, tree_pair in enumerate(mbdata):
            c,corr,tot = self.forward_prop_all(tree_pair, test)

            cost += c
            correct += corr
            total += tot
            if test and i % 100 == 0:
                print i
        if test:
            return (1./len(mbdata))*cost,correct,total

        # Back prop each tree in mini-batch
        for tree_pair in mbdata:
            self.back_prop_all(tree_pair)

        # scale cost and grad by mb size
        scale = (1./self.mbSize)
        for v in self.dL.itervalues():
            v *=scale

        # Add L2 Regularization
        cost += (self.rho/2)*np.sum(self.W**2)
        cost += (self.rho/2)*np.sum(self.Ws**2)
        cost += (self.rho/2)*np.sum(self.Wc**2)
        cost += (self.rho/2)*np.sum(self.We**2)

        return scale*cost,[self.dL,scale*(self.dW + self.rho*self.W),scale*self.db,
                           scale*(self.dWs+self.rho*self.Ws),scale*self.dbs,scale*(self.dWc+self.rho*self.Wc),scale*self.dbc,scale*(self.dWe+self.rho*self.We),scale*self.dbe]


    def forward_prop_all(self, tree_pair, test=False):
        cost = correct =  total = 0.0

        # Get the feature of both sentences
        # TODO: do some additional initializations to each rnn's parameters?

        # sent_left/sent_right: Node class. we need to use sent_left.hActs for using representation!!!
        sent_left = self.forwardProp(tree_pair.tree1.root, test)
        sent_right = self.forwardProp(tree_pair.tree2.root, test)

        # > we used dropout (Srivastavaet al., 2014) at the input to the comparison layer (10%)
        input = np.hstack([tree_pair.tree1.root.hActs, tree_pair.tree2.root.hActs])
        if self.dropout and not test:
            input *= np.random.binomial(1, 1.-0.1, input.shape)

        # Propagate to the comparison layer.
        # Use the representations!
        # Affine
        if self.dropout and test:
            # >  At test time, the weights are scaled as W^(l)test = pW^(l)
            hActs = np.dot(self.Wc*(1.-0.1), input) + self.bc
        else:
            hActs = np.dot(self.Wc, input) + self.bc
        # Relu
        hActs[hActs<0] = 0

        # Softmax layer
        probs = np.dot(self.Ws,hActs) + self.bs
        probs -= np.max(probs)
        probs = np.exp(probs)
        probs = probs/np.sum(probs)

        # TODO: Create a class for softmax layer
        # Save softmax layer values as member vars (temporary solution)
        tree_pair.probs = probs
        tree_pair.hActs = hActs

        label = tree_pair.label

        return cost - np.log(probs[label]), correct + (np.argmax(probs)==label), total+1

    # USED: RNN.forwardProp() will not be called from forward_prop_all()
    # Forward propagate each RNN. Returns the output vector of each sentence!
    def forwardProp(self,node, test=False):
        cost = correct = total = 0.0

        if node.isLeaf:
            # > Before any embedding is used as an input to a recursive layer, it is passed
            # > through an additional tanh neural network layer with the same output dimension as the recursive layer
            if self.dropout and test:
                node.hActs = np.tanh(np.dot(self.We*(1.-0.25), self.L[node.word]) + self.be)
            else:
                node.hActs = np.tanh(np.dot(self.We, self.L[node.word]) + self.be)

            # > we used dropout (Srivastavaet al., 2014) ... at the output from the embedding transform layer (25%)
            if self.dropout and not test:
                node.hActs *= np.random.binomial(1, 1.-0.25, node.hActs.shape)
            node.fprop = True

        else:
            if not node.left.fprop:
                c,corr,tot = self.forwardProp(node.left, test)
                cost += c
                correct += corr
                total += tot
            if not node.right.fprop:
                c,corr,tot = self.forwardProp(node.right, test)
                cost += c
                correct += corr
                total += tot
            # Affine
            node.hActs = np.dot(self.W,
                    np.hstack([node.left.hActs, node.right.hActs])) + self.b
            # Relu
            node.hActs[node.hActs<0] = 0

        # Softmax
        node.probs = np.dot(self.Ws,node.hActs) + self.bs
        node.probs -= np.max(node.probs)
        node.probs = np.exp(node.probs)
        node.probs = node.probs/np.sum(node.probs)
        node.fprop = True
        return cost - np.log(node.probs[node.label]), correct + (np.argmax(node.probs)==node.label),total + 1


    def back_prop_all(self, tree_pair):
        # Softmax layer grad
        # cross entropy error: dE/da = y-t = deltas
        # E.g. [0.11, 0.0008, 0.5] - [0, 1, 0]
        self.deltas = tree_pair.probs
        self.deltas[tree_pair.label] -= 1.0 # target value = 1
        # derivative of weights(softmax layer):
        # dE/dw = delta*z = deltas*hActs
        self.dWs += np.outer(self.deltas, tree_pair.hActs)
        self.dbs += self.deltas
        # TODO: Save splitted deltas into a custom comparison layer class
        # derivative of weights(comparison hidden layer)
        # delta_j = h'(a_j)*sigma_k(w_kj*delta_k)
        # delta^(l)=h'(a)*sum(w*delta^(l+1)
        # (for calc) => delta^(l) = W^(l).T * delta^(l+1) (element-wise *) f'(z^(l))
        self.deltas = np.dot(self.Ws.T,self.deltas)
        self.deltas *= (tree_pair.hActs != 0)

        # Comparison layer grad
        #self.dWe += np.dot(np.atleast_2d(deltas_local).T, np.atleast_2d(self.L[node.word]))
        #self.dbe += deltas_local
        # dE/dW^(l) = delta^(l+1) * (a^(l))^T + lamda*W^(l)
        # The regularization will be added later

        # This layer isn't a recursive layer, so perhaps I shouldn't use outer product?
        #self.dWc += np.outer(self.deltas, np.hstack([tree_pair.tree1.root.hActs, tree_pair.tree2.root.hActs]))
        self.dWc += np.dot(np.atleast_2d(self.deltas).T, np.atleast_2d(np.hstack([tree_pair.tree1.root.hActs, tree_pair.tree2.root.hActs])) )
        self.dbc += self.deltas


        # Calc deltas at the top layers of the trees
        # delta^(l) = W^(l).T * delta^(l+1) (element-wise *) f'(z^(l))
        self.deltas = np.dot(self.Wc.T, self.deltas)

        # NOTE: make sure to create deep copy
        self.deltas_left = np.empty_like(self.deltas[:self.wvecDim])
        self.deltas_left[:] = self.deltas[:self.wvecDim]
        self.deltas_right = np.empty_like(self.deltas[self.wvecDim:])
        self.deltas_right[:] = self.deltas[self.wvecDim:]

        # Composition layers grad. dL, dW, db will be calculated.
        self.backProp(tree_pair.tree1.root, self.deltas_left)
        self.backProp(tree_pair.tree2.root, self.deltas_right)

    def backProp(self,node,deltas,error=None):
        # Clear nodes
        node.fprop = False

        # self.deltas is already calculated
        # [NOTE: DO NOT assign the instance itself. make sure to create a deep copied value!!!]
        deltas_local = np.empty_like(deltas)
        deltas_local[:] = deltas    # self.deltas has wvecDim

        if error is not None:
            deltas_local += error

        # Complete the calculation of delta at the current layer
        # delta^(l) = W^(l).T * delta^(l+1) (element-wise *) f'(z^(l))
        if node.isLeaf:
            deltas_local *= (node.hActs != 0)
        else:
            deltas_local *= (node.hActs != 0)


        # Leaf nodes update word vecs
        if node.isLeaf:
            # original:
            #self.dL[node.word] += deltas
            # leaf nodes have no split
            #self.dW += np.outer(deltas_local, node.hActs)
            #self.db += deltas_local

            # NOW we compute the gradients of the embedding transformation layer
            #delta = np.dot(self.We.T, deltas_local)
            #delta *= tanh_deriv(self.L[node.word])

            #delta = np.atleast_2d(delta)
            #self.dWe += np.outer(deltas, self.L[node.word])
            #self.dWe += np.dot(np.atleast_2d(node.hActs).T, np.atleast_2d(delta))

            # deltas_local IS ALREADY the delta of this hidden layer.
            self.dWe += np.dot(np.atleast_2d(deltas_local).T, np.atleast_2d(self.L[node.word]))
            self.dbe += deltas_local
            return

        # Hidden grad
        if not node.isLeaf:
            # calc gradient for a composition layer
            self.dW += np.outer(deltas_local,
                    np.hstack([node.left.hActs, node.right.hActs]))
            self.db += deltas_local

            # Error signal to children.
            # f'(z^(l)) will be multiplied at the begging of the next recursion
            if node.left.isLeaf:
                deltas = np.dot(self.W.T, deltas_local)                 # deltas.size = wvecDim*2 at this point
            else:
                deltas = np.dot(self.W.T, deltas_local)                 # deltas.size = wvecDim*2 at this point
            self.backProp(node.left, deltas[:self.wvecDim])
            self.backProp(node.right, deltas[self.wvecDim:])


    def updateParams(self,scale,update,log=False):
        """
        Updates parameters as
        p := p - scale * update.
        If log is true, prints root mean square of parameter
        and update.
        """
        if log:
            for P,dP in zip(self.stack[1:],update[1:]):
                pRMS = np.sqrt(np.mean(P**2))
                dpRMS = np.sqrt(np.mean((scale*dP)**2))
                print "weight rms=%f -- update rms=%f"%(pRMS,dpRMS)

        self.stack[1:] = [P+scale*dP for P,dP in zip(self.stack[1:],update[1:])]
        # handle dictionary update sparsely
        # don't update it while we're using word2vec vectors
        """
        dL = update[0]
        for j in dL.iterkeys():
            self.L[:,j] += scale*dL[j]
        """

    def toFile(self,fid, last=False):
        pickle.dump(self.stack,fid)
        if last:
            print ''

    def fromFile(self,fid):
        # Load params other than word vector dictionary
        self.stack[1:] = pickle.load(fid)[1:]

    # NOT USED
    def from_file_denotation(self, fid):
        inputs = pickle.load(fid)

        # Softmax weights
        self.Ws = 0.01*np.random.randn(self.outputDim,self.wvecDim)
        self.bs = np.zeros((self.outputDim))
        self.stack = [self.L, inputs[1], inputs[2], self.Ws, self.bs, inputs[5], inputs[6], inputs[7], inputs[8]]



    # 1:W, 2:b, 3:Ws, 4:bs, 5: Wc, 6:bc, 7:We, 8:be
    def check_grad(self,data,epsilon=1e-6):

        cost, grad = self.costAndGrad(data)
        wegihts= ['W', 'b', 'Ws', 'bs', 'Wc', 'bc', 'We', 'be']
        index = 0
        for W,dW in zip(self.stack[1:],grad[1:]):# makes a list of tuple(stack[i], grad[i])
            W = W[...,None] # add dimension since bias is flat
            dW = dW[...,None]
            print wegihts[index]
            index += 1
            for i in xrange(W.shape[0]):#line. E.g. W.shape=(10,20,1)
                for j in xrange(W.shape[1]):#row
                    W[i,j] += epsilon
                    costP,_ = self.costAndGrad(data)
                    W[i,j] -= epsilon
                    numGrad = (costP - cost)/epsilon
                    err = np.abs(dW[i,j] - numGrad)
                    print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dW[i,j],numGrad,err)

        # This model uses word2vec
        # check dL separately since dict
        """
        dL = grad[0]
        L = self.stack[0]
        for j in dL.iterkeys():
            for i in xrange(L.shape[0]):
                L[i,j] += epsilon
                costP,_ = self.costAndGrad(data)
                L[i,j] -= epsilon
                numGrad = (costP - cost)/epsilon
                err = np.abs(dL[j][i] - numGrad)
                print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dL[j][i],numGrad,err)
        """

if __name__ == '__main__':

    import tree_rte as treeM
    train, vocab = treeM.loadTrees()
    numW = len(treeM.loadWordMap('sick'))

    wvecDim = 10
    outputDim = 5
    print numW

    x = pickle.load(open("mr.p","rb"))
    W = x[0]
    rnn = RNNRTE(wvecDim,outputDim,200,numW,mbSize=4)
    rnn.initParams(W)

    mbData = train[:4]

    print "Numerical gradient check..."
    rnn.check_grad(mbData)
    #rnn.check_single_grad(mbData, 7)
    #rnn.check_single_grad(mbData, 8)






