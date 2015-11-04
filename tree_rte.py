import optparse
import collections
import pygraphviz as PG
import process_data as pd
import numpy as np
import cPickle as pickle
UNK = 'UNK'

# Container of two trees
class TreePair:
    def __init__(self, tree1, tree2, label=None, relatedness=5.0, index=None):
        self.tree1 = tree1
        self.tree2 = tree2
        self.label = label
        self.relatedness = relatedness
        self.index = index

class Node:
    #def __init__(self,label,word=None):
    def __init__(self,label=None,word=None):
        self.label = label
        self.word = word
        self.parent = None
        self.left = None
        self.right = None
        self.isLeaf = False
        self.fprop = False

# Read consitutuency parse trees. Tuned for reading SICK dataset.
class Tree:

    # treeString: class, sentence_A, sentence_B separated by '\t'.
    # E.g. NEUTRAL\t((( A group ) ( of ( kids ( is...\t((( A group ) ( of ( boys ( is...
    def __init__(self,treeString,openChar='(',closeChar=')'):
        tokens = []
        self.open = '('
        self.close = ')'

        # Remove all whitespaces from the start and end of the treeString.
        # Then split the string by space.
        for toks in treeString.strip().split():
            tokens += list(toks)

        # tokens[1:-1] is not necessary cuz we set split=1 in the parse method
        self.root = self.parse(tokens)


    def parse(self, tokens, parent=None):
        # no asset for no-sentiment analysis tasks
        # position after open and label
        split = 1
        after_label = split
        countOpen = countClose = 0

        #if tokens[split] == ' ': split += 1
        if tokens[split] == self.open:
            countOpen += 1
            split += 1
        # Find where left child and right child split
        while countOpen != countClose:
            if tokens[split] == self.open:
                countOpen += 1
            if tokens[split] == self.close:
                countClose += 1
            split += 1

        # New node
        node = Node(None, tokens[after_label:-1]) # no labels for no-sentiment analysis tasks
        node.parent = parent

        # leaf Node
        if countOpen == 0:
            node.word = ''.join(tokens[1:-1]).lower() # lower case?
            node.isLeaf = True
            return node

        node.left = self.parse(tokens[1:split],parent=node)
        node.right = self.parse(tokens[split:-1],parent=node)    # split should point the start point of a parenthese
        return node




def leftTraverse(root,nodeFn=None,args=None):
    """
    Recursive function traverses tree
    from left to right.
    Calls nodeFn at each node
    """
    nodeFn(root,args)
    if root.left is not None:
        leftTraverse(root.left,nodeFn,args)
    if root.right is not None:
        leftTraverse(root.right,nodeFn,args)

def countWords(node,words):
    if node.isLeaf:
        words[node.word] += 1

def mapWords(node,wordMap):
    if node.isLeaf:
        if node.word not in wordMap:
            node.word = wordMap[UNK]
        else:
            node.word = wordMap[node.word]


def create_graph(root, graph, is_root=False, label='root'):
    """
    Recursive function traverses tree
    from left to right.
    Calls nodeFn at each node
    """
    print root.word

    if is_root:
        graph.add_node(label)

    if root.left is not None:
        left_label = 'a'
        if root.left.isLeaf:
            graph.add_node(root.left.word)
            graph.add_edge(label, root.left.word)
        else:
            left_label = label+'L'
            graph.add_node(left_label)
            graph.add_edge(label, left_label)
        create_graph(root.left, graph, False, left_label)
    if root.right is not None:
        right_label = 'r'
        if root.right.isLeaf:
            graph.add_node(root.right.word)
            graph.add_edge(label, root.right.word)
        else:
            right_label = label+'R'
            graph.add_node(right_label)
            graph.add_edge(label, right_label)
        create_graph(root.right, graph, False, right_label)

# Debug scanned trees by visualizing them.
def debug_tree(root):
    graph = PG.AGraph(directed=True, strict=True)
    create_graph(root, graph, True)
    # save the graph in dot format
    graph.write('tree_debug.dot')


def loadWordMap(dataset):
    if dataset == 'denotation' or dataset=='denotation_sample':
        with open('wordMap_denotation.bin','r') as fid:
            return pickle.load(fid)
    else:
        with open('wordMap_%s.bin' % dataset,'r') as fid:
            return pickle.load(fid)


# Split a list into evenly sized chunks
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

# Get the tree instance of all sentences from the dataset
def buildWordMap(dataset='sick', data='train_parsed'):
    """
    Builds map of all words in training set
    to integer values.
    """

    import cPickle as pickle
    file = 'trees/%s_%s' % (dataset, data)
    print "Reading trees.."

    with open(file,'r') as fid:
        l = list(inputarray(fid.readlines(), dataset))
        tree_pairs = l

    print "Counting words.."
    words = collections.defaultdict(int)
    for tree_pair in tree_pairs:
        leftTraverse(tree_pair.tree1.root,nodeFn=countWords,args=words)
        leftTraverse(tree_pair.tree2.root,nodeFn=countWords,args=words)

    wordMap = dict(zip(words.iterkeys(),xrange(len(words))))
    wordMap[UNK] = len(words) # Add unknown as word

    with open('wordMap.bin','w') as fid:
        pickle.dump(wordMap,fid)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def get_lines(dataset, data):
    file = 'trees/%s_%s' % (dataset, data)

    with open(file,'r') as fid:
        lines = fid.readlines()
    return lines

# Build a word map from a very large dataset.
# To avoid memory problems, load them batch by batch.
def build_wordmap_with_batch(dataset='sick', data='train_parsed'):
    """
    Builds map of all words in training set
    to integer values.
    """

    file = 'trees/%s_%s' % (dataset, data)
    print "Reading trees.."

    lines = []
    i = 0
    CHUNK_SIZE = 10000
    with open(file,'r') as fid:
        lines = fid.readlines()
        #l = list(inputarray(fid.readlines(), dataset))
        #tree_pairs = l
    current_tree_pairs=[]
    chunked_lines=list(chunks(lines, CHUNK_SIZE))
    print len(lines)
    print len(chunked_lines)

    print "Counting words.."
    words = collections.defaultdict(int)
    index=0
    for chunk in chunked_lines:
        tree_pairs = list(inputarray(chunk, dataset))
        i=0
        for tree_pair in tree_pairs:
            leftTraverse(tree_pair.tree1.root,nodeFn=countWords,args=words)
            leftTraverse(tree_pair.tree2.root,nodeFn=countWords,args=words)
            if i%1000==0: print 'words counted in %d pairs' % (index*CHUNK_SIZE+i)
            i+=1
        index+=1

    wordMap = dict(zip(words.iterkeys(),xrange(len(words))))
    wordMap[UNK] = len(words) # Add unknown as word


    with open('wordMap_%s.bin' % dataset,'w') as fid:
        pickle.dump(wordMap,fid)

# Map words with wordmap
def map_words_to_trees(tree_pairs, dataset='denotation'):
    wordMap = loadWordMap(dataset)
    for tree_pair in tree_pairs:
        leftTraverse(tree_pair.tree1.root,nodeFn=mapWords,args=wordMap)
        leftTraverse(tree_pair.tree2.root,nodeFn=mapWords,args=wordMap)

# To get the word that has an arbitrary index.
def createIndexWordMap(wordMap):
    idx_word_map = []
    index=0
    keys=wordMap.keys()
    values=wordMap.values()
    for word in wordMap:
        idx_word_map.append(keys[values.index(index)])
        index+=1
    return idx_word_map



# For creating TreePair instances.
def inputarray(lines, dataset='sick'):
    if dataset == 'sick':
        labels = {'ENTAILMENT':0, 'NEUTRAL':1, 'CONTRADICTION':2}
        for l in lines:
            tmp = l.split('\t')
            # Note the order of arguments.
            # Change below depends on your RTE dataset.
            yield TreePair(Tree(tmp[1]), Tree(tmp[2]), labels[tmp[0]], tmp[3], tmp[4])
    elif dataset == 'denotation' or dataset == 'denotation_sample':
        labels = {'ENTAILMENT':0, 'NOT_ENTAILMENT':1}
        i=0
        for l in lines:
            tmp = l.split('\t')
            #if i % 10000 == 0:
            #    print 'parsing trees: index=%d'%i
            i+=1
            yield TreePair(Tree(tmp[1]), Tree(tmp[2]), labels[tmp[0]])

def input_quant(lines):
    labels = {'<':0, '>':1, '=':2, '|':3, '^':4, 'v':5, '#':6}
    for l in lines:
        tmp = l.split('\t')
        # Note the order of arguments.
        # Change below depends on your RTE dataset.
        yield TreePair(Tree(tmp[1]), Tree(tmp[2]), labels[tmp[0]])

# Get word matrix and word map based on dataset's vocab.
def get_W(word_vecs, word_map, idx_word_map, k=300):
    vocab_size = len(word_vecs)
    W = np.zeros(shape=(vocab_size+1, k))

    for i in range(0,len(idx_word_map)):
         W[i] = word_vecs[idx_word_map[i]]
    return W


def loadTrees(dataset='sick', data='train_parsed'):
    """
    Loads training trees. Maps leaf node words to word ids.
    """
    wordMap = loadWordMap(dataset)
    file = 'trees/%s_%s' % (dataset, data)

    print file
    print "Reading trees.."

    with open(file, 'r') as fid:
        l = list(inputarray(fid.readlines(), dataset))
        tree_pairs = l


    # Run mapWords on each node.
    # Comment out this section when you need to check pygraphviz tree visualization.
    for tree_pair in tree_pairs:
        leftTraverse(tree_pair.tree1.root,nodeFn=mapWords,args=wordMap)
        leftTraverse(tree_pair.tree2.root,nodeFn=mapWords,args=wordMap)
    return tree_pairs, wordMap


def run(args=None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--test",action="store_true",dest="test",default=False)
    parser.add_option("--data",dest="data",type="string",default="train")
    parser.add_option("--dataset",dest="dataset",type="string",default="sick")
    (opts,args)=parser.parse_args(args)

    print 'Building word map...'
    #buildWordMap(opts.dataset,'train_parsed')
    build_wordmap_with_batch(opts.dataset, 'train_parsed')

    #train, wordMap = loadTrees(opts.dataset, 'train_parsed')
    wordMap = loadWordMap(opts.dataset)
    idx_word_map = createIndexWordMap(wordMap)
    print 'vocab size:%s' % len(wordMap)

    print 'Loading word2vec...'
    # word2vec
    #w2v_file = "/Users/pentiumx/Projects/word2vec/GoogleNews-vectors-negative300.bin"
    #word_vecs = pd.load_bin_vec(w2v_file, wordMap)
    # Glove
    w2v_file = "/Users/pentiumx/Projects/ms_scripts/semantic-rntn/word_vectors/glove.6B.200d.txt"
    word_vecs = pd.load_glove_vec(w2v_file, wordMap)
    print 'asian:%s'%word_vecs['asian']
    print type(word_vecs)
    print len(word_vecs)

    print 'Adding unknown words...'
    pd.add_unknown_words(word_vecs, wordMap, 0, 200)# wordMap is not countWOrds array.
    W = get_W(word_vecs, wordMap, idx_word_map, 200)
    print 'word matrix:\n%s' % W
    #print 'word_idx_map:\n%s' % word_idx_map
    pickle.dump([W], open("mr_%s.p" % opts.dataset, "wb"))
    #debug_tree(train[0].tree1.root)


if __name__=='__main__':
    run()