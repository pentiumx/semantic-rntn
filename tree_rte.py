

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
            #print toks
            tokens += list(toks)

        # Remove the top parenthesis
        #self.root = self.parse(treeString.strip()[6:-2])
        #self.root = self.parse(treeString.strip()[2:-2])

        # tokens[1:-1] is not necessary cuz we set split=1 in the parse method
        self.root = self.parse(tokens)


    """def parse(self, tokens, parent=None, is_leaf=None):
        # no asset for no-sentiment analysis tasks
        #assert tokens[0] == self.open, "Malformed tree"
        #assert tokens[-1] == self.close, "Malformed tree"

        #split = 2 # position after open and label
        split = 0 # position after open
        after_label = split
        countOpen = countClose = 0

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
        #node = Node(int(tokens[1])-1) # zero index labels
        node = Node(None, tokens[after_label:-1]) # no labels for no-sentiment analysis tasks
        node.parent = parent

        # debugging
        print tokens[2:split-2]
        print tokens[split+3:-2]

        # leaf Node
        if countOpen == 0 and countClose == 0:
            tmp = tokens.replace('(','').replace(')','').strip().split()
            left = Node(None, tmp[0])
            left.parent = node
            right = Node(None, tmp[1])
            right.parent = node

            node.left = left
            node.right = right
            return node


        # If it's a parent node of two leaves
        #if countOpen == 1 and countClose == 1:
        #    node.left = self.parse(tokens[0:split],parent=node, is_leaf=True)
        #    node.right = self.parse(tokens[split:-1],parent=node, is_leaf=True)
        #else:
        node.left = self.parse(tokens[2:split-2],parent=node)
        node.right = self.parse(tokens[split+3:-2],parent=node)
        return node
    """

    def parse(self, tokens, parent=None):
        # no asset for no-sentiment analysis tasks
        #assert tokens[0] == self.open, "Malformed tree"
        #assert tokens[-1] == self.close, "Malformed tree"

        #split = 2 # position after open and label
        #split = 1 # position after open
        # position after open and label
        split = 1
        """while tokens[split] != ' ':
            split += 1
        split += 1
        """
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
        #node = Node(int(tokens[1])-1) # zero index labels
        node = Node(None, tokens[after_label:-1]) # no labels for no-sentiment analysis tasks
        node.parent = parent

        # leaf Node
        if countOpen == 0:
        # DEBUG
        #    print tokens

            #node.word = ''.join(tokens[2:-1]).lower() # lower case?
            node.word = ''.join(tokens[1:-1]).lower() # lower case?
            node.isLeaf = True
            return node
        #else:
        # DEBUG
        #    print tokens[1:split]
        #    print tokens[split:-1]


        node.left = self.parse(tokens[1:split],parent=node)
        #if tokens[split:-1] ==
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


def loadWordMap():
    import cPickle as pickle
    with open('wordMap.bin','r') as fid:
        return pickle.load(fid)

# Get the tree instance of all sentences from the dataset
def buildWordMap(dataSet='sick', data='train_parsed'):
    """
    Builds map of all words in training set
    to integer values.
    """

    import cPickle as pickle
    file = 'trees/%s_%s' % (dataSet, data)
    #file = 'trees/train_parsed'
    #file = 'trees/train_parsed_debug'
    print "Reading trees.."

    """with open(file,'r') as fid:
        trees = [Tree(l) for l in fid.readlines()]"""
    with open(file,'r') as fid:
        if dataSet == 'sick':
            l = list(inputarray(fid.readlines()))
        elif dataSet == 'quant':
            l = list(input_quant(fid.readlines()))
        tree_pairs = l#TreePair(l[0], l[1], l[2], l[3])

    print "Counting words.."
    words = collections.defaultdict(int)
    for tree_pair in tree_pairs:
        leftTraverse(tree_pair.tree1.root,nodeFn=countWords,args=words)
        leftTraverse(tree_pair.tree2.root,nodeFn=countWords,args=words)

    wordMap = dict(zip(words.iterkeys(),xrange(len(words))))
    wordMap[UNK] = len(words) # Add unknown as word

    # debug
    """"print "debug1"
    print wordMap["awesome"]
    fds"""

    with open('wordMap.bin','w') as fid:
        pickle.dump(wordMap,fid)

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
def inputarray(lines):
    labels = {'ENTAILMENT':0, 'NEUTRAL':1, 'CONTRADICTION':2}
    for l in lines:
        tmp = l.split('\t')
        # Note the order of arguments.
        # Change below depends on your RTE dataset.
        yield TreePair(Tree(tmp[1]), Tree(tmp[2]), labels[tmp[0]], tmp[3], tmp[4])

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
    #word_idx_map = dict()

    W = np.zeros(shape=(vocab_size+1, k))
    #W[0] = np.zeros(k)

    for i in range(0,len(idx_word_map)):
        W[i] = word_vecs[idx_word_map[i]]
        #word_idx_map[word] = i
    return W


def loadTrees(dataSet='sick', data='train_parsed'):
    """
    Loads training trees. Maps leaf node words to word ids.
    """
    wordMap = loadWordMap()
    file = 'trees/%s_%s' % (dataSet, data)

    #file = 'trees/train_parsed_debug'
    #file = 'trees/train_parsed'
    print file
    print "Reading trees.."

    with open(file, 'r') as fid:
        #trees = [Tree(l) for l in fid.readlines()]
        if dataSet == 'sick':
            l = list(inputarray(fid.readlines()))
        elif dataSet == 'quant':
            l = list(input_quant(fid.readlines()))
        tree_pairs = l


    # Run mapWords on each node.
    # Comment out this section when you need to check pygraphviz tree visualization.
    for tree_pair in tree_pairs:
        leftTraverse(tree_pair.tree1.root,nodeFn=mapWords,args=wordMap)
        leftTraverse(tree_pair.tree2.root,nodeFn=mapWords,args=wordMap)
    return tree_pairs, wordMap


def dummy_word_vecs():
    word_vecs = dict()
    word_vecs['stick']=[ 0.0546875 , -0.00958252,  0.26171875,  0.35351562, -0.14453125,
        0.00352478,  0.38867188, -0.07861328, -0.02905273,  0.11523438,
       -0.04394531, -0.14257812, -0.20996094,  0.16894531, -0.30273438,
       -0.14550781,  0.09179688, -0.04541016,  0.00234985, -0.31445312,
        0.12304688,  0.20507812, -0.03088379,  0.07226562,  0.22949219,
       -0.04980469,  0.03271484,  0.04541016,  0.2734375 , -0.14648438,
        0.04858398,  0.20605469,  0.12207031, -0.04418945,  0.05786133,
       -0.15039062,  0.11572266,  0.23828125,  0.02722168,  0.06298828,
        0.22265625,  0.18945312,  0.30078125, -0.09912109, -0.09912109,
        0.01306152,  0.02807617, -0.24023438,  0.28515625,  0.15332031,
       -0.12988281,  0.19628906,  0.06835938, -0.19628906,  0.06835938,
       -0.22949219, -0.00491333, -0.22851562, -0.26757812, -0.19824219,
       -0.01647949,  0.10986328,  0.17773438,  0.15429688, -0.06933594,
       -0.13378906, -0.16210938,  0.06738281,  0.20800781,  0.11035156,
        0.14648438, -0.00372314,  0.34765625, -0.21484375, -0.3125    ,
       -0.125     , -0.02648926,  0.16015625,  0.04663086,  0.03344727,
       -0.11181641, -0.00793457,  0.04614258, -0.00375366,  0.1484375 ,
        0.10009766, -0.19140625,  0.00297546,  0.14550781,  0.15429688,
        0.09570312,  0.06005859,  0.11572266,  0.02929688,  0.125     ,
       -0.1875    ,  0.1796875 ,  0.12890625, -0.05639648, -0.00622559,
       -0.00744629, -0.10449219,  0.1484375 ,  0.20507812, -0.10498047,
       -0.38476562,  0.15332031, -0.06494141,  0.32617188, -0.22265625,
       -0.11083984, -0.18652344, -0.06884766, -0.09667969, -0.08154297,
        0.06640625, -0.07373047, -0.14453125, -0.24414062, -0.01733398,
       -0.08251953, -0.15234375,  0.00595093,  0.05493164,  0.12011719,
       -0.19824219,  0.14453125, -0.07080078, -0.06982422,  0.0213623 ,
       -0.140625  , -0.26171875, -0.10058594,  0.01239014, -0.08642578,
        0.2109375 ,  0.21582031,  0.14648438,  0.00415039,  0.20703125,
        0.27539062,  0.0378418 , -0.18359375,  0.03344727, -0.01556396,
        0.13476562, -0.11669922, -0.3125    ,  0.390625  , -0.07128906,
        0.15234375, -0.07519531, -0.04956055,  0.01318359, -0.06738281,
       -0.16894531, -0.08984375, -0.07275391,  0.11132812,  0.27929688,
       -0.12792969,  0.15625   , -0.09082031, -0.26367188,  0.0402832 ,
       -0.17285156, -0.00567627, -0.10888672, -0.07861328, -0.13671875,
       -0.07275391,  0.10253906, -0.04589844,  0.05517578, -0.14746094,
        0.01965332,  0.15136719, -0.32421875,  0.14355469,  0.20898438,
       -0.16503906, -0.16015625,  0.13671875,  0.18164062, -0.13574219,
       -0.07568359,  0.00106049, -0.18164062, -0.09814453,  0.06933594,
        0.19238281, -0.01202393,  0.00119781, -0.0859375 ,  0.00300598,
        0.02392578, -0.03149414, -0.03588867, -0.34960938,  0.04711914,
       -0.09423828, -0.15820312, -0.14453125,  0.05151367,  0.05249023,
       -0.0859375 , -0.15234375, -0.12597656,  0.17382812,  0.0324707 ,
        0.02514648, -0.01928711,  0.17871094,  0.30273438, -0.02697754,
       -0.12060547,  0.09277344,  0.06176758,  0.06835938,  0.00352478,
       -0.08105469,  0.00836182, -0.18554688,  0.07373047,  0.20703125,
       -0.11279297, -0.00909424,  0.14453125,  0.22167969, -0.14941406,
       -0.05859375,  0.08447266, -0.00909424,  0.08789062,  0.15820312,
        0.01196289, -0.24121094, -0.33007812, -0.07617188,  0.11230469,
       -0.07958984, -0.0402832 , -0.03442383, -0.28125   ,  0.11669922,
       -0.14648438, -0.18359375,  0.14648438,  0.06005859,  0.01928711,
       -0.28125   , -0.04736328,  0.08056641,  0.11425781,  0.13671875,
       -0.00176239, -0.3046875 , -0.10498047,  0.06152344,  0.07324219,
        0.03881836,  0.13867188, -0.14257812,  0.00296021, -0.07226562,
        0.203125  , -0.10546875, -0.10498047,  0.08300781, -0.09228516,
        0.14453125,  0.0612793 ,  0.09814453, -0.1796875 ,  0.1484375 ,
       -0.08398438, -0.24414062, -0.125     , -0.20507812, -0.05664062,
        0.23046875,  0.14648438,  0.13769531,  0.03100586, -0.11035156,
       -0.07666016, -0.15527344, -0.06542969, -0.13183594, -0.0324707 ,
       -0.07958984,  0.23828125,  0.07910156,  0.14941406, -0.13476562,
       -0.19042969,  0.07421875,  0.07910156, -0.24511719, -0.0612793 ]
    return word_vecs

if __name__=='__main__':
    #buildWordMap('quant','train_parsed')
    buildWordMap('sick','train_parsed')
    #train, vocab = loadTrees('quant', 'train_parsed')
    train, wordMap = loadTrees('sick', 'train_parsed')
    idx_word_map=createIndexWordMap(wordMap)

    print 'Loading word2vec...'
    w2v_file = "/Users/pentiumx/Projects/word2vec/GoogleNews-vectors-negative300.bin"
    word_vecs = pd.load_bin_vec(w2v_file, wordMap)
    #word_vecs = dummy_word_vecs()
    #print word_vecs
    print type(word_vecs)
    print len(word_vecs)
    pd.add_unknown_words(word_vecs, wordMap, 0)# wordMap is not countWOrds array.
    W = get_W(word_vecs, wordMap, idx_word_map)
    print 'word matrix:\n%s' % W
    #print 'word_idx_map:\n%s' % word_idx_map
    pickle.dump([W], open("mr.p", "wb"))

    debug_tree(train[0].tree1.root)


