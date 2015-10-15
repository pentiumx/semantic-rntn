

import collections
import pygraphviz as PG
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
            print toks
            tokens += list(toks)

        # Remove the top parenthesis
        #self.root = self.parse(treeString.strip()[6:-2])
        #self.root = self.parse(treeString.strip()[2:-2])

        # tokens[1:-1] is not necessary cuz we set split=1 in the parse method
        self.root = self.parse(tokens)
        print 'test'


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
        #print tokens
        if tokens == '(NNS kids)':
            print 'debug'
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
            print tokens

            #node.word = ''.join(tokens[2:-1]).lower() # lower case?
            node.word = ''.join(tokens[1:-1]).lower() # lower case?
            node.isLeaf = True
            return node
        else:
            print tokens[1:split]
            print tokens[split:-1]


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
def buildWordMap():
    """
    Builds map of all words in training set
    to integer values.
    """

    import cPickle as pickle
    #file = 'trees/train.txt'
    #file = 'trees/SICK_train_parsed_sample.txt'
    file = 'trees/SICK_parsed'
    print "Reading trees.."

    """with open(file,'r') as fid:
        trees = [Tree(l) for l in fid.readlines()]"""
    with open(file,'r') as fid:
        l = list(inputarray(fid.readlines()))
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


# For creating TreePair instances.
def inputarray(lines):
    for l in lines:
        tmp = l.split('\t')
        # Note the order of arguments.
        # Change below depends on your RTE dataset.
        yield TreePair(Tree(tmp[1]), Tree(tmp[2]), tmp[0], tmp[3], tmp[4])


def loadTrees(dataSet='train'):
    """
    Loads training trees. Maps leaf node words to word ids.
    """
    wordMap = loadWordMap()
    #file = 'trees/%s.txt'%dataSet
    #file = 'trees/SICK_train_parsed_sample.txt'
    file = 'trees/SICK_parsed'
    print "Reading trees.."

    with open(file, 'r') as fid:
        #trees = [Tree(l) for l in fid.readlines()]
        l = list(inputarray(fid.readlines()))
        tree_pairs = l


    # Run mapWords on each node
    #for tree_pair in tree_pairs:
    #    leftTraverse(tree_pair.tree1.root,nodeFn=mapWords,args=wordMap)
    #    leftTraverse(tree_pair.tree2.root,nodeFn=mapWords,args=wordMap)
    return tree_pairs

if __name__=='__main__':
    buildWordMap()
    train = loadTrees()
    debug_tree(train[0].tree1.root)



