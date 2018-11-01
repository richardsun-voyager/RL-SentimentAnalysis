import networkx as nx
from nltk.tree import Tree
import numpy as np
from stanfordcorenlp import StanfordCoreNLP
stfnlp = StanfordCoreNLP(r'../data/stanford-corenlp-full-2018-02-27')
# import spacy
# spanlp = spacy.load('en')
import en_core_web_sm
spanlp = en_core_web_sm.load()
class dependency_path:
    def __init__(self, text=None):
        self.text = text

    def build_graph(self, text):
        '''
        Build a graph based on dependency parsing
        '''
        document = spanlp(text)
        edges = []
        for token in document:
            # FYI https://spacy.io/docs/api/token
            for child in token.children:
                edges.append(('{0}-{1}'.format(token.lower_,token.i),
                            '{0}-{1}'.format(child.lower_,child.i)))
        graph = nx.Graph(edges)
        return graph

    def get_shortest_path_len(self, graph, node1, node2):
        '''
        Compute the path between two nodes
        '''
        try:
            shortest_path_length = nx.shortest_path_length(graph, source=node1, target=node2)
        except:
            #Penalize the distance between two non-adjacent nodes
            shortest_path_length = 1000
        return shortest_path_length

    def compute_node_distance(self, graph, word_num):
        '''
        Compute the path for each node pair
        '''
        nodes = list(graph.nodes())
        node_order = np.array([int(node.split('-')[1]) for node in nodes])
        #Sort the words according to the original order
        # indice = node_order.argsort()
        # nodes = [nodes[i] for i in indice]
        # node_num = len(nodes)
        #mat = np.ones([word_num, word_num]) * 1000
        mat = (1-np.identity(word_num)) * 1000
        #Calculate the path for each node pair
        for i in np.arange(len(nodes)-1):
            for j in np.arange(i+1, len(nodes)):
                loc_i = node_order[i]
                loc_j = node_order[j]
                mat[loc_i, loc_j] = self.get_shortest_path_len(graph, nodes[i], nodes[j])
                mat[loc_j, loc_i] = mat[loc_i, loc_j]
        return mat

    def compute_soft_targets_weights(self, mat, target_nodes):
        '''
        compute the normalized path values for the targets
        mat: matrix, [node_num, node_num]
        target_nodes: [nodes], index of the target words
        '''
        target_weights = np.zeros([len(target_nodes), len(mat)])
        for i, node in enumerate(target_nodes):
            target_weights[i] = np.exp(-mat[node]**2/max(mat[node]))
        max_target_weight = target_weights.max(0)
        min_target_weight = target_weights.min(0)
        avg_target_weight = target_weights.mean(0)
        return max_target_weight, min_target_weight, avg_target_weight

    def compute_hard_targets_weights(self, mat, target_nodes):
        '''
        compute the normalized path values for the targets
        mat: matrix, [node_num, node_num]
        target_nodes: [nodes], index of the target words
        '''
        target_weights = np.zeros([len(target_nodes), len(mat)])
        for i, node in enumerate(target_nodes):
            target_weights[i] = np.where(mat[node]<5, 1, 0)
        max_target_weight = target_weights.max(1)
        min_target_weight = target_weights.min(1)
        avg_target_weight = target_weights.mean(1)
        return max_target_weight, min_target_weight, avg_target_weight


class constituency_path:
    def __init__(self):
        self.index = None

    def build_parser(self, text):
        '''
        Build a graph based on dependency parsing
        args: text, a sentence string
        '''
        document = stfnlp.parse(text)
        parsed_sent = Tree.fromstring(document)
        return parsed_sent

    def get_leaves(self, parsed_sent):
        '''
        Get the tokens 
        '''
        return parsed_sent.leaves()

    def get_leave_pos(self, parsed_sent):
        '''
        Get the position of the leave
        '''
        positions = []
        leaves_num = len(parsed_sent.leaves())
        for i in np.arange(leaves_num):
            pos = parsed_sent.leaf_treeposition(i)
            positions.append(pos)
        return positions


    def compute_node_distance(self, pos1, pos2):
        '''
        Compute the path for each node pair
        Args:
        pos1: a list, [0, 1, 0, 0]
        pos2: a list, [0, 0, 1, 1, 0]
        '''
        if len(pos1) > len(pos2):
            pos1, pos2 = pos2, pos1
        distance = 0
        for i, num in enumerate(pos1):
            if num != pos2[i]:#The distance between two words
                distance = np.abs(num-pos2[i]) + len(pos1) + len(pos2) - 2*i - 4
                break
        return distance

    def compute_target_distance(self, positions, target_pos):
        '''
        Compute distance between each context word and the taget word
        Args:
        positions: a list of position vector
        target_pos: index
        '''
        distances = []
        for pos in positions:
            distance = self.compute_node_distance(pos, positions[target_pos])
            distances.append(distance)
        return distances

    def compute_soft_targets_weights(self, positions, target_nodes):
        '''
        compute the normalized path values for the targets
        target_nodes: [nodes], index of the target words
        '''
        target_weights = np.zeros([len(target_nodes), len(positions)])
        for i, node in enumerate(target_nodes):
            target_weights[i] = np.array(self.compute_target_distance(positions, node))
            #target_weights[i] /= sum(target_weights[i])
            target_weights[i] = np.exp(-target_weights[i]**2/max(target_weights[i]))
        max_target_weight = target_weights.max(0)
        min_target_weight = target_weights.min(0)
        avg_target_weight = target_weights.mean(0)
        return max_target_weight, min_target_weight, avg_target_weight

    def compute_hard_targets_weights(self, positions, target_nodes):
        '''
        compute the normalized path values for the targets
        mat: matrix, [node_num, node_num]
        target_nodes: [nodes], index of the target words
        '''
        target_weights = np.zeros([len(target_nodes), len(positions)])
        for i, node in enumerate(target_nodes):
            target_weights[i] = np.array(self.compute_target_distance(positions, node))
            target_weights[i] = np.where(target_weights[i] < 5, 1, 0)
        max_target_weight = target_weights.max(1)
        min_target_weight = target_weights.min(1)
        avg_target_weight = target_weights.mean(1)
        return max_target_weight, min_target_weight, avg_target_weight

    def proceed(self, text, target_nodes):
        '''Calculate the weights for the context words'''
        #print(text)
        parsed_sent = self.build_parser(text)
        positions = self.get_leave_pos(parsed_sent)
        #print(target_nodes)
        max_target_weight, min_target_weight, avg_target_weight = self.compute_soft_targets_weights(positions, target_nodes)
        return max_target_weight, min_target_weight, avg_target_weight

    # def get_context_weights(self, text, target_nodes):
    #     '''Calculate the weights for the context words of a batch of contexxts'''
    #     parsed_sent = self.build_parser(text)
    #     positions = self.get_leave_pos(parsed_sent)
    #     max_target_weight, min_target_weight, avg_target_weight = self.compute_soft_targets_weights(positions, target_nodes)
    #     return max_target_weight, min_target_weight, avg_target_weight



 
