import numpy as np
import os
import sys
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from src.data.preprocessing.frequent_motif import find_repeat_subseqs, encode_node_types
from src.data.preprocessing.utils import (
    remove_useless_node,
    GraphAdj,
    remove_loose_input
)
# from src.data.utils import visualize_graph
from graphviz import Digraph
import pickle

def visualize_graph(edges_list, filename):

    unique_nodes = []
    # print(network.edges_list)
    for node_i, node_j in edges_list:
        unique_nodes.append(node_i)
        unique_nodes.append(node_j)
    unique_nodes = set(unique_nodes)

    graph_dot = Digraph('round-table', node_attr=dict(style='filled'))
    for node in unique_nodes:
        # if 'CONCAT' in node:
        #     graph_dot.node(node, shape='box', fillcolor='palegoldenrod')
        # elif 'ADD' in node:
        #     graph_dot.node(node, shape='diamond', fillcolor='lightblue1')
        if '&' in node:
            graph_dot.node(node, shape='box', fillcolor='darkseagreen1')
        elif '+' in node:
            graph_dot.node(node, shape='box', fillcolor='lightblue1')
        else:
            graph_dot.node(node, fillcolor='lightgrey')

    graph_dot.edges(edges_list)

    graph_dot.render(filename=filename)


def test():
 
    query = 'resnet18'
    with open(os.path.join('datasets', query+'.pkl'), 'rb') as f:
        data = pickle.load(f)
        
    graph_edges = data['graph_edges']
    print("len(graph_edges):", len(graph_edges))
    graph_edges = remove_useless_node(graph_edges=graph_edges, node_name='AccumulateGrad')
    graph_edges = remove_useless_node(graph_edges, node_name='TBackward')
    graph = GraphAdj(graph_edges=graph_edges)
    loose_input = [graph.nodes_list[_i] for _i in np.where(graph.in_degree == 0)[0]]
    graph_edges = remove_loose_input(graph_edges, loose_input)
    graph = GraphAdj(graph_edges=graph_edges)
    print("len(graph_edges):", len(graph_edges))

    print(graph_edges[-1])
    type_of_nodes, node_types_idxs = encode_node_types(graph.adj_matrix)
    print("graph.adj_matrix: ", graph.adj_matrix)

    print("len(node_types_idxs): ", len(node_types_idxs))
    subseqs_0 = find_repeat_subseqs(node_types_idxs)
    print("len(subseqs_0):", len(subseqs_0))
    lens_subseqs_0 = [len(ss) for ss in subseqs_0]

    i = 0

    subseqs_node_list = []
    for l in lens_subseqs_0:
        subseqs_node_list.append(graph.nodes_list[i:i + l])
        i += l

    print("lens_subseqs_0: ", len(lens_subseqs_0))
    print("subseqs_node_list: ", len(subseqs_node_list))
    
    # for i, ssn in enumerate(subseqs_node_list):

    #     print('\r%03d' % i, end='')

    #     ss_edges_list = []
    #     for node_i, node_j in graph_edges:
    #         if node_i in ssn or node_j in ssn:
    #             ss_edges_list.append((node_i, node_j))

    #     visualize_graph(edges_list=ss_edges_list, filename='subgraph/subgraph-%03d' % i)

    ssn = subseqs_node_list[-1]
    # print(ssn)
    ss_edges_list = []
    for node_i, node_j in graph_edges:
        if node_i in ssn or node_j in ssn:
            ss_edges_list.append((node_i, node_j))
            
    visualize_graph(edges_list=ss_edges_list, filename='subgraph/subgraph-%03d' % i)
    print('\r%03d' % i)

if __name__ == '__main__':
    test()
