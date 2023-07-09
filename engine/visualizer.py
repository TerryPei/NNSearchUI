from graphviz import Digraph
import shutil
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
import pickle
# from src.data.utils import visualize_graph

def extract_subgraphs(graph_edges):

    print("len(graph_edges):", len(graph_edges)) # right 507
    graph_edges = remove_useless_node(graph_edges=graph_edges, node_name='AccumulateGrad')
    graph_edges = remove_useless_node(graph_edges, node_name='TBackward')
    graph = GraphAdj(graph_edges=graph_edges)
    loose_input = [graph.nodes_list[_i] for _i in np.where(graph.in_degree == 0)[0]]
    graph_edges = remove_loose_input(graph_edges, loose_input)
    # graph = GraphAdj(graph_edges=graph_edges)
    # print(graph_edges[-1]) #right
    print("len(graph_edges):", len(graph_edges)) # right 187
    print(graph_edges[-1])
    type_of_nodes, node_types_idxs = encode_node_types(graph.adj_matrix)
    print("graph.adj_matrix: ", graph.adj_matrix)
    print("len(node_types_idxs): ", len(node_types_idxs)) # wrong

    subseqs_0 = find_repeat_subseqs(node_types_idxs)
    print("len(subseqs_0):", len(subseqs_0)) # wrong

    lens_subseqs_0 = [len(ss) for ss in subseqs_0]

    i = 0
    subseqs_node_list = []
    for l in lens_subseqs_0:
        subseqs_node_list.append(graph.nodes_list[i:i + l])
        i += l

    print("lens_subseqs_0: ", len(lens_subseqs_0)) # wrong, should be 10 but get 108
    print("subseqs_node_list: ", len(subseqs_node_list))
    return subseqs_node_list, graph_edges

def draw_graph_edges(model_name: str, graph_edges: list, ssn: list, i: str):
    # print("num of subseqs: ", len(subseqs_node_list))

    # ssn = subseqs_node_list[-1]
    ss_edges_list = []
    for node_i, node_j in graph_edges:
        if node_i in ssn or node_j in ssn:
            ss_edges_list.append((node_i, node_j))
    # edges_list = ss_edges_list
    unique_nodes = []
    # print(network.edges_list)
    for node_i, node_j in ss_edges_list:
        unique_nodes.append(node_i)
        unique_nodes.append(node_j)
    unique_nodes = set(unique_nodes)

    graph_dot = Digraph('round-table', node_attr=dict(style='filled'))

    for node in unique_nodes:

        if '&' in node:
            graph_dot.node(node, shape='box', fillcolor='darkseagreen1')
        elif '+' in node:
            graph_dot.node(node, shape='box', fillcolor='lightblue1')
        elif 'features' in node:
            graph_dot.node(node, shape='box', fillcolor='orange')
        else:
            graph_dot.node(node, fillcolor='lightgrey')

    graph_dot.edges(ss_edges_list)
    # dot = Digraph(name=model_name, node_attr=node_attr)
    graph_dot.render(model_name+i, format='svg', view=False)
    # flask only can load img file from static
    shutil.move(model_name+i+'.svg', 'static/'+model_name+i+'.svg')
    return f"<img src='static/{model_name}{i}.svg/>"

def visualize_graph(edges_list, model_name):
    
    unique_nodes = []
    # print(network.edges_list)
    for node_i, node_j in edges_list:
        unique_nodes.append(node_i)
        unique_nodes.append(node_j)
    unique_nodes = set(unique_nodes)

    graph_dot = Digraph('round-table', node_attr=dict(style='filled'))
    
    for node in unique_nodes:
        if '&' in node:
            graph_dot.node(node, shape='box', fillcolor='darkseagreen1')
        elif '+' in node:
            graph_dot.node(node, shape='box', fillcolor='lightblue1')
        else:
            graph_dot.node(node, fillcolor='lightgrey')

    graph_dot.edges(edges_list)
    # dot = Digraph(name=model_name, node_attr=node_attr)
    graph_dot.graph_attr['bgcolor'] = 'transparent'
    graph_dot.render(model_name, format='svg', view=False)
    # flask only can load img file from static
    shutil.move(model_name+'.svg', 'static/figs/'+model_name+'.svg')
    return f"<img src='static/figs/{model_name}.svg'/>"

def visualize_retrieval(model_name):

    with open(os.path.join('datasets', model_name+'.pkl'), 'rb') as f:
        data = pickle.load(f)
    graph_edges = data['graph_edges']
    graph_edges = remove_useless_node(graph_edges=graph_edges, node_name='AccumulateGrad')
    graph_edges = remove_useless_node(graph_edges, node_name='TBackward')
    graph = GraphAdj(graph_edges=graph_edges)
    loose_input = [graph.nodes_list[_i] for _i in np.where(graph.in_degree == 0)[0]]
    graph_edges = remove_loose_input(graph_edges, loose_input)
    graph = GraphAdj(graph_edges=graph_edges)

    type_of_nodes, node_types_idxs = encode_node_types(graph.adj_matrix)

    subseqs_0 = find_repeat_subseqs(node_types_idxs)

    lens_subseqs_0 = [len(ss) for ss in subseqs_0]

    i = 0

    subseqs_node_list = []
    for l in lens_subseqs_0:
        subseqs_node_list.append(graph.nodes_list[i:i + l])
        i += l

    ssn = subseqs_node_list[-1]
    ss_edges_list = []
    for node_i, node_j in graph_edges:
        if node_i in ssn or node_j in ssn:
            ss_edges_list.append((node_i, node_j))
            
    fig_path = visualize_graph(edges_list=ss_edges_list, model_name=model_name)
    return fig_path


def remove_svg():
    directory = "static/figs"
    try:
        for filename in os.listdir(directory):
            if filename.endswith(".svg"):
                os.remove(directory + "/" + filename)
                print(f"File Removed: {filename}!")
    except Exception as error:
        print("Error occurred while trying to remove file.", error)


if __name__ == '__main__':

    model_name = 'resnet18'
    # with open(os.path.join('datasets', model_name+'.pkl'), 'rb') as f:
    #     data = pickle.load(f)
    # graph_edges = data['graph_edges']

    fig_path = visualize_retrieval(model_name)
    print(fig_path)

    ################ visualize_retrieval(graph_edges) ################

    # graph_edges = remove_useless_node(graph_edges=graph_edges, node_name='AccumulateGrad')
    # graph_edges = remove_useless_node(graph_edges, node_name='TBackward')
    # graph = GraphAdj(graph_edges=graph_edges)
    # loose_input = [graph.nodes_list[_i] for _i in np.where(graph.in_degree == 0)[0]]
    # graph_edges = remove_loose_input(graph_edges, loose_input)
    # graph = GraphAdj(graph_edges=graph_edges)

    # type_of_nodes, node_types_idxs = encode_node_types(graph.adj_matrix)

    # subseqs_0 = find_repeat_subseqs(node_types_idxs)

    # lens_subseqs_0 = [len(ss) for ss in subseqs_0]

    # i = 0

    # subseqs_node_list = []
    # for l in lens_subseqs_0:
    #     subseqs_node_list.append(graph.nodes_list[i:i + l])
    #     i += l

    # ssn = subseqs_node_list[-1]
    # ss_edges_list = []
    # for node_i, node_j in graph_edges:
    #     if node_i in ssn or node_j in ssn:
    #         ss_edges_list.append((node_i, node_j))
            
    # fig_path = visualize_graph(edges_list=ss_edges_list, model_name=query)

    # print(fig_path)

    ################ visualize_retrieval(graph_edges) ################

    