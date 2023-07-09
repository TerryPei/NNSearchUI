import math

import numpy as np

from src.data.preprocessing.utils import remove_useless_node, GraphAdj, remove_loose_input


def node_removal(graph_edges):
    graph_edges = remove_useless_node(graph_edges=graph_edges, node_name='AccumulateGrad')
    graph_edges = remove_useless_node(graph_edges=graph_edges, node_name='TBackward')
    graph = GraphAdj(graph_edges=graph_edges)
    loose_input = [graph.nodes_list[_i] for _i in np.where(graph.in_degree == 0)[0]]
    graph_edges = remove_loose_input(graph_edges, loose_input)
    return graph_edges


def split_by_node_num(graph_edges, node_num, with_remove=True):
    if with_remove:
        graph_edges = node_removal(graph_edges=graph_edges)

    graph = GraphAdj(graph_edges=graph_edges)

    num_nodes_total = len(graph.nodes_list)
    indices_start = list(range(0, num_nodes_total - 1, node_num))
    indices_end = list(range(node_num, num_nodes_total - 1, node_num))

    if len(indices_start) != len(indices_end):
        assert len(indices_start) - len(indices_end) == 1, (indices_start, indices_end)
        indices_end.append(num_nodes_total)

    list_of_subgraph_node = []
    for ni, nj in zip(indices_start, indices_end):
        list_of_subgraph_node.append(graph.nodes_list[ni:nj])

    list_of_subgraph_edges = []
    for i, ssn in enumerate(list_of_subgraph_node):
        sg_edges = []
        for node_i, node_j in graph_edges:
            if node_i in ssn or node_j in ssn:
                sg_edges.append((node_i, node_j))
        list_of_subgraph_edges.append(sg_edges)
    return list_of_subgraph_edges


def split_by_motif_num(graph_edges, motif_num, with_remove=True):
    if with_remove:
        graph_edges = node_removal(graph_edges=graph_edges)

    graph = GraphAdj(graph_edges=graph_edges)

    num_nodes_total = len(graph.nodes_list)
    if num_nodes_total % motif_num > 0:
        node_num = math.floor(num_nodes_total / motif_num)
    else:
        node_num = num_nodes_total // motif_num

    return split_by_node_num(
        graph_edges=graph_edges,
        node_num=node_num,
        with_remove=with_remove
    )


def split_random(graph_edges, min_size, max_size, with_remove=True):
    if with_remove:
        graph_edges = node_removal(graph_edges=graph_edges)

    graph = GraphAdj(graph_edges=graph_edges)

    num_nodes_total = len(graph.nodes_list)

    list_of_subgraph_node = []
    i = 0
    while i < num_nodes_total:
        node_num = np.random.randint(low=min_size, high=max_size)
        list_of_subgraph_node.append(graph.nodes_list[i:i + node_num])
        i += node_num

    list_of_subgraph_edges = []
    for i, ssn in enumerate(list_of_subgraph_node):
        sg_edges = []
        for node_i, node_j in graph_edges:
            if node_i in ssn or node_j in ssn:
                sg_edges.append((node_i, node_j))
        list_of_subgraph_edges.append(sg_edges)

    return list_of_subgraph_edges
