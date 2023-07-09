import os

from flask import Flask, render_template, request, jsonify
from engine import search, draw_graph_edges, extract_subgraphs, visualize_graph, visualize_retrieval
import torch
import pickle
import numpy as np
import base64
from io import BytesIO
from matplotlib.figure import Figure
from graphviz import Digraph

import io, json
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from src.data.preprocessing.frequent_motif import find_repeat_subseqs, encode_node_types
from src.data.preprocessing.utils import (
    remove_useless_node,
    GraphAdj,
    remove_loose_input
)
# from src.data.utils import visualize_graph
from graphviz import Digraph

app = Flask(__name__)

DATA_PATH = 'datasets'
embeddings = torch.load('pretrained/embeddings.pt')
embeddings = embeddings.data.detach().numpy().astype('float32')
# basedir = os.path.abspath(os.path.dirname(__file__))
# app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search_engine', methods=['POST'])
def search_engine():
    if request.method == "POST":
        js_data = request.get_json()
        query = js_data[0]['query']
        assert type(query) == str
        path = os.path.join(DATA_PATH, query+'.pkl')
        
        try:
            print(path)
            with open(path, 'rb') as f:
                data = pickle.load(f)

            query = data['model_name']

            # graph_emb = data['graph_emb'].numpy().astype('float32')
            graph_emb = np.array(data['graph_emb']).astype('float32')
            torch.manual_seed(1234)
            database = embeddings.data.detach().numpy().astype('float32')
            top_k_index, top_k_emb = search(graph_emb, database, top_k=5)

            data['sim_top_k'] = top_k_index.tolist()
            data['graph_emb'] = graph_emb.tolist()

            data['fig_path'] = []
            
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
            
            # Visualizer
            # max_len_ssn, max_ssn_idx = 0, 0
            top_k = 10
            data['model_names'] = []
            for idx in range(10):
                data['model_names'].append("<h1>"+"subgraph"+str(idx+1)+"</h1>")
            data['model_names'] = data['model_names'][:top_k]
            
            for idx, ssn in enumerate(subseqs_node_list):
                if idx >= 10:
                    break
                if len(ssn) > 1:
                    ss_edges_list = []
                    for node_i, node_j in graph_edges:
                        if node_i in ssn or node_j in ssn:
                            ss_edges_list.append((node_i, node_j))
                    
                    filename = query + str(idx)
                    fig_path = visualize_graph(edges_list=ss_edges_list, filename=filename)
                    data['fig_path'].append(fig_path)

            data['fig_path'] = data['fig_path'][:top_k]
            
            print(data['model_names'])
            print(data['fig_path'])
            return jsonify(data)

        except Exception as e:
            print(e)
            data = dict()
            data['graph_edges'] = "Can't find this model, please try re-type the name..."
            return data
        

@app.route('/retrieval', methods=['POST'])
def retrieval():
    if request.method == "POST":

        js_data = request.get_json()
        # print(js_data)
        query = js_data[0]['query']
        assert type(query) == str
        # path = os.path.join(DATA_PATH, query+'.pkl')
        
        try:

            print(query)

            model2idx = {}
            idx2model = {}

            # Load model2idx and idx2model from JSON
            with open('maps/model2idx.json', 'r') as f:
                model2idx = json.load(f)
            with open('maps/idx2model.json', 'r') as f:
                idx2model = json.load(f)
            
            idx = model2idx[query]
            # print(idx)

            query_emb = embeddings[idx].reshape(1, -1)
            # print(query_emb.shape, embeddings.shape)
            top_k_index, top_k_emb = search(query_emb, embeddings)
            
            # print(top_k_index)
            top_k_models = dict()

            for i, idx in enumerate(top_k_index[0]):
                
                model_name = idx2model[str(idx)]
                
                sim_score = top_k_emb[0][i].reshape(-1) @ query_emb.reshape(-1).T / (np.linalg.norm(top_k_emb[0][i]) * np.linalg.norm(query_emb))
                # Initialize dictionary if it doesn't exist
                if model_name not in top_k_models:
                    top_k_models[model_name] = dict()
                
                top_k_models[model_name]['score'] = str(sim_score)

                path = os.path.join(DATA_PATH, model_name+'.pkl')
                top_k_models[model_name]['graph_path'] = path

                fig_path = visualize_retrieval(model_name)
                top_k_models[model_name]['fig_path'] = fig_path
                
            print("Searched similar models: ", top_k_models)

            return jsonify(top_k_models)

        except Exception as e:
            print(e)
            top_k_models = dict()
            # data['graph_edges'] = "Can't find this model, please try re-type the name..."
            return top_k_models
        # print(text)
        # rows = text
    # results = {'rows': rows}
    # return jsonify(results)
def draw_buffer(graph_edges: list):

    dot = Digraph(comment='The Round Table', format='gv')
    dot.node('A', 'King Arthur')

    buf = io.StringIO()

    print("writing")
    buf.write(dot.pipe().decode('utf-8') )

    print("reading")
    buf.seek(0)
    print(buf.read())

    buf.close()

def draw_model(graph_edges: list):
        # Generate the figure **without using pyplot**.
    fig = Figure()
    ax = fig.subplots()
    ax.plot([1, 2])
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"

def process(model):
    print(model, "hahhahh can process")

if __name__ == "__main__":
    try:
        app.run(debug=True, port=int(os.environ.get('PORT', 5007)))
    except:
        exit(0)

