from .neo4j_connection import Neo4jConnection
from .d3_graph import neo4j_to_d3
from dotenv import load_dotenv
import os
import torch
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from tqdm import tqdm
import time
import h5py
import glob
import re

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
embedding_model = SentenceTransformer("all-mpnet-base-v2").to(DEVICE)

load_dotenv()
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER_NAME")
PASSWORD = os.getenv("NEO4J_PASSWD")
AUTH = (os.getenv("NEO4J_USER_NAME"), os.getenv("NEO4J_PASSWD"))

# Instantiate Neo4j connection
neo4j_obj = Neo4jConnection(uri=URI, 
                    user=USER,
                    pwd=PASSWORD)

_file_path_cache = {}

def find_file_path(filename, search_directory="."):
    if filename in _file_path_cache:
        return _file_path_cache[filename]
    for file_path in glob.iglob(f"{search_directory}/**/{filename}", recursive=True):
        path = os.path.abspath(file_path)
        _file_path_cache[filename] = path
        return path
    _file_path_cache[filename] = None
    return None


_models_cache = None
_model_embedding_ids_cache = None
_model_embeddings_tensor_cache = None

def _ensure_model_cache():
    global _models_cache, _model_embedding_ids_cache, _model_embeddings_tensor_cache
    if _models_cache is not None:
        return
    _models_cache = get_models()
    filename = 'model_embeddings_all.h5'
    filepath = find_file_path(filename=filename)
    if filepath is None:
        raise FileNotFoundError(
            f"Embedding file '{filename}' not found. Run compute_embeddings.py first."
        )
    with h5py.File(filepath, 'r') as f:
        embedding_ids = f['embedding_ids'][:]
        embeddings = f['embeddings'][:]
    _model_embedding_ids_cache = [eid.decode('utf-8') for eid in embedding_ids]
    _model_embeddings_tensor_cache = torch.tensor(embeddings).to(DEVICE)

def create_tokens(tid):
    return [t.lower() for t in re.split(r'[-\s]+', tid) if t]

def convert_json(result):
    data_dict = {}
    for item in result:
        curr_dict = dict(item[0])
        item_id = curr_dict['itemID']
        name = curr_dict['name']
        curr_dict['tokens'] = create_tokens(name)
        data_dict[item_id] = curr_dict
    return data_dict

def get_models():
    nodes= """MATCH (n:Model) RETURN properties(n)""" 
    res = neo4j_obj.query(nodes)
    data_dict = convert_json(res)
    return data_dict

def get_model_node(id):
    nodes= """MATCH (n:Model {itemID:$id}) RETURN properties(n)""" 
    parameters = {'id':id}
    res = neo4j_obj.query(nodes, parameters)
    data_dict = convert_json(res)
    return data_dict


def get_result_pipelines(model_ids):
    """
    For the given model ids (top similar ones) get the entire pipeline and return to the main function
    """
    results = []
    for model_id in model_ids:
        query_str = """
            MATCH (model:Model {itemID:$model_id})
            OPTIONAL MATCH (model)-[r6]-(artifact:Artifact)
            OPTIONAL MATCH (artifact)-[r7]-(metric:Metric)
            OPTIONAL MATCH (artifact)-[r4]-(execution:Execution)
            OPTIONAL MATCH (execution)-[r3]-(stage:Stage)
            OPTIONAL MATCH (stage)-[r2]-(pipeline:Pipeline)
            OPTIONAL MATCH (pipeline)-[r1]-(task:Task)
            OPTIONAL MATCH (artifact)-[r5]-(dataset:Dataset)
            OPTIONAL MATCH (pipeline)-[r8]-(framework:Framework)
            OPTIONAL MATCH (pipeline)-[r9]-(report:Report)
            RETURN task, pipeline, stage, execution, artifact, dataset, model, metric, framework, report, r1, r2, r3, r4, r5, r6, r7, r8, r9
            limit 20
            """
        parameters = {'model_id': model_id}
        res = neo4j_obj.query(query_str, parameters)
        results.append(res)
    return results


def get_explanations(query_model, top_model_ids, top_sim_scores, data_dict):
    explanations = []
    explanations.append({'title': 'Query', 'content': {'Name': query_model.title(), 'Label': 'Model',
                         'Properties Computed': {'Model Class': ''}}})
    for i, mid in enumerate(top_model_ids):
        curr_item = data_dict[mid]
        explanations.append({'title': 'Recommendation-' + str(i + 1),
                              'content': {'Name': curr_item['name'].title(),
                                          'Similarity Score': str(round(top_sim_scores[i].item(), 3)),
                                          'Similar Properties': {'Tokens': curr_item['tokens'],
                                                                  'Model Class': curr_item.get('modelClass', '')}}})
    return explanations




def get_similar_models(query_model, num_res=3):
    start_time = time.time()
    # test - compute just embedding similarity from all the files
    _ensure_model_cache()
    data_dict = _models_cache
    model_ids = _model_embedding_ids_cache
    embeddings = _model_embeddings_tensor_cache

    query_embedding = torch.tensor(embedding_model.encode(str(query_model))).view(1, -1).to(DEVICE)
    

    cos_sim =  F.cosine_similarity(query_embedding, embeddings, dim=1)
    print(len(cos_sim), cos_sim.device)
    # Sort the tensor in descending order and get the indices
    sorted_tensor, sorted_indices = torch.sort(cos_sim, descending=True)
    indices = sorted_indices[:num_res]
    top_ids = [model_ids[idx] for idx in indices]
    top_sim_scores = sorted_tensor[:num_res]
    explanations = get_explanations(query_model, top_ids, top_sim_scores, data_dict)
    neo4j_results = get_result_pipelines(top_ids)
    result_d3_graphs = neo4j_to_d3(neo4j_results)
    result_items = {'nodes': result_d3_graphs['nodes'], 'links':result_d3_graphs['links'], 'explanations':explanations}
    print("Time Taken",time.time()-start_time)
    batch_query = "MATCH (n:Model) WHERE n.itemID IN $ids RETURN properties(n)"
    res = neo4j_obj.query(batch_query, {'ids': top_ids})
    similar_item_dict = convert_json(res)
    return result_items, similar_item_dict

# get_similar_models("clinical llama")