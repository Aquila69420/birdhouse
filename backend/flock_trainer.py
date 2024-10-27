from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import logging
from flock_model import FlockModel
from llm_flock_model import LLMFlockModel
import requests
from base64 import b64decode
from client import TokenManager

# Load the MongoDB URI from the file
with open ("backend/mongo_uri.txt", "r") as myfile:
    mongo_uri=b64decode(myfile.readline().strip()).decode("utf-8")
token_manager = TokenManager(mongo_uri)

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

global model
nodes = set()  # To store connected nodes by their addresses
weights_dict = {}  # To store weights from all nodes

@app.route('/login_client', methods=['POST'])
def login():
    """
    Route to login a user based on wallet ID.
    """
    data = request.json
    wallet_id = data.get('wallet_address')
    success = token_manager.login(wallet_id)
    if not success:
        initial_tokens = data.get('initial_tokens', 10)
        token_manager.register_client(wallet_id, initial_tokens)
        success_v2 = token_manager.login(wallet_id)
        if success_v2:
            return jsonify({"message": f"User {wallet_id} registered and logged in successfully"}), 201
        else:
            return jsonify({"message": f"User {wallet_id} not found"}), 404
    return jsonify({"message": f"User {wallet_id} logged in successfully"}), 200

# @app.route('/register_client', methods=['POST'])
# def register():
#     """
#     Route to register a new client with an initial amount of tokens.
#     """
#     data = request.json
#     wallet_id = data.get('wallet_address')
#     initial_tokens = data.get('initial_tokens', 10)
#     token_manager.register_client(wallet_id, initial_tokens)
#     return jsonify({"message": f"User {wallet_id} registered successfully"}), 201

@app.route('/pay_tokens', methods=['POST'])
def pay_tokens():
    """
    Route to pay tokens for training.
    Tokens are paid for training by the future
    """
    data = request.json
    wallet_id = data.get('wallet_id')
    tokens = data.get('tokens')
    if not token_manager.update_client_tokens(wallet_id, tokens):
        return jsonify({"message": "Insufficient tokens"}), 400
    return jsonify({"message": f"{tokens} tokens paid successfully"}), 200

@app.route('/join_network', methods=['POST'])
def join_network():
    """
    Input route to join a network.
    Expects JSON with 'node_address'.
    """
    data = request.json
    node_address = data.get('node_address')
    nodes.add(node_address)
    logger.info(f"Node {node_address} joined the network")
    return jsonify({"message": f"{node_address} joined network successfully"}), 200

@app.route('/instantiate_flock_model', methods=['POST'])
def instantiate_flock_model():
    """
    Input route to instantiate a FlockModel.
    """
    global model
    data = request.json
    print(request)
    model_name = data.get('model_name')
    loss_function_name = data.get('loss_function')
    # Other parameters can be retrieved similarly
    model = FlockModel(model=model_name, loss_function=loss_function_name, classes = ["1", "2"]) # Instantiate the model using user inputs
    logger.info(f"FlockModel {model_name} instantiated")
    return jsonify({"model_name": model_name, "loss_function_name": loss_function_name}), 201

@app.route('/instantiate_llm_flock_model', methods=['POST'])
def instantiate_llm_flock_model():
    """
    Input route to instantiate an LLMFlockModel.
    """
    global model
    data = request.json
    model_name = data.get('model_name')
    try:
        model = LLMFlockModel(model_name=model_name, output_dir='.')
        print(f"LLMFlockModel {model_name} instantiated")
        return jsonify({"model_name": model_name}), 201
    except Exception as e:
        print(f"Error instantiating LLMFlockModel: {e}")
        return jsonify({"error": f"Error instantiating LLMFlockModel: {e}"}), 500

@app.route('/send_llm_flock_model', methods=['POST'])
def send_llm_flock_model():
    """
    Output route to send the instantiated LLMFlockModel object to a node.
    """
    if isinstance(model, LLMFlockModel):
        llm_flock_model = model
    else:
        return jsonify({"error": "LLMFlockModel not instantiated"}), 400
    
    serialized_llm_flock_model = pickle.dumps(llm_flock_model,protocol=2)

    for node_address in nodes:
        try:
            requests.post(f"{node_address}/receive_llm_flock_model", json={"model": serialized_llm_flock_model})
            logger.info(f"LLMFlockModel sent to node {(node_address)}")
            # return jsonify({"message": f"LLMFlockModel sent to node {(node_address)}", "model": serialized_llm_flock_model}), 200
        except requests.exceptions.RequestException as e:
            nodes.remove(node_address)
            logger.error(f"Node {node_address} disconnected")
            return jsonify({"error": f"Node {node_address} disconnected"}), 500
        
    return jsonify({"message": f"LLMFlockModel sent to connected {len(nodes)}"}), 200

@app.route('/send_flock_model', methods=['POST'])
def send_flock_model(model_id):
    """
    Output route to send the instantiated FlockModel object to a node.
    """
    # node_address = "http://10.154.36.81:5001" # Store IP addresses after joiningand Iterate over all nodes
    global model

    if isinstance(model, FlockModel):
        flock_model = model
    else:
        return jsonify({"error": "FlockModel not instantiated"}), 400
    
    serialized_flock_model = pickle.dumps(flock_model,protocol=2)

    for node_address in nodes:
        try:
            requests.post(f"{node_address}/receive_flock_model", json={"model": serialized_flock_model})
            logger.info(f"FlockModel sent to node {(node_address)}")
            # return jsonify({"message": f"FlockModel with ID {model_id} sent to connected {len(nodes)} nodes", "model": serialized_flock_model}), 200
        except requests.exceptions.RequestException as e:
            nodes.remove(node_address)
            logger.error(f"Node {node_address} disconnected")
            return jsonify({"error": f"Node {node_address} disconnected"}), 500
        
    return jsonify({"message": f"FlockModel sent to connected {len(nodes)}"}), 200

@app.route('/receive_model_weights', methods=['POST'])
def receive_model_weights():
    """
    Input route to receive the trained FlockModel's weights from a node.
    """
    global model
    data = request.json
    node_address = data.get('node_address')
    serialized_weights = data.get('weights')
    weights = pickle.loads(serialized_weights)
    # Maybe add the weights to a list and aggregate them later?
    weights_dict[node_address] = weights
    logger.info(f"Model weights received from node {node_address}")
    return jsonify({"message": f"Model weights received successfully from node {node_address}"}), 200

@app.route('/aggregate_weights', methods=['POST'])
def aggregate_weights():
    """
    Input route to aggregate weights from all nodes.
    """
    # Aggregate weights from all nodes
    node_selection_for_weights = request.json.get('node_selection_for_weights') # node addresses to select for aggregation
    if node_selection_for_weights:
        aggregated_weights_dict = {k: v for k, v in weights_dict.items() if k in node_selection_for_weights}
    aggregated_weights = model.aggregate(aggregated_weights_dict.values())
    aggregated_accuracy = model.evaluate(aggregated_weights)
    logger.info("Weights aggregated")
    return jsonify({"message": "Weights aggregated successfully", "aggregated_accuracy": aggregated_accuracy}), 200

@app.route('/fine_tune', methods=['POST'])
def fine_tune():
    """
    Input route to fine-tune the model on a custom dataset.
    """
    global model

    if isinstance(model, LLMFlockModel):
        flock_model = model
    else:
        return jsonify({"error": "LLMFlockModel not instantiated"}), 400
    
    serialized_llm_flock_model = pickle.dumps(flock_model,protocol=2)

    for node_address in nodes:
        try:
            requests.post(f"{node_address}/fine_tune_llm_flock_model", json={"model": serialized_llm_flock_model})
            logger.info(f"LLMFlockModel sent to node {(node_address)}")
            # return jsonify({"message": f"FlockModel with ID {model_id} sent to connected {len(nodes)} nodes", "model": serialized_flock_model}), 200
        except requests.exceptions.RequestException as e:
            nodes.remove(node_address)
            logger.error(f"Node {node_address} disconnected")
            return jsonify({"error": f"Node {node_address} disconnected"}), 500
    return jsonify({"message": f"LLMFlockModel sent to connected {len(nodes)}"}), 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
