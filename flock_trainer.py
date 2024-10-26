from flask import Flask, request, jsonify
import pickle
import logging
from flock_model import FlockModel
from llm_flock_model import LLMFlockModel
import requests

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

global model
nodes = set()  # To store connected nodes by their addresses

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

@app.route('/instantiate_flock_model', methods=['GET'])
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

@app.route('/instantiate_llm_flock_model', methods=['GET'])
def instantiate_llm_flock_model():
    """
    Input route to instantiate an LLMFlockModel.
    Expects JSON with 'model_name' and 'output_dir'.
    """
    global model
    data = request.json
    model_name = data.get('model_name')
    model = LLMFlockModel(model_name=model_name, output_dir='.')
    logger.info(f"LLMFlockModel {model_name} instantiated")
    return jsonify({"model_name": model_name}), 201

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


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
