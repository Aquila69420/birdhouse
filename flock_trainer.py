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

models = {}  # To store instantiated models by their IDs
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
    Expects JSON with 'model', 'loss_function', and other params as needed.
    """
    data = request.json
    print(request)
    model_name = data.get('model_name')
    loss_function_name = data.get('loss_function')
    # Other parameters can be retrieved similarly
    # Instantiate the model using user inputs
    model = FlockModel(model=model_name, loss_function=loss_function_name, classes = ["1", "2",])
    
    model_id = f"flock_model_{len(models)}"
    models[model_id] = model
    logger.info(f"FlockModel instantiated with ID: {model_id}")

    return jsonify({"model_id": model_id}), 201

@app.route('/instantiate_llm_flock_model', methods=['GET'])
def instantiate_llm_flock_model():
    """
    Input route to instantiate an LLMFlockModel.
    Expects JSON with 'model_name' and 'output_dir'.
    """
    data = request.json
    model_name = data.get('model_name')
    # current directory is used as default output directory

    llm_model = LLMFlockModel(model_name=model_name, output_dir='.')
    
    model_id = f"llm_flock_model_{len(models)}"
    models[model_id] = llm_model
    logger.info(f"LLMFlockModel instantiated with ID: {model_id}")

    return jsonify({"model_id": model_id}), 201

@app.route('/send_llm_flock_model', methods=['POST'])
def send_llm_flock_model(model_id):
    """
    Output route to send the instantiated LLMFlockModel object to a node.
    Expects JSON with 'node_url' where the model will be sent.
    """
    data = request.json
    node_url = data.get('node_url')

    if model_id not in models:
        return jsonify({"error": "Model ID not found"}), 404

    llm_model = models[model_id]
    # Here, you can implement the logic to serialize the model and send it to the node
    # For demonstration, we're just logging it
    logger.info(f"Sending LLMFlockModel with ID {model_id} to {node_url}")
    # Send the model (consider using requests.post() to send it)

    return jsonify({"message": f"LLMFlockModel with ID {model_id} sent to {node_url}", "model": models.get(model_id)}), 200

@app.route('/send_flock_model', methods=['POST'])
def send_flock_model(model_id):
    """
    Output route to send the instantiated FlockModel object to a node.
    Expects JSON with 'node_url' where the model will be sent.
    """
    # node_address = "http://10.154.36.81:5001" # Store IP addresses after joiningand Iterate over all nodes
    data = request.json
    node_url = data.get('node_url')

    if model_id not in models:
        return jsonify({"error": "Model ID not found"}), 404

    flock_model = models.get(model_id)
    serialized_flock_model = pickle.dumps(flock_model,protocol=2)
    # Here, logic to serialize the model and send it to the node
    logger.info(f"Sending FlockModel with ID {model_id} to {node_url}")
    # Send the model (consider using requests.post() to send it)
    for node_address in nodes:
        try:
            requests.post(f"{node_address}/receive_model", json={"model": serialized_flock_model})
            return jsonify({"message": f"FlockModel with ID {model_id} sent to connected {len(nodes)} nodes", "model": serialized_flock_model}), 200
        except requests.exceptions.RequestException as e:
            nodes.remove(node_address)
            return jsonify({"error": f"Node {node_address} disconnected"}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
