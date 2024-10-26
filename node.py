from flask import Flask, request, jsonify
import logging
import pickle
from flock_model import FlockModel
from llm_flock_model import LLMFlockModel
import requests

# Initialize the Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

global model
# user_address = "http://10.154.36.81:5000"

# Join network - server
@app.route('/join_network', methods=['POST'])
def join_network():
    """
    Send request to join a network.
    """
    # May have to use this instead of request.remote_addr
    # headers_list = request.headers.getlist("X-Forwarded-For")
    # node_address = headers_list[0] if headers_list else request.remote_addr
    node_address = request.remote_addr
    user_address = request.json.get('user_address')
    requests.post(f"{user_address}/join_network", json={"node_address": node_address })
    logger.info(f"Node {node_address} joined the network {user_address}")
    return jsonify({"message": f"Joined network {user_address} successfully"}), 200

@app.route('/receive_flock_model', methods=['POST'])
def receive_flock_model():
    """
    Route to receive a model from the flock_trainer.
    """
    global model
    data = request.json
    serialized_flock_model = data.get('model')
    loaded_model: FlockModel = pickle.loads(serialized_flock_model)
    model = loaded_model
    logger.info(f"Model Data: {model}")

    # Return a success response
    return jsonify({"message": f"Model {model} received successfully"}), 200

@app.route('/receive_llm_flock_model', methods=['POST'])
def receive_llm_flock_model():
    """
    Route to receive a LLM model from the flock_trainer.
    """
    global model
    data = request.json
    serialized_llm_flock_model = data.get('model')
    loaded_model: LLMFlockModel = pickle.loads(serialized_llm_flock_model)
    model = loaded_model
    logger.info(f"Model Data: {model}")

    # Return a success response
    return jsonify({"message": f"Model {model} received successfully"}), 200

@app.route('send_model_weights', methods=['POST'])
def send_model_weights():
    """
    Route to send the trained FlockModel's weights back to the flock_trainer.
    """
    weights = train_model(model)
    serialized_weights = pickle.dumps(weights)
    # May have to use this instead of request.remote_addr
    # headers_list = request.headers.getlist("X-Forwarded-For")
    # node_address = headers_list[0] if headers_list else request.remote_addr
    node_address = request.remote_addr
    user_address = request.json.get('user_address')
    requests.post(f"{user_address}/receive_model_weights", json={"node_address": node_address, "weights": serialized_weights})
    return jsonify({"weights": weights}), 200

def train_model(object: FlockModel):
    """
    Train the model and return the weights.
    """
    weights = object.train()
    return weights

def test_model(object: FlockModel):
    """
    Test the model and return the accuracy.
    """
    accuracy = object.evaluate()
    return accuracy

# TODO: Implement training and evaluating/inferencing for LLM model
def train_llm_model(object: LLMFlockModel):
    """
    Train the LLM model and return the weights.
    """
    pass


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)