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
    user_address = request.json.get('node_address')
    requests.post(f"{user_address}/join_network", json={"node_address": request.remote_addr })
    return jsonify({"message": f"Joined network {user_address} successfully"}), 200

@app.route('/receive_model', methods=['POST'])
def receive_model():
    """
    Route to receive a model from the flock_trainer.
    Expects JSON containing 'model_id', 'model_data', and 'model_type'.
    """
    data = request.json
    model = data.get('model')  # This should contain the serialized model data

    # Here you would typically deserialize the model and store or use it
    deserialized_flock_model = pickle.loads(model)
    # For demonstration, we're just logging the received model data
    logger.info(f"Model Data: {deserialized_flock_model}")

    # Return a success response
    return jsonify({"message": f"Model {deserialized_flock_model} received successfully"}), 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
