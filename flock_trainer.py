from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import json
import logging
from flock_model import FlockModel
from llm_flock_model import LLMFlockModel

# Assuming the classes are imported
# from your_model_file import FlockModel, LLMFlockModel

app = Flask(__name__)
api = Api(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlockTrainer:
    def __init__(self):
        self.models = {}  # To store instantiated models by their IDs

    @api.route('/instantiate_flock_model', methods=['POST'])
    def instantiate_flock_model(self):
        """
        Input route to instantiate a FlockModel.
        Expects JSON with 'model', 'loss_function', and other params as needed.
        """
        data = request.json
        model_name = data.get('model')
        loss_function_name = data.get('loss_function')
        # Other parameters can be retrieved similarly
        # Instantiate the model using user inputs
        model = FlockModel(model=model_name, loss_function=loss_function_name)
        
        model_id = f"flock_model_{len(self.models)}"
        self.models[model_id] = model
        logger.info(f"FlockModel instantiated with ID: {model_id}")

        return jsonify({"model_id": model_id}), 201

    @api.route('/instantiate_llm_flock_model', methods=['POST'])
    def instantiate_llm_flock_model(self):
        """
        Input route to instantiate an LLMFlockModel.
        Expects JSON with 'model_name' and 'output_dir'.
        """
        data = request.json
        model_name = data.get('model_name')
        output_dir = data.get('output_dir')

        llm_model = LLMFlockModel(model_name=model_name, output_dir=output_dir)
        
        model_id = f"llm_flock_model_{len(self.models)}"
        self.models[model_id] = llm_model
        logger.info(f"LLMFlockModel instantiated with ID: {model_id}")

        return jsonify({"model_id": model_id}), 201

    @api.route('/send_llm_flock_model', methods=['POST'])
    def send_llm_flock_model(self, model_id):
        """
        Output route to send the instantiated LLMFlockModel object to a node.
        Expects JSON with 'node_url' where the model will be sent.
        """
        data = request.json
        node_url = data.get('node_url')

        if model_id not in self.models:
            return jsonify({"error": "Model ID not found"}), 404

        llm_model = self.models[model_id]
        # Here, you can implement the logic to serialize the model and send it to the node
        # For demonstration, we're just logging it
        logger.info(f"Sending LLMFlockModel with ID {model_id} to {node_url}")
        # Send the model (consider using requests.post() to send it)

        return jsonify({"message": f"LLMFlockModel with ID {model_id} sent to {node_url}"}), 200

    @api.route('/send_flock_model', methods=['POST'])
    def send_flock_model(self, model_id):
        """
        Output route to send the instantiated FlockModel object to a node.
        Expects JSON with 'node_url' where the model will be sent.
        """
        data = request.json
        node_url = data.get('node_url')

        if model_id not in self.models:
            return jsonify({"error": "Model ID not found"}), 404

        flock_model = self.models[model_id]
        # Here, you can implement the logic to serialize the model and send it to the node
        # For demonstration, we're just logging it
        logger.info(f"Sending FlockModel with ID {model_id} to {node_url}")
        # Send the model (consider using requests.post() to send it)

        return jsonify({"message": f"FlockModel with ID {model_id} sent to {node_url}"}), 200


if __name__ == "__main__":
    # Create an instance of FlockTrainer
    flock_trainer = FlockTrainer()
    app.run(debug=True, host='0.0.0.0', port=5000)
