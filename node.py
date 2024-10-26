from flask import Flask, request, jsonify
import logging
import pickle
from flock_model import FlockModel
from llm_flock_model import *
import requests
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import io

# Initialize the Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

global model
global trained 
trained = False
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

@app.route('send_model_accuarcy', methods=['POST'])
def send_accuracy():
    """
    Route to send the local accuracy of the trained model back to the flock_trainer.
    """
    if not trained:
        train_model(model)
    accuracy = test_model(model)
    # May have to use this instead of request.remote_addr
    # headers_list = request.headers.getlist("X-Forwarded-For")
    # node_address = headers_list[0] if headers_list else request.remote_addr
    node_address = request.remote_addr
    user_address = request.json.get('user_address')
    requests.post(f"{user_address}/receive_accuracy", json={"node_address": node_address, "accuracy": accuracy})
    return jsonify({"accuracy": accuracy}), 200

@app.route('fine_tune_llm_flock_model', methods=['POST'])
def fine_tune_llm_flock_model():
    dataset_texts = [
    "I absolutely love this product! It has exceeded all my expectations.",
    "This is the worst experience I’ve ever had. Totally disappointed.",
    "Amazing service and friendly staff. Will come back again!",
    "I wouldn't recommend this to anyone. It was a complete waste of time.",
    "I'm so happy with my purchase. Great value for money!",
    "Terrible quality. It broke after just one use.",
    "Fantastic! Everything was perfect.",
    "Not worth the price at all.",
    "The movie was fantastic! Highly recommend it.",
    "Worst movie I've seen in years. Don't waste your time."
    ]

    dataset_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    fine_tune_llm_model(
            model_obj=model,
            dataset_texts=dataset_texts,
            dataset_labels=dataset_labels,
            epochs=3,
            batch_size=4,
            learning_rate=2e-5
        )
    # Prepare test prompts and expected outputs for evaluation
    test_prompts = [
        "I enjoyed the product.",
        "This was a waste of money.",
        "The service was excellent!",
        "I'm not satisfied with my purchase."
    ]
    expected_outputs = [1, 0, 1, 0]

    # Evaluate the model's performance
    accuracy = model._run_evaluation_script(test_prompts, expected_outputs)

    return jsonify({"accuracy": accuracy * 100})

def train_model(object: FlockModel):
    """
    Train the model and return the weights.
    """
    global trained
    weights = object.train()
    trained = True
    return weights

def test_model(object: FlockModel):
    """
    Test the model and return the accuracy.
    """
    accuracy = object.evaluate()
    return accuracy

# TODO: Implement fine-tuning and evaluating/inferencing for LLM model
def fine_tune_llm_model(model_obj: LLMFlockModel, dataset_texts, dataset_labels, epochs: int = 3, batch_size: int = 4, learning_rate: float = 2e-5) -> bytes:
    """
    Fine-tunes the LLM model on a sentiment analysis dataset and returns the updated weights.
    
    Args:
        model_obj (LLMFlockModel): The instance of the language model to fine-tune.
        dataset_texts (list): List of texts for training.
        dataset_labels (list): List of labels (1 for positive, 0 for negative).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        
    Returns:
        bytes: Serialized model weights after fine-tuning.
    """
    # Step 1: Load the dataset for sentiment analysis
    tokenizer = model_obj.tokenizer
    dataset = SentimentTextDataset(dataset_texts, dataset_labels, tokenizer, max_length=64)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Step 2: Set the model to training mode and define optimizer and loss
    model = model_obj.model
    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Step 3: Fine-tuning loop
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in data_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(outputs.view(-1), labels.float())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Step 4: Save model weights and return them as bytes
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    return buffer.getvalue()

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)