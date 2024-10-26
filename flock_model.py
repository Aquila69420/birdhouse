import torch
import io
import logging
from data_preprocessing import IndexesDataset, get_loader
import numpy as np
import random
import json
from model_class import *


class FlockModel:
    """
    FlockModel is a class designed to handle the training, evaluation, and aggregation of machine learning models.
    Attributes:
        model (str): The type of model to be used (supervised learning).
        loss_function (str): The type of loss function to be used (model/task dependent).
        classes (list): List of class names.
        batch_size (int): The size of the batches for training. Default is 256.
        epochs (int): The number of epochs for training. Default is 1.
        lr (float): The learning rate for the optimizer. Default is 0.03.
        emb_size (int): The embedding size. Default is 100.
        vocab_size (int): The vocabulary size. Default is 30000.
        device (torch.device): The device to run the model on ('cuda' or 'cpu').
    Methods:
        __init__(self, model, loss_function, classes, batch_size=256, epochs=1, lr=0.03, emb_size=100, vocab_size=30000):
            Initializes the FlockModel with the specified hyperparameters and device settings.
        init_dataset(self, dataset_path: str) -> None:
            Initializes the dataset from the given path and prepares the data loader.
        get_starting_model(self):
            Returns the initialized model with a fixed random seed for reproducibility.
        train(self, parameters: bytes | None) -> bytes:
        evaluate(self, parameters: bytes | None) -> float:
            Evaluates the model using the provided parameters and returns the accuracy on the dataset.
        aggregate(self, parameters_list: list[bytes]) -> bytes:
            Aggregates a list of model weights using averaging and returns the aggregated parameters.
    """
    def __init__(
            self,
            model, # model: Custom model class object defined by the user
            loss_function,  # loss_function: Loss function also defined by the user
            classes,
            batch_size=256,
            epochs=1,
            lr=0.03,
            emb_size=100,
            vocab_size=30000
    ):
        if model == "CNN":
            self.model = CNNClassifier()
        elif model == "LR":
            self.model = LogisticRegressionModel()
        elif model == "DT":
            self.model = DecisionTreeModel()
        
        if loss_function == "BCE":
            self.loss_function = torch.nn.BCELoss()
        elif loss_function == "MSE":
            self.loss_function = torch.nn.MSELoss()
        elif loss_function == "CE":
            self.loss_function = torch.nn.CrossEntropyLoss()
        elif loss_function == "NLL":
            self.loss_function = torch.nn.NLLLoss()

        self.batch_size = batch_size
        self.epochs = epochs
        self.classes = classes
        self.class_to_idx = {_class: idx for idx, _class in enumerate(self.classes)}
        self.lr = lr
        self.emb_size = emb_size
        self.vocab_size = vocab_size

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.device = torch.device(device)      
    
    def init_dataset(self, dataset_path: str) -> None:
        """
        Initializes the dataset for the model.

        This method loads a dataset from the specified JSON file path, processes it into a DataFrame,
        and sets up a data loader for testing purposes. It also logs the processing step and stores
        the size of the dataset.

        Args:
            dataset_path (str): The file path to the dataset JSON file.

        Returns:
            None
        """
        self.datasetpath = dataset_path
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        dataset_df = IndexesDataset(dataset, max_samples_count=10000, device=self.device)
        logging.info("Processing dataset")
        self.test_data_loader = get_loader(
            dataset_df, batch_size=self.batch_size
        )
        self.dataset_size = len(dataset_df)


    def get_starting_model(self):
        """
        Initializes the starting model with a fixed random seed for reproducibility.

        This method sets the random seed for Python's `random` module, NumPy, and PyTorch 
        (both CPU and CUDA) to ensure that the model initialization is deterministic. 
        It then returns an instance of the model with the specified vocabulary size and 
        embedding size.

        Returns:
            torch.nn.Module: An instance of the model initialized with the given 
            vocabulary size and embedding size.
        """
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        return self.model(vocab_size=self.vocab_size, emb_size=self.emb_size)

    def train(self, parameters: bytes | None) -> bytes:
        """
        Trains the model using the provided parameters and returns the trained model's state dictionary.
        Args:
            parameters (bytes | None): Serialized model parameters to initialize the model. If None, training starts from scratch.
        Returns:
            bytes: Serialized state dictionary of the trained model.
        """
        model = self.get_starting_model()
        if parameters is not None:
            model.load_state_dict(torch.load(io.BytesIO(parameters)))
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
        )
        criterion = self.loss_function
        model.to(self.device)

        for epoch in range(self.epochs):
            logging.debug(f"Epoch {epoch}")
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for batch_idx, (inputs, targets) in enumerate(self.test_data_loader):
                optimizer.zero_grad()

                inputs, targets = inputs.to(self.device), targets.to(self.device).unsqueeze(1)
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                loss.backward()

                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                predicted = torch.round(outputs).squeeze()
                # self.progress += len(inputs)
                train_total += targets.size(0)
                train_correct += (predicted == targets.squeeze()).sum().item()
                if batch_idx < 2:
                    logging.debug(
                        f"Batch {batch_idx}, Acc: {round(100.0 * train_correct / train_total, 2)}, Loss: {round(train_loss / train_total, 4)}"
                    )

            logging.info(
                f"Training Epoch: {epoch}, Acc: {round(100.0 * train_correct / train_total, 2)}, Loss: {round(train_loss / train_total, 4)}"
            )

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        return buffer.getvalue()

    def evaluate(self, parameters: bytes | None) -> float:
        """
        Evaluate the model using the provided parameters and test data.
        Args:
            parameters (bytes | None): Serialized model parameters in bytes format. 
                        If None, the model will use the starting parameters.
        Returns:
            float: The accuracy of the model on the test dataset.
        This method performs the following steps:
        1. Loads the model with the given parameters or initializes it with starting parameters.
        2. Moves the model to the specified device and sets it to evaluation mode.
        3. Iterates over the test data, computes the loss and accuracy.
        4. Logs the accuracy and loss.
        5. Returns the accuracy of the model on the test dataset.
        """
        criterion = self.loss_function

        model = self.get_starting_model()
        if parameters is not None:
            model.load_state_dict(torch.load(io.BytesIO(parameters)))
        model.to(self.device)
        model.eval()

        test_correct = 0
        test_loss = 0.0
        test_total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_data_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device).unsqueeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item() * inputs.size(0)
                predicted = torch.round(outputs).squeeze()
                test_total += targets.size(0)
                test_correct += (predicted == targets.squeeze()).sum().item()

        accuracy = test_correct / test_total
        logging.info(
            f"Model test, Acc: {accuracy}, Loss: {round(test_loss / test_total, 4)}"
        )

        return accuracy


    def aggregate(self, parameters_list: list[bytes]) -> bytes:
        """
        Aggregates a list of serialized model parameters by averaging them.
        Args:
            parameters_list (list[bytes]): A list of serialized model parameters in bytes format.
        Returns:
            bytes: The aggregated model parameters in bytes format.
        """
        parameters_list = [
            torch.load(io.BytesIO(parameters)) for parameters in parameters_list
        ]
        averaged_params_template = parameters_list[0]
        for k in averaged_params_template.keys():
            temp_w = []
            for local_w in parameters_list:
                temp_w.append(local_w[k])
            averaged_params_template[k] = sum(temp_w) / len(temp_w)

        # Create a buffer
        buffer = io.BytesIO()

        # Save state dict to the buffer
        torch.save(averaged_params_template, buffer)

        # Get the byte representation
        aggregated_parameters = buffer.getvalue()

        return aggregated_parameters


if __name__ == "__main__":
    """
    Hyper parameters
    """
    device = "cuda"
    max_seq_len = 64
    epochs = 3
    lr = 0.001
    emb_size = 100
    batch_size = 64
    classes = [
        "1",
        "2",
    ]