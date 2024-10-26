import torch
import io
import logging
from flock_sdk import FlockSDK
from data_preprocessing import IndexesDataset, get_loader
from pandas import DataFrame
import numpy as np
import random
import json
import torchvision
from model_class import *


class FlockModel:
    def __init__(
            self,
            model, # model: Custom model class object defined by the user
            loss_function,  # loss_function: Loss function also defined by the user
            classes,
            batch_size=256,
            epochs=1,
            lr=0.03,
            emb_size=100,
            vocab_size=30000,
            client_id=1
    ):
        """
        Hyper parameters
        """
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

        """
            Device setting
        """
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.device = torch.device(device)        
    
    def init_dataset(self, dataset_path: str) -> None:
        self.datasetpath = dataset_path
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        dataset_df = IndexesDataset(dataset, max_samples_count=10000, device=device)
        logging.info("Processing dataset")
        self.test_data_loader = get_loader(
            dataset_df, batch_size=batch_size
        )


    def get_starting_model(self):
        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        return self.model(vocab_size=self.vocab_size, emb_size=self.emb_size)

    """
    train() should:
    1. Take in the model weights as bytes and load them into your model
    2. If parameters passed are None, initialize them to match your untrained model's parameters (i.e. clean slate)
    3. If needed pre-process the dataset which is passed as a list of rows parsed as dicts
    4. Output the model parameters retrained on the dataset AS BYTES
    """

    def train(self, parameters: bytes | None) -> bytes:

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

    """
    evaluate() should:
    1. Take in the model weights as bytes and load them into your model
    3. If parameters passed are None, initialize them to match your untrained model's parameters (i.e. clean slate)
    4. If needed pre-process the dataset which is passed as a list of rows parsed as dicts
    5. Output the accuracy of the model parameters on the dataset as a float
    """

    def evaluate(self, parameters: bytes | None) -> float:
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

    """
    aggregate() should take in a list of model weights (bytes),
    aggregate them using avg and output the aggregated parameters as bytes.
    """

    def aggregate(self, parameters_list: list[bytes]) -> bytes:
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


    # flock_model = FlockModel(
    #     classes=classes,
    #     batch_size=batch_size,
    #     epochs=epochs,
    #     lr=lr,
    #     emb_size=emb_size,
    #     loss_function=torch.nn.BCELoss(),
    #     model=torchvision.models.resnet18,
    # )

    # sdk = FlockSDK(flock_model)
    # sdk.run()