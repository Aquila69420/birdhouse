from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier

class ModelClass(ABC):
    """
    ModelClass is an abstract base class for both traditional and neural network models. 
    It provides a unified structure for model classes with a flexible set of parameters.
    Attributes:
        params (dict): A dictionary of parameters passed to the model.
    Methods:
        forward(x):
            Abstract method to be implemented by subclasses. Defines the forward pass of the model.
        train_model(x, y):
            Optional method for non-deep learning models to define their training procedure.
    """
    def __init__(self, **kwargs):
        """
        Base class for both traditional and neural network models. This provides a
        unified structure for model classes with a flexible set of parameters.
        """
        super().__init__()
        self.params = kwargs

    @abstractmethod
    def forward(self, x):
        """Forward method to be implemented by subclasses"""
        pass

    def train_model(self, x, y):
        """Optional method for non-deep learning models to define their training procedure"""
        pass


# Neural Network Example: CNN Classifier
class CNNClassifier(ModelClass, nn.Module):
    """
    A Convolutional Neural Network (CNN) based classifier for text data.
    Args:
        vocab_size (int, optional): Size of the vocabulary. Default is 30000.
        emb_size (int, optional): Size of the embedding vectors. Default is 100.
        num_filters (int, optional): Number of convolutional filters. Default is 32.
        kernel_size (int, optional): Size of the convolutional kernel. Default is 3.
    Attributes:
        embedding (nn.Embedding): Embedding layer that converts input tokens to dense vectors.
        conv (nn.Conv1d): 1D convolutional layer.
        pool (nn.AdaptiveMaxPool1d): Adaptive max pooling layer.
        fc (nn.Linear): Fully connected layer that outputs the final classification score.
    Methods:
        forward(x):
            Defines the forward pass of the model.
            Args:
                x (torch.Tensor): Input tensor containing token indices.
            Returns:
                torch.Tensor: Output tensor containing the classification scores.
    """
    def __init__(self, vocab_size=30000, emb_size=100, num_filters=32, kernel_size=3):
        """
        Initializes the CNNClassifier model.

        Args:
            vocab_size (int, optional): Size of the vocabulary. Default is 30000.
            emb_size (int, optional): Size of the embedding vectors. Default is 100.
            num_filters (int, optional): Number of filters for the convolutional layer. Default is 32.
            kernel_size (int, optional): Size of the convolutional kernel. Default is 3.
        """
        super(CNNClassifier, self).__init__(vocab_size=vocab_size, emb_size=emb_size)
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.conv = nn.Conv1d(emb_size, num_filters, kernel_size)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters, 1)
    
    def forward(self, x):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output tensor after applying embedding, convolution, pooling, and fully connected layers.
        """
        x = self.embedding(x).permute(0, 2, 1)
        x = self.pool(torch.relu(self.conv(x))).squeeze(-1)
        return self.fc(x)

# Traditional Model Example: Logistic Regression
class LogisticRegressionModel(ModelClass, nn.Module):
    """
    A logistic regression model implemented using PyTorch.
    Inherits from:
        ModelClass: A custom base class for models.
        nn.Module: Base class for all neural network modules in PyTorch.
    Attributes:
        linear (nn.Linear): A linear transformation layer.
    Args:
        input_dim (int): The number of input features.
    Methods:
        forward(x):
            Performs a forward pass of the model.
            Args:
                x (torch.Tensor): Input tensor.
            Returns:
                torch.Tensor: Output tensor after applying the sigmoid function.
    """
    def __init__(self, input_dim):
        """
        Initializes the LogisticRegressionModel.

        Args:
            input_dim (int): The number of input features for the linear layer.
        """
        super(LogisticRegressionModel, self).__init__(input_dim=input_dim)
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        """
        Applies a linear transformation followed by a sigmoid activation function to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the linear transformation and sigmoid activation.
        """
        return torch.sigmoid(self.linear(x))

# Traditional Model Example: Decision Tree (sklearn-based)
class DecisionTreeModel(ModelClass):
    """
    DecisionTreeModel is a subclass of ModelClass that encapsulates a decision tree classifier.
    Attributes:
        model (DecisionTreeClassifier): An instance of sklearn's DecisionTreeClassifier.
    Methods:
        __init__(max_depth=None, random_state=0):
            Initializes the DecisionTreeModel with the specified maximum depth and random state.
        train_model(x, y):
            Trains the decision tree model using the provided features (x) and target labels (y).
        predict(x):
            Predicts the target labels for the given features (x) using the trained model.
    """
    def __init__(self, max_depth=None, random_state=0):
        """
        Initialize the DecisionTreeModel.

        Parameters:
        max_depth (int, optional): The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples. Defaults to None.
        random_state (int, optional): Controls the randomness of the estimator. Pass an int for reproducible output across multiple function calls. Defaults to 0.
        """
        super(DecisionTreeModel, self).__init__(max_depth=max_depth, random_state=random_state)
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    
    def train_model(self, x, y):
        """
        Trains the model using the provided features and target values.

        Parameters:
        x (numpy.ndarray): The input features for training the model.
        y (numpy.ndarray): The target values for training the model.

        Returns:
        None
        """
        # Fit the model using sklearn's API (requires x and y to be numpy arrays)
        self.model.fit(x, y)
    
    def predict(self, x):
        """
        Predict the output using the trained model.

        Parameters:
        x (array-like): Input data to be predicted.

        Returns:
        array-like: Predicted output.
        """
        return self.model.predict(x)
