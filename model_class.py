from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class ModelClass(ABC):
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
    def __init__(self, vocab_size, emb_size, num_filters=32, kernel_size=3):
        super(CNNClassifier, self).__init__(vocab_size=vocab_size, emb_size=emb_size)
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.conv = nn.Conv1d(emb_size, num_filters, kernel_size)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters, 1)
    
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.pool(torch.relu(self.conv(x))).squeeze(-1)
        return self.fc(x)

# Traditional Model Example: Logistic Regression
class LogisticRegressionModel(ModelClass, nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__(input_dim=input_dim)
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Traditional Model Example: Decision Tree (sklearn-based)
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeModel(ModelClass):
    def __init__(self, max_depth=None, random_state=0):
        super(DecisionTreeModel, self).__init__(max_depth=max_depth, random_state=random_state)
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    
    def train_model(self, x, y):
        # Fit the model using sklearn's API (requires x and y to be numpy arrays)
        self.model.fit(x, y)
    
    def predict(self, x):
        return self.model.predict(x)
