import os
import logging
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_linear_schedule_with_warmup

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMFlockModel:
    """
    LLMFlockModel is a class for managing and fine-tuning a language model from Hugging Face's model hub.
    Attributes:
        model: Loaded Hugging Face model for causal language modeling.
        tokenizer: Tokenizer corresponding to the loaded model.
    Methods:
        __init__(model_name: str, output_dir: str, verbose: bool = False):
        fine_tune(texts, epochs=3, batch_size=8, learning_rate=5e-5):
        generate_text(prompt: str, max_length: int = 50) -> str:
        _run_evaluation_script(test_prompts: list, expected_outputs: list) -> float:
        save_model(save_path: str):
    """
    def __init__(self, model_name: str, output_dir: str, verbose: bool = False):
        """
        Initializes the LLMFlockModel with a specified model and output directory.
        
        Args:
            model_name (str): Name of the Hugging Face model to load.
            output_dir (str): Directory to save outputs.
            verbose (bool): Verbosity flag for logging.
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.verbose = verbose

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.verbose:
            logger.setLevel(logging.DEBUG)

        logger.info(f"Loaded model: {self.model_name}")

    def fine_tune(self, texts, epochs=3, batch_size=8, learning_rate=5e-5):
        """
        Fine-tunes the model on a custom dataset.
        
        Args:
            texts (list): List of training texts for fine-tuning.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for the optimizer.
        """
        # Prepare the dataset and dataloader
        dataset = TextDataset(texts, self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Define the optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(data_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        self.model.train()  # Set model to training mode

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (input_ids, attention_mask) in enumerate(data_loader):
                input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()

                # Log batch-level details if verbose
                if self.verbose and batch_idx % 10 == 0:
                    logger.debug(f"Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item()}")

            avg_loss = epoch_loss / len(data_loader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss}")

        logger.info("Fine-tuning complete!")

    def generate_text(self, prompt: str, max_length: int = 50) -> str:
        """
        Generates text from a given prompt using the loaded model.
        
        Args:
            prompt (str): The input prompt to generate text from.
            max_length (int): The maximum length of the generated text.
        
        Returns:
            str: The generated text.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length)
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated text: {generated_text}")
        return generated_text

    def _run_evaluation_script(self, test_prompts: list, expected_outputs: list) -> float:
        """
        Runs an evaluation on the model using provided test prompts and expected outputs.
        
        Args:
            test_prompts (list): List of prompts for evaluation.
            expected_outputs (list): Corresponding expected outputs for evaluation.
        
        Returns:
            float: Average accuracy of the model's responses.
        """
        total_correct = 0
        total_tests = len(test_prompts)

        for prompt, expected in zip(test_prompts, expected_outputs):
            generated = self.generate_text(prompt)
            # Here, you could use more sophisticated evaluation (e.g., BLEU score)
            if expected.lower() in generated.lower():  # Simplistic check
                total_correct += 1
        
        accuracy = total_correct / total_tests if total_tests > 0 else 0.0
        logger.info(f"Evaluation accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def save_model(self, save_path: str):
        """
        Saves the model and tokenizer to the specified directory.
        
        Args:
            save_path (str): Path to save the model and tokenizer.
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")


class TextDataset(Dataset):
    """
    A custom dataset class for handling text data and tokenization.
    Args:
        texts (list of str): A list of text strings to be tokenized.
        tokenizer (PreTrainedTokenizer): A tokenizer instance from the Hugging Face Transformers library.
        max_length (int, optional): The maximum length of the tokenized sequences. Defaults to 512.
    Methods:
        __len__(): Returns the number of text samples in the dataset.
        __getitem__(idx): Tokenizes the text at the given index and returns the input IDs and attention mask.
    Returns:
        tuple: A tuple containing the input IDs and attention mask for the tokenized text.
    """
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encodings["input_ids"].squeeze(), encodings["attention_mask"].squeeze()

class SentimentTextDataset(TextDataset):
    """
    A dataset class for sentiment analysis tasks, extending the TextDataset class.
    Args:
        texts (list of str): A list of text samples.
        labels (list of int): A list of sentiment labels corresponding to the text samples.
        tokenizer (Tokenizer): A tokenizer instance to convert text samples into token IDs.
        max_length (int, optional): The maximum length of tokenized sequences. Defaults to 512.
    Methods:
        __getitem__(idx):
            Retrieves the tokenized input IDs, attention mask, and label for a given index.
            Args:
                idx (int): The index of the item to retrieve.
            Returns:
                tuple: A tuple containing input IDs, attention mask, and the corresponding label as a tensor.
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        super().__init__(texts, tokenizer, max_length)
        self.labels = labels  # Add labels to the dataset

    def __getitem__(self, idx):
        input_ids, attention_mask = super().__getitem__(idx)
        label = self.labels[idx]
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)  # Include label in each item