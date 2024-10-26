import json
import torch
import os
import subprocess
import platform
import psutil
import logging

class DeviceManager:
    """
    DeviceManager is a class responsible for managing device configurations and settings for machine learning models.
    It includes methods to load configuration files, check the compute platform, and set up devices for training.
    Methods:
        __init__(self, config_file=None):
            Initializes the DeviceManager with an optional configuration file.
        load_config(self, config_file):
            Loads the configuration from a JSON file if provided, otherwise returns an empty dictionary.
        check_compute_platform(self):
            Checks the availability of CUDA devices and logs their details. If CUDA is not available, it logs CPU and system memory information.
            Also runs the nvidia-smi command to check for specific GPU models and sets environment variables if necessary.
        setup_device(self, model_name, proposer_train_batch_size, proposer_train_micro_batch_size, device_map=None):
            Sets up the device and DDP (Distributed Data Parallel) settings based on the model name and training parameters.
                device_map (str or dict, optional): The mapping of devices to be used for training. Defaults to None.
    """
    def __init__(self, config_file=None):
        """
        Initializes the DeviceManager instance.

        Args:
            config_file (str, optional): Path to the configuration file. Defaults to None.

        Attributes:
            config (dict): Configuration settings loaded from the config file.
        """
        self.config = self.load_config(config_file)
        self.check_compute_platform()

    def load_config(self, config_file):
        """
        Loads configuration from a JSON file.

        Args:
            config_file (str): Path to the JSON configuration file.

        Returns:
            dict: The configuration data loaded from the file. Returns an empty dictionary if no file is provided.
        """
        if config_file:
            with open(config_file, 'r') as file:
                return json.load(file)
        return {}

    def check_compute_platform(self):
        """
        Checks the compute platform available on the system, including CUDA-enabled GPUs and CPU information.
        This method performs the following checks:
        1. Determines if CUDA is available and logs the number of CUDA devices.
        2. Logs details of each CUDA device, including its name, memory allocation, and CUDA capability.
        3. If CUDA is not available, logs CPU model information and total system memory.
        4. Runs the `nvidia-smi` command to get GPU names and checks for the presence of RTX 4090 GPUs.
        5. Sets environment variables to disable NCCL P2P and NCCL IB if RTX 4090 GPUs are detected due to incompatibility.
        Logs warnings and errors appropriately if CUDA is not available or if there are issues running the `nvidia-smi` command.
        Exceptions:
            - Logs a warning if CUDA is not available.
            - Logs an error if there is an issue running the `nvidia-smi` command.
            - Logs an error if any unexpected exception occurs.
        """
        try:
            # Check if CUDA is available
            if torch.cuda.is_available():
                num_devices = torch.cuda.device_count()
                logging.debug(f"CUDA is available with {num_devices} device(s).")

                # Iterate over each CUDA device and log its details
                for i in range(num_devices):
                    device_name = torch.cuda.get_device_name(i)
                    device_properties = torch.cuda.get_device_properties(i)
                    logging.debug(f"Device {i}: {device_name}")
                    logging.debug(f"  Memory Allocation: {device_properties.total_memory / 1e9:.2f} GB")
                    logging.debug(f"  CUDA Capability: {device_properties.major}.{device_properties.minor}")
            else:
                logging.warning("CUDA is not available. Checking CPU and system memory...")

                # Get CPU model information
                cpu_model = platform.processor()
                if not cpu_model:
                    cpu_model = platform.machine()  # Fallback to machine info if processor info is not available
                logging.debug(f"CPU Model: {cpu_model}")

                # Get total system memory
                total_memory = psutil.virtual_memory().total / (1024 ** 3)  # Convert bytes to GB
                logging.debug(f"Total System Memory: {total_memory:.2f} GB")

            # Run the nvidia-smi command to get GPU names
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Check if there was an error running the nvidia-smi command
            if result.returncode != 0:
                logging.warning(f"Error running nvidia-smi: {result.stderr.strip()}")
                return

            # Check for RTX 4090 GPU and set environment variables if necessary
            gpu_names = result.stdout.splitlines()
            if any("4090" in name for name in gpu_names):
                os.environ["NCCL_P2P_DISABLE"] = "1"
                os.environ["NCCL_IB_DISABLE"] = "1"
                logging.warning("RTX 4090 detected. NCCL_P2P and NCCL_IB are disabled due to incompatibility.")
            else:
                logging.debug("No RTX 4090 GPUs detected. No environment variables changed.")

        except subprocess.SubprocessError as e:
            logging.error(f"Subprocess error: {str(e)}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {str(e)}")

    def setup_device(self, model_name, proposer_train_batch_size, proposer_train_micro_batch_size, device_map=None):
        """
        Set up the device and DDP (Distributed Data Parallel) settings based on model_name and training parameters.

        Args:
            model_name (str): The name of the model to determine special conditions.
            proposer_train_batch_size (int): The total batch size for training.
            proposer_train_micro_batch_size (int): The micro batch size for training.

        Returns:
            device_map (str or dict): The mapping of devices to be used for training.
            ddp (bool): Whether Distributed Data Parallel is enabled.
            gradient_accumulation_steps (int): Adjusted gradient accumulation steps.
        """

        # Check model-specific conditions from the config
        if device_map is None:
            if any(keyword in model_name for keyword in self.config.get("models_without_nccl", [])):
                # Specific models that don't support NCCL or require specific device settings
                device_map = "cuda:0"
            else:
                device_map = "auto"

        # Determine if we're in a distributed environment
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = world_size > 1

        gradient_accumulation_steps = proposer_train_batch_size // proposer_train_micro_batch_size

        if ddp:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device_map = {"": local_rank}
            gradient_accumulation_steps = max(1, gradient_accumulation_steps // world_size)

        return device_map, ddp, gradient_accumulation_steps