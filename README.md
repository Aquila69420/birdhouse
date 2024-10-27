# Birdhouse
![Project Logo](logo.jpg)
<!-- ![Flock.io logo](flock_logo.svg) -->
<!-- ![Event Logo](encode.png) -->

This repository hosts the code for Birdhouse, a decentralized Federated Learning web app using Flock.io’s design principles, focusing on accessible and secure machine learning training across distributed nodes, built as part of the Encode London Hackathon hosted October 25 to October 27, 2024. The application is built with a **backend** for managing federated learning processes and a **frontend** for an intuitive user experience.

## Table of Contents
- [Project Structure](#project-structure)
- [Built With](#built-with)
- [Features](#features)
- [Flock.io Principles](#flock.io-principles)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Presentation](#presentation)
- [Video](#video)
- [License](#license)



## Project Structure

 * /backend
   * client.py
   *  data_preprocessing.py
   * device_manager.py
   * flock_model.py
   * flock_trainer.py
   * flnn_flock_model.py
   * model_classes.py
   * mongo_url.txt
   * node.py
   * utils/
      * constants.py
      * flock_api.py
      * gpu_utils.py
   * ui_cmd_line/
   * .pre-commit-config.yaml
   * full_automation.py
   * merge.py
   * training_args.yaml
   * src/
      * client/
         * \_\_init\_\_.py
         * fed_ledger.py
      * core/
         * collator.py
         * dataset.py
         * exception.py
         * gpu_utils.py
         * template.py
      * data/
         * dummy_data.jsonl
      * start.sh
      * .env
      * validate.py
      * validation_config.json.example
 * /frontend
   * public/
   * src/
      * assets/
      * components/
      * contexts/
      * images/
      * layouts/
      * theme/
      * variables/
      * views/
      * App.js
      * index.js
      * routes.js
  * LICENSE.txt
  * README.md
## Built With

* [![Flask][Flask.com]][Flask-url]
* [![React][React.js]][React-url]
* [![Redux][Redux.js]][Redux-url]
* [![Express][Express.js]][Express-url]
* [![PyTorch][Torch]][Torch-url]
* [![Transformers][Transformers]][Transformers-url]
* [![Scikit-Learn][Scikit-Learn.com]][Scikit-Learn-url]

## Features

- **Federated Learning Model**: Decentralized model training using nodes across the network.
- **Decentralized Incentivization**: Proof-of-stake and reward systems with token staking.
- **User Roles**: Support for task creators/delegators and training/validation nodes.
- **Comprehensive UI**: Frontend for model setup, training node management, and performance tracking.
- **Attack Mitigation**: Security features against Sybil, DoS, Free-rider, and Lookup attacks.

## Flock.io Principles

The Flock.io framework is designed to democratize and decentralize AI model training, making high-quality machine learning accessible and secure for users worldwide. This project leverages key Flock.io principles, integrating them into a decentralized platform that promotes equitable access, incentivization, and robust security for federated learning. Below are the core principles of Flock.io that we implement in this project:

### 1. AI Arena
The **AI Arena** is the core environment where federated learning tasks are managed and executed. In this system, users can submit model training tasks, which are distributed across multiple nodes in a decentralized manner. The arena allows users to select a model architecture and adjust hyperparameters. The training workload is divided among nodes (Training Nodes), and upon task completion, the aggregated model is evaluated and prepared for deployment. This approach provides:

- **Distributed training** that reduces the computational burden on individual users.
- **Fault tolerance** and resilience, as training is spread across many nodes.
- **Enhanced accessibility**, allowing users without extensive resources to participate in machine learning model training.

### 2. Proof-of-Stake + Rewards + Staking Tokens System
Incentivization is fundamental to Flock.io's model, encouraging active participation through a **Proof-of-Stake (PoS)** mechanism combined with a **rewards** system. The system works as follows:

- **Staking**: Both users (Task Creators) and nodes (Training Nodes) stake tokens to participate in the training and validation processes.
- **Rewards**: Upon successful model aggregation and validation, rewards are distributed based on contributions. Rewards can be earned by training nodes for providing quality updates, validators for securing the aggregation process, and task creators for engaging with the platform.
- **Security**: The staking system disincentivizes malicious actions and contributes to platform security. Malicious nodes may lose their staked tokens if they attempt to undermine the aggregation process.

### 3. Aggregation
**Aggregation** is a crucial step in federated learning where individual models trained on different nodes are combined into a single global model. In our system, aggregation is performed after training nodes send their model updates to the AI Arena. The aggregated model is then evaluated for quality, ensuring it meets the required standards before deployment. This approach provides:

- **Efficient use of distributed data** by aggregating insights from multiple sources.
- **Enhanced privacy** as raw data never leaves the node but contributes to the global model.

### 4. Roles in the Flock Ecosystem

- **Task Creators**: Users who initiate training tasks, specifying the model architecture and hyperparameters. They stake tokens to create tasks and, upon successful completion, receive rewards. Task creators benefit from a low barrier to entry, able to access machine learning training without large computational investments.
  
- **Training Nodes**: These nodes perform the distributed training, receiving portions of the model and data segments. They complete training based on the task creator’s specifications and submit their weights to the AI Arena for aggregation. Training nodes stake tokens to participate and earn rewards based on their contributions and the quality of their updates.

- **Validators**: Validators verify the integrity and quality of the model aggregation process, ensuring that only valid, high-quality models are accepted. They play a critical role in maintaining the reliability and security of the system by detecting anomalies, preventing model poisoning attacks, and validating genuine contributions.

- **Delegators**: Delegators support validators and training nodes by staking tokens, allowing them to participate in tasks. Delegators earn a portion of the rewards based on the performance and honesty of the nodes they support, aligning incentives for collaboration and accountability within the ecosystem.

This structure, inspired by the Flock.io framework, provides a robust system for **fault tolerance**, **decentralization**, and **equity** in AI training. Each role is essential to the network’s functionality, security, and success, as outlined in the whitepaper. Combining these principles enables a decentralized training ecosystem that is both accessible and secure, with built-in protections against Sybil attacks, DoS, model poisoning, and other adversarial threats.

## Getting Started

### Prerequisites

- **Node.js 20.18.0**: (For the frontend)
- **Python 3.12.2**: (For the backend)

### Installation

1. Clone the repository on each client and navigate to the backend folder:
   ```bash
   git clone https://github.com/Aquila69420/birdhouse.git
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. With Node.js installed run:
   ```bash
   cd frontend
   npm i
   cd ..
   ```

### Usage
1. On each training node client run the node script:
   ```bash
   python backend/node.py
   ```
2. On the user's client run the flock_trainer script:
   ```bash
   python backend/flock_trainer.py
   ```
3. Navigate to the frontend directory and start the frontend
   ```bash
   cd frontend
   npm start
   ```

The application should now be accessible in your browser.

## Presentation
[View Presentation](https://prezi.com/view/vC9ths61d6mv1dR4NQqy/)

## Video
[View Demo](https://youtu.be/dMFcckJwb2I)

<!-- LICENSE -->
## License

Distributed under the BSD 3-Clause "New" or "Revised" License License. See `LICENSE.txt` for more information.

<!-- MARKDOWN LINKS & IMAGES -->
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Redux.js]: https://img.shields.io/badge/Redux-764ABC?style=for-the-badge&logo=redux&logoColor=white
[Redux-url]: https://redux.js.org/
[Express.js]: https://img.shields.io/badge/Express-000000?style=for-the-badge&logo=express&logoColor=white
[Express-url]: https://expressjs.com/
[Flask.com]: https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white
[Flask-url]: https://flask.palletsprojects.com/
[Torch]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[Torch-url]: https://pytorch.org/
[Scikit-Learn.com]: https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white
[Scikit-Learn-url]: https://scikit-learn.org/
[Transformers]: https://img.shields.io/badge/Transformers-FF6F00?style=for-the-badge&logo=transformers&logoColor=white
[Transformers-url]: https://huggingface.co/transformers/
