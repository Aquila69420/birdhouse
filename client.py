# Registration via wallet id (wallet id, tokens) in MongoDB with initial amount of tokens when the user registers

# function to update node's token count and updating the total token count in the network (also stored in MongoDB)
# x2 - decrement when node places stake, increment when node receives reward

# function to update the user's token count and updating the total token count in the network (also stored in MongoDB)
# x1 - decrement when user places fee


from pymongo import MongoClient

mongo_uri = "mongodb+srv://janetjarrontester:qcZnyLVpARV7rDAU@birdhouse.s729d.mongodb.net/?retryWrites=true&w=majority&appName=Birdhouse"
db_name = "token_db"
collection_name = "tokens"

# # Connect to MongoDB
# client = MongoClient(mongo_uri)
# db = client[db_name]

# # Create the collection if it doesn't exist
# collection = db[collection_name]

# # Initialize the total_tokens collection
# total_tokens_collection = db['total_tokens']  # This collection will store total tokens

# # Create the collection schema
# # Example document to initialize the total tokens
# total_tokens_doc = {
#     '_id': 'total',  # Identifier for the total tokens document
#     'count': 1000    # Initialize total tokens to 1000
# }

# # Insert total tokens document if it doesn't already exist
# if total_tokens_collection.count_documents({'_id': 'total'}) == 0:
#     total_tokens_collection.insert_one(total_tokens_doc)
#     print("Total tokens initialized to 1000.")
# else:
#     print("Total tokens document already exists.")

# # Initialize the clients collection with a sample document (optional)
# sample_client_doc = {
#     'wallet_id': 'sample_wallet_001',
#     'tokens': 10  # Initial tokens for the sample client
# }

# # Insert sample client document if it doesn't already exist
# if collection.count_documents({'wallet_id': 'sample_wallet_001'}) == 0:
#     collection.insert_one(sample_client_doc)
#     print("Sample client added to the clients collection.")
from pymongo import MongoClient

class TokenManager:
    def __init__(self, mongo_uri: str):
        """
        Initializes the TokenManager and sets up the database and collections.

        Args:
            mongo_uri (str): The MongoDB connection URI.
        """
        self.db_name = "token_db"
        self.collection_name = "tokens"
        self.client = MongoClient(mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        self.total_tokens_collection = self.db['total_tokens']

    def login(self, wallet_id: str):
        """
        Logs in a user based on wallet ID.

        Args:
            wallet_id (str): The wallet ID of the user.
        
        Returns:
            str: A message indicating the login status.
        """
        # Check if the wallet_id exists in the database
        return self.collection.count_documents({'wallet_id': wallet_id}) > 0
    

    def register_client(self, wallet_id: str, initial_tokens: int = 10):
        """
        Registers a new client with an initial amount of tokens and decrements total tokens.

        Args:
            wallet_id (str): The wallet ID of the client.
            initial_tokens (int): The initial amount of tokens to assign (default is 10).
        """
        # Check if the wallet_id already exists
        if self.collection.count_documents({'wallet_id': wallet_id}) > 0:
            print(f"Client with wallet_id {wallet_id} already exists.")
            return
        # Check and decrement total tokens
        total_tokens_doc = self.total_tokens_collection.find_one({'_id': 'total'})
        if total_tokens_doc and total_tokens_doc['count'] >= 10:
            # Decrement total tokens
            self.total_tokens_collection.update_one(
                {'_id': 'total'},
                {'$inc': {'count': -10}}
            )
            print(f"10 tokens deducted from total tokens. New total: {total_tokens_doc['count'] - 10}")

            # Register the client
            client_doc = {
                'wallet_id': wallet_id,
                'tokens': initial_tokens  # Assign initial tokens to the client
            }
            self.collection.insert_one(client_doc)
            print(f"Client with wallet_id {wallet_id} registered successfully.")
        else:
            print("Not enough tokens available to register the client.")

    def update_client_tokens(self, wallet_id: str, delta: int):
        """
        Updates the token count for a client and adjusts the total tokens in the network.

        Args:
            wallet_id (str): The wallet ID of the Client to update.
            delta (int): The amount to change the Client's tokens by (positive or negative).
        """
        # Update the Client's tokens
        result = self.collection.update_one(
            {'wallet_id': wallet_id},
            {'$inc': {'tokens': delta}}
        )

        # Check if the Client existed and was updated
        if result.matched_count == 0:
            print(f"No Client found with wallet_id {wallet_id}.")
            return
        
        # Update the total tokens based on the delta
        if delta > 0:
            # Client gains tokens: decrease total tokens
            self.total_tokens_collection.update_one(
                {'_id': 'total'},
                {'$inc': {'count': -delta}}
            )
            print(f"Client {wallet_id} gained {delta} tokens. Total tokens decremented by {delta}.")
        else:
            # Client loses tokens: increase total tokens
            self.total_tokens_collection.update_one(
                {'_id': 'total'},
                {'$inc': {'count': -delta}}  # delta is negative, so this adds the absolute value
            )
            print(f"Client {wallet_id} lost {-delta} tokens. Total tokens incremented by {-delta}.")