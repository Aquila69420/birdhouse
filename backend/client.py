from pymongo import MongoClient
from base64 import b64decode

# Load the MongoDB URI from the file
with open ("backend/mongo_uri.txt", "r") as myfile:
    mongo_uri=b64decode(myfile.readline().strip()).decode("utf-8")
db_name = "token_db"
collection_name = "tokens"

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