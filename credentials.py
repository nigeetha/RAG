#Embedding
GOOGLE_API_KEY = ""  # Replace with your actual key
GOOGLE_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"
# Hugging face
HF_API_KEY = "" 

# MongoDB connection
client = MongoClient("mongodb+srv://:@cluster0.g0poq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["pdf_embeddings_db"]

# Define collections
text_collection = db["embeddings"]
image_collection = db["image_embeddings"]

HF_REPO = "/Image-Store"