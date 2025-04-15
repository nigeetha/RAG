from flask import Flask, request, jsonify,render_template
import requests
from pymongo import MongoClient
from bson.objectid import ObjectId

app = Flask(__name__)

# Configuration
MONGO_URI = "mongodb+srv://arunmass:senthil3226w@wardrobe-products.sjzov.mongodb.net/?retryWrites=true&w=majority&appName=wardrobe-products"
GOOGLE_API_KEY = "AIzaSyBrCisSoUqfhFvP2L3bXLhOUUZl9kHLbL0"
GOOGLE_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"

# Initialize MongoDB client
client = MongoClient(MONGO_URI)
db = client['pdf_embeddings_db']
collection = db['embeddings']
image_collection = db['image_embeddings']

def get_text_embedding(text):
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "models/text-embedding-004",
        "content": {
            "parts": [{"text": text}]
        }
    }
    
    try:
        response = requests.post(
            f"{GOOGLE_API_URL}?key={GOOGLE_API_KEY}",
            headers=headers,
            json=payload
        )
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Extract embedding from response
        data = response.json()
        embedding = data.get("embedding", None)
        
        if embedding is not None:
            return embedding
        else:
            print("No embedding returned in the response.")
            return None
    
    except requests.exceptions.RequestException as e:
        print(f"Error getting embedding: {str(e)}")
        return None

@app.route('/query_mongodb', methods=['POST'])
def query_mongodb():
    """
    Execute an aggregation pipeline on MongoDB
    """
    try:
        text = request.json.get('text')
        # print(request.json)
        n = request.json.get('n')
        if not text:
            return jsonify({'error': 'text is required'}), 400
        
        embeddings = get_text_embedding(text)['values']
        
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": embeddings,
                    "numCandidates": 50,
                    "limit": n
                }
            },
            {
                "$project":{
                    "_id":0,
                    "ObjectID":0,
                    "embedding":0
                }
            }
        ]
        
        # print(pipeline)
        # Execute the aggregation pipeline
        txt_results = list(collection.aggregate(pipeline))
        img_results = list(image_collection.aggregate(pipeline))
        
        results = {
            "txt":txt_results,
            "img":img_results
        }
        return jsonify({
            'status': 'success',
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/',methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/list/text',methods=['GET'])
def list_text_data():
    data = collection.find({},{"embedding":0})
    
    # return jsonify({"data":data})
    return render_template('data.html',data=data)

@app.route('/list/image',methods=['GET'])
def list_image_data():
    data = image_collection.find({},{"embedding":0})
    
    # return jsonify({"data":data})
    return render_template('data.html',data=data)

@app.route('/delete/image/<Id>',methods=['DELETE'])
def delete_image_doc(Id):
    objectid = ObjectId(Id)
    image_collection.delete_one({"_id":objectid})
    
    return jsonify({"result":"success"}),200


@app.route('/delete/text/<Id>',methods=['DELETE'])
def delete_text_doc(Id):
    objectid = ObjectId(Id)
    collection.delete_one({"_id":objectid})
    
    return jsonify({"result":"success"}),200

if __name__ == '__main__':
    
    app.run(debug=True)