// MongoDB Playground
// Use Ctrl+Space inside a snippet or a string literal to trigger completions.

// The current database to use.
use("pdf_embeddings_db");

// Find a document in a collection.
db.embeddings.aggregate(
    [
        {'$vectorSearch': {
        'index': 'vector_index', 
        'path': 'embedding', 
        'queryVector':"<embedding array>",
        'numCandidates': 50, 
        'limit': 2}
        }, 
        {'$project': {'_id': 0, 'ObjectID': 0}}]
)
