# @title requirements
# !pip install watchdog requests  PyMuPDF huggingface_hub pymongo[srv]
# !mkdir pdfs

import os
from pymongo import MongoClient
import fitz  # PyMuPDF
import re
import io
from PIL import Image
from huggingface_hub import InferenceClient, upload_file
import requests
import time
from bson.objectid import ObjectId
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from credentials import *

#------------------------------------------------------#
# helper functions
def format_text(raw_text):
    # Step 1: Normalize whitespace
    text = re.sub(r'\n+', '\n', raw_text.strip())  # Remove excessive newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Replace tabs and extra spaces with single space

    # Step 2: Add formatting for section titles
    text = re.sub(r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b:', r'\n\033[1m\1\033[0m:\n', text)  # Bold section titles

    # Step 3: Format key-value pairs and bullet points
    text = re.sub(r'(\b\d+[kKMGTPEZYQR]B\b.*)', r'  - \1', text)  # Bullet points for units
    text = re.sub(r'([a-zA-Z]+):([^\n]+)', r'  \1: \2', text)  # Key-value pairs

    # Step 4: Add better line breaks for lists or subcategories
    text = re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', text)  # Add space between number and text if missing
    text = re.sub(r'\b(\d+)\b', r'\n  - \1', text)  # Add bullets for numeric values

    # Step 5: Add space between sections for better readability
    text = re.sub(r'(\n\033\[1m.*?\033\[0m:)', r'\n\1', text)  # Add newline before section headers

    return text


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file on a per-page basis.

    Parameters:
        pdf_path (str): Path to the PDF file.

    Returns:
        list: A list of dictionaries containing page text and page numbers.
    """
    pdf_document = fitz.open(pdf_path)
    text_data = []

    for page_no in range(len(pdf_document)):
        page = pdf_document[page_no]
        text = format_text(page.get_text())
        if text.strip():
            text_data.append({"text": text, "page_no": page_no + 1})

    return text_data

# pdf image extraction and uploading to hugging face and then get the description of the image
client = InferenceClient(api_key=HF_API_KEY)

def extract_images_from_pdf(pdf_path):
    """
    Extract images from a PDF file.

    Parameters:
        pdf_path (str): Path to the PDF file.

    Returns:
        list: A list of dictionaries containing image data and page numbers.
    """
    pdf_document = fitz.open(pdf_path)
    images = []

    for page_index in range(len(pdf_document)):
        page = pdf_document[page_index]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append({"page_no": page_index + 1, "image_index": img_index, "image": image})

    return images

def upload_image_to_huggingface(image, pdf_name, page_number, image_index):
    """
    Upload an image to Hugging Face and return its public URL.

    Parameters:
        image (PIL.Image.Image): The image to upload.
        filename (str): The filename to use for the upload.

    Returns:
        str: The public URL of the uploaded image.
    """
    # Save the image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    sanitized_pdf_name = os.path.splitext(os.path.basename(pdf_name))[0].replace(" ", "_")
    filename = f"{sanitized_pdf_name}/page_{page_number}_image_{image_index}.jpg"

    # Upload the file to Hugging Face Hub
    url = upload_file(
        path_or_fileobj=image_bytes,
        path_in_repo=filename,
        repo_id=HF_REPO,
        token=HF_API_KEY
    )
    downloadable_url = url.replace("blob/main/", "resolve/main/")
    return downloadable_url

def describe_image(image, pdf_name, page_number, image_index):
    """
    Generate a description for an image using Hugging Face.

    Parameters:
        image (PIL.Image.Image): The image to describe.
        filename (str): The filename for the image.

    Returns:
        str: The generated description.
    """
    # Upload the image and get the public URL
    public_url = upload_image_to_huggingface(image, pdf_name, page_number, image_index)

    # Prepare the messages for inference
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image_url", "image_url": {"url": public_url}}
            ]
        }
    ]

    # Perform inference
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct",
        messages=messages,
        max_tokens=100
    )
    return completion.choices[0].message["content"],public_url

# pdf - text or description emebedding request function
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
            return embedding['values']
        else:
            print("No embedding returned in the response.")
            return None

    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        return None

# pdf - texts and description storing in the mongo db
def store_text_embedding(pdf_path, page_no, embedding, text):
    """
    Stores text embeddings in MongoDB.

    Parameters:
        pdf_path (str): Path to the PDF file.
        page_no (int): Page number of the text.
        embedding (list): The embedding vector.
        text (str): Text content.
    """
    document = {
        "text": text,
        "page_no": page_no,
        "pdf": pdf_path,
        "embedding": embedding,
        "timestamp": time.time()
    }
    text_collection.insert_one(document)

def store_image_embedding(image_path, page_no, image_index, text_object_id, embedding, description):
    """
    Stores image embeddings in MongoDB.

    Parameters:
        image_path (str): Path to the image.
        page_no (int): Page number of the image.
        image_index (int): Index of the image on the page.
        text_object_id (ObjectId): MongoDB ObjectID of the related text.
        embedding (list): The embedding vector.
        description (str): Image description.
    """
    document = {
        "image_path": image_path,
        "description": description,
        "page_no": page_no,
        "image_index": image_index,
        "text_object_id": text_object_id,
        "embedding": embedding,
        "timestamp": time.time()
    }
    image_collection.insert_one(document)

# pdf processing
def process_pdf(pdf_path):
    pdf_name = os.path.basename(pdf_path)

    # Open PDF
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {str(e)}")
        return

    for page_number in range(len(doc)):
        try:
            page = doc.load_page(page_number)
            # Extract text from the page
            page_text = page.get_text()

            # Embed text content
            try:
                text_embedding = get_text_embedding(page_text)
            except Exception as e:
                print(f"Error embedding text on page {page_number + 1}: {str(e)}")
                continue

            # Store the text content and its embedding in MongoDB
            text_data = {
                "text": page_text,
                "page no": page_number + 1,
                "pdf": pdf_path,
                "embedding": text_embedding
            }
            text_collection.insert_one(text_data)

            # Extract images from the page
            images = page.get_images(full=True)
            for image_index, image_info in enumerate(images):
                try:
                    xref = image_info[0]
                    image = doc.extract_image(xref)
                    img_data = image["image"]
                    pil_image = Image.open(io.BytesIO(img_data))

                    # Describe the image and create embedding for the description
                    try:
                        description, public_url = describe_image(pil_image, pdf_name, page_number + 1, image_index)
                    except Exception as e:
                        print(f"Error describing image on page {page_number + 1}, image index {image_index}: {str(e)}")
                        continue

                    # Embed the description text
                    try:
                        image_embedding = get_text_embedding(description)
                    except Exception as e:
                        print(f"Error embedding description on page {page_number + 1}, image index {image_index}: {str(e)}")
                        continue

                    # Store the image metadata and embedding in MongoDB
                    image_data = {
                        "image-path": public_url,
                        "description": description,
                        "image_index": image_index,
                        "ObjectID": text_data["_id"],
                        "embedding": image_embedding
                    }
                    image_collection.insert_one(image_data)
                except Exception as e:
                    print(f"Error processing image on page {page_number + 1}, image index {image_index}: {str(e)}")
                    continue
            print(f"Finished processing page {page_number + 1}")

        except Exception as e:
            print(f"Error processing page {page_number + 1}: {str(e)}")

    print("pdf processed")

# pdf_embeddings_module/monitor.py
class DirectoryEventHandler(FileSystemEventHandler):
    """Event handler for monitoring directory changes."""
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.pdf'):
            print(f"New PDF detected: {event.src_path}")
            process_pdf(event.src_path)

def monitor_directory(directory_path):
    """Monitor the directory for new PDF files."""
    event_handler = DirectoryEventHandler()
    observer = Observer()
    observer.schedule(event_handler, directory_path, recursive=False)
    observer.start()
    print(f"Monitoring directory: {directory_path}")

    try:
        while True:
            time.sleep(1)  # Keep the script running
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# main.py

if __name__ == "__main__":
    # Specify the directory to monitor
    directory_to_watch = "./pdfs"

    # Ensure the directory exists
    if not os.path.exists(directory_to_watch):
        os.makedirs(directory_to_watch)

    monitor_directory(directory_to_watch)