from langchain_core.tools import tool
from typing import Annotated
import os

import chromadb

from google import genai
from google.genai import types
from google.api_core import retry
from chromadb import Documents, EmbeddingFunction, Embeddings

# Define the embedding function
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key, document_mode=True):
        self.client = genai.Client(api_key=api_key)
        self.document_mode = document_mode

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        embedding_task = "retrieval_document" if self.document_mode else "retrieval_query"
        response = self.client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(task_type=embedding_task),
        )
        return [e.values for e in response.embeddings]

def process_python_files(directory):
    """
    Reads all Python files in a directory, adds a title based on the filename,
    and combines their content into a single string for LLM parsing.
    """
    combined_content = ""
    for file_name in os.listdir(directory):
        if file_name.endswith(".py") or file_name.endswith(".md"):
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
            title = f"### {file_name[:-3].title()} ###\n"
            combined_content += title + file_content + "\n\n"
    return combined_content

@tool
def code_analysis_rag(
    directory_path: Annotated[str, "Path to the directory containing code files"],
    query: Annotated[str, "Query about the codebase to analyze"]
) -> str:
    """
    Analyzes code in a directory using RAG (Retrieval-Augmented Generation) to answer queries
    about code quality, structure, and potential improvements.
    """
    try:
        # Initialize the Gemini client
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        
        # Process all Python files in the directory
        combined_content = process_python_files(directory_path)
        
        # Create a ChromaDB collection with Gemini embeddings
        embed_fn = GeminiEmbeddingFunction(api_key=os.environ["GOOGLE_API_KEY"])
        chroma_client = chromadb.Client()
        db_name = f"code_analysis_{os.path.basename(directory_path)}"
        db = chroma_client.get_or_create_collection(name=db_name, embedding_function=embed_fn)
        
        # Add the document to the collection
        db.add(documents=[combined_content], ids=["0"])
        
        # Set embedding function to query mode
        embed_fn.document_mode = False
        
        # Query the database
        result = db.query(query_texts=[query], n_results=1)
        [all_passages] = result["documents"]
        
        # Generate response using Gemini
        answer = client.models.generate_content(
            model="gemini-2.5-pro-exp-03-25",
            contents=all_passages + [query]
        )
        
        # Save the output to a markdown file
        output_file = f"code_analysis_{os.path.basename(directory_path)}.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(answer.text)
        
        return answer.text
    
    except Exception as e:
        return f"Error analyzing code: {str(e)}"
