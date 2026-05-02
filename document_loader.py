import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_document(file_name="sample_report.txt"):
    """
    Loads a text document and splits it into manageable chunks for processing.
    """
    # Dynamically find the absolute path of the file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'data', file_name)
    
    print(f"📄 Reading file: {file_path}")
    

    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,       # Maximum 300 characters per chunk
        chunk_overlap=50      # 50 characters overlap to maintain semantic context
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"✂️ Document successfully split into {len(chunks)} chunks.\n")
    return chunks

# Test block if the file is executed directly
if __name__ == "__main__":
    test_chunks = load_and_split_document()
    
    for i, chunk in enumerate(test_chunks):
        print(f"--- Chunk {i+1} ---")
        print(chunk.page_content)
        print("-" * 20)