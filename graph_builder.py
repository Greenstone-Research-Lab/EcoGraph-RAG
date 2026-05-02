import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Import the loader function from the previous module
from document_loader import load_and_split_document

# Load environment variables
load_dotenv()

def extract_triplets(chunks):
    """
    Uses an LLM to extract knowledge graph triplets from text chunks.
    """
    # Force the LLM to return a JSON object
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", 
        temperature=0, 
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    
    # System Prompt: Engineered for Entity-Relationship extraction in English
    prompt_template = PromptTemplate.from_template(
        """You are an expert AI analyzing corporate social responsibility (CSR) and human resources data.
Read the following text chunk and extract entities (companies, projects, technologies, skills, goals).
Identify the relationships between these entities. 
Ensure all extracted entities and relations are in English.

Please respond ONLY in the following JSON format, without any additional text or markdown formatting:
{{
  "triplets": [
    {{"source": "Entity 1", "relation": "RELATION_TYPE", "target": "Entity 2"}},
    {{"source": "RWE", "relation": "TARGETS", "target": "Net-zero emissions"}}
  ]
}}

Text to analyze:
{text}
"""
    )
    
    all_triplets = []
    
    print("🧠 LLM is analyzing text and extracting knowledge graph triplets, please wait...\n")
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}...")
        
        formatted_prompt = prompt_template.format(text=chunk.page_content)
        
        try:
            response = llm.invoke(formatted_prompt)
            result_dict = json.loads(response.content)
            
            if "triplets" in result_dict:
                all_triplets.extend(result_dict["triplets"])
                
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            
    return all_triplets

# Test block
if __name__ == "__main__":
    chunks = load_and_split_document()
    
    if chunks:
        extracted_triplets = extract_triplets(chunks)
        
        print("\n✅ EXTRACTION COMPLETE! Knowledge Graph Triplets:")
        print("-" * 50)
        for t in extracted_triplets:
            print(f"[{t.get('source')}] --({t.get('relation')})--> [{t.get('target')}]")
        print("-" * 50)