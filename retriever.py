import os
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Import custom modules
from document_loader import load_and_split_document
from graph_builder import extract_triplets

load_dotenv()

def build_networkx_graph(triplets):
    """Builds a NetworkX directed graph in memory from a list of triplets."""
    G = nx.DiGraph()
    
    for t in triplets:
        source = t.get("source")
        target = t.get("target")
        relation = t.get("relation")
        
        if source and target and relation:
            # Add nodes and the directed edge (relationship) between them
            G.add_edge(source, target, label=relation)
            
    print(f"🕸️ Knowledge Graph built: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges.")
    return G

def visualize_and_save_graph(G, filename="knowledge_graph.png"):
    """Visualizes the NetworkX graph and saves it as a PNG image."""
    print(f"🎨 Generating graph visualization...")
    
    # Set the size of the image
    plt.figure(figsize=(12, 8))
    
    # Calculate positions for the nodes using a spring layout algorithm
    # k regulates the distance between nodes, iterations makes it look better
    pos = nx.spring_layout(G, k=0.8, iterations=50)
    
    # 1. Draw the nodes (circles)
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', edgecolors='gray')
    
    # 2. Draw the edges (arrows)
    nx.draw_networkx_edges(G, pos, width=2, edge_color='gray', arrowsize=20)
    
    # 3. Draw the text inside nodes
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_family="sans-serif")
    
    # 4. Draw the relationship text on the edges
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_color='red')
    
    # Finalize and save the plot
    plt.title("Neuro-Symbolic Knowledge Graph", fontsize=16, fontweight="bold")
    plt.axis('off') # Hide the grid axis
    plt.tight_layout()
    
    # Save as image
    plt.savefig(filename, dpi=300) # dpi=300 for high resolution
    print(f"🖼️ Graph successfully saved as '{filename}' in your project folder.")
    
    # Optional: Display the graph in a popup window
    plt.show()

def get_graph_context(G, entity_name):
    """Retrieves the immediate neighborhood of a target entity from the graph to build context."""
    context = []
    

    found_node = None
    for node in G.nodes():
        if entity_name.lower() in node.lower():
            found_node = node
            break
            
    if found_node:
        for neighbor in G.successors(found_node):
            relation = G.edges[found_node, neighbor]['label']
            context.append(f"{found_node} --({relation})--> {neighbor}")
            
        for neighbor in G.predecessors(found_node):
            relation = G.edges[neighbor, found_node]['label']
            context.append(f"{neighbor} --({relation})--> {found_node}")
            
    return "\n".join(context)

def answer_question_with_graph(question, target_entity, G):
    """Generates an answer using only the extracted graph context."""
    graph_context = get_graph_context(G, target_entity)
    
    if not graph_context:
        return f"Unfortunately, no information regarding '{target_entity}' was found in the graph."
        
    print(f"\n🔍 Context retrieved from Graph (for {target_entity}):\n{graph_context}\n")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    prompt = PromptTemplate.from_template(
        """You are a knowledge graph analyst. Answer the user's question using ONLY the provided Neuro-Symbolic Graph context below.
Do not use any external knowledge. If the answer is not present in the graph context, state that you do not know.

Graph Context:
{context}

Question: {question}

Answer:"""
    )
    
    formatted_prompt = prompt.format(context=graph_context, question=question)
    response = llm.invoke(formatted_prompt)
    
    return response.content

# Main Execution Pipeline
if __name__ == "__main__":
    print("🚀 Initializing Neuro-Symbolic Graph RAG Pipeline...\n")
    
    chunks = load_and_split_document()
    
    if chunks:
        triplets = extract_triplets(chunks)
        G = build_networkx_graph(triplets)
        
        # --- NEW VISUALIZATION STEP ---
        visualize_and_save_graph(G)
        # ------------------------------
        
        question = "What is RWE doing to close the skills gap and transform its human capital?"
        target_entity = "RWE" 
        
        print(f"\n❓ Question: {question}")
        answer = answer_question_with_graph(question, target_entity, G)
        
        print("\n💡 Graph RAG Output:")
        print("-" * 50)
        print(answer)
        print("-" * 50)