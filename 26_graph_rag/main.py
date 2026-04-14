"""
Graph RAG (Knowledge Graphs)
=============================
Combine vector retrieval with a knowledge graph:
1. Extract entities and relationships from documents
2. Build an in-memory knowledge graph
3. Retrieve via both vector similarity and graph traversal
4. Merge results for richer context
"""

import os
import sys
from collections import defaultdict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field


def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)


# ---------- Sample documents ----------

DOCUMENTS = [
    Document(
        page_content=(
            "LangChain is an open-source framework created by Harrison Chase. "
            "It integrates with OpenAI, Anthropic, and Google for LLM providers. "
            "LangChain supports retrieval-augmented generation through vector stores like ChromaDB and Pinecone."
        ),
        metadata={"source": "langchain"},
    ),
    Document(
        page_content=(
            "LangGraph is a library built on top of LangChain for creating stateful, "
            "graph-based agent workflows. It was also created by the LangChain team. "
            "LangGraph supports checkpointing, human-in-the-loop, and multi-agent patterns."
        ),
        metadata={"source": "langgraph"},
    ),
    Document(
        page_content=(
            "ChromaDB is an open-source vector database. It stores embeddings alongside "
            "metadata and documents. ChromaDB integrates with LangChain and LlamaIndex. "
            "It can run in-memory or as a persistent server."
        ),
        metadata={"source": "chromadb"},
    ),
    Document(
        page_content=(
            "Pinecone is a managed vector database service. It provides high-performance "
            "similarity search at scale. Pinecone integrates with LangChain, LlamaIndex, "
            "and Haystack. It offers serverless and pod-based deployment options."
        ),
        metadata={"source": "pinecone"},
    ),
]


# ---------- Entity/relationship extraction ----------

class Entity(BaseModel):
    name: str = Field(description="Entity name")
    entity_type: str = Field(description="Type: tool, person, company, or concept")


class Relationship(BaseModel):
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relation: str = Field(description="Relationship type (e.g., created_by, integrates_with)")


class ExtractionResult(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)


class SimpleKnowledgeGraph:
    """In-memory knowledge graph using adjacency lists."""

    def __init__(self):
        self.entities: dict[str, str] = {}  # name -> type
        self.edges: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)  # source -> [(target, relation)]
        self.reverse_edges: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)

    def add_entity(self, name: str, entity_type: str):
        self.entities[name.lower()] = entity_type

    def add_relationship(self, source: str, target: str, relation: str):
        self.edges[source.lower()].append((target.lower(), relation))
        self.reverse_edges[target.lower()].append((source.lower(), relation))

    def get_neighbors(self, entity: str, max_hops: int = 2) -> list[str]:
        """Get all entities within max_hops of the given entity."""
        visited = set()
        queue = [(entity.lower(), 0)]
        results = []

        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_hops:
                continue
            visited.add(current)
            if current != entity.lower():
                results.append(current)

            for target, relation in self.edges.get(current, []):
                if target not in visited:
                    queue.append((target, depth + 1))
            for source, relation in self.reverse_edges.get(current, []):
                if source not in visited:
                    queue.append((source, depth + 1))

        return results

    def describe_entity(self, entity: str) -> str:
        """Get a text description of an entity and its relationships."""
        key = entity.lower()
        if key not in self.entities:
            return f"No graph information found for '{entity}'."

        lines = [f"{entity} (type: {self.entities[key]})"]
        for target, relation in self.edges.get(key, []):
            lines.append(f"  -> {relation} -> {target}")
        for source, relation in self.reverse_edges.get(key, []):
            lines.append(f"  <- {relation} <- {source}")
        return "\n".join(lines)

    def __repr__(self):
        return f"KnowledgeGraph(entities={len(self.entities)}, edges={sum(len(v) for v in self.edges.values())})"


def extract_and_build_graph(documents: list[Document]) -> SimpleKnowledgeGraph:
    """Extract entities and relationships from documents, build a graph."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(ExtractionResult)

    prompt = ChatPromptTemplate.from_template(
        "Extract entities and relationships from this text.\n"
        "Entity types: tool, person, company, concept\n"
        "Relationship types: created_by, integrates_with, built_on, supports, part_of\n\n"
        "Text: {text}\n"
    )

    graph = SimpleKnowledgeGraph()

    for doc in documents:
        result = structured_llm.invoke(prompt.format(text=doc.page_content))
        for entity in result.entities:
            graph.add_entity(entity.name, entity.entity_type)
        for rel in result.relationships:
            graph.add_relationship(rel.source, rel.target, rel.relation)

    return graph


# ---------- Graph-enhanced RAG ----------

def build_graph_rag_chain(documents: list[Document], graph: SimpleKnowledgeGraph):
    """Build a RAG chain that combines vector retrieval with graph context."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        "Answer the question using both the retrieved documents and knowledge graph context.\n\n"
        "Retrieved documents:\n{vector_context}\n\n"
        "Knowledge graph context:\n{graph_context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )

    def get_combined_context(question: str) -> dict:
        # Vector retrieval
        docs = retriever.invoke(question)
        vector_context = "\n\n".join(doc.page_content for doc in docs)

        # Graph retrieval — find entities mentioned in the question
        graph_parts = []
        for entity_name in graph.entities:
            if entity_name in question.lower():
                graph_parts.append(graph.describe_entity(entity_name))
                for neighbor in graph.get_neighbors(entity_name, max_hops=1):
                    graph_parts.append(graph.describe_entity(neighbor))

        graph_context = "\n".join(graph_parts) if graph_parts else "No graph matches found."

        return {
            "vector_context": vector_context,
            "graph_context": graph_context,
            "question": question,
        }

    chain = get_combined_context | prompt | llm | StrOutputParser()
    return chain, vectorstore


def main():
    check_api_key()

    # Step 1: Extract entities and build knowledge graph
    print("Extracting entities and relationships...")
    graph = extract_and_build_graph(DOCUMENTS)
    print(f"Built {graph!r}\n")

    for entity_name in sorted(graph.entities):
        print(graph.describe_entity(entity_name))
    print()

    # Step 2: Build graph-enhanced RAG chain
    print("Building Graph RAG chain...")
    chain, vectorstore = build_graph_rag_chain(DOCUMENTS, graph)

    # Step 3: Query
    questions = [
        "What does LangChain integrate with?",
        "How is LangGraph related to LangChain?",
        "What vector databases work with LangChain?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        answer = chain.invoke(q)
        print(f"A: {answer}")

    vectorstore.delete_collection()
    print("\nDone!")


if __name__ == "__main__":
    main()
