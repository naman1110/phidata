from typing import Optional

from phi.assistant import Assistant
from phi.knowledge import AssistantKnowledge
from phi.llm.groq import Groq
from phi.embedder.openai import OpenAIEmbedder
from phi.embedder.ollama import OllamaEmbedder
from phi.vectordb.pgvector import PgVector2
from phi.storage.assistant.postgres import PgAssistantStorage

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"


def get_groq_assistant(
    llm_model: str = "llama3-70b-8192",
    embeddings_model: str = "text-embedding-3-large",
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Assistant:
    """Get a Groq RAG Assistant."""

    # Define the embedder based on the embeddings model
    embedder = (
        OllamaEmbedder(model=embeddings_model, dimensions=768)
        if embeddings_model == "nomic-embed-text"
        else OpenAIEmbedder(model=embeddings_model, dimensions=1536)
    )
    # Define the embeddings table based on the embeddings model
    # extra check for collection of tables/embedding models
    groq_rag_documents_openai= "groq_rag_documents_openai"

    if user_id is not None:
        if user_id !='':
            groq_rag_documents_openai = groq_rag_documents_openai+"_"+user_id
    
    print("before embedding model init groq_rag_documents_openai ===== ",groq_rag_documents_openai," >> for user_id :",user_id)
    embeddings_table = (
        "groq_rag_documents_ollama" if embeddings_model == "nomic-embed-text" else groq_rag_documents_openai
    )

    return Assistant(
        name="groq_rag_assistant",
        run_id=run_id,
        user_id=user_id,
        llm=Groq(model=llm_model),
        storage=PgAssistantStorage(table_name="groq_rag_assistant", db_url=db_url),
        knowledge_base=AssistantKnowledge(
            vector_db=PgVector2(
                db_url=db_url,
                collection=embeddings_table,
                embedder=embedder,
            ),
            # 2 references are added to the prompt
            num_documents=2,
        ),
        description="You are an AI called 'Trainer Assistant' and your task is to answer questions using the provided information",
        instructions=[
            "When a user asks a question, you will be provided with information about the question.",
            "Carefully read this information and provide a clear and concise answer to the user.",
            "Do not use phrases like 'based on my knowledge' or 'depending on the information'.",
            "It is mandatory to add references from where or which document or knowledge base document the response was generated.",
            # "If no user prompt is asked then respond saying how can i help in various styles with the list of documents based on knowledge base documents for the current user_id. this is mandatory condition.",
        ],
        # This setting adds references from the knowledge_base to the user prompt
        add_references_to_prompt=True,
        # This setting tells the LLM to format messages in markdown
        markdown=True,
        # This setting adds chat history to the messages
        add_chat_history_to_messages=True,
        # This setting adds 4 previous messages from chat history to the messages
        num_history_messages=4,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )
