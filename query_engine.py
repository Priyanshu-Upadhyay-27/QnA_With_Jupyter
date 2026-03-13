import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from retrieval import RelationalRetriever

load_dotenv()


class NotebookChatbot:
