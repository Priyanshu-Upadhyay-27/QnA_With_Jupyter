from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from retrieval import RelationalRetriever
from dotenv import load_dotenv
load_dotenv()

print("Starting Llama3:70b...")
Chatbot = ChatGroq(model="llama-3.3-70b-versatile",
                   temperature=0.1)

print("Reading your jupyter notebook...")
retriever = RelationalRetriever()

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Data Scientist and AI Assistant analyzing a Jupyter Notebook.
    Use the provided context (which contains actual code cells and their explanations) to answer the user's question.

    CRITICAL RULES:
    1. Base your answer strictly on the provided context. 
    2. If the answer is not in the context, say "I don't see that in the notebook's code."
    3. Always reference specific variable names, datasets, or algorithms if they appear in the code.

    NOTEBOOK CONTEXT:
    {context}"""),
    ("human", "{question}")
])


print("Starting the chatbot, ask whatever you want to ask regarding the jupyter notebook uploaded")
print("Type 'q' for exiting the chatbot")
while(True):
    user_input = input("😀 You:")

    if user_input.lower() == "q":
        break
    if not user_input.strip():
        continue

    context = retriever.retrieve(user_input, 3)

    formatted_context = retriever.format_for_llm(context)

    prompt = prompt_template.invoke({
        "context": formatted_context,
        "question": user_input
    })

    results = Chatbot.invoke(prompt)

    print(f"🤖 AI: {results.content}\n")
    print("------------------------result ended-------------------------")

print("Have a Nice Day 😎")





