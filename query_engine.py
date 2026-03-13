import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from retrieval import RelationalRetriever

load_dotenv()


class NotebookChatbot:
    def __init__(self):
        """
        The Setup.
        1. Instantiate your RelationalRetriever here (Composition).
        2. Initialize your ChatGroq LLM instance.
        3. Create an empty list for self.chat_history to manage state.
        """
        print("🤖 Booting up Data Science Assistant (Llama 3 70B)...")
        self.retriever = RelationalRetriever()
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=2048
        )
        self.chat_history = []

    def _build_system_prompt(self) -> str:
        """
        The Rulebook (Private Helper).
        Returns a hardcoded string instructing the LLM that it is a Data Scientist.
        It MUST strictly tell the LLM to only use the provided context and
        to say "I don't know" if the answer isn't in the notebook chunks.
        """
        return """
            You are an expert Data Scientist and a highly analytical AI assistant. 
            Your job is to answer the user's questions based strictly on the provided Jupyter Notebook context.

            You will receive context blocks containing:
            - [INTENT / PURPOSE]: What the code was trying to do.
            - [EXPLANATION]: Why it was done.
            - [PYTHON CODE]: The exact code executed.
            - [TERMINAL OUTPUT] & [STATISTICAL RESULT]: The real-world results of that code.

            STRICT RULES:
            1. Ground your answers ENTIRELY in the provided context. 
            2. If the user asks for code, provide the exact snippet from the [PYTHON CODE] section. Do not invent new code unless explicitly asked to modify it.
            3. If the user asks about metrics, accuracy, or outputs, quote the exact numbers from the [STATISTICAL RESULT] or [TERMINAL OUTPUT].
            4. If the provided context does not contain the answer to the user's question, politely say: "I cannot find the answer to that in the retrieved notebook cells." DO NOT guess.
            5. Be concise, professional, and directly answer the question.
            """

    def _format_memory(self) -> str:
        """
        The Memory Manager (Private Helper).
        Loops through self.chat_history and formats it into a readable string
        (e.g., "User: ... \n Assistant: ...") so the LLM remembers the last few questions.
        """

    def ask(self, user_query: str) -> str:
        """
        The Main Orchestrator (Public Method).
        This is the only function the user actually calls.

        Execution Flow to code here:
        1. Call self.retriever.retrieve(user_query)
        2. Guardrail: If no chunks returned, return a fallback message.
        3. Format the retrieved chunks using self.retriever.format_for_llm()
        4. Grab the system prompt and the formatted memory.
        5. Combine everything into one massive HumanMessage/Prompt.
        6. Invoke self.llm.
        7. Append the user_query and the LLM's answer to self.chat_history.
        8. Return the LLM's text response.
        """
        from langchain_core.messages import SystemMessage, HumanMessage

        # 1. The Bait and Switch (Fetch the context!)
        results = self.retriever.retrieve(user_query, max_cells=3)

        # 2. The Guardrail (If ChromaDB finds absolutely nothing)
        if not results:
            fallback = "I couldn't find any relevant code or data in the notebook to answer that."
            self.chat_history.append(("User", user_query))
            self.chat_history.append(("Assistant", fallback))
            return fallback

        # 3. Format the retrieved context and gather our prompt pieces
        context = self.retriever.format_for_llm(results)
        system_prompt = self._build_system_prompt()
        memory = self._format_memory()

        # 4. Construct the ultimate Human Prompt
        human_content = f"""
            {memory}

            --- NEW USER QUESTION ---
            {user_query}

            --- RETRIEVED NOTEBOOK CONTEXT ---
            {context}
            """

        # 5. Invoke Llama 3 70B via Groq
        print("\n🧠 Llama 3 is analyzing your notebook...")
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content)
        ])

        answer = response.content

        # 6. Save this interaction to memory for the next follow-up question
        self.chat_history.append(("User", user_query))
        self.chat_history.append(("Assistant", answer))

        return answer

    def clear_history(self):
        """
        The Reset Button (Optional but recommended).
        Simply clears out self.chat_history so the user can start a fresh topic
        without the LLM getting confused by old context.
        """
        self.chat_history = []
        print("\n🧹 Chat history has been cleared! Starting a fresh conversation.")


if __name__ == "__main__":
    bot = NotebookChatbot()
    print("\n" + "=" * 50)
    print("✅ Data Science Assistant Ready! (Type 'exit' to stop)")
    print("=" * 50)

    while True:
        query = input("\n🧑‍💻 You: ")
        if query.lower() in ['exit', 'quit']:
            print("Goodbye! 👋")
            break

        answer = bot.ask(query)
        print(f"\n🤖 Assistant:\n{answer}")