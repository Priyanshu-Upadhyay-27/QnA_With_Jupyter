Have you ever worked on a large-scale dataset, performed Exploratory Data Analysis on a dataset of tremendous amount and very large dimensions. If yes, then you know, how hard it becomes for you to revisit the notebook and study each cell and understands why that code exist or if someone shared you a notebook, then it will be monotonous to ask every time from the owner, why this is written.
Hey, this is jupyter parser, an ipynb file parser which traverse the code inside your notebook and understands what’s happening within a particular context. What you can do is to ask questions related to the code present in the notebook. While it may not be 100% accurate, but it can assist millions of data scientist and analyst who struggles with large jupyter notebook. 
It implements Retrieval Augmented Generation, which retrieves the code similar to the user query and LLM answers the user query with respect to the retrieved code.

From where this idea comes: I was working on lending club data and it was difficult for me to come again and see each and every cell. So, I



**Retrieval Logic**:
Relational RAG Retrieval is a two-stage information retrieval process designed to maintain semantic coherence in fragmented documents (like Jupyter Notebooks). It overcomes the limitations of standard chunk-based retrieval by treating data as interconnected entities rather than isolated text segments.

The logic follows a precise "Anchor-and-Expand" execution flow:

Vector Anchor Search (Stage 1):
The system queries a Vector Database to identify the most semantically relevant child chunk (a small, searchable text fragment) based on the user's query. This chunk serves as the "Anchor."

Parent Resolution (Stage 2 - The Vertical Hop):
Using the unique parent_id embedded in the Anchor's metadata, the system retrieves the full, original document object (the complete code cell or text paragraph) from a secondary Key-Value Store (DocStore). This ensures the context is complete and not fragmented.

Sibling Expansion (Stage 3 - The Horizontal Hop):
Using the shared cell_id from the Parent document, the system queries the DocStore for all "Sibling" documents—specifically, the corresponding Explanation document linked to a Code document (or vice versa).