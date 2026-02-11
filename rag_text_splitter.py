from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_text_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[Document]:
    """
    Split text RAG documents (WHAT / WHY).
    Uses RecursiveCharacterTextSplitter.
    Preserves metadata and adds chunk_index.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",   # paragraph boundary
            "\n",     # line boundary
            ". ",     # sentence boundary
            " ",      # word boundary
            ""        # fallback
        ]
    )

    split_docs = []

    for doc in documents:
        content = doc.page_content.strip()

        # Do not split short docs
        if len(content) <= chunk_size:
            split_docs.append(doc)
            continue

        chunks = splitter.split_text(content)

        for idx, chunk in enumerate(chunks):
            metadata = dict(doc.metadata)
            metadata["chunk_index"] = idx

            split_docs.append(
                Document(
                    page_content=chunk.strip(),
                    metadata=metadata
                )
            )

    print(f"✂️ Text docs split: {len(documents)} → {len(split_docs)}")
    return split_docs
