from langchain_text_splitters import RecursiveCharacterTextSplitter
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

def split_code_documents(
    documents: list[Document],
    max_length: int = 800
) -> list[Document]:
    """
    Custom splitter for notebook code cells.
    Splits only very long cells.
    Splits by lines to preserve execution logic.
    """

    split_docs = []

    for doc in documents:
        content = doc.page_content.strip()

        # Keep atomic if small
        if len(content) <= max_length:
            split_docs.append(doc)
            continue

        lines = content.splitlines()
        buffer = []
        chunk_index = 0

        for line in lines:
            buffer.append(line)

            if len("\n".join(buffer)) > max_length:
                metadata = dict(doc.metadata)
                metadata["chunk_index"] = chunk_index

                split_docs.append(
                    Document(
                        page_content="\n".join(buffer).strip(),
                        metadata=metadata
                    )
                )

                buffer = []
                chunk_index += 1

        # Add remainder
        if buffer:
            metadata = dict(doc.metadata)
            metadata["chunk_index"] = chunk_index

            split_docs.append(
                Document(
                    page_content="\n".join(buffer).strip(),
                    metadata=metadata
                )
            )

    print(f"✂️ Code docs split: {len(documents)} → {len(split_docs)}")
    return split_docs

