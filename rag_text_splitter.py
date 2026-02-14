import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_text_documents(
        documents: list[Document],
        chunk_size: int = 500,
        chunk_overlap: int = 50
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    split_docs = []

    for doc in documents:
        # 1. Generate & Attach ID to the ORIGINAL document object
        # This modification persists outside the function!
        parent_id = str(uuid.uuid4())
        doc.metadata["parent_id"] = parent_id

        content = doc.page_content.strip()

        # Handle short docs
        if len(content) <= chunk_size:
            metadata = dict(doc.metadata)
            metadata["chunk_index"] = 0
            split_docs.append(Document(page_content=content, metadata=metadata))
            continue

        # Handle long docs
        chunks = splitter.split_text(content)
        for idx, chunk in enumerate(chunks):
            metadata = dict(doc.metadata)
            metadata["chunk_index"] = idx
            # The child inherits the parent_id we just added
            metadata["parent_id"] = parent_id

            split_docs.append(
                Document(
                    page_content=chunk.strip(),
                    metadata=metadata
                )
            )

    print(f"✂️ Text docs split: {len(documents)} → {len(split_docs)}")
    return split_docs


# The same logic applies to split_code_documents
def split_code_documents(
        documents: list[Document],
        max_length: int = 800
) -> list[Document]:
    split_docs = []

    for doc in documents:
        # 1. Attach ID to Original
        parent_id = str(uuid.uuid4())
        doc.metadata["parent_id"] = parent_id

        content = doc.page_content.strip()

        if len(content) <= max_length:
            metadata = dict(doc.metadata)
            metadata["chunk_index"] = 0
            split_docs.append(Document(page_content=content, metadata=metadata))
            continue

        lines = content.splitlines()
        buffer, chunk_index = [], 0

        for line in lines:
            buffer.append(line)
            if len("\n".join(buffer)) > max_length:
                metadata = dict(doc.metadata)
                metadata["chunk_index"] = chunk_index
                # Child inherits ID
                metadata["parent_id"] = parent_id

                split_docs.append(Document(page_content="\n".join(buffer).strip(), metadata=metadata))
                buffer, chunk_index = [], chunk_index + 1

        if buffer:
            metadata = dict(doc.metadata)
            metadata["chunk_index"] = chunk_index
            metadata["parent_id"] = parent_id
            split_docs.append(Document(page_content="\n".join(buffer).strip(), metadata=metadata))

    print(f"✂️ Code docs split: {len(documents)} → {len(split_docs)}")
    return split_docs