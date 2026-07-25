from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_text(text):
    """
    Split text into LangChain Document chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )

    chunks = splitter.split_text(text)

    documents = [
        Document(page_content=chunk)
        for chunk in chunks
    ]

    return documents