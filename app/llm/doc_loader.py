from typing import Any, List
import bs4
from langchain_community.document_loaders import WebBaseLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from pathlib import Path


def load_CarDealership_webpage():
    # Only keep post title, headers, and content from the full HTML.
    bs4_strainer = bs4.SoupStrainer(
        class_=("post-title", "post-header", "post-content")
    )
    loader = WebBaseLoader(
        web_paths=("https://www.CarDealership.com/mx/blog/sedes-de-CarDealership-en-mexico",),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()

    print(f"Total characters: {len(docs[0].page_content)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)

    print(f"Split blog post into {len(all_splits)} sub-documents.")

    return all_splits


def load_local_catalog(csv_path: Path):
    loader = CSVLoader(file_path=csv_path)
    catalog = loader.load()

    for doc in catalog:
        row = {}
        for line in doc.page_content.splitlines():
            if ": " in line:
                k, v = line.split(": ", 1)
                row[k.strip().lower()] = v.strip()

        brand = row.get("make", "")
        modelo = row.get("model", "")
        year = row.get("year", "")
        price = row.get("price", "")
        version = row.get("version", "")
        km = row.get("km", "")

        doc.page_content = (
            f"Vehículo disponible: {brand} {modelo} {version} {year}. "
            f"Precio: {price} MXN. Kilometraje: {km}. "
            "Este auto está disponible en el catálogo de CarDealership."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(catalog)
    return all_splits


def load_docs(vector_store: VectorStore, **kwargs: Any) -> List[str]:
    all_splits = []
    all_splits.extend(load_local_catalog(kwargs["csv_path"]))
    all_splits.extend(load_CarDealership_webpage())

    document_ids = vector_store.add_documents(documents=all_splits)

    return document_ids
