# 数据分块，向量化，持久化存储,支持多种文件格式
import os
import argparse

from langchain.text_splitter import CharacterTextSplitter, MarkdownTextSplitter
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader, UnstructuredMarkdownLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from constants import CHROMA_SETTINGS


def main(source_documents):
    print("source_documents:", source_documents)
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese")
    documents = []
    for root, dirs, files in os.walk(source_documents):
        for file in files:
            if file.endswith(".txt"):
                loader = TextLoader(os.path.join(root, file), encoding="utf8")
            elif file.endswith(".pdf"):
                loader = PDFMinerLoader(os.path.join(root, file))
            elif file.endswith(".csv"):
                loader = CSVLoader(os.path.join(root, file))
            elif file.endswith(".md"):
                loader = UnstructuredMarkdownLoader(os.path.join(root, file), mode="elements")
            else:
                print(f"Unsupported file type: {file}")
                continue
        # 导入文本
        documents.extend(loader.load())

    # 初始化文本分割器
    text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=0)
    # 切分文本
    texts = text_splitter.split_documents(documents)
    # Create and store locally vectorstore
    db = Chroma.from_documents(texts, embeddings, persist_directory="./db", client_settings=CHROMA_SETTINGS)

    db.persist()
    db = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='支持参数')
    parser.add_argument('--source', type=str, required=True, help='数据源路径', default="./source_documents")

    args = parser.parse_args()
    main(args.source)
