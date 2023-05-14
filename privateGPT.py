from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS
from langchain import PromptTemplate

embeddings = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese")


# embeddings = OpenAIEmbeddings()

def main():
    prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
    如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
    已知内容:
    {context}
    问题:
    {question}"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": prompt}

    # 加载持久化DB数据
    db = Chroma(persist_directory="./db", embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.8), chain_type="stuff",
                                     retriever=retriever,
                                     chain_type_kwargs=chain_type_kwargs,
                                     return_source_documents=False)
    qa.combine_documents_chain.document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}"
    )

    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        # Get the answer from the chain
        res = qa(query)
        answer, docs = res['result'], res.get('source_documents', [])

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)


if __name__ == "__main__":
    main()
