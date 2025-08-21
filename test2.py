from langchain_community.retrievers import TFIDFRetriever


retriever = TFIDFRetriever.load_local("tf_idf",allow_dangerous_deserialization=True)

results = retriever.invoke("what would be the cost of green hydrogen to charge to consumer")

print(">> Results:", results)