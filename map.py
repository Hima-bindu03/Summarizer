# map.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains.combine_documents.reduce import (
    split_list_of_docs,
    acollapse_docs,
)

load_dotenv()

llm = ChatGroq(model="llama3-70b-8192", temperature=0)

# Prompts
map_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a concise summary of the following:\n\n{context}")
])

reduce_prompt = ChatPromptTemplate.from_messages([
    ("human", "The following is a set of summaries:\n{docs}\nTake these and distill it into a final, consolidated summary of the main themes.")
])

# Token length calculator
def length_function(documents):
    # Roughly 1 token ~ 4 characters
    return sum(len(doc.page_content) // 4 for doc in documents)

# Async reduction
async def _reduce(input: dict) -> str:
    prompt = reduce_prompt.invoke(input)
    response = await llm.ainvoke(prompt)
    return response.content

# Async map-reduce logic
async def map_reduce_summary(url: str) -> str:
    loader = WebBaseLoader(url)

    # ✅ Set custom User-Agent header
    loader.requests_kwargs = {
        "headers": {
            "User-Agent": os.getenv("USER_AGENT", "Mozilla/5.0 (compatible; CustomSummarizer/1.0)")
        }
    }

    docs = loader.load()
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,    # ~500 tokens
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", "!", "?", ",", " "]) 

    split_docs = text_splitter.split_documents(docs)

    # Step 1: Map
    summaries = []
    for chunk in split_docs:
        prompt = map_prompt.invoke(chunk.page_content)
        response = await llm.ainvoke(prompt)
        summaries.append(Document(page_content=response.content))
   
    # Step 2: Reduce
    doc_lists = split_list_of_docs(summaries, length_function, 1000)
    reduced_docs = []
    for doc_list in doc_lists:
        reduced_docs.append(await acollapse_docs(doc_list, _reduce))

    # Final Reduction
    final_summary = await _reduce({"docs": [doc.page_content for doc in reduced_docs]})
    return final_summary

# Wrapper for sync use
def summarize_with_map(url: str) -> str:
    import asyncio
    return asyncio.run(map_reduce_summary(url))


# For testing
if __name__ == "__main__":
    test_url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    summary = summarize_with_map(test_url)
    print(summary)
