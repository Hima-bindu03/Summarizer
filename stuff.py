from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

llm = ChatGroq(model="llama3-70b-8192", temperature=0)

def summarize_with_stuff(url: str) -> str:
    loader = WebBaseLoader(url)
    loader.requests_kwargs = {
        "headers": {
            "User-Agent": os.getenv("USER_AGENT", "Mozilla/5.0 (compatible; CustomSummarizer/1.0)")
        }
    }
    docs = loader.load()

    # Split docs to chunks small enough for model limits
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=3000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Write a concise summary of the following:\n\n{context}")
    ])
    chain = create_stuff_documents_chain(llm, prompt)

    # Only send the first chunk to stay within token limits
    result = chain.invoke({"context": split_docs[0:1]})
    return result


if __name__ == "__main__":
    test_url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    summary = summarize_with_stuff(test_url)
    print(summary)
