# LangChain Local RAG
Simple RAG app to compare the result between naive document retrieval and parent document retrieval and the effect of re-ranker.

This project processes documents (PDF or text), stores them in a vector database, and retrieves relevant documents based on user queries. It compares both naive retrieval, parent document retrieval and re-ranker to improve the relevance of the results.

## Features
* Document Processing: Load and parse PDF or text documents.
* Vector Store: Store document chunks in a vector database for efficient retrieval.
* Naive Retrieval: Retrieve relevant documents using a naive approach.
* Re-Ranker Retrieval: Improve retrieval relevance using a re-ranker.
* Parent Document Retrieval: Retrieve parent documents based on child chunks.

## Requirements
Python 3.12+
Required Python packages (see requirements.txt)
