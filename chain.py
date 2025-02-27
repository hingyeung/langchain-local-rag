# https://www.youtube.com/watch?v=yF9kGESAi3M&list=WL&index=25
# https://blog.gopenai.com/implementing-a-local-rag-with-langchain-and-llama3-a-quick-guide-1c0fe37341cc
# https://www.reddit.com/r/LocalLLaMA/comments/1cqolrb/comment/l3tyeg9/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
import click
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
# Using ollama to pull and use the Llama3 embeddings (using weights directly
# or from HuggingFace involves more maintenance and model specific code snippets)
from langchain_community.embeddings import OllamaEmbeddings
# Using chroma vector store for storing the parsed chunks
from langchain_community.vectorstores import Chroma
import re
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessage, SystemMessage
from click import command, option, Option, UsageError
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.globals import set_verbose, set_debug
from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
# embedding model: mxbai-embed-large:v1
# embeddings = OllamaEmbeddings(model="mxbai-embed-large:v1")
#
# https://huggingface.co/blog/langchain#huggingfaceembeddings
# https://huggingface.co/spaces/mteb/leaderboard
# downloaded models are in ~/.cache/huggingface/
# embedding model: dunzhang/stella_en_1.5B_v5
# embeddings = HuggingFaceEmbeddings(model_name="dunzhang/stella_en_1.5B_v5")
#
# embedding model: Alibaba-NLP/gte-large-en-v1.5
# Loading Alibaba-NLP/gte-large-en-v1.5 requires you to execute the
# configuration file in that repo on your local machine. Make sure 
# you have read the code there to avoid malicious use, then set the 
# option `trust_remote_code=True` to remove this error.
embeddings = HuggingFaceEmbeddings(model_name="Alibaba-NLP/gte-large-en-v1.5", model_kwargs={"trust_remote_code": True})
PRIMARY_CHUNK_SIZE = 1000
PRIMARY_CHUNK_OVERLAP = 100
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 50

class MutuallyExclusiveOption(Option):
    def __init__(self, *args, **kwargs):
        self.mutually_exclusive = set(kwargs.pop('mutually_exclusive', []))
        help = kwargs.get('help', '')
        if self.mutually_exclusive:
            ex_str = ', '.join(self.mutually_exclusive)
            kwargs['help'] = help + (
                ' NOTE: This argument is mutually exclusive with '
                ' arguments: [' + ex_str + '].'
            )
        super(MutuallyExclusiveOption, self).__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        if self.mutually_exclusive.intersection(opts) and self.name in opts:
            raise UsageError(
                "Illegal usage: `{}` is mutually exclusive with "
                "arguments `{}`.".format(
                    self.name,
                    ', '.join(self.mutually_exclusive)
                )
            )

        return super(MutuallyExclusiveOption, self).handle_parse_result(
            ctx,
            opts,
            args
        )

class VectorStore:
    def __init__(self, store_name):
        self.store_name = store_name
        self.persistent_dir = os.path.join(db_dir, store_name)
        self.db = None

    def exists(self):
        return os.path.exists(self.persistent_dir)
    
def load_vector_store_for_parent_document_retrieval(vectordb, loader, persistent_parent_store):
    # if the store already exists, return None
    if vectordb.exists():
        print(f"Vector store {vectordb.store_name} already exists at {vectordb.persistent_dir}.")
        return None
    
    # This text splitter is used to create the parent documents
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=PRIMARY_CHUNK_SIZE, chunk_overlap=PRIMARY_CHUNK_OVERLAP)
    # This text splitter is used to create the child documents
    # It should create documents smaller than the parent
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP)
    # The vectordb to use to index the child chunks
    # The storage layer for the parent documents
    # InMemoryStore does not survive across runs, so we use a persistent store
    # store = InMemoryStore()
    store = persistent_parent_store

    # create a new vector store
    vectordb.db = Chroma(
        persist_directory=vectordb.persistent_dir,
        embedding_function=embeddings,
        # collection_metadata={"hnsw:space": "cosine"}
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectordb.db,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    retriever.add_documents(loader.load())
    print(f"docs in parent store: {len(list(store.yield_keys()))}")
    
def load_vector_store_for_naive_retrieval(vectordb, loader):
    splitter = RecursiveCharacterTextSplitter(chunk_size=PRIMARY_CHUNK_SIZE, chunk_overlap=PRIMARY_CHUNK_OVERLAP)
    docs = loader.load()
    splits = splitter.split_documents(docs)

    # if the store already exists, return None
    if vectordb.exists():
        print(f"Vector store {vectordb.store_name} already exists at {vectordb.persistent_dir}.")
        return None
    
    # create a new vector store
    vectordb.db = Chroma(
        persist_directory=vectordb.persistent_dir,
        embedding_function=embeddings,
        # collection_metadata={"hnsw:space": "cosine"}
    )
    
    # add the documents to the vector store
    print(f"\n--- Create vector store: {vectordb.store_name} ---")
    vectordb.db = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=vectordb.persistent_dir)
    print(f"--- Vector store created: {vectordb.store_name} ---")

def get_naive_retriever(vectordb, search_type="mmr", k=5, fetch_k=30, lambda_mult=0.8):
    if not vectordb.exists():
        raise ValueError(f"Vector store {vectordb.store_name} does not exist at {vectordb.persistent_dir}.")
    
    # load vector db from persistent path
    vectordb.db = Chroma(
        persist_directory=vectordb.persistent_dir,
        embedding_function=embeddings,
    )
    
    retriever = vectordb.db.as_retriever(
        ## similarity_score_threshold search type
        # search_type="similarity_score_threshold",
        # search_kwargs={"score_threshold": 0.1, "k": 2}

        ## similarity search type
        # search_type="similarity",
        # search_kwargs={"k": 3}

        ## mmr search type
        search_type="mmr",
        # If not using re-ranker, limit "k" to smaller value and rely on Chroma to
        # return the closest matches
        # If using re-ranker, get more top matching docs from Chroma and give them
        # to the re-ranker to re-rank to docs
        search_kwargs = {'k': k, 'fetch_k': fetch_k, 'lambda_mult': lambda_mult}
    )
    return retriever

def get_naive_with_reranker_retriever(vectordb):
    # use higher k and fetch_k values to get more top matching docs from Chroma
    # for re-ranker to re-rank
    base_retriever = get_naive_retriever(vectordb, k=50, fetch_k=100)
    # top_n is the number of documents to return after re-ranking
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    return compression_retriever

# https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/parent_document_retriever/#retrieving-larger-chunks
# https://github.com/damiangilgonzalez1995/AdvancedRetrievalRags/blob/main/2_parent_document_retriever.ipynb
# https://stackoverflow.com/a/77397998
def get_parent_doc_retriever(vectordb, persistent_parent_store):
    if not vectordb.exists():
        raise ValueError(f"Vector store {vectordb.store_name} does not exist at {vectordb.persistent_dir}.")
    
    # The vectordb to use to index the child chunks
    # The storage layer for the parent documents
    # InMemoryStore does not survive across runs, so we use a persistent store
    # store = InMemoryStore()
    store = persistent_parent_store
    
    # load vector db from persistent path
    vectordb.db = Chroma(
        persist_directory=vectordb.persistent_dir,
        embedding_function=embeddings,
    )
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectordb.db,
        docstore=store,
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP),
        parent_splitter=RecursiveCharacterTextSplitter(chunk_size=PRIMARY_CHUNK_SIZE, chunk_overlap=PRIMARY_CHUNK_OVERLAP),
        search_type="mmr",
        search_kwargs={'k': 5, 'fetch_k': 30, 'lambda_mult': 0.8}
    )
    print(f"docs in parent store: {len(list(store.yield_keys()))}")
    return retriever

def get_parent_doc_retriever_with_reranker(vectordb, persistent_parent_store):
    base_retriever = get_parent_doc_retriever(vectordb, persistent_parent_store)
    # top_n is the number of documents to return after re-ranking
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    return compression_retriever
    
def sanitise_filename(full_path_to_file):
    # replace all non-alphanumeric characters in the basename with underscores
    return re.sub(r"[^a-zA-Z0-9]", "_", os.path.basename(full_path_to_file))

@click.command()
@click.option("--pdf", type=click.Path(exists=True), cls=MutuallyExclusiveOption, mutually_exclusive=["text"], help="Path to the PDF file to be processed")
@click.option("--text", type=click.Path(exists=True), cls=MutuallyExclusiveOption, mutually_exclusive=["pdf"], help="Path to the text file to be processed")
@click.option("--query", help="User query to be processed")
@click.option("--use-reranker", is_flag=True, help="Use reranker to re-rank docs retrieved from vector database")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def main(pdf, text, query, use_reranker, debug):
    file_path = pdf or text
    if pdf:
        loader = PyPDFLoader(file_path=pdf)
    else:
        loader = TextLoader(file_path=text)

    PROMPT_TEMPLATE = (
        "Here are some documents that might help answering the question: "
        + "{query}"
        + "\n\nRelevant documents:\n"
        # + "\n\n".join([doc.page_content for doc in relevant_docs])
        + "{relevant_docs}"
        + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the provided documents, please state so."
    )
    
    # debug mode
    set_debug(debug)
    
    ### store the parsed chunks in the vector store for naive retrieval
    naive_rag_store_name = sanitise_filename(file_path)
    naive_db = VectorStore(naive_rag_store_name)
    load_vector_store_for_naive_retrieval(naive_db, loader)

    rag_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOllama(model="llama3.1", temperature=0)
    output_parser = StrOutputParser()
    
    print("\n\nNAIVE RETRIEVAL")
    naive_retrieval = RunnableParallel({"query": RunnablePassthrough(), "relevant_docs": get_naive_retriever(naive_db)})
    naive_chain = naive_retrieval | rag_prompt | llm | output_parser
    print(naive_chain.invoke(query))

    print("\n\nNAIVE RETRIEVAL WITH RE-RANKER RETRIEVAL")
    naive_with_reranker_retrieval = RunnableParallel({"query": RunnablePassthrough(), "relevant_docs": get_naive_with_reranker_retriever(naive_db)})
    naive_with_reranker_chain = naive_with_reranker_retrieval | rag_prompt | llm | output_parser
    print(naive_with_reranker_chain.invoke(query))

    ### store the parsed chunks in the vector store for parent document retrieval
    parent_doc_rag_store_name = f"{naive_rag_store_name}_parent"
    child_chunk_db = VectorStore(parent_doc_rag_store_name)
    # https://stackoverflow.com/a/77397998
    persistent_parent_store = create_kv_docstore(LocalFileStore(f"{child_chunk_db.persistent_dir}/child_chunks"))
    load_vector_store_for_parent_document_retrieval(child_chunk_db, loader, persistent_parent_store)

    print("\n\nPARENT DOCUMENT RETRIEVAL")
    parent_doc_retrieval = RunnableParallel({"query": RunnablePassthrough(), "relevant_docs": get_parent_doc_retriever(child_chunk_db, persistent_parent_store)})
    parent_doc_chain = parent_doc_retrieval | rag_prompt | llm | output_parser
    print(parent_doc_chain.invoke(query))

    print("\n\nPARENT DOCUMENT WITH RE-RANKER RETRIEVAL")
    parent_doc_with_reranker_retrieval = RunnableParallel({"query": RunnablePassthrough(), "relevant_docs": get_parent_doc_retriever_with_reranker(child_chunk_db, persistent_parent_store)})
    parent_doc_with_reranker_chain = parent_doc_with_reranker_retrieval | rag_prompt | llm | output_parser
    print(parent_doc_with_reranker_chain.invoke(query))
    
if __name__ == "__main__":
    main()