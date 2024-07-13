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

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
# embeddings = OllamaEmbeddings(model="llama3")
embeddings = OllamaEmbeddings(model="mxbai-embed-large:v1")

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

# a class the represents a vector store
class VectorStore:
    def __init__(self, store_name):
        self.store_name = store_name
        self.persistent_dir = os.path.join(db_dir, store_name)
        self.db = None
        
    def query(self, query, use_reranker=False):
        if not self.exists():
            print(f"Vector store {self.store_name} does not exist at {self.persistent_dir}.")
            return None
        
        if self.db is None:
            self.db = Chroma(
                persist_directory=self.persistent_dir,
                embedding_function=embeddings,
                # collection_metadata={"hnsw:space": "cosine"}
            )
        
        retriever = self.db.as_retriever(
            # # similarity_score_threshold search type
            # search_type="similarity_score_threshold",
            # search_kwargs={"score_threshold": 0.1, "k": 2}

            # # similarity search type
            # search_type="similarity",
            # search_kwargs={"k": 3}

            # mmr search type
            search_type="mmr",
            # If not using re-ranker, limit "k" to smaller value and rely on Chroma to
            # return the closest matches
            # If using re-ranker, get more top matching docs from Chroma and give them
            # to the re-ranker to re-rank to docs
            search_kwargs =
                {'k': 20, 'fetch_k': 50, 'lambda_mult': 0.8}
                    if use_reranker
                    else {'k': 5, 'fetch_k': 30, 'lambda_mult': 0.8}
        )

        if use_reranker:
            # use re-ranker to re-rank to retrieved docs from the Chroma db
            compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=retriever
            )
            reranked_docs = compression_retriever.invoke(query)
            return reranked_docs
        else:
            # return the docs retrieved from the Chroma db
            relevant_docs = retriever.invoke(query)
            return relevant_docs

    def exists(self):
        return os.path.exists(self.persistent_dir)
    
    def populate(self, docs):
        if not self.exists():
            print(f"\n--- Create vector store: {self.store_name} ---")
            self.db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=self.persistent_dir)
            print(f"--- Vector store created: {self.store_name} ---")
        else:
            print(f"Vector store {self.store_name} already exists at {self.persistent_dir}. No need to initialise.")

def sanitise_filename(full_path_to_file):
    # replace all non-alphanumeric characters in the basename with underscores
    return re.sub(r"[^a-zA-Z0-9]", "_", os.path.basename(full_path_to_file))

def validate_options(ctx, param, value):
    pdf = ctx.params.get('pdf')
    print(pdf)
    text = ctx.params.get('text')
    if not pdf and not text:
        raise click.BadParameter('You must provide at least one of --pdf or --text.')
    if pdf and text:
        raise click.BadParameter('You cannot provide both --pdf and --text.')
    return value

@click.command()
@click.option("--pdf", type=click.Path(exists=True), cls=MutuallyExclusiveOption, mutually_exclusive=["text"], help="Path to the PDF file to be processed")
@click.option("--text", type=click.Path(exists=True), cls=MutuallyExclusiveOption, mutually_exclusive=["pdf"], help="Path to the text file to be processed")
@click.option("--query", help="User query to be processed")
@click.option("--use-reranker", is_flag=True, help="Use reranker to re-rank docs retrieved from vector database")
def main(pdf, text, query, use_reranker):
    file_path = pdf or text
    store_name = sanitise_filename(file_path)
    db = VectorStore(store_name)
    if pdf:
        loader = PyPDFLoader(file_path=pdf)
    else:
        loader = TextLoader(file_path=text)

    if db.exists():
        print(f"Vector store {db.store_name} already exists at {db.persistent_dir}. No need to reload the pdf.")
    else:
        ### store the parsed chunks in the vector store for naive retrieval
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = loader.load()
        splits = splitter.split_documents(docs)
        db.populate(splits)
    
    relevant_docs = db.query(query, use_reranker)
    for doc in relevant_docs:
        print(doc.page_content[:40])
    print(f"number of relevant docs {len(relevant_docs)}")
    combined_text = (
        "Here are some documents that might help answering the question: "
        + "{query}"
        + "\n\nRelevant documents:\n"
        # + "\n\n".join([doc.page_content for doc in relevant_docs])
        + "{relevant_docs}"
        + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the provided documents, please state so."
    )
    llm = ChatOllama(model="llama3", temperature=0)
    template = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant"), ("human", combined_text)]
    )
    messages = template.format_messages(query=query, relevant_docs=relevant_docs)
    result = llm.invoke(messages)
    print(result.content)

if __name__ == '__main__':
    main()