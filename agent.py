import os
import git
import argparse
import shutil
from pathlib import Path

from langchain.text_splitter import Language
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser

# Load Env Variables
from dotenv import load_dotenv

load_dotenv()

# For BedRock
from langchain_aws import BedrockEmbeddings


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Clone and vectorize a Git repository for LLM analysis")
    
    parser.add_argument(
        "--repo-url",
        default="https://github.com/redpointsec/vtm.git",
        help="URL of the Git repository to clone (default: current repo)"
    )
    
    parser.add_argument(
        "--repo-path", 
        default="./repo",
        help="Local path where repository will be cloned (default: ./repo)"
    )
    
    parser.add_argument(
        "--vector-db-path",
        default="./vector_databases",
        help="Path to save vector database (default: ./vector_databases)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reclone repository and revectorize code (deletes existing repo and vector database)"
    )
    
    return parser.parse_args()


def setup_vector_db_directory(vector_db_path):
    """Create vector database directory if it doesn't exist."""
    Path(vector_db_path).mkdir(parents=True, exist_ok=True)
    print(f"Vector database directory: {vector_db_path}")


def clone_repository(repo_url, repo_path, force=False):
    """Clone repository if it doesn't exist or if force is True."""
    if force and os.path.exists(repo_path):
        print(f"Force flag set. Removing existing repository at: {repo_path}")
        shutil.rmtree(repo_path)
    
    if os.path.isdir(repo_path) and os.path.isdir(os.path.join(repo_path, ".git")):
        print("Directory already contains a git repository.")
        return True
    else:
        try:
            repo = git.Repo.clone_from(repo_url, repo_path)
            print(f"Repository cloned into: {repo_path}")
            return True
        except Exception as e:
            print(f"An error occurred while cloning the repository: {e}")
            return False


def vectorize_repository(repo_path, vector_db_path, force=False):
    """Vectorize the repository if not already done or if force is True."""
    faiss_db_path = os.path.join(vector_db_path, "vtm_faiss")
    
    if force and os.path.exists(faiss_db_path):
        print(f"Force flag set. Removing existing vector database at: {faiss_db_path}")
        shutil.rmtree(faiss_db_path)
    
    if os.path.exists(faiss_db_path):
        print("Vector database already exists. Skipping vectorization.")
        return
    
    print("Starting vectorization process...")
    
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON),
        show_progress=True,
    )

    documents = loader.load()
    
    if not documents:
        print("No documents found to vectorize.")
        return
    
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=8000, chunk_overlap=100
    )
    texts = text_splitter.split_documents(documents)
    
    print(f"Creating vector database with {len(texts)} text chunks...")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(faiss_db_path)
    print(f"Vector database saved to: {faiss_db_path}")


def main():
    """Main function to orchestrate the repository cloning and vectorization."""
    args = parse_arguments()
    
    print("=== LLM Repository Analysis Tool ===")
    print(f"Repository URL: {args.repo_url}")
    print(f"Repository Path: {args.repo_path}")
    print(f"Vector DB Path: {args.vector_db_path}")
    print(f"Force Mode: {args.force}")
    print("=" * 40)
    
    # Setup vector database directory
    setup_vector_db_directory(args.vector_db_path)
    
    # Clone repository
    if not clone_repository(args.repo_url, args.repo_path, args.force):
        print("Failed to clone repository. Exiting.")
        return
    
    # Vectorize repository
    vectorize_repository(args.repo_path, args.vector_db_path, args.force)
    
    print("Process completed successfully!")


if __name__ == "__main__":
    main()
