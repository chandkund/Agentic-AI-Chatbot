# db_utils.py
import chromadb
from chromadb.config import Settings
import os
import shutil
from pathlib import Path

def get_chroma_client():
    chroma_dir = "chroma_db"
    try:
        # First try normal connection
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True  # Enable reset functionality
            )
        )
        
        # Test basic operations to validate schema
        try:
            client.heartbeat()  # Simple connection test
            test_col = client.get_or_create_collection("schema_test")
            test_col.add(ids=["1"], documents=["test"])
            test_col.query(query_texts=["test"], n_results=1)
            client.delete_collection("schema_test")
            return client
        except Exception as schema_error:
            print(f"Schema validation failed, resetting ChromaDB: {schema_error}")
            shutil.rmtree(chroma_dir, ignore_errors=True)
            Path(chroma_dir).mkdir(exist_ok=True)
            return chromadb.PersistentClient(
                path=chroma_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            
    except Exception as e:
        print(f"ChromaDB initialization failed: {e}")
        raise