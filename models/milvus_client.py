from pymilvus import connections, CollectionSchema, DataType, FieldSchema, Collection, utility
from tqdm.notebook import tqdm 
import numpy as np
import torch, re
import pandas as pd

class MilvusClient:

    def __init__(self, collection_name: str, host="127.0.0.1", port="19530", alias="default"):
        self.collection_name = collection_name
        self.collection = None
        self.alias = alias
        self.connect(host, port, alias)
        self.load_collection()

    def connect(self, host="127.0.0.1", port="19530", alias="default"):
        if connections.has_connection(alias):
            print(f"Already connected to Milvus with alias '{alias}'")
        else:
            try:
                connections.connect(alias=alias, host=host, port=port)
                print(f"Connected to Milvus at {host}:{port} with alias '{alias}'")
            except Exception as e:
                print(f"Failed to connect to Milvus: {e}")
                raise e

    def load_collection(self):
        if self.collection is None:
            self.collection = self.get_collection()
            print(f"Collection '{self.collection_name}' loaded")

    def drop_collection(self):
        self.collection.drop()
        print(f"Collection '{self.collection_name}' dropped.")

    def insert_embeddings(self, records):
        self.load_collection()
        self.collection.insert(records, _async=True)
        self.collection.flush()

    def get_collection(self):
        # This method should be extended to be dataset agnostic
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True), 
            FieldSchema(name="plot_embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="plot_text", dtype=DataType.VARCHAR, max_length=40000),
            FieldSchema(name="release_year", dtype=DataType.INT64),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=255),
        ]
        schema = CollectionSchema(fields, description="Wikipedia Movie Plots with vector embeddings and original plot text")
        collection = Collection(name=self.collection_name, schema=schema)
        return collection
    
    def clean_text(self, text):
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            return text.lower()

    def ingest_data(self, csv, embeddings_model, batch_size=32):
        df = pd.read_csv(csv)
        df = df[['Release Year', 'Title', 'Plot']].dropna()
        for i in tqdm(range(0, len(df), batch_size), desc="Inserting Batches", unit="batch"):
            batch_texts = [self.clean_text(text) for text in df['Plot'].iloc[i:i+batch_size].tolist()]
            batch_titles = df['Title'].iloc[i:i+batch_size].tolist()
            batch_release_year = df['Release Year'].iloc[i:i+batch_size].tolist()
            batch_ids = df.index[i:i+batch_size].tolist() 
            batch_embeddings = embeddings_model.get_batch_embeddings(batch_texts)
            batch_embeddings_cpu = batch_embeddings.cpu().numpy()

            records = [
                {
                    "id": id_value,
                    "release_year": release_year,
                    "title": title,
                    "plot_embedding": embedding.tolist(),  # Convert to list for insertion
                    "plot_text": text
                }
                for id_value, release_year, title, embedding, text in zip(batch_ids, batch_release_year, batch_titles, batch_embeddings_cpu, batch_texts)
            ]
            # self.insert_embeddings(records)
            self.collection.insert(records, _async=True)
    
        self.collection.flush()
        self.collection.create_index(field_name="plot_embedding", index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 100}})
        self.collection.load()
        self.count_records()
        
    def count_records(self):
        num_records = self.collection.num_entities
        print(f"Number of records in the collection: {num_records}")

    def create_index(self, field_name: str, index_type: str = "IVF_FLAT", metric_type: str = "L2", nlist: int = 100):
        self.load_collection()
        index_params = {"index_type": index_type, "metric_type": metric_type, "params": {"nlist": nlist}}
        self.collection.create_index(field_name=field_name, index_params=index_params)
        self.collection.load()

    def search(self, embedding, anns_field="plot_embedding", limit=5):
        self.load_collection()
        search_params = {"metric_type": "L2"}
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        if len(embedding.shape) > 1:
            embedding = embedding.flatten()
        embedding_list = embedding.tolist()
        print(f"Embedding type: {type(embedding_list)}, First element type: {type(embedding_list[0])}")
        results = self.collection.search(
            data=[embedding_list],
            anns_field=anns_field,
            param=search_params,
            limit=limit,
            output_fields=["id", "plot_text", "title", "release_year"]
        )
        return results
