import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

es = Elasticsearch(
    "http://localhost:9200"
)

model = SentenceTransformer("all-mpnet-base-v2")

st.title("Movie Recommendation")

user_input = st.text_input("Enter your query:")

if st.button('submit'):
    if user_input:
        query_vector = model.encode(user_input)
        query = es.search(index = 'elastic-demo',body={
            "query": {
                "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",
                    "params": {
                    "query_vector": query_vector
                    }
                }
                }
            }
            })['hits']['hits']
        for item in query:
            source = item["_source"]
            st.json({
                "Title":source["title"],
                "Rating":source["rating"],
                "Overview":source["overview"],
                "Tagline":source["tagline"]
            })