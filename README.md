# Image_Search_VecDB

The Image_Search_VecDB repository is a project focused on implementing vector indexing, a fundamental technique used in modern image search engines. Vector indexing is chosen for its efficacy in efficiently handling high-dimensional feature embeddings, making it suitable for large-scale image databases.
The core component of this repository is the vector database powered by Pinecone, a vector similarity search service. The vector database is designed to store and manage the feature embeddings extracted from images, enabling fast and accurate similarity searches for image retrieval.

## Current Issue 
During the development of the app, an issue has been encountered when upserting data into the Pinecone index. The primary reason for this issue is that the size of the HTTP request exceeds the request limit. The issue stems from the large size of the involved images and their corresponding feature vectors, which contributes to the data payload of the HTTP request.

## Work in Progress
The Image_Search_VecDB repository is currently a work in progress, and the issue with HTTP request size is being addressed. The potential solutions to optimize the data transfer process and ensure that upsert operations do not exceed the request limits set by Pinecone are under development. Also, one simpler solution is to not use a cloud based vector database like Pinecone and options like ChromaDB and FAISS.
