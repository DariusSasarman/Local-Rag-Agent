Roadmap
- [x] Initial pipeline setup
- [ ] Fullstack setup implementation
- [ ] Make the full hybrid retrieval engine
  - [x] Vector search 
  - [x] Keyword search 
  - [ ] SQL + Small model translation
  - [ ] Web information retrieval + Search query generating model
- [ ] HNSW optimisation for vector search
- [ ] Reciprocal Rank Fusion
- [ ] Augment initial query with additional ones via query generation
- [ ] Remove distracting vector search results using a Cross-encoder
- [ ] Use a reranker and Top K selection to retrieve the best results 
- [ ] Advanced techniques : Windowing and hierarchical retrieval
- [ ] Use Ragas for evaluation

"For example window sizes are variable. 
A too small one, might lead to not enough context, 
a window too big might lead to irrelevant contest 
being passed"