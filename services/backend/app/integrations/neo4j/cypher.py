from __future__ import annotations

CONSTRAINTS_CYPHER: list[str] = [
    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE INDEX entity_doc_id_idx IF NOT EXISTS FOR (e:Entity) ON (e.doc_id)",
    "CREATE INDEX rel_doc_id_idx IF NOT EXISTS FOR ()-[r:REL]-() ON (r.doc_id)",
    "CREATE INDEX corr_rel_doc_id_idx IF NOT EXISTS FOR ()-[r:CORRELATED_WITH]-() ON (r.doc_id)",
    "CREATE INDEX corr_rel_id_idx IF NOT EXISTS FOR ()-[r:CORRELATED_WITH]-() ON (r.id)",
]


UPSERT_GRAPH_CYPHER = """
UNWIND $nodes AS n
MERGE (e:Entity {id: n.id})
SET e.name = n.name,
    e.type = n.type,
    e.doc_id = n.doc_id
"""


UPSERT_REL_CYPHER = """
UNWIND $edges AS r
MATCH (s:Entity {id: r.source_id})
MATCH (t:Entity {id: r.target_id})
MERGE (s)-[rel:REL {id: r.id}]->(t)
SET rel.type = r.type,
    rel.label = r.label,
    rel.doc_id = r.doc_id,
    rel.evidence = r.evidence
"""


READ_GRAPH_BY_DOC_ID = """
// 获取指定 doc_id 的所有实体
MATCH (e:Entity {doc_id: $doc_id})
// 获取这些实体之间的 REL 关系（同文档内）- 双向
OPTIONAL MATCH (e)-[r1:REL {doc_id: $doc_id}]-(t1:Entity {doc_id: $doc_id})
WITH e, r1, t1
// 获取这些实体参与的 CORRELATED_WITH 关系（跨文档相似度）- 双向
OPTIONAL MATCH (e)-[r2:CORRELATED_WITH]-(t2:Entity)
WHERE t2.doc_id <> $doc_id OR t2.doc_id IS NULL
RETURN e AS source,
       r1 AS rel, t1 AS target,
       r2 AS corr_rel, t2 AS corr_target
"""


DELETE_GRAPH_BY_DOC_ID = """
MATCH (e:Entity {doc_id: $doc_id})
DETACH DELETE e
"""

# 读取所有图谱数据（限制数量避免过大）
READ_ALL_GRAPH = """
MATCH (e:Entity)
WHERE ($types IS NULL OR e.type IN $types)
OPTIONAL MATCH (e)-[r:REL|CORRELATED_WITH]-(t:Entity)
RETURN e AS source, r AS rel, t AS target
LIMIT $limit
"""

# 向量相关查询
SET_NEWS_EMBEDDING = """
MATCH (n:Entity {type: 'NewsItem', doc_id: $doc_id})
SET n.embedding = $embedding,
    n.embedding_model = $model,
    n.embedding_updated = datetime()
RETURN n.doc_id as doc_id
"""

GET_NEWS_WITH_EMBEDDING = """
MATCH (n:Entity {type: 'NewsItem'})
WHERE n.embedding IS NOT NULL
RETURN n.doc_id as doc_id, n.name as title, n.embedding as embedding
LIMIT $limit
"""

GET_NEWS_WITHOUT_EMBEDDING = """
MATCH (n:Entity {type: 'NewsItem'})
WHERE n.embedding IS NULL
RETURN n.doc_id as doc_id, n.name as title
LIMIT $limit
"""

# 使用欧几里得距离计算向量相似度（Neo4j 原生支持）
# 注意：distance 返回的是欧几里得距离，需要转换为相似度分数
VECTOR_SIMILARITY_SEARCH = """
MATCH (n:Entity {type: 'NewsItem'})
WHERE n.embedding IS NOT NULL AND n.doc_id <> $doc_id
WITH n, gds.similarity.cosine(n.embedding, $embedding) AS similarity
WHERE similarity >= $min_score
RETURN n.doc_id as doc_id, n.name as title, similarity
ORDER BY similarity DESC
LIMIT $limit
"""
