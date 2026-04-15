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
MERGE (e:Entity {name: n.name, type: n.type})
SET e.id = n.id,
    e.doc_id = CASE WHEN n.type = 'NewsItem' THEN n.doc_id ELSE coalesce(e.doc_id, n.doc_id) END
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
// 从该文档的 NewsItem 出发获取关联实体和边
MATCH (news:Entity {type: 'NewsItem', doc_id: $doc_id})
// 获取同文档内的 REL 关系
OPTIONAL MATCH (news)-[r1:REL {doc_id: $doc_id}]-(e:Entity)
WITH news, r1, e
// 获取跨文档的 CORRELATED_WITH 关系
OPTIONAL MATCH (news)-[r2:CORRELATED_WITH]-(other:Entity)
WHERE other.doc_id <> $doc_id OR other.doc_id IS NULL
RETURN news AS source,
       r1 AS rel, e AS target,
       r2 AS corr_rel, other AS corr_target
"""


DELETE_GRAPH_BY_DOC_ID = """
// 删除该文档的 NewsItem 和该文档创建的 REL 边
MATCH (news:Entity {type: 'NewsItem', doc_id: $doc_id})
OPTIONAL MATCH (news)-[r:REL {doc_id: $doc_id}]-()
DELETE r, news
WITH count(news) as deleted_news
// 清理不再关联任何节点的孤立共享实体
MATCH (e:Entity)
WHERE NOT (e)-[]-() AND e.type <> 'NewsItem'
DELETE e
"""

# 读取所有图谱数据（限制数量避免过大）
READ_ALL_GRAPH = """
MATCH (e:Entity)
WHERE ($types IS NULL OR e.type IN $types)
WITH e LIMIT $node_limit
OPTIONAL MATCH (e)-[r:REL|CORRELATED_WITH]-(t:Entity)
RETURN e AS source, r AS rel, t AS target
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
