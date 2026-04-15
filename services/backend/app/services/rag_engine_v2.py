"""
RAG Engine V2 - 完全动态的 LLM 驱动检索

核心原则：
1. 不预设任何查询格式 - LLM 自己决定如何查询
2. 动态 Schema 发现 - 实时获取图谱结构传给 LLM
3. 自我修正机制 - LLM 评估结果并决定是否需要重新查询
4. 直接推理支持 - 不只依赖 Cypher，也支持子图分析
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.settings import settings


def json_serializable(obj: Any) -> Any:
    """将对象转换为 JSON 可序列化的格式"""
    if hasattr(obj, "isoformat"):  # DateTime 对象
        return obj.isoformat()
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        return [json_serializable(i) for i in obj]
    return obj


def clean_node_for_json(node: dict[str, Any]) -> dict[str, Any]:
    """清理节点数据，确保可以 JSON 序列化"""
    return {k: json_serializable(v) for k, v in node.items()}


def extract_json_from_response(text: str) -> dict[str, Any]:
    """从 LLM 响应中提取 JSON 内容"""
    # 尝试直接解析
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块提取
    patterns = [
        r'```json\s*(.*?)\s*```',  # ```json ... ```
        r'```\s*(.*?)\s*```',       # ``` ... ```
        r'\{.*\}',                  # 最外层的大括号
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

    # 如果都失败了，返回空 dict
    logger.warning(f"Could not extract JSON from response: {text[:200]}...")
    return {}

logger = logging.getLogger(__name__)

# 常量配置
MAX_ITERATIONS = 3
MAX_LABEL_SAMPLES = 5
MAX_CANDIDATE_NODES = 5
MAX_QUERIES_PER_ITERATION = 3
MAX_RESULTS_FOR_LLM = 20
MAX_SOURCES_IN_RESPONSE = 10
KEYWORD_MIN_LENGTH = 2


@dataclass
class RetrievalContext:
    """检索上下文 - 包含所有相关信息供 LLM 推理"""
    question: str
    graph_schema: dict[str, Any]  # 动态获取的图谱结构
    sample_data: list[dict[str, Any]]  # 样本数据帮助 LLM 理解格式
    previous_queries: list[dict[str, Any]] = field(default_factory=list)  # 之前的查询记录（用于自我修正）
    retrieved_data: list[dict[str, Any]] = field(default_factory=list)  # 已检索到的数据
    max_results: int = MAX_RESULTS_FOR_LLM  # 限制返回结果数量


@dataclass
class LLMQueryPlan:
    """LLM 生成的查询计划"""
    thinking: str  # LLM 的思考过程
    queries: list[str]  # Cypher 查询列表（可以是空，表示直接推理）
    needs_direct_analysis: bool = False  # 是否需要直接分析子图而非 Cypher
    follow_up_needed: bool = False  # 是否需要后续查询


@dataclass
class RAGAnswerV2:
    """RAG 回答结果"""
    answer: str
    reasoning_process: str  # 完整的推理过程
    sources: list[dict[str, Any]]
    confidence: str
    query_plans: list[LLMQueryPlan]  # 记录所有查询计划（展示思考过程）


class AdaptiveRAGEngine:
    """
    自适应 RAG 引擎 - 完全依赖 LLM 能力

    流程：
    1. 动态获取图谱 Schema
    2. 让 LLM 决定检索策略（Cypher vs 直接分析）
    3. 执行检索
    4. 让 LLM 评估结果，决定是否需要修正
    5. 生成最终回答
    """

    # 类级 Schema 缓存，避免每次请求都重新 discovery
    _schema_cache: dict[str, Any] | None = None
    _schema_cache_time: float = 0.0
    _SCHEMA_CACHE_TTL: float = 300.0  # 5 分钟

    def __init__(
        self,
        neo4j_client: Any | None = None,
        sqlite_store: Any | None = None,
    ) -> None:
        self.neo4j = neo4j_client
        self.store = sqlite_store

    def _get_neo4j_driver(self) -> Any | None:
        """获取 Neo4j 驱动（提取重复的驱动检查逻辑）"""
        if not self.neo4j:
            return None
        return getattr(self.neo4j, "_driver", None)

    async def answer(
        self,
        question: str,
        doc_id: str | None = None,
        top_k: int = MAX_RESULTS_FOR_LLM,
    ) -> RAGAnswerV2:
        """主入口 - 完全动态的流程"""

        # 步骤 1: 动态获取图谱 Schema
        schema = await self._discover_schema()
        sample_data = await self._get_sample_data(3)

        context = RetrievalContext(
            question=question,
            graph_schema=schema,
            sample_data=sample_data,
            max_results=top_k,
        )

        # 如果指定了 doc_id，可以过滤 schema 或样本数据
        if doc_id and self.neo4j:
            sample_data = await self._get_samples_for_doc(doc_id, 3)
            if sample_data:
                context.sample_data = sample_data

        # 步骤 2-4: 迭代检索（支持自我修正）
        query_plans: list[LLMQueryPlan] = []

        for iteration in range(MAX_ITERATIONS):
            # 让 LLM 制定查询计划
            plan = await self._generate_query_plan(context)
            query_plans.append(plan)

            if plan.needs_direct_analysis:
                # 直接分析模式：获取子图让 LLM 分析
                await self._direct_graph_analysis(context, plan)
            elif plan.queries:
                # Cypher 查询模式
                await self._execute_queries(context, plan.queries)

            # 检查是否需要继续
            if not plan.follow_up_needed or iteration == MAX_ITERATIONS - 1:
                break

            context.previous_queries.append({
                "iteration": iteration,
                "plan": plan,
                "results": context.retrieved_data.copy(),
            })

        # 步骤 5: 生成最终回答
        return await self._generate_final_answer(context, query_plans)

    async def _discover_schema(self) -> dict[str, Any]:
        """动态发现图谱 Schema - 带缓存避免重复查询"""
        now = time.time()
        if (
            AdaptiveRAGEngine._schema_cache is not None
            and (now - AdaptiveRAGEngine._schema_cache_time) < AdaptiveRAGEngine._SCHEMA_CACHE_TTL
        ):
            return dict(AdaptiveRAGEngine._schema_cache)

        driver = self._get_neo4j_driver()
        if not driver:
            return {}

        async with driver.session() as session:
            labels = await self._get_node_labels(session)
            rel_types = await self._get_relationship_types(session)
            node_samples = await self._get_node_samples(session, labels[:MAX_LABEL_SAMPLES])
            type_distribution = await self._get_type_distribution(session)

            schema = {
                "labels": labels,
                "relationship_types": rel_types,
                "node_samples": node_samples,
                "type_distribution": type_distribution,
            }
            AdaptiveRAGEngine._schema_cache = dict(schema)
            AdaptiveRAGEngine._schema_cache_time = now
            return schema

    async def _get_node_labels(self, session: Any) -> list[str]:
        """获取所有节点标签"""
        result = await session.run("""
            CALL db.labels() YIELD label
            RETURN collect(label) as labels
        """)
        record = await result.single()
        return record["labels"] if record else []

    async def _get_relationship_types(self, session: Any) -> list[str]:
        """获取所有关系类型"""
        result = await session.run("""
            CALL db.relationshipTypes() YIELD relationshipType
            RETURN collect(relationshipType) as types
        """)
        record = await result.single()
        return record["types"] if record else []

    async def _get_node_samples(
        self,
        session: Any,
        labels: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """获取每种节点的属性样本"""
        node_samples: dict[str, list[dict[str, Any]]] = {}

        for label in labels:
            result = await session.run(f"""
                MATCH (n:{label})
                RETURN n LIMIT 3
            """)
            samples = [clean_node_for_json(dict(record["n"])) async for record in result]
            if samples:
                node_samples[label] = samples

        return node_samples

    async def _get_type_distribution(self, session: Any) -> dict[str, int]:
        """获取节点类型分布"""
        type_distribution: dict[str, int] = {}
        result = await session.run("""
            MATCH (n)
            RETURN labels(n)[0] as label, count(n) as count
            ORDER BY count DESC
            LIMIT 10
        """)
        async for record in result:
            type_distribution[record["label"]] = record["count"]
        return type_distribution

    async def _get_sample_data(self, limit: int = 3) -> list[dict[str, Any]]:
        """获取样本数据帮助 LLM 理解格式"""
        driver = self._get_neo4j_driver()
        if not driver:
            return []

        async with driver.session() as session:
            return await self._get_path_samples(session, limit)

    async def _get_path_samples(self, session: Any, limit: int) -> list[dict[str, Any]]:
        """获取路径样本"""
        samples: list[dict[str, Any]] = []
        result = await session.run(f"""
            MATCH path = (n)-[r]->(m)
            RETURN n, r, m
            LIMIT {limit}
        """)
        async for record in result:
            samples.append(self._build_relationship_data(record))
        return samples

    async def _get_samples_for_doc(
        self,
        doc_id: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """获取指定文档的样本数据"""
        driver = self._get_neo4j_driver()
        if not driver:
            return []

        async with driver.session() as session:
            samples: list[dict[str, Any]] = []
            result = await session.run("""
                MATCH (n)-[r]-(m)
                WHERE n.doc_id = $doc_id OR m.doc_id = $doc_id
                RETURN n, r, m
                LIMIT $limit
            """, doc_id=doc_id, limit=limit)
            async for record in result:
                samples.append(self._build_relationship_data(record))
            return samples

    def _build_relationship_data(self, record: Any) -> dict[str, Any]:
        """构建关系数据字典（提取重复的代码）"""
        return {
            "source": clean_node_for_json(dict(record["n"])),
            "relationship": {
                "type": record["r"].type,
                "properties": clean_node_for_json(dict(record["r"])),
            },
            "target": clean_node_for_json(dict(record["m"])),
        }

    async def _generate_query_plan(self, context: RetrievalContext) -> LLMQueryPlan:
        """
        让 LLM 自主决定检索策略
        """
        system_prompt = self._build_query_plan_prompt(context)

        try:
            response = await self._call_llm(system_prompt)
            plan_data = extract_json_from_response(response)

            if not plan_data:
                logger.warning("Could not parse LLM response, using fallback")
                return LLMQueryPlan(
                    thinking="Failed to parse response, using direct analysis",
                    queries=[],
                    needs_direct_analysis=True,
                )

            return LLMQueryPlan(
                thinking=plan_data.get("thinking", ""),
                queries=plan_data.get("queries", []),
                needs_direct_analysis=plan_data.get("needs_direct_analysis", False),
                follow_up_needed=plan_data.get("follow_up_needed", False),
            )
        except Exception as e:
            logger.error(f"Failed to generate query plan: {e}")
            return LLMQueryPlan(
                thinking=f"Error: {e}",
                queries=[],
                needs_direct_analysis=True,
            )

    def _build_query_plan_prompt(self, context: RetrievalContext) -> str:
        """构建查询计划的 prompt"""
        schema_str = json.dumps(context.graph_schema, ensure_ascii=False, indent=2)
        samples_str = json.dumps(context.sample_data, ensure_ascii=False, indent=2)

        previous_queries_str = ""
        if context.previous_queries:
            previous_queries_str = f"""
之前已尝试的查询：
{json.dumps(context.previous_queries, ensure_ascii=False, default=json_serializable)}

检索结果分析：
{self._summarize_retrieved(context.retrieved_data)}

请评估：
1. 之前的结果是否足够回答问题？
2. 是否需要调整查询策略？
"""

        return f"""你是一个智能知识检索专家。

你的任务是根据用户问题，决定如何从知识图谱中检索信息。

图谱结构信息（动态获取）：
{schema_str}

样本数据（帮助理解格式）：
{samples_str}

用户问题：
{context.question}
{previous_queries_str}

【重要提示 - 图谱结构】
1. 所有节点的标签都是 "Entity"，不同类型用 "type" 属性区分
2. 节点名称存储在 "name" 属性中（不是 label）
3. 数据库中实际存在的 type 值（示例）：
   - NewsItem: 新闻标题
   - Organization: 组织机构（如来源网站）
   - Entity: 通用实体（OneKE 提取的未分类实体）
   - Time: 时间信息
   - ThemeTag/ProvinceTag/CityTag/IndustryTag: 各类标签
4. 关系类型：REL(OneKE提取的关系), CORRELATED_WITH(相似度计算的关系)
5. 查询策略：
   - 政策信息通常在 NewsItem 中
   - 优先使用 CONTAINS 进行模糊匹配，不要假设精确值存在
   - 如果找不到特定类型，尝试在 Entity 类型中搜索
6. 查询示例（将"关键词"替换为实际查询词）：
   - MATCH (n:Entity {{type: 'NewsItem'}}) WHERE n.name CONTAINS '关键词' RETURN n
   - MATCH (n:Entity)-[r:REL|CORRELATED_WITH]-(m:Entity) WHERE n.name CONTAINS '关键词' RETURN n, r, m
   - MATCH (n:Entity {{type: 'Organization'}}) WHERE n.name CONTAINS '关键词' RETURN n

请用 JSON 格式输出你的思考：
{{
  "thinking": "你的分析思考过程...",
  "strategy": "cypher|direct_analysis|hybrid",
  "queries": ["如果需要Cypher查询，写在这里"],
  "needs_direct_analysis": false,
  "follow_up_needed": false,
  "reason": "为什么这样选择"
}}"""

    async def _direct_graph_analysis(
        self,
        context: RetrievalContext,
        plan: LLMQueryPlan,
    ) -> None:
        """
        直接分析模式：获取相关子图让 LLM 分析
        """
        driver = self._get_neo4j_driver()
        if not driver:
            logger.warning("[_direct_graph_analysis] Neo4j driver not available")
            return

        keywords = self._extract_keywords(context.question)
        logger.info(f"[_direct_graph_analysis] Extracted keywords: {keywords}")

        async with driver.session() as session:
            # 获取候选节点
            candidate_nodes = await self._find_candidate_nodes(session, keywords)
            logger.info(f"[_direct_graph_analysis] Found {len(candidate_nodes)} candidate nodes")

            # 获取子图
            await self._fetch_subgraphs(session, context, candidate_nodes)

    def _extract_keywords(self, question: str) -> list[str]:
        """提取关键词（过滤停用词和短词）"""
        # 简单停用词表
        stopwords = {"的", "是", "在", "和", "了", "有", "我", "都", "个", "与", "也", "对",
                     "为", "能", "很", "可以", "就", "不", "会", "要", "没有", "我们的"}

        words = question.split()
        return [
            word.strip("。？！，；：\"'（）【】《》")
            for word in words
            if len(word) >= KEYWORD_MIN_LENGTH and word not in stopwords
        ]

    async def _find_candidate_nodes(
        self,
        session: Any,
        keywords: list[str],
    ) -> list[dict[str, Any]]:
        """查找候选节点（串行执行避免 Neo4j 并发问题）"""
        if not keywords:
            return []

        # 串行执行关键词查询
        all_nodes: list[dict[str, Any]] = []
        for keyword in keywords[:5]:  # 限制关键词数量
            nodes = await self._search_nodes_by_keyword(session, keyword)
            all_nodes.extend(nodes)

        # 合并结果并去重（使用 Set 实现 O(1) 查找）
        seen_ids: set[str] = set()
        unique_nodes: list[dict[str, Any]] = []

        for node in all_nodes:
            node_id = node.get("id") or node.get("name")
            if node_id and node_id not in seen_ids:
                seen_ids.add(node_id)
                unique_nodes.append(node)

        return unique_nodes[:MAX_CANDIDATE_NODES]

    async def _search_nodes_by_keyword(self, session: Any, keyword: str) -> list[dict[str, Any]]:
        """根据关键词搜索节点"""
        try:
            # 实际图谱结构：所有节点都是 Entity 标签，类型存储在 type 属性中
            result = await session.run("""
                MATCH (n:Entity)
                WHERE n.name CONTAINS $keyword
                RETURN n LIMIT 10
            """, keyword=keyword)

            nodes = [clean_node_for_json(dict(record["n"])) async for record in result]
            logger.info(f"[_search_nodes_by_keyword] Keyword '{keyword}' found {len(nodes)} nodes")
            return nodes[:10]
        except Exception as e:
            logger.warning(f"[_search_nodes_by_keyword] Keyword '{keyword}' search failed: {e}")
            return []

    async def _fetch_subgraphs(
        self,
        session: Any,
        context: RetrievalContext,
        candidate_nodes: list[dict[str, Any]],
    ) -> None:
        """获取候选节点的子图（串行执行避免 Neo4j 并发问题）"""
        if not candidate_nodes:
            logger.info("[_fetch_subgraphs] No candidate nodes to fetch")
            return

        # 限制节点数量
        nodes_to_fetch = candidate_nodes[:MAX_CANDIDATE_NODES]
        logger.info(f"[_fetch_subgraphs] Fetching subgraphs for {len(nodes_to_fetch)} nodes")

        # 串行获取子图
        total_subgraph_items = 0
        for node in nodes_to_fetch:
            subgraph_data = await self._fetch_single_subgraph(session, node)
            context.retrieved_data.extend(subgraph_data)
            total_subgraph_items += len(subgraph_data)

        logger.info(f"[_fetch_subgraphs] Total subgraph items: {total_subgraph_items}")

        # 限制总结果数量，避免内存溢出
        if len(context.retrieved_data) > context.max_results * 2:
            context.retrieved_data = context.retrieved_data[:context.max_results * 2]

    async def _fetch_single_subgraph(self, session: Any, node: dict[str, Any]) -> list[dict[str, Any]]:
        """获取单个节点的子图"""
        try:
            result = await session.run("""
                MATCH (n)-[r]-(m)
                WHERE id(n) = $node_id
                RETURN n, r, m
                LIMIT 20
            """, node_id=node.get("id"))

            return [self._build_relationship_data(record) async for record in result]
        except Exception as e:
            logger.warning(f"Subgraph fetch failed for node {node.get('id')}: {e}")
            return []

    async def _execute_queries(
        self,
        context: RetrievalContext,
        queries: list[str],
    ) -> None:
        """执行 Cypher 查询（串行执行，避免 Neo4j 会话并发问题）"""
        driver = self._get_neo4j_driver()
        if not driver:
            logger.warning("[_execute_queries] Neo4j driver not available")
            return

        logger.info(f"[_execute_queries] Executing {len(queries)} queries")

        async with driver.session() as session:
            # 串行执行查询（Neo4j 会话不支持并发）
            limited_queries = queries[:MAX_QUERIES_PER_ITERATION]
            total_results = 0

            for i, query in enumerate(limited_queries):
                data_list = await self._execute_single_query(session, query, i)
                context.retrieved_data.extend(data_list)
                total_results += len(data_list)

            logger.info(f"[_execute_queries] Total results collected: {total_results}")

            # 限制总结果数量
            if len(context.retrieved_data) > context.max_results * 2:
                context.retrieved_data = context.retrieved_data[:context.max_results * 2]

    def _validate_and_fix_query(self, query: str) -> str:
        """验证并修复常见的 Cypher 查询错误"""
        original = query
        # 修复错误标签：将 :NewsItem, :Organization 等改为 :Entity {type: 'NewsItem'}
        import re

        # 常见错误标签映射
        # 注意：使用 {{ 和 }} 来转义 Python f-string 中的大括号
        type_patterns = [
            (r'\(:NewsItem\)', "(n:Entity {{type: 'NewsItem'}})"),
            (r'\(:Organization\)', "(n:Entity {{type: 'Organization'}})"),
            (r'\(:Person\)', "(n:Entity {{type: 'Person'}})"),
            (r'\(:Policy\)', "(n:Entity {{type: 'Policy'}})"),
            (r'\(:ThemeTag\)', "(n:Entity {{type: 'ThemeTag'}})"),
            (r'\(:ProvinceTag\)', "(n:Entity {{type: 'ProvinceTag'}})"),
            (r'\(:CityTag\)', "(n:Entity {{type: 'CityTag'}})"),
            (r'\(:Location\)', "(n:Entity {{type: 'Location'}})"),
            (r'\(:Event\)', "(n:Entity {{type: 'Event'}})"),
            (r'\(:Technology\)', "(n:Entity {{type: 'Technology'}})"),
            (r'\(:Time\)', "(n:Entity {{type: 'Time'}})"),
            # 变量形式的标签
            (r'\(([a-zA-Z_]+):NewsItem\)', r"(\1:Entity {{type: 'NewsItem'}})"),
            (r'\(([a-zA-Z_]+):Organization\)', r"(\1:Entity {{type: 'Organization'}})"),
            (r'\(([a-zA-Z_]+):Person\)', r"(\1:Entity {{type: 'Person'}})"),
            (r'\(([a-zA-Z_]+):ProvinceTag\)', r"(\1:Entity {{type: 'ProvinceTag'}})"),
            (r'\(([a-zA-Z_]+):ThemeTag\)', r"(\1:Entity {{type: 'ThemeTag'}})"),
            (r'\(([a-zA-Z_]+):Time\)', r"(\1:Entity {{type: 'Time'}})"),
        ]

        for pattern, replacement in type_patterns:
            query = re.sub(pattern, replacement, query)

        if query != original:
            logger.info(f"[_validate_and_fix_query] Fixed query: {original} -> {query}")

        return query

    async def _execute_single_query(
        self,
        session: Any,
        query: str,
        query_index: int = 0,
    ) -> list[dict[str, Any]]:
        """执行单个 Cypher 查询"""
        # 先验证和修复查询
        query = self._validate_and_fix_query(query)
        logger.info(f"[_execute_single_query] Query {query_index}: {query[:100]}...")
        try:
            result = await session.run(query)
            data_list: list[dict[str, Any]] = []

            async for record in result:
                data: dict[str, Any] = {}
                for key, value in record.items():
                    if hasattr(value, "keys"):  # Node/Relationship
                        data[key] = dict(value)
                    else:
                        data[key] = value
                data_list.append(data)

            logger.info(f"[_execute_single_query] Query {query_index} returned {len(data_list)} results")
            return data_list
        except Exception as e:
            logger.warning(f"[_execute_single_query] Query {query_index} failed: {e}")
            return []

    async def _generate_final_answer(
        self,
        context: RetrievalContext,
        query_plans: list[LLMQueryPlan],
    ) -> RAGAnswerV2:
        """生成最终回答"""

        if not context.retrieved_data:
            return RAGAnswerV2(
                answer="未找到相关信息。",
                reasoning_process="经过多次尝试，未能从知识库中检索到相关信息。",
                sources=[],
                confidence="low",
                query_plans=query_plans,
            )

        prompt = self._build_final_answer_prompt(context)

        try:
            response = await self._call_llm(prompt)
            # 优先尝试 JSON 解析（兼容旧逻辑）
            result = extract_json_from_response(response)

            if result and ("answer" in result or "reasoning" in result):
                return RAGAnswerV2(
                    answer=str(result.get("answer", "")),
                    reasoning_process=str(result.get("reasoning", "")),
                    sources=context.retrieved_data[:MAX_SOURCES_IN_RESPONSE],
                    confidence=str(result.get("confidence", "medium")),
                    query_plans=query_plans,
                )

            # Markdown 格式：直接使用全文作为 answer
            return RAGAnswerV2(
                answer=response.strip(),
                reasoning_process="",
                sources=context.retrieved_data[:MAX_SOURCES_IN_RESPONSE],
                confidence="medium",
                query_plans=query_plans,
            )
        except Exception as e:
            logger.error(f"Failed to generate final answer: {e}")
            return RAGAnswerV2(
                answer=f"生成回答时出错: {e}",
                reasoning_process="",
                sources=[],
                confidence="low",
                query_plans=query_plans,
            )

    def _format_data_for_llm(self, data: list[dict[str, Any]]) -> str:
        """将检索数据格式化为 LLM 可读的文本格式"""
        if not data:
            return "无检索结果"

        formatted_items = []
        for i, item in enumerate(data):
            # 处理不同类型的数据
            if isinstance(item, dict):
                # 提取关键信息
                parts = []

                # 处理关系型数据 (source-relationship-target)
                if "source" in item and "target" in item:
                    src = item.get("source", {})
                    tgt = item.get("target", {})
                    rel = item.get("relationship", {})

                    src_name = src.get("name", "未知") if isinstance(src, dict) else str(src)
                    tgt_name = tgt.get("name", "未知") if isinstance(tgt, dict) else str(tgt)
                    rel_type = rel.get("type", "关联") if isinstance(rel, dict) else str(rel)

                    parts.append(f"[{i}] 关系: {src_name} --{rel_type}--> {tgt_name}")

                    # 添加额外属性
                    if isinstance(src, dict) and src.get("type"):
                        parts.append(f"    源类型: {src.get('type')}")
                    if isinstance(tgt, dict) and tgt.get("type"):
                        parts.append(f"    目标类型: {tgt.get('type')}")

                # 处理节点数据 (n, m 等键)
                elif "n" in item:
                    node = item["n"]
                    if isinstance(node, dict):
                        name = node.get("name", "未知")
                        node_type = node.get("type", "")
                        parts.append(f"[{i}] 节点: {name}")
                        if node_type:
                            parts.append(f"    类型: {node_type}")
                        # 添加其他属性
                        for k, v in node.items():
                            if k not in ("name", "type", "id", "embedding_model") and v:
                                parts.append(f"    {k}: {v}")

                # 处理其他字典数据
                else:
                    parts.append(f"[{i}] 数据:")
                    for k, v in item.items():
                        if isinstance(v, dict):
                            v_str = v.get("name", str(v))
                        else:
                            v_str = str(v)
                        parts.append(f"    {k}: {v_str}")

                formatted_items.append("\n".join(parts))
            else:
                formatted_items.append(f"[{i}] {str(item)}")

        return "\n\n".join(formatted_items)

    def _build_final_answer_prompt(self, context: RetrievalContext) -> str:
        """构建最终回答的 prompt"""
        data_str = self._format_data_for_llm(context.retrieved_data[:context.max_results])

        return f"""基于以下检索结果，回答用户问题。

用户问题：{context.question}

检索结果：
{data_str}

任务要求：
1. 仔细阅读检索结果中的每条信息
2. 只基于提供的数据回答，不要编造
3. 如果不确定，明确说明"无法确定"
4. 引用来源时标注 [数字]

回答时必须包含以下四个部分（使用 Markdown 标题）：

## 直接答案
用 2-4 句话给出核心结论，不要绕弯子。

## 详细分析
分点列出支持该结论的关键信息。每一点都必须引用具体的实体、关系或文本片段作为证据。

## 数据依据
以 Markdown 表格形式列出引用的关键来源（最多 10 条）。
| 序号 | 来源类型 | 名称/关系 | 说明 |
|------|----------|-----------|------|

## 信息缺口
明确指出哪些子问题在现有数据中找不到答案，禁止编造。

约束：
- 禁止输出 "根据提供的检索结果..." 这类套话
- 禁止编造不存在的数据
- 如果有定量问题，必须基于数据依据中的条目计数"""

    def _summarize_retrieved(self, data: list[dict[str, Any]]) -> str:
        """总结已检索的数据（用于自我修正）"""
        if not data:
            return "暂无检索结果"

        entities: set[str] = set()
        for item in data:
            if "source" in item:
                entities.add(item["source"].get("name", ""))
            if "target" in item:
                entities.add(item["target"].get("name", ""))

        return f"已检索到 {len(data)} 条记录，涉及实体: {', '.join(list(entities)[:10])}"

    async def _call_llm(self, prompt: str) -> str:
        """调用 LLM"""
        if not settings.openai_api_key:
            raise RuntimeError("LLM not configured")

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.openai_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.openai_model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant. Always respond in valid JSON format when asked, otherwise use Markdown."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 4000,
                },
            )
            response.raise_for_status()
            result = response.json()

            choices = result.get("choices", [])
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"Invalid LLM response: {result}")

            first = choices[0]
            if not isinstance(first, dict):
                raise ValueError(f"Invalid LLM response: {result}")
            message = first.get("message", {})
            if not isinstance(message, dict):
                raise ValueError(f"Invalid LLM response: {result}")
            content = message.get("content", "")
            content_str = str(content)
            logger.debug(f"LLM raw response: {content_str[:500]}...")
            return content_str
