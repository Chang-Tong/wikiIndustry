from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

from app.integrations.neo4j.client import Neo4jClient

router = APIRouter()


class GraphSchemaResponse(BaseModel):
    """图谱结构信息，帮助 LLM 理解图谱"""
    node_types: list[dict[str, str]] = Field(default_factory=list, description="节点类型列表")
    edge_types: list[dict[str, str]] = Field(default_factory=list, description="关系类型列表")
    example_queries: list[dict[str, str]] = Field(default_factory=list, description="示例Cypher查询")
    stats: dict[str, int] = Field(default_factory=dict, description="统计信息")


@router.get("/graph/schema", response_model=GraphSchemaResponse)
async def get_graph_schema(request: Request) -> GraphSchemaResponse:
    """获取知识图谱的结构信息，用于LLM理解和前端展示"""
    neo4j: Neo4jClient | None = request.app.state.neo4j
    if neo4j is None:
        raise HTTPException(status_code=503, detail="Neo4j not available")

    try:
        stats = await neo4j.get_schema_stats()
    except Exception:
        stats = {}

    return GraphSchemaResponse(
        node_types=[
            {
                "type": "NewsItem",
                "description": "新闻条目，文档中的每条新闻摘要",
                "example": "【1】教育部发布新政策",
                "fields": "id, name, type, doc_id"
            },
            {
                "type": "Organization",
                "description": "组织机构，如教育部、学校、公司等",
                "example": "教育部、北京市教委",
                "fields": "id, name, type, doc_id"
            },
            {
                "type": "Person",
                "description": "人物，如教育专家、领导、教师等",
                "example": "张三、李局长",
                "fields": "id, name, type, doc_id"
            },
            {
                "type": "Policy",
                "description": "政策文件或法规",
                "example": "《教育信息化2.0行动计划》",
                "fields": "id, name, type, doc_id"
            },
            {
                "type": "ThemeTag",
                "description": "主题标签，如教育信息化、课程改革等",
                "example": "教育信息化、人工智能教育",
                "fields": "id, name, type, doc_id"
            },
            {
                "type": "ProvinceTag",
                "description": "省份标签，如北京、上海、广东等",
                "example": "北京市、上海市、广东省",
                "fields": "id, name, type, doc_id"
            },
            {
                "type": "CityTag",
                "description": "城市标签",
                "example": "深圳市、杭州市",
                "fields": "id, name, type, doc_id"
            },
            {
                "type": "Location",
                "description": "地点或区域",
                "example": "中关村、浦东新区",
                "fields": "id, name, type, doc_id"
            },
            {
                "type": "Event",
                "description": "事件或活动",
                "example": "全国教育工作会议、高考改革",
                "fields": "id, name, type, doc_id"
            },
            {
                "type": "Technology",
                "description": "技术或平台",
                "example": "人工智能、在线教育平台",
                "fields": "id, name, type, doc_id"
            },
        ],
        edge_types=[
            {
                "type": "REL",
                "description": "通用关联关系（OneKE 实体提取）",
                "example": "发布、参与、涉及",
            },
            {
                "type": "CORRELATED_WITH",
                "description": "相似度关联（混合相似度计算）",
                "example": "新闻之间的相似度关联 (entity + vector)",
            },
        ],
        example_queries=[
            {
                "description": "查找所有教育机构",
                "cypher": "MATCH (n:Entity {type: 'Organization'}) RETURN n.name LIMIT 10"
            },
            {
                "description": "查找某机构相关的所有新闻",
                "cypher": "MATCH (o:Entity {name: '教育部'})-[:REL]-(n:Entity {type: 'NewsItem'}) RETURN n.name LIMIT 10"
            },
            {
                "description": "查找包含特定主题的新闻",
                "cypher": "MATCH (t:Entity {type: 'ThemeTag', name: '教育信息化'})-[:REL]-(n:Entity {type: 'NewsItem'}) RETURN n.name LIMIT 10"
            },
            {
                "description": "查找某省份的教育新闻",
                "cypher": "MATCH (p:Entity {type: 'ProvinceTag', name: '北京市'})-[:REL]-(n:Entity {type: 'NewsItem'}) RETURN n.name LIMIT 10"
            },
            {
                "description": "查找两个实体之间的关联路径",
                "cypher": "MATCH path = (a:Entity {name: '教育部'})-[:REL*1..3]-(b:Entity {name: '人工智能'}) RETURN path LIMIT 5"
            },
        ],
        stats=stats
    )


class CytoscapeNodeData(BaseModel):
    id: str
    label: str
    type: str
    doc_id: str


class CytoscapeEdgeData(BaseModel):
    id: str
    source: str
    target: str
    label: str
    type: str
    doc_id: str
    evidence: str | None = None


class CytoscapeNode(BaseModel):
    data: CytoscapeNodeData


class CytoscapeEdge(BaseModel):
    data: CytoscapeEdgeData


class CytoscapeElements(BaseModel):
    nodes: list[CytoscapeNode] = Field(default_factory=list)
    edges: list[CytoscapeEdge] = Field(default_factory=list)


class GraphResponse(BaseModel):
    elements: CytoscapeElements


@router.get("/graph", response_model=GraphResponse)
async def get_graph(
    request: Request,
    doc_id: str | None = Query(default=None, min_length=1),
    node_limit: int = Query(default=25, ge=1, le=200, description="Maximum nodes to return"),
) -> GraphResponse:
    """获取知识图谱数据

    Args:
        doc_id: 可选，指定文档ID获取特定图谱；不传则返回所有数据
        node_limit: 返回的最大节点数量，默认25
    """
    neo4j: Neo4jClient | None = request.app.state.neo4j
    if neo4j is None:
        return GraphResponse(elements=CytoscapeElements())

    try:
        if doc_id:
            nodes, edges = await neo4j.read_graph_by_doc_id(doc_id=doc_id)
        else:
            nodes, edges = await neo4j.read_all_graph(limit=node_limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return GraphResponse(
        elements=CytoscapeElements(
            nodes=[
                CytoscapeNode(
                    data=CytoscapeNodeData(
                        id=n.id,
                        label=n.name,
                        type=n.type,
                        doc_id=n.doc_id,
                    )
                )
                for n in nodes
            ],
            edges=[
                CytoscapeEdge(
                    data=CytoscapeEdgeData(
                        id=e.id,
                        source=e.source_id,
                        target=e.target_id,
                        label=e.label,
                        type=e.type,
                        doc_id=e.doc_id,
                        evidence=e.evidence,
                    )
                )
                for e in edges
            ],
        )
    )


class ProvinceDistributionItem(BaseModel):
    province: str
    news_count: int


class ProvinceStatsResponse(BaseModel):
    """省份统计响应"""
    total_provinces: int = Field(description="省份标签总数")
    news_with_province: int = Field(description="有省份标签的新闻数量")
    province_distribution: list[ProvinceDistributionItem] = Field(description="每个省份的新闻数量分布")


@router.get("/graph/provinces", response_model=ProvinceStatsResponse)
async def get_province_stats(request: Request) -> ProvinceStatsResponse:
    """获取省份统计信息

    返回图谱中省份标签的分布情况，包括每个省份关联的新闻数量。
    """
    neo4j: Neo4jClient | None = request.app.state.neo4j
    if neo4j is None:
        raise HTTPException(status_code=503, detail="Neo4j not available")

    try:
        result = await neo4j.get_province_stats()

        return ProvinceStatsResponse(
            total_provinces=result["total_provinces"],
            news_with_province=result["news_with_province"],
            province_distribution=[
                ProvinceDistributionItem(province=item["province"], news_count=item["news_count"])
                for item in result["province_distribution"]
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get province stats: {str(e)}")


class ThemeTagItem(BaseModel):
    """主题标签项"""
    theme: str = Field(..., description="主题名称")
    news_count: int = Field(..., description="关联新闻数量")


class ThemeTagsResponse(BaseModel):
    """主题标签列表响应"""
    themes: list[ThemeTagItem] = Field(default_factory=list, description="主题标签列表")
    total: int = Field(..., description="主题总数")


class ThemeGraphResponse(BaseModel):
    """主题相关图谱响应"""
    elements: CytoscapeElements
    theme: str = Field(..., description="查询的主题")
    related_news_count: int = Field(..., description="相关新闻数量")
    related_entities_count: int = Field(..., description="相关实体数量")


@router.get("/graph/themes", response_model=ThemeTagsResponse)
async def get_theme_tags(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500, description="Maximum number of themes to return"),
) -> ThemeTagsResponse:
    """获取所有主题标签及其关联新闻数量。

    Returns:
        主题标签列表，按关联新闻数量降序排列
    """
    neo4j: Neo4jClient | None = request.app.state.neo4j
    if neo4j is None:
        raise HTTPException(status_code=503, detail="Neo4j not available")

    try:
        themes = await neo4j.get_theme_tags(limit=limit)

        return ThemeTagsResponse(
            themes=[ThemeTagItem(theme=t["theme"], news_count=t["news_count"]) for t in themes],
            total=len(themes),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get theme tags: {str(e)}")


@router.get("/graph/theme/{theme}", response_model=ThemeGraphResponse)
async def get_graph_by_theme(
    request: Request,
    theme: str = Path(..., min_length=1, description="Theme tag name to filter by"),
    news_limit: int = Query(default=30, ge=1, le=200, description="Maximum news items to include"),
) -> ThemeGraphResponse:
    """获取与指定主题相关的所有节点和边。

    Args:
        theme: 主题标签名称，如"义务教育"
        news_limit: 最多包含的新闻数量，避免数据过大

    Returns:
        主题相关的图谱数据，包含新闻节点和所有关联实体
    """
    neo4j: Neo4jClient | None = request.app.state.neo4j
    if neo4j is None:
        raise HTTPException(status_code=503, detail="Neo4j not available")

    try:
        nodes, edges = await neo4j.get_nodes_by_theme(theme=theme, news_limit=news_limit)

        # 统计信息
        news_nodes = [n for n in nodes if n.type == "NewsItem"]
        entity_nodes = [n for n in nodes if n.type != "NewsItem"]

        return ThemeGraphResponse(
            elements=CytoscapeElements(
                nodes=[
                    CytoscapeNode(
                        data=CytoscapeNodeData(
                            id=n.id,
                            label=n.name,
                            type=n.type,
                            doc_id=n.doc_id,
                        )
                    )
                    for n in nodes
                ],
                edges=[
                    CytoscapeEdge(
                        data=CytoscapeEdgeData(
                            id=e.id,
                            source=e.source_id,
                            target=e.target_id,
                            label=e.label,
                            type=e.type,
                            doc_id=e.doc_id,
                            evidence=e.evidence,
                        )
                    )
                    for e in edges
                ],
            ),
            theme=theme,
            related_news_count=len(news_nodes),
            related_entities_count=len(entity_nodes),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get graph by theme: {str(e)}")
