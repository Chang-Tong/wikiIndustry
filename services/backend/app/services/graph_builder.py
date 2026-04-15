"""Graph builder for constructing Neo4j graph from extraction results."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.domain.extraction.models import ExtractionResult
from app.integrations.neo4j.client import GraphNode, GraphEdge


class GraphBuilder:
    """Builds Neo4j graph from extraction results."""

    def __init__(self) -> None:
        """Initialize graph builder."""
        self._node_cache: dict[str, GraphNode] = {}

    def build_from_extraction(
        self,
        extraction: ExtractionResult,
        doc_id: str,
        metadata: dict[str, Any],
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Build graph nodes and edges from extraction result.

        Args:
            extraction: Entity/relation extraction result
            doc_id: Document ID
            metadata: News metadata (title, site, date, etc.)

        Returns:
            Tuple of (nodes, edges)
        """
        self._node_cache = {}
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []

        # Build entity name -> type mapping from extracted entities
        entity_type_map: dict[str, str] = {
            entity.name: entity.type for entity in extraction.entities
        }

        # Create central NewsItem node
        news_node = self._create_news_node(doc_id, metadata)
        nodes.append(news_node)
        self._node_cache[news_node.id] = news_node

        # Create entity nodes
        for entity in extraction.entities:
            node = self._create_entity_node(
                name=entity.name,
                node_type=entity.type,
                doc_id=doc_id,
            )
            if node.id not in self._node_cache:
                nodes.append(node)
                self._node_cache[node.id] = node

            # Create relationship from news to entity (提及)
            edge = self._create_edge(
                source_id=news_node.id,
                target_id=node.id,
                rel_type="提及",
                label="提及",
                doc_id=doc_id,
                evidence=f"来自: {metadata.get('title', '')}",
            )
            edges.append(edge)

        # Create explicit relations
        for relation in extraction.relations:
            # Look up entity types from the extraction result
            source_type = entity_type_map.get(relation.source, "Entity")
            target_type = entity_type_map.get(relation.target, "Entity")

            source_node = self._find_or_create_entity(
                name=relation.source,
                doc_id=doc_id,
                nodes=nodes,
                default_type=source_type,
            )
            target_node = self._find_or_create_entity(
                name=relation.target,
                doc_id=doc_id,
                nodes=nodes,
                default_type=target_type,
            )

            edge = self._create_edge(
                source_id=source_node.id,
                target_id=target_node.id,
                rel_type=relation.type,
                label=relation.type,
                doc_id=doc_id,
                evidence=relation.evidence or metadata.get("title", ""),
            )
            edges.append(edge)

        # Add structured metadata as additional nodes and edges
        metadata_edges = self._build_metadata_edges(
            news_node=news_node,
            metadata=metadata,
            doc_id=doc_id,
            nodes=nodes,
        )
        edges.extend(metadata_edges)

        return nodes, edges

    def _create_news_node(self, doc_id: str, metadata: dict[str, Any]) -> GraphNode:
        """Create central NewsItem node."""
        title = metadata.get("title", "未命名")
        return GraphNode(
            id=doc_id,
            name=title,
            type="NewsItem",
            doc_id=doc_id,
        )

    def _create_entity_node(
        self,
        name: str,
        node_type: str,
        doc_id: str,
    ) -> GraphNode:
        """Create entity node with stable ID."""
        # Map OneKE types to Neo4j types
        mapped_type = self._map_entity_type(node_type)

        # Generate stable ID globally by type+name so same entity is shared across docs
        node_id = self._stable_id(mapped_type, name)

        return GraphNode(
            id=node_id,
            name=name,
            type=mapped_type,
            doc_id=doc_id,
        )

    def _find_or_create_entity(
        self,
        name: str,
        doc_id: str,
        nodes: list[GraphNode],
        default_type: str = "Entity",
    ) -> GraphNode:
        """Find existing entity or create new one."""
        # Try to find by name in existing nodes
        for i, node in enumerate(nodes):
            if node.name == name:
                # Upgrade generic types to more specific ones if provided
                generic_types = {"Entity", "Location", "Other"}
                if node.type in generic_types and default_type not in generic_types:
                    # Create a new node with the updated type (since GraphNode is frozen)
                    new_node = GraphNode(
                        id=node.id,
                        name=node.name,
                        type=default_type,
                        doc_id=node.doc_id,
                    )
                    nodes[i] = new_node
                    self._node_cache[new_node.id] = new_node
                    return new_node
                return node

        # Create new entity node
        node = self._create_entity_node(
            name=name,
            node_type=default_type,
            doc_id=doc_id,
        )
        nodes.append(node)
        self._node_cache[node.id] = node
        return node

    def _create_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        label: str,
        doc_id: str,
        evidence: str | None = None,
    ) -> GraphEdge:
        """Create graph edge."""
        edge_id = self._stable_id(doc_id, "rel", source_id, target_id, rel_type)

        return GraphEdge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            type=rel_type,
            label=label,
            doc_id=doc_id,
            evidence=evidence,
        )

    def _build_metadata_edges(
        self,
        news_node: GraphNode,
        metadata: dict[str, Any],
        doc_id: str,
        nodes: list[GraphNode],
    ) -> list[GraphEdge]:
        """Build edges from metadata."""
        edges: list[GraphEdge] = []

        # Site/Organization
        site = metadata.get("site", "")
        if site:
            site_node = self._find_or_create_entity(
                name=site,
                doc_id=doc_id,
                nodes=nodes,
                default_type="Organization",
            )
            edges.append(
                self._create_edge(
                    source_id=news_node.id,
                    target_id=site_node.id,
                    rel_type="来自",
                    label="来自",
                    doc_id=doc_id,
                    evidence=f"来源: {site}",
                )
            )

        # Province tag
        province = metadata.get("province", "")
        if province:
            province_node = self._find_or_create_entity(
                name=province,
                doc_id=doc_id,
                nodes=nodes,
                default_type="ProvinceTag",
            )
            edges.append(
                self._create_edge(
                    source_id=news_node.id,
                    target_id=province_node.id,
                    rel_type="省份标签",
                    label="省份标签",
                    doc_id=doc_id,
                    evidence=f"涉及省份: {province}",
                )
            )

        # Channel/Category
        channel = metadata.get("channel", "")
        if channel:
            channel_node = self._find_or_create_entity(
                name=channel,
                doc_id=doc_id,
                nodes=nodes,
                default_type="Category",
            )
            edges.append(
                self._create_edge(
                    source_id=news_node.id,
                    target_id=channel_node.id,
                    rel_type="一级分类",
                    label="一级分类",
                    doc_id=doc_id,
                    evidence=f"栏目: {channel}",
                )
            )

        # Tag
        tag = metadata.get("tag", "")
        if tag:
            tag_node = self._find_or_create_entity(
                name=tag,
                doc_id=doc_id,
                nodes=nodes,
                default_type="ThemeTag",
            )
            edges.append(
                self._create_edge(
                    source_id=news_node.id,
                    target_id=tag_node.id,
                    rel_type="主题标签",
                    label="主题标签",
                    doc_id=doc_id,
                    evidence=f"标签: {tag}",
                )
            )

        # Date
        date = metadata.get("date", "")
        if date:
            date_node = self._find_or_create_entity(
                name=date,
                doc_id=doc_id,
                nodes=nodes,
                default_type="Time",
            )
            edges.append(
                self._create_edge(
                    source_id=news_node.id,
                    target_id=date_node.id,
                    rel_type="日期",
                    label="日期",
                    doc_id=doc_id,
                    evidence=f"日期: {date}",
                )
            )

        # URL
        url = metadata.get("url", "")
        if url:
            url_node = self._find_or_create_entity(
                name=url[:100],  # Truncate long URLs
                doc_id=doc_id,
                nodes=nodes,
                default_type="URL",
            )
            edges.append(
                self._create_edge(
                    source_id=news_node.id,
                    target_id=url_node.id,
                    rel_type="网址",
                    label="网址",
                    doc_id=doc_id,
                    evidence="链接",
                )
            )

        return edges

    def _map_entity_type(self, oneke_type: str) -> str:
        """Map OneKE entity type to Neo4j node type.

        已知类型做标准化映射，未知类型直接透传保留 OneKE 原始标签，
        避免丢失语义信息。
        """
        mapping: dict[str, str] = {
            "Organization": "Organization",
            "Person": "Person",
            "Policy": "Policy",
            "Project": "Project",
            "Time": "Time",
            "Location": "Location",
            "Province": "ProvinceTag",
            "City": "CityTag",
            "Industry": "IndustryTag",
            "Theme": "ThemeTag",
            "Meeting": "Event",
            "Technology": "Technology",
            "Device": "Technology",
            "Other": "Entity",
        }
        return mapping.get(oneke_type, oneke_type)

    def _stable_id(self, *parts: str) -> str:
        """Generate stable ID from parts."""
        content = json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(content).hexdigest()[:32]
