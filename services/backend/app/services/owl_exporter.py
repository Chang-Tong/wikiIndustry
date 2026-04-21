"""OWL/RDF export service for knowledge graph."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

from app.integrations.neo4j.client import Neo4jClient

logger = logging.getLogger(__name__)

# OWL/RDF Namespaces
NAMESPACES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "kg": "http://example.org/knowledge-graph#",
    "xml": "http://www.w3.org/XML/1998/namespace",
}


@dataclass
class OWLClass:
    """OWL Class definition."""

    name: str
    label: str
    comment: str
    parent: str | None = None


@dataclass
class OWLProperty:
    """OWL Property definition."""

    name: str
    label: str
    domain: str
    range: str
    property_type: str = "object"  # object or datatype


@dataclass
class OWLIndividual:
    """OWL Individual instance."""

    id: str
    type: str
    label: str
    properties: dict[str, Any]


class OWLExporter:
    """Export Neo4j graph to OWL/RDF format."""

    # Entity type to OWL class mapping
    TYPE_CLASS_MAP: dict[str, str] = {
        "NewsItem": "NewsArticle",
        "Organization": "Organization",
        "Person": "Person",
        "Location": "Place",
        "ProvinceTag": "Province",
        "CityTag": "City",
        "ThemeTag": "Theme",
        "IndustryTag": "Industry",
        "Category": "Category",
        "Time": "Time",
        "URL": "URL",
        "Entity": "Entity",
    }

    # Relationship type to OWL property mapping
    REL_PROPERTY_MAP: dict[str, str] = {
        "提及": "mentions",
        "来自": "source",
        "一级分类": "category",
        "主题标签": "hasTheme",
        "日期": "date",
        "网址": "url",
        "相关部门": "relatedDepartment",
        "省份标签": "province",
        "地市标签": "city",
        "行业标签": "industry",
        "内容类型": "contentType",
    }

    def __init__(self, neo4j: Neo4jClient) -> None:
        """Initialize with Neo4j client.

        Args:
            neo4j: Neo4j client instance
        """
        self.neo4j = neo4j

    async def export_to_owl(
        self,
        doc_id: str | None = None,
        include_individuals: bool = True,
    ) -> str:
        """Export graph to OWL/XML format.

        Args:
            doc_id: Optional specific document to export
            include_individuals: Whether to include instance data

        Returns:
            OWL/XML string
        """
        logger.info(f"Exporting to OWL: doc_id={doc_id}, include_individuals={include_individuals}")

        # Create root element
        root = ET.Element("rdf:RDF")

        # Add namespace declarations
        for prefix, uri in NAMESPACES.items():
            root.set(f"xmlns:{prefix}", uri)

        # Add ontology header
        ontology = ET.SubElement(root, "owl:Ontology")
        ontology.set(f"{{{NAMESPACES['rdf']}}}about", "http://example.org/knowledge-graph")

        # Add ontology metadata
        label = ET.SubElement(ontology, "rdfs:label")
        label.text = "Knowledge Graph Ontology"

        comment = ET.SubElement(ontology, "rdfs:comment")
        comment.text = "Auto-generated ontology from Neo4j knowledge graph"

        # Add class definitions
        self._add_class_definitions(root)

        # Add property definitions
        self._add_property_definitions(root)

        # Add individuals if requested
        if include_individuals:
            await self._add_individuals(root, doc_id)

        # Convert to string
        ET.indent(root, space="  ")
        xml_string = ET.tostring(root, encoding="unicode")

        # Add XML declaration
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_string

    def _add_class_definitions(self, root: ET.Element) -> None:
        """Add OWL class definitions."""
        classes = [
            OWLClass("NewsArticle", "新闻文章", "一条新闻或文章"),
            OWLClass("Organization", "组织机构", "政府部门、企业、学校等组织"),
            OWLClass("Person", "人物", "相关人物"),
            OWLClass("Place", "地点", "地理位置"),
            OWLClass("Province", "省份", "省级行政区", "Place"),
            OWLClass("City", "城市", "城市或区县", "Place"),
            OWLClass("Theme", "主题", "新闻主题标签"),
            OWLClass("Industry", "行业", "行业分类"),
            OWLClass("Category", "分类", "内容分类"),
            OWLClass("Time", "时间", "时间信息"),
            OWLClass("URL", "链接", "网络链接"),
            OWLClass("Entity", "实体", "通用实体"),
        ]

        for cls in classes:
            class_elem = ET.SubElement(root, "owl:Class")
            class_elem.set(f"{{{NAMESPACES['rdf']}}}ID", cls.name)

            label_elem = ET.SubElement(class_elem, "rdfs:label")
            label_elem.set(f"{{{NAMESPACES['xml']}}}lang", "zh")
            label_elem.text = cls.label

            comment_elem = ET.SubElement(class_elem, "rdfs:comment")
            comment_elem.text = cls.comment

            if cls.parent:
                parent_elem = ET.SubElement(class_elem, "rdfs:subClassOf")
                parent_elem.set(f"{{{NAMESPACES['rdf']}}}resource", f"#{cls.parent}")

    def _add_property_definitions(self, root: ET.Element) -> None:
        """Add OWL property definitions."""
        properties = [
            # Object properties
            OWLProperty("mentions", "提及", "NewsArticle", "Entity"),
            OWLProperty("source", "来源", "NewsArticle", "Organization"),
            OWLProperty("category", "分类", "NewsArticle", "Category"),
            OWLProperty("hasTheme", "主题", "NewsArticle", "Theme"),
            OWLProperty("province", "省份", "NewsArticle", "Province"),
            OWLProperty("city", "城市", "NewsArticle", "City"),
            OWLProperty("industry", "行业", "NewsArticle", "Industry"),
            OWLProperty("relatedDepartment", "相关部门", "NewsArticle", "Organization"),
            OWLProperty("correlatedWith", "关联", "NewsArticle", "NewsArticle"),
            # Datatype properties
            OWLProperty("date", "日期", "NewsArticle", "xsd:string", "datatype"),
            OWLProperty("url", "链接", "NewsArticle", "xsd:anyURI", "datatype"),
            OWLProperty("docId", "文档ID", "NewsArticle", "xsd:string", "datatype"),
        ]

        for prop in properties:
            if prop.property_type == "datatype":
                prop_elem = ET.SubElement(root, "owl:DatatypeProperty")
            else:
                prop_elem = ET.SubElement(root, "owl:ObjectProperty")

            prop_elem.set(f"{{{NAMESPACES['rdf']}}}ID", prop.name)

            label_elem = ET.SubElement(prop_elem, "rdfs:label")
            label_elem.set(f"{{{NAMESPACES['xml']}}}lang", "zh")
            label_elem.text = prop.label

            domain_elem = ET.SubElement(prop_elem, "rdfs:domain")
            domain_elem.set(f"{{{NAMESPACES['rdf']}}}resource", f"#{prop.domain}")

            range_elem = ET.SubElement(prop_elem, "rdfs:range")
            if prop.property_type == "datatype":
                range_elem.set(f"{{{NAMESPACES['rdf']}}}resource", prop.range)
            else:
                range_elem.set(f"{{{NAMESPACES['rdf']}}}resource", f"#{prop.range}")

    async def _add_individuals(
        self,
        root: ET.Element,
        doc_id: str | None = None,
    ) -> None:
        """Add OWL individuals from graph data."""
        driver = self.neo4j._driver
        if not driver:
            raise RuntimeError("Neo4j driver not initialized")

        async with driver.session() as session:
            # Get all entities
            if doc_id:
                entity_query = """
                MATCH (e:Entity {doc_id: $doc_id})
                RETURN e.id as id, e.name as name, e.type as type, e.doc_id as doc_id
                """
                entity_result = await session.run(entity_query, doc_id=doc_id)
            else:
                entity_query = """
                MATCH (e:Entity)
                RETURN e.id as id, e.name as name, e.type as type, e.doc_id as doc_id
                LIMIT 1000
                """
                entity_result = await session.run(entity_query)

            entities: dict[str, dict[str, Any]] = {}
            async for record in entity_result:
                entity_id = record["id"]
                entities[entity_id] = {
                    "id": entity_id,
                    "name": record["name"],
                    "type": record["type"],
                    "doc_id": record["doc_id"],
                }

            # Create individuals
            for entity_id, entity in entities.items():
                owl_class = self.TYPE_CLASS_MAP.get(entity["type"], "Entity")

                individual = ET.SubElement(root, f"{owl_class}")
                individual.set(f"{{{NAMESPACES['rdf']}}}ID", self._safe_id(entity_id))

                # Add type assertion
                type_elem = ET.SubElement(individual, "rdf:type")
                type_elem.set(f"{{{NAMESPACES['rdf']}}}resource", f"#{owl_class}")

                # Add label
                label_elem = ET.SubElement(individual, "rdfs:label")
                label_elem.text = entity["name"]

                # Add doc_id property
                docid_elem = ET.SubElement(individual, "docId")
                docid_elem.text = str(entity["doc_id"])

            # Get all relationships
            if doc_id:
                rel_query = """
                MATCH (s:Entity {doc_id: $doc_id})-[r:REL]->(t:Entity {doc_id: $doc_id})
                RETURN s.id as source_id, t.id as target_id,
                       r.type as rel_type, r.label as rel_label
                """
                rel_result = await session.run(rel_query, doc_id=doc_id)
            else:
                rel_query = """
                MATCH (s:Entity)-[r:REL]->(t:Entity)
                RETURN s.id as source_id, t.id as target_id,
                       r.type as rel_type, r.label as rel_label
                LIMIT 500
                """
                rel_result = await session.run(rel_query)

            # Add relationships as property assertions
            async for record in rel_result:
                source_id = record["source_id"]
                target_id = record["target_id"]
                rel_type = record["rel_type"]

                if source_id not in entities:
                    continue

                prop_name = self.REL_PROPERTY_MAP.get(rel_type, "relatedTo")

                # Find the source individual and add property
                source_class = self.TYPE_CLASS_MAP.get(
                    entities[source_id]["type"], "Entity"
                )

                # Create a simplified relationship representation
                # In practice, you'd look up the existing individual element
                rel_elem = ET.SubElement(root, f"{source_class}")
                rel_elem.set(f"{{{NAMESPACES['rdf']}}}about", f"#{self._safe_id(source_id)}")

                prop_elem = ET.SubElement(rel_elem, f"{prop_name}")
                prop_elem.set(f"{{{NAMESPACES['rdf']}}}resource", f"#{self._safe_id(target_id)}")

    def _safe_id(self, entity_id: str) -> str:
        """Convert entity ID to safe OWL ID."""
        # Remove or replace problematic characters
        safe = entity_id.replace(":", "_").replace("/", "_").replace("#", "_")
        return f"id_{safe[:50]}"  # Limit length

    async def export_to_turtle(
        self,
        doc_id: str | None = None,
    ) -> str:
        """Export graph to Turtle format (more readable).

        Args:
            doc_id: Optional specific document to export

        Returns:
            Turtle format string
        """
        logger.info(f"Exporting to Turtle: doc_id={doc_id}")

        lines = [
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            "@prefix kg: <http://example.org/knowledge-graph#> .",
            "",
            "kg:Ontology rdf:type owl:Ontology ;",
            '    rdfs:label "Knowledge Graph Ontology"@zh ;',
            '    rdfs:comment "Auto-generated from Neo4j" .',
            "",
        ]

        # Add class definitions
        classes = [
            ("kg:NewsArticle", "新闻文章", "News"),
            ("kg:Organization", "组织机构", "Organization"),
            ("kg:Person", "人物", "Person"),
            ("kg:Place", "地点", "Place"),
            ("kg:Theme", "主题", "Theme"),
            ("kg:Industry", "行业", "Industry"),
        ]

        for cls, label, parent in classes:
            lines.extend([
                f"{cls} rdf:type owl:Class ;",
                f'    rdfs:label "{label}"@zh ;',
                f"    rdfs:subClassOf {parent} .",
                "",
            ])

        driver = self.neo4j._driver
        if driver:
            async with driver.session() as session:
                # Get entities
                if doc_id:
                    query = "MATCH (e:Entity {doc_id: $doc_id}) RETURN e LIMIT 500"
                    result = await session.run(query, doc_id=doc_id)
                else:
                    query = "MATCH (e:Entity) RETURN e LIMIT 500"
                    result = await session.run(query)

                async for record in result:
                    entity = record["e"]
                    entity_id = entity.get("id", "")
                    name = entity.get("name", "").replace('"', '\\"')
                    entity_type = entity.get("type", "Entity")

                    owl_class = self.TYPE_CLASS_MAP.get(entity_type, "Entity")
                    safe_id = self._safe_id(entity_id)

                    lines.extend([
                        f"kg:{safe_id} rdf:type kg:{owl_class} ;",
                        f'    rdfs:label "{name}"@zh ;',
                        f'    kg:docId "{entity.get("doc_id", "")}" .',
                        "",
                    ])

        return "\n".join(lines)

    async def get_export_stats(self) -> dict[str, Any]:
        """Get statistics about the exportable graph.

        Returns:
            Stats dict
        """
        driver = self.neo4j._driver
        if not driver:
            return {"error": "Neo4j not connected"}

        node_types: dict[str, int] = {}
        relation_types: dict[str, int] = {}
        stats: dict[str, Any] = {
            "total_nodes": 0,
            "total_edges": 0,
            "node_types": node_types,
            "relation_types": relation_types,
        }

        async with driver.session() as session:
            # Count nodes
            result = await session.run("MATCH (e:Entity) RETURN count(e) as cnt")
            record = await result.single()
            stats["total_nodes"] = record["cnt"] if record else 0

            # Count edges
            result = await session.run("MATCH ()-[r:REL]->() RETURN count(r) as cnt")
            record = await result.single()
            stats["total_edges"] = record["cnt"] if record else 0

            # Node type distribution
            result = await session.run(
                "MATCH (e:Entity) RETURN e.type as type, count(*) as cnt ORDER BY cnt DESC"
            )
            async for record in result:
                stats["node_types"][record["type"]] = record["cnt"]

            # Relation type distribution
            result = await session.run(
                "MATCH ()-[r:REL]->() RETURN r.type as type, count(*) as cnt ORDER BY cnt DESC LIMIT 20"
            )
            async for record in result:
                stats["relation_types"][record["type"]] = record["cnt"]

        return stats
