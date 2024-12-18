import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from difflib import SequenceMatcher
import numpy as np

from src.pubtator import PubTatorAPI
from src.models.cerebras_inference import CerebrasInference, EntityInfo, RelationInfo

logger = logging.getLogger(__name__)

class KnowledgeGraphUpdater:
    def __init__(self, graph_path: str, entity_aliases_path: str, model: CerebrasInference):
        # Initialize empty data structs
        self.graph = {"nodes": {}, "edges": {}}
        self.entity_aliases = {}
        
        # Store paths for later saving
        self.graph_path = graph_path
        self.entity_aliases_path = entity_aliases_path
        
        # Load existing graph and aliases
        self.load_graph(graph_path)
        self.load_entity_aliases(entity_aliases_path)
        
        # Build name map
        self.name_to_id_map = self.build_name_map()

        # Initialize model and PubTator
        self.model = model
        self.pubtator_api = PubTatorAPI()

    def load_graph(self, path: str) -> None:
        """Load existing knowledge graph or create new if missing."""
        try:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                with open(path, 'r') as f:
                    self.graph = json.load(f)
                self.graph.setdefault("nodes", {})
                self.graph.setdefault("edges", {})
                logger.info(f"Successfully loaded knowledge graph from {path}")
            else:
                logger.info(f"No existing graph found at {path}, initializing new graph")
                os.makedirs(os.path.dirname(path), exist_ok=True)  # create directory if it doesn't exist
                self.save_graph()
        except json.JSONDecodeError as e:
            logger.error(f"Error reading knowledge graph file: {e}")
            logger.info("Initializing new graph")
            self.save_graph()
        except Exception as e:
            logger.error(f"Unexpected error loading knowledge graph: {e}")
            self.save_graph()
            
    def load_entity_aliases(self, path: str) -> None:
        """Load known entity aliases or create new if missing."""
        try:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                with open(path, 'r') as f:
                    self.entity_aliases = json.load(f)
                logger.info(f"Successfully loaded entity aliases from {path}")
            else:
                logger.info(f"No existing aliases found at {path}, initializing empty aliases")
                os.makedirs(os.path.dirname(path), exist_ok=True)    # create directory if it doesn't exist
                self.save_entity_aliases()
        except json.JSONDecodeError as e:
            logger.error(f"Error reading entity aliases file: {e}")
            logger.info("Initializing empty aliases")
            self.save_entity_aliases()
        except Exception as e:
            logger.error(f"Unexpected error loading entity aliases: {e}")
            self.save_entity_aliases()

    def save_graph(self) -> None:
        """Save the current state of the knowledge graph."""
        try:
            with open(self.graph_path, 'w') as f:
                json.dump(self.graph, f, indent=2)
            logger.info(f"Successfully saved knowledge graph to {self.graph_path}")
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {e}")

    def save_entity_aliases(self) -> None:
        """Save the current state of entity aliases."""
        try:
            with open(self.entity_aliases_path, 'w') as f:
                json.dump(self.entity_aliases, f, indent=2)
            logger.info(f"Successfully saved entity aliases to {self.entity_aliases_path}")
        except Exception as e:
            logger.error(f"Error saving entity aliases: {e}")

    def build_name_map(self) -> Dict:
        """Build mapping from all known names (including aliases) to node IDs."""
        name_map = {}
        for node_id, node_data in self.graph["nodes"].items():
            name_map[node_data["properties"]["primary_name"].lower()] = node_id
            for alt_name in node_data["properties"].get("alternative_names", []):
                name_map[alt_name.lower()] = node_id
        return name_map

    def find_matching_entity(self, entity: EntityInfo, threshold: float = 0.5) -> Optional[str]:
        name_lower = entity.name.lower()
        entity_type = entity.type

        # Exact match
        if name_lower in self.name_to_id_map:
            node_id = self.name_to_id_map[name_lower]
            node_data = self.graph["nodes"][node_id]
            if node_data["properties"]["entity_type"] == entity_type:
                logger.info(f"Exact match found for entity '{entity.name}' with node_id '{node_id}'")
                return node_id
            else:
                logger.warning(f"Type mismatch for entity '{entity.name}' (found type: {node_data['properties']['entity_type']})")

        # Fuzzy match
        candidate_entities = []
        for node_id, node_data in self.graph["nodes"].items():
            if node_data["properties"]["entity_type"] != entity_type:
                continue
            known_names = [node_data["properties"]["primary_name"].lower()] + \
                          [alt_name.lower() for alt_name in node_data["properties"].get("alternative_names", [])]
            for known_name in known_names:
                similarity = SequenceMatcher(None, name_lower, known_name).ratio()
                if similarity >= threshold:
                    candidate_entities.append(node_id)
                    break  # Avoid duplicates

        # If multiple candidates, disambiguate
        if candidate_entities:
            match_id = self.model.disambiguate_entity(entity, candidate_entities)
            if match_id:
                logger.info(f"Disambiguation matched '{entity.name}' to node_id '{match_id}'")
                return match_id

        logger.info(f"No match found for entity '{entity.name}'")
        return None

    def create_node(self, entity_info: Dict) -> str:
        # Final check for existing nodes before creation
        node_id = self.find_matching_entity(EntityInfo(**entity_info))
        if node_id:
            logger.info(f"Found a match during final check, skipping node creation for '{entity_info['name']}'")
            return node_id

        node_id = f"node_{len(self.graph['nodes'])}"
        self.graph["nodes"][node_id] = {
            "type": "string",
            "properties": {
                "entity_type": entity_info["type"],
                "primary_name": entity_info["name"],
                "alternative_names": [],
                "external_ids": entity_info.get("external_ids", {}),
                "description": entity_info.get("description", ""),
                "last_updated": datetime.now().isoformat(),
                "creation_date": datetime.now().isoformat()
            }
        }
        # Add to name mapping
        self.name_to_id_map[entity_info["name"].lower()] = node_id
        logger.info(f"Created new node '{node_id}' for entity '{entity_info['name']}'")
        return node_id

    def update_node(self, node_id: str, entity_info: EntityInfo) -> None:
        """Update an existing node with new information from entity_info."""
        node = self.graph["nodes"][node_id]
        properties = node["properties"]
        # Update description if the new one is longer
        if entity_info.description:
            if not properties.get("description") or len(entity_info.description) > len(properties["description"]):
                properties["description"] = entity_info.description

        # Update external_ids
        if entity_info.external_ids:
            existing_external_ids = properties.get("external_ids", {})
            existing_external_ids.update(entity_info.external_ids)
            properties["external_ids"] = existing_external_ids

        # Add alternative names
        alternative_names = properties.get("alternative_names", [])
        if entity_info.name != properties["primary_name"] and entity_info.name not in alternative_names:
            alternative_names.append(entity_info.name)
        properties["alternative_names"] = alternative_names

        # Update last_updated
        properties["last_updated"] = datetime.now().isoformat()
        # Update the name_to_id_map with the new alternative names
        self.name_to_id_map[entity_info.name.lower()] = node_id

    def create_update_edge(self, source_id: str, target_id: str, relation_info: Dict) -> str:
        """Create new edge or update existing one with new evidence."""
        # Create unique edge identifier
        edge_key = f"{source_id}_{target_id}_{relation_info['relationship_type']}"
        
        if edge_key not in self.graph["edges"]:
            # Create new edge
            self.graph["edges"][edge_key] = {
                "type": "string",
                "source_node": source_id,
                "target_node": target_id,
                "relationship_type": relation_info["relationship_type"],
                "evidence": [],
                "aggregated_metadata": {
                    "total_papers": 0,
                    "earliest_evidence": None,
                    "latest_evidence": None,
                    "evidence_strength": 0.0,
                    "contradictory_evidence": False
                },
                "last_updated": datetime.now().isoformat()
            }
        
        # add new evidence
        evidence = {
            "paper_id": relation_info["paper_id"],
            "citation_metadata": relation_info["citation_metadata"],
            "experimental_context": relation_info["experimental_context"],
            "statistical_evidence": relation_info.get("statistical_evidence", {}),
            "extracted_text": relation_info["extracted_text"],
            "extraction_confidence": relation_info["confidence"],
            "last_verified": datetime.now().isoformat()
        }
        
        # check for duplicate evidence
        if not self._is_duplicate_evidence(edge_key, evidence):
            self.graph["edges"][edge_key]["evidence"].append(evidence)
            self._update_edge_metadata(edge_key)
        
        return edge_key

    def _is_duplicate_evidence(self, edge_key: str, new_evidence: Dict) -> bool:
        """Check if this evidence is already recorded for the given edge."""
        existing_evidences = self.graph["edges"][edge_key]["evidence"]
        for ev in existing_evidences:
            if ev["paper_id"] == new_evidence["paper_id"]:
                return True
        return False

    def _update_edge_metadata(self, edge_key: str):
        """Update aggregated metadata for the given edge."""
        edge = self.graph["edges"][edge_key]
        evidences = edge["evidence"]
        
        years = [e["citation_metadata"]["year"] for e in evidences if e["citation_metadata"]["year"]]
        earliest = min(years) if years else None
        latest = max(years) if years else None
        
        edge["aggregated_metadata"].update({
            "total_papers": len(evidences),
            "earliest_evidence": earliest,
            "latest_evidence": latest,
            "evidence_strength": float(np.mean([e["extraction_confidence"] for e in evidences])),
            "last_updated": datetime.now().isoformat()
        })

    def process_abstract(self, abstract_info: Dict) -> List[Dict]:
        try:
            # 1: Extract entities and relationships with LLM
            entities, relations = self.model.process_abstract(abstract_info)
            logger.info(f"Extracted {len(entities)} entities and {len(relations)} relationships.")

            # 2: PubTator normalization
            for ent in entities:
                logger.debug(f"Looking up PubTator ID for entity: {ent.name}")
                try:
                    entity_ids = self.pubtator_api.find_entity_id(ent.name)
                    if entity_ids:
                        ent.external_ids = ent.external_ids or {}
                        ent.external_ids["PubTatorID"] = entity_ids[0]
                        print()
                        logger.debug(f"Found PubTator ID {entity_ids[0]} for {ent.name}")
                except Exception as e:
                    logger.warning(f"Failed to fetch PubTator ID for {ent.name}: {e}")
            
            # Validate Relationships (commented out/optional for now)
            # for relation in relations:
            #     source_id = relation.source_entity.external_ids.get("PubTatorID")
            #     target_id = relation.target_entity.external_ids.get("PubTatorID")
            #     if source_id and target_id:
            #         # Validate using PubTator relations
            #         pass

            updates = []
            for relation in relations:
                # process source entity
                source_entity = relation.source_entity
                source_id = self.find_matching_entity(source_entity)
                if source_id:
                    self.update_node(source_id, source_entity)
                else:
                    source_id = self.create_node(source_entity.__dict__)

                # process entity in question
                target_entity = relation.target_entity
                target_id = self.find_matching_entity(target_entity)
                if target_id:
                    self.update_node(target_id, target_entity)
                else:
                    target_id = self.create_node(target_entity.__dict__)

                # create or update the edge
                print('################')
                logger.debug(f"abstract_info: {abstract_info} (type: {type(abstract_info)})")
                print(abstract_info, type(abstract_info))
                print('################')

                edge_id = self.create_update_edge(source_id, target_id, {
                    "relationship_type": relation.relationship_type,
                    "paper_id": abstract_info["pmid"],
                    "citation_metadata": {
                        "title": abstract_info["title"],
                        "authors": abstract_info["authors"],
                        "journal": abstract_info["journal"],
                        "year": abstract_info["year"]
                    },
                    "experimental_context": relation.context,
                    "extracted_text": relation.supporting_text,
                    "confidence": relation.confidence
                })

                updates.append({
                    "edge_id": edge_id,
                    "source_id": source_id,
                    "target_id": target_id,
                    "action": "updated" if edge_id in self.graph["edges"] else "created"
                })

            return updates

        except Exception as e:
            print('################')
            logger.debug(f"abstract_info: {abstract_info} (type: {type(abstract_info)})")
            print(abstract_info, type(abstract_info))
            print('################')
            logger.error(f"Error processing abstract {abstract_info.get('pmid', 'unknown')}: {e}")
            raise