import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import jsonschema
import time
from difflib import SequenceMatcher

from cerebras.cloud.sdk import Cerebras

logger = logging.getLogger(__name__)

@dataclass
class EntityInfo:
    name: str
    type: str
    description: Optional[str] = None
    external_ids: Optional[Dict[str, str]] = None

@dataclass
class RelationInfo:
    source_entity: EntityInfo
    target_entity: EntityInfo
    relationship_type: str
    context: Dict
    supporting_text: str
    confidence: float

SYSTEM_PROMPT = """You are an expert biomedical knowledge extractor. Your task is to analyze scientific abstracts 
and extract exclusively biomedical entities and their relationships of the designated types only. Follow these rules strictly:

1. Entity Types: GENE, PROTEIN, DISEASE, CHEMICAL, GENETIC VARIANT (Protein Mutation and DNA Mutation, SNP), SPECIES
2. Relationship Types: ASSOCIATE, CAUSE, COMPARE, COTREAT, DRUG_INTERACT, INHIBIT, INTERACT, NEGATIVE_CORRELATE, POSITIVE_CORRELATE, PREVENT, STIMULATE, TREAT, SUBTYPE
3. Format all output as valid JSON
4. Include confidence scores (0-1) for each relation extraction
5. Extract experimental context (study type, model system, methods)
6. Include specific supporting text for each relationship
7. Be precise with entity names and types
8. Do not infer relationships not stated in the abstract
9. Include any available entity identifiers (UMLS, etc.) that you have found from external sources and specify

Output must be in this exact format:
{
    "entities": [
        {
            "name": "entity_name",
            "type": "entity_type",
            "description": "brief description",
            "external_ids": {"system": "id"}
        }
    ],
    "relations": [
        {
            "source_entity": {entity object},
            "target_entity": {entity object},
            "relationship_type": "type",
            "context": {
                "study_type": "type",
                "model_system": {"type": "system", "details": "details"},
                "methods": ["method1", "method2"]
            },
            "supporting_text": "exact text from abstract",
            "confidence": 0.95
        }
    ]
}
"""

class CerebrasInference:
    def __init__(self, model: str, api_key=None):
        self.model = model
        self.api_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        if not self.api_key:
            logger.error("CEREBRAS_API_KEY not set.")
            raise ValueError("CEREBRAS_API_KEY required.")
        self.client = Cerebras(api_key=self.api_key)
        logger.info(f"CerebrasInference initialized with model: {self.model}")

        # Define JSON schemas for validation
        self.entity_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string"},
                "description": {"type": "string"},
                "external_ids": {
                    "type": "object",
                    "additional_properties": {"type": "string"}
                }
            },
            "required": ["name", "type"]
        }

        self.relation_schema = {
            "type": "object",
            "properties": {
                "source_entity": {"type": "object"},
                "target_entity": {"type": "object"},
                "relationship_type": {"type": "string"},
                "context": {"type": "object"},
                "supporting_text": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["source_entity", "target_entity", "relationship_type", 
                         "context", "supporting_text", "confidence"]
        }

        # Log file path for API calls
        self.api_log_path = "./logs/api_calls_log.ndjson"
        os.makedirs(os.path.dirname(self.api_log_path), exist_ok=True)

    def chat_completion(self, messages: List[Dict]) -> str:
        """
        Sends a list of messages to the Cerebras LLM and returns the response content.

        Args:
            messages (list): List of message dictionaries with 'role' and 'content'.

        Returns:
            str: The content of the LLM's response.
        """
        try:
            # temperature and max_tokens can vary
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.3,
                max_tokens=2000
            )
            content = response.choices[0].message.content.strip()
            logger.debug(f"Cerebras LLM response: {content}")
            return content
        except Exception as e:
            logger.error(f"Cerebras API call failed: {e}")
            raise

    def _log_api_response(self, response_content: str, abstract_info: Dict, start_time: float, messages: List[Dict], fix_attempt: bool = False, previous_extraction: Dict = None) -> None:
        """Log API response to an NDJSON file."""
        end_time = time.time()
        duration = end_time - start_time
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "abstract_info": abstract_info,
            "model": self.model,
            "messages": messages,
            "api_response": response_content
        }
        if fix_attempt:
            log_entry["fix_attempt"] = True
            log_entry["previous_extraction"] = previous_extraction

        # Append log entry as a line in NDJSON format
        try:
            with open(self.api_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")
            logger.info(f"API response logged to {self.api_log_path}")
        except Exception as e:
            logger.error(f"Failed to log API response: {e}")

    def _validate_entity(self, entity: Dict) -> bool:
        """Validate entity against schema."""
        try:
            jsonschema.validate(instance=entity, schema=self.entity_schema)
            return True
        except jsonschema.exceptions.ValidationError as e:
            logger.warning(f"Entity validation failed: {e}")
            return False

    def _validate_relation(self, relation: Dict) -> bool:
        """Validate relation against schema."""
        try:
            jsonschema.validate(instance=relation, schema=self.relation_schema)
            return True
        except jsonschema.exceptions.ValidationError as e:
            logger.warning(f"Relation validation failed: {e}")
            return False

    def _fix_extraction(self, extraction: Dict, abstract_info: Dict) -> Dict:
        fix_prompt = f"""The previous extraction was invalid. Please fix this extraction to match the required format:

        Previous extraction: {json.dumps(extraction, indent=2)}
        
        Original abstract:
        Title: {abstract_info['title']}
        Abstract: {abstract_info['abstract']}

        Ensure all entities and relations follow the exact schema specified.
        """

        start_time = time.time()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": fix_prompt}
        ]
        response_content = self.chat_completion(messages)
        self._log_api_response(response_content, abstract_info, start_time, messages, fix_attempt=True, previous_extraction=extraction)

        try:
            fixed_result = json.loads(response_content.strip())
            return fixed_result
        except json.JSONDecodeError:
            raise ValueError("Unable to fix extraction format")

    def process_abstract(self, abstract_info: Dict) -> Tuple[List[EntityInfo], List[RelationInfo]]:
        logger.info(f"Processing abstract PMID: {abstract_info.get('pmid', 'N/A')}")
        
        user_prompt = f"""Analyze this biomedical abstract and extract biomedical entities and their relationships:

        Title: {abstract_info['title']}
        Abstract: {abstract_info['abstract']}
        Journal: {abstract_info['journal']}
        Year: {abstract_info['year']}

        Provide all entities and relationships found in the exact JSON format specified."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        # API Call to get response
        start_time = time.time()
        response_content = self.chat_completion(messages)
        self._log_api_response(response_content, abstract_info, start_time, messages)

        # Parse response
        try:
            # Attempt to extract JSON block if wrapped in code fences
            json_block_pattern = re.compile(r'```json\s*(\{.*?\})\s*```', re.DOTALL)
            match = json_block_pattern.search(response_content)
            if match:
                json_str = match.group(1)
            else:
                json_str = response_content
            extraction = json.loads(json_str.strip())
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            raise ValueError("Invalid JSON response from LLM")

        # Validate extracted data
        attempts = 0
        while attempts < 3:
            entities_valid = all(self._validate_entity(ent) for ent in extraction.get("entities", []))
            relations_valid = all(self._validate_relation(rel) for rel in extraction.get("relations", []))

            if entities_valid and relations_valid:
                break
            attempts += 1
            extraction = self._fix_extraction(extraction, abstract_info)

        if attempts == 3:
            if not entities_valid:
                raise ValueError(f"Validation failed: Unable to validate entities after {len(attempts)} attempts")
            elif not relations_valid:
                raise ValueError(f"Validation failed: Unable to validate relations after {len(attempts)} attempts")

        # Convert to dataclass instances
        entities_info = [
            EntityInfo(
                name=e['name'],
                type=e['type'],
                description=e.get('description'),
                external_ids=e.get('external_ids')
            ) for e in extraction.get("entities", [])
        ]

        relations_info = [
            RelationInfo(
                source_entity=EntityInfo(
                    name=r['source_entity']['name'],
                    type=r['source_entity']['type'],
                    description=r['source_entity'].get('description'),
                    external_ids=r['source_entity'].get('external_ids')
                ),
                target_entity=EntityInfo(
                    name=r['target_entity']['name'],
                    type=r['target_entity']['type'],
                    description=r['target_entity'].get('description'),
                    external_ids=r['target_entity'].get('external_ids')
                ),
                relationship_type=r['relationship_type'],
                context=r['context'],
                supporting_text=r['supporting_text'],
                confidence=r['confidence']
            ) for r in extraction.get("relations", [])
        ]

        return entities_info, relations_info

    def disambiguate_entity(self, new_entity: EntityInfo, candidate_entities: List[Dict]) -> Optional[str]:
        """
        Determine if the new entity matches any of the candidate entities.
        Return the entity_id of the matching entity, or None if no match.
        """
        logger.info(f"Disambiguating entity: {new_entity.name} using Cerebras Inference")

        prompt = """You are an expert biomedical entity resolver. 
        Given a new entity and a list of candidate existing entities, determine if the new entity matches any of the candidates.

        Match criteria: The entities should refer to the same real-world biomedical concept. Consider name, type, description, external IDs.

        If there's a match, return the 'entity_id' of the matching entity. 
        If no match, return "No Match".

        Return answer in JSON:
        {"match": "entity_id"} or {"match": "No Match"}
        """
        
        user_message = {
            "role": "user",
            "content": prompt.format(
                new_entity=json.dumps(new_entity.__dict__, indent=2),
                candidate_entities="\n".join([
                    f"{idx+1}.\n{json.dumps({'entity_id': candidate['entity_id'], 'name': candidate['name'], 'type': candidate['type'], 'description': candidate.get('description', ''), 'external_ids': candidate.get('external_ids', {})}, indent=2)}"
                    for idx, candidate in enumerate(candidate_entities)
                ])
            )
        }

        messages = [
            {"role": "system", "content": "You are a helpful assistant that follows instructions carefully."},
            user_message
        ]

        start_time = time.time()  # Start timing the API call
        response_content = self.chat_completion(messages)
        self._log_api_response(response_content, {"disambiguation": True}, start_time, messages)

        try:
            result = json.loads(response_content.strip())
            match = result.get("match")
            if match and match != "No Match":
                return match
        except json.JSONDecodeError:
            logger.error("Failed to parse disambiguation response as JSON")

        return None