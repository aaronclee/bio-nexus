import requests
import logging
from typing import List

logger = logging.getLogger(__name__)

class PubTatorAPI:
    def __init__(self, base_url="https://www.ncbi.nlm.nih.gov/research/pubtator3-api/"):
        """Initialize the PubTatorAPI with the specified base URL with the Pubtator API endpoints."""
        self.base_url = base_url

    def find_entity_id(self, entity_name: str, limit: int = 5) -> List[str]:
        """Find entity IDs in PubTator for a given entity name."""
        url = f"{self.base_url}entity/autocomplete/"
        params = {"query": entity_name, "limit": limit}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        # list of entity IDs matching the entity name.
        return [item["id"] for item in data if "id" in item]

    def find_related_entities(self, entity_id: str, relation_type: str, entity_type: str, limit: int = 5):
        """Find related entities in PubTator for a given entity ID and relation type, optionally."""
        url = f"{self.base_url}relations"
        params = {"e1": entity_id, "type": relation_type, "e2": entity_type, "limit": limit}
        response = requests.get(url, params=params)
        response.raise_for_status()
        # list of dictionaries representing related entities and their relations
        return response.json().get("relations", [])

    def search(self, query: str, page: int = 1):
        """Search PubTator for a given query."""
        url = f"{self.base_url}search/"
        params = {"text": query, "page": page}
        response = requests.get(url, params=params)
        response.raise_for_status()
        # dictionary with JSON response from the PubTator search API containing search results
        return response.json()