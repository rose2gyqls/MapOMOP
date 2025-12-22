"""
MapOMOP

A 3-stage medical entity mapping system for OMOP CDM.

Stages:
    1. Candidate Retrieval: Multi-strategy search (lexical, semantic, combined)
    2. Standard Collection: Convert non-standard to standard concepts
    3. Hybrid Scoring: LLM or embedding-based final ranking

Usage:
    from MapOMOP import EntityMappingAPI, EntityInput, DomainID
    
    api = EntityMappingAPI()
    entity = EntityInput(entity_name="aspirin", domain_id=DomainID.DRUG)
    results = api.map_entity(entity)
"""

from .elasticsearch_client import ElasticsearchClient, SearchResult
from .entity_mapping_api import (
    DomainID,
    EntityInput,
    EntityMappingAPI,
    MappingResult,
)

# Local LLM support
try:
    from .local_llm import LocalLLM, get_local_llm, cleanup_global_llm
    HAS_LOCAL_LLM = True
except ImportError:
    HAS_LOCAL_LLM = False
    LocalLLM = None
    get_local_llm = None
    cleanup_global_llm = None

__version__ = "1.0.0"
__author__ = "rose"

__all__ = [
    # Main API
    "EntityMappingAPI",
    "EntityInput",
    "DomainID",
    "MappingResult",
    # Elasticsearch
    "ElasticsearchClient",
    "SearchResult",
    # Local LLM
    "LocalLLM",
    "get_local_llm",
    "cleanup_global_llm",
    "HAS_LOCAL_LLM",
]
