"""
MapOMOP

A 3-stage pipeline that maps source terms to OMOP CDM Standard Concepts.

Stages:
    1. Candidate Retrieval: Lexical, semantic, and combined search
    2. Standard Concept Collection: Non-standard to Standard Concept conversion
    3. LLM Scoring: LLM-based final ranking

LLM Providers:
    - OpenAI (gpt-5-mini-2025-08-07, etc.)
    - Together AI serverless models

Usage:
    from MapOMOP import EntityMappingAPI, EntityInput, DomainID

    api = EntityMappingAPI()
    entity = EntityInput(entity_name="aspirin", domain_id=DomainID.DRUG)
    results = api.map_entity(entity)
"""

from .elasticsearch_client import ElasticsearchClient
from .entity_mapping_api import (
    DomainID,
    EntityInput,
    EntityMappingAPI,
    MappingResult,
)
from .llm_client import (
    LLMClient,
    LLMProvider,
    get_llm_client,
    create_llm_client,
)
from .utils import deduplicate_by_concept, sigmoid_normalize

__version__ = "1.0.0"
__author__ = "rose"

__all__ = [
    # Main API
    "EntityMappingAPI",
    "EntityInput",
    "DomainID",
    "MappingResult",
    # LLM Client
    "LLMClient",
    "LLMProvider",
    "get_llm_client",
    "create_llm_client",
    # Elasticsearch
    "ElasticsearchClient",
    # Utils
    "deduplicate_by_concept",
    "sigmoid_normalize",
]
