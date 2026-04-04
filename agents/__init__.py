import logging

# Unified logging format for all MARS agents
_fmt = logging.Formatter(
    "%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_handler = logging.StreamHandler()
_handler.setFormatter(_fmt)

_mars_logger = logging.getLogger("mars")
if not _mars_logger.handlers:
    _mars_logger.addHandler(_handler)
    _mars_logger.setLevel(logging.INFO)

from .base_agent import BaseAgent
from .orchestrator import Orchestrator
from .kg_agent import KnowledgeGraphAgent
from .diagnostic_agent import DiagnosticAgent
from .confidence_agent import ConfidenceAgent
from .recommendation_agent import RecommendationAgent
from .prediction_agent import PredictionAgent
from .personalization_agent import PersonalizationAgent

__all__ = [
    "BaseAgent", "Orchestrator", "KnowledgeGraphAgent",
    "DiagnosticAgent", "ConfidenceAgent", "RecommendationAgent",
    "PredictionAgent", "PersonalizationAgent",
]
