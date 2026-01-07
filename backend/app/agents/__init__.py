"""
Agents Module

Multi-agent system for financial research queries:
- Router: Classifies query complexity
- Planner: Decomposes complex queries
- Retriever: Fetches relevant documents
- Analyst: Extracts data and performs calculations
- Synthesizer: Generates responses with citations
- Validator: Quality checks responses
"""

from app.agents.router import QueryRouter
from app.agents.planner import QueryPlanner
from app.agents.retriever_agent import RetrieverAgent
from app.agents.analyst_agent import AnalystAgent
from app.agents.synthesizer import Synthesizer
from app.agents.validator import Validator
from app.agents.workflow import FinAgentWorkflow

__all__ = [
    "QueryRouter",
    "QueryPlanner",
    "RetrieverAgent",
    "AnalystAgent",
    "Synthesizer",
    "Validator",
    "FinAgentWorkflow"
]
