"""
Domain interfaces (ports) for the PDF processing feature.

Following clean architecture principles:
- Domain defines interfaces (ports)
- Infrastructure implements interfaces (adapters)
- Application orchestrates via interfaces

Each interface is defined in its own file for better organization.
"""

from .ichunker import IChunker

__all__ = ["IChunker"]

