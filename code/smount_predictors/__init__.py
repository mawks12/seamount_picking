"""
Package for seamount prediction and scoring.
Contains several classes which can be interfaces
with sklearn pipelines, as well as some additional
supporting functions for reading and working with
data.
"""
from .src import SeamountHelp
from .src.SeamountPredictor import SeamountPredictor
from .src.SeamountScorer import SeamountScorer
from .src.SeamountCVSplitter import SeamountCVSplitter

__all__ = ['SeamountPredictor', 'SeamountCVSplitter', 'SeamountScorer', 'SeamountHelp']
