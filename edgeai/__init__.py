"""
EdgeAI Framework v1.0.0
Lightweight Edge AI Framework for Any Device

Usage:
    from edgeai import EdgeAI
    
    ai = EdgeAI(domain="healthcare")
    result = ai.predict([72, 120, 36.5, 98, 1])
    print(result)
"""

__version__ = "1.0.0"
__author__ = "EdgeAI Team"

from edgeai.EdgeAI import EdgeAI

__all__ = ["EdgeAI"]
