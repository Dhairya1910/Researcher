"""
Vercel Serverless Function entry point.
Re-exports the existing FastAPI app so Vercel can discover it.
No backend/frontend code is modified — this file only bridges
the existing server.py app into Vercel's expected layout.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so Backend/Frontend imports work
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Re-export the FastAPI application
from Frontend.server import app
