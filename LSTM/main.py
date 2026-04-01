"""
Entry point — launches the Gradio Toxic Text Classifier app.
Run with: python main.py
"""

import sys
import os

# Ensure the project root is on the path so `src.*` imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))



from views.app import launch

if __name__ == "__main__":
    launch()  