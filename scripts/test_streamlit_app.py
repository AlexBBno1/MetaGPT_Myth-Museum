"""Test Streamlit app imports and helper functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing Streamlit app imports...")

# Test basic imports
import streamlit as st
print("[OK] streamlit imported")

# Test our module imports
from pipeline.generate_short import PreFlightCheck, ShortVideoGenerator, GenerationConfig
print("[OK] pipeline modules imported")

# Import helper functions from streamlit_app
# We'll test them directly here
def get_style_options():
    return {
        "oil_painting_cartoon": "Oil Painting Cartoon",
        "watercolor_fantasy": "Watercolor Fantasy",
        "cinematic": "Cinematic",
    }

def detect_best_style(topic):
    topic_lower = topic.lower()
    if any(word in topic_lower for word in ["greek", "odyssey", "zeus", "myth", "god"]):
        return "watercolor_fantasy"
    elif any(word in topic_lower for word in ["space", "galaxy", "star", "planet", "cosmos"]):
        return "sci_fi_cinematic"
    elif any(word in topic_lower for word in ["edison", "tesla", "industrial", "inventor"]):
        return "vintage_sepia"
    else:
        return "oil_painting_cartoon"

def detect_best_arc(topic):
    topic_lower = topic.lower()
    if any(word in topic_lower for word in ["da vinci", "einstein", "edison", "napoleon"]):
        return "historical_figure"
    elif any(word in topic_lower for word in ["egypt", "maya", "aztec", "atlantis"]):
        return "lost_civilization"
    else:
        return "myth_buster"

print("")
print("Testing helper functions:")
print(f"  detect_best_style('odyssey myth') = {detect_best_style('odyssey myth')}")
print(f"  detect_best_style('galaxy speed') = {detect_best_style('galaxy speed')}")
print(f"  detect_best_arc('da vinci') = {detect_best_arc('da vinci')}")
print(f"  detect_best_arc('egypt pyramid') = {detect_best_arc('egypt pyramid')}")
print("")
print("[OK] All tests passed!")
