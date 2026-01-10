"""
Test script for Smart Suggestions System.

Tests the Gemini-powered suggestion generator with various topics.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Reconfigure stdout for Windows Unicode support
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


def test_smart_suggestions():
    """Test the smart suggestions generator."""
    
    # Import after path setup
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Test importing the function
    print("=" * 60)
    print("Testing Smart Suggestions System")
    print("=" * 60)
    
    # We'll test with a simple import and function call
    try:
        import google.generativeai as genai
        print("[OK] google-generativeai is installed")
    except ImportError:
        print("[FAIL] google-generativeai not installed")
        print("  Run: pip install google-generativeai")
        return False
    
    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        print("[OK] Gemini API key found")
    else:
        print("[WARN] No Gemini API key in environment")
        print("  Set GEMINI_API_KEY or GOOGLE_API_KEY")
    
    # Test topics
    test_topics = ["Galaxy", "Odyssey"]
    
    for topic in test_topics:
        print(f"\n{'=' * 60}")
        print(f"Testing topic: {topic}")
        print("=" * 60)
        
        # Import the function from streamlit_app
        try:
            from streamlit_app import generate_smart_suggestions
            
            suggestions = generate_smart_suggestions(topic)
            
            # Validate structure
            assert "myths" in suggestions, "Missing 'myths' key"
            assert "arcs" in suggestions, "Missing 'arcs' key"
            assert "recommended_style" in suggestions, "Missing 'recommended_style'"
            
            print(f"\n[OK] Generated suggestions for '{topic}'")
            
            # Print myths
            print(f"\n  Myths ({len(suggestions['myths'])} options):")
            for myth in suggestions['myths']:
                title = myth.get('title', 'Unknown')
                summary = myth.get('summary', '')[:50]
                print(f"    - {myth.get('id', '?')}. {title}")
                print(f"      Summary: {summary}...")
            
            # Print arcs
            print(f"\n  Narrative Arcs ({len(suggestions['arcs'])} options):")
            for arc in suggestions['arcs']:
                name = arc.get('name', 'Unknown')
                description = arc.get('description', '')[:40]
                print(f"    - {arc.get('id', '?')}. {name}")
                print(f"      {description}...")
            
            # Print style recommendation
            print(f"\n  Recommended Style: {suggestions.get('recommended_style', 'N/A')}")
            print(f"  Reason: {suggestions.get('style_reasons', 'N/A')[:60]}...")
            
        except Exception as e:
            print(f"[FAIL] Error testing '{topic}': {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_smart_suggestions()
    sys.exit(0 if success else 1)
