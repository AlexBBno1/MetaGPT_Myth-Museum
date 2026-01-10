"""Check Google Cloud credentials setup."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=== Google Cloud Credentials Check ===\n")

# Check environment variable
creds_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
print(f"GOOGLE_APPLICATION_CREDENTIALS: {creds_env or 'Not set'}")

# Check for service account files
print("\nLooking for service account JSON files...")
json_files = list(Path(".").glob("*-*.json"))
for jf in json_files:
    try:
        import json
        with open(jf, "r") as f:
            data = json.load(f)
            if data.get("type") == "service_account":
                print(f"  Found service account: {jf.name}")
                print(f"    Project: {data.get('project_id', 'unknown')}")
    except Exception as e:
        print(f"  Error reading {jf}: {e}")

# Try google.auth.default()
print("\nTrying google.auth.default()...")
try:
    from google.auth import default
    credentials, project = default()
    print(f"  Credentials: {type(credentials).__name__}")
    print(f"  Project: {project}")
except Exception as e:
    print(f"  Failed: {e}")

# Check image provider
print("\nChecking Vertex AI Image Provider...")
try:
    from pipeline.image_providers.vertex_imagen import VertexImagenProvider
    provider = VertexImagenProvider()
    print(f"  Project ID: {provider.project_id}")
    print(f"  Location: {provider.location}")
    print(f"  Available: {provider.is_available()}")
except Exception as e:
    print(f"  Error: {e}")

# Check TTS provider
print("\nChecking Google TTS Provider...")
try:
    from pipeline.tts import GoogleTTSProvider
    tts = GoogleTTSProvider()
    print(f"  Initialized: Yes")
    if hasattr(tts, 'service_account_file'):
        print(f"  Service account: {tts.service_account_file}")
except Exception as e:
    print(f"  Error: {e}")

print("\n=== Done ===")
