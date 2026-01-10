"""
Test Subtitle Quality Guard

Tests the QualityGuard.validate_subtitles() method to ensure
it correctly detects truncated subtitles.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.generate_short import QualityGuard


def test_galaxy_video():
    """Test with Galaxy video - should PASS (subtitles are complete)."""
    print("=" * 60)
    print("Test 1: Galaxy video (should PASS - subtitles are complete)")
    print("=" * 60)
    
    output_dir = Path("outputs/shorts/space-myths_galaxy-speed")
    script_path = output_dir / "voiceover.txt"
    srt_path = output_dir / "captions.srt"
    
    if not script_path.exists() or not srt_path.exists():
        print(f"[SKIP] Galaxy video files not found")
        return
    
    guard = QualityGuard()
    result = guard.validate_subtitles(script_path, srt_path)
    
    print(f"Script: {script_path}")
    print(f"SRT: {srt_path}")
    print(f"Result: {'PASS' if result else 'FAIL'}")
    
    if guard.errors:
        print("Errors:")
        for e in guard.errors:
            print(f"  - {e}")
    
    if guard.warnings:
        print("Warnings:")
        for w in guard.warnings:
            print(f"  - {w}")
    
    if result:
        print("\n[OK] Galaxy video subtitles are complete!")
    else:
        print("\n[FAIL] Galaxy video subtitles have issues!")
    
    return result


def test_truncated_subtitles():
    """Test with simulated truncated subtitles - should FAIL."""
    print("\n" + "=" * 60)
    print("Test 2: Simulated truncated subtitles (should FAIL)")
    print("=" * 60)
    
    output_dir = Path("outputs/shorts/space-myths_galaxy-speed")
    script_path = output_dir / "voiceover.txt"
    
    if not script_path.exists():
        print("[SKIP] Script file not found")
        return
    
    # Create a fake truncated SRT (missing many words)
    fake_srt = """1
00:00:00,000 --> 00:00:05,000
The stars look the same

2
00:00:05,000 --> 00:00:10,000
Dead wrong
"""
    
    # Write to temp file
    temp_srt = output_dir / "test_truncated.srt"
    temp_srt.write_text(fake_srt, encoding="utf-8")
    
    guard = QualityGuard()
    result = guard.validate_subtitles(script_path, temp_srt)
    
    print(f"Result: {'PASS' if result else 'FAIL (expected)'}")
    
    if guard.errors:
        print("Errors (expected):")
        for e in guard.errors:
            print(f"  - {e}")
    
    # Clean up temp file
    temp_srt.unlink()
    
    if not result:
        print("\n[OK] Truncated subtitles correctly detected!")
        return True
    else:
        print("\n[FAIL] Should have detected truncated subtitles!")
        return False


def main():
    print("\n=== SUBTITLE QUALITY GUARD TESTS ===\n")
    
    test1_passed = test_galaxy_video()
    test2_passed = test_truncated_subtitles()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if test1_passed:
        print("[PASS] Test 1: Complete subtitles correctly validated")
    else:
        print("[FAIL] Test 1: Complete subtitles validation failed")
    
    if test2_passed:
        print("[PASS] Test 2: Truncated subtitles correctly rejected")
    else:
        print("[FAIL] Test 2: Truncated subtitles not detected")
    
    if test1_passed and test2_passed:
        print("\nAll tests passed! Subtitle Quality Guard is working correctly.")
        return 0
    else:
        print("\nSome tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
