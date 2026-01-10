"""
Myth Museum - Streamlit Video Generator

Interactive web interface for generating myth-busting YouTube Shorts.
Features smart LLM-powered suggestions for myths, narrative arcs, and visual styles.

Run with:
    streamlit run streamlit_app.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure pipeline modules are importable
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Myth Museum Video Generator",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1E3A5F;
    }
    .myth-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        margin: 0.5rem 0;
    }
    .recommended-badge {
        background-color: #10B981;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        margin-left: 0.5rem;
    }
    .arc-description {
        color: #6B7280;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Smart Suggestions Generator (using Gemini)
# ============================================================================

def generate_smart_suggestions(topic: str) -> dict:
    """
    Generate smart suggestions using Google Gemini API.
    
    Returns a dictionary with:
    - myths: List of myth options with title, hook, summary, script
    - arcs: List of narrative arcs specific to the topic
    - recommended_style: Best visual style for this topic
    - style_reasons: Why this style is recommended
    """
    import google.generativeai as genai
    
    # Configure Gemini
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # Try to get from service account
        try:
            from google.auth import default
            credentials, project = default()
            # If we have credentials, we can use vertex AI instead
        except:
            pass
    
    if api_key:
        genai.configure(api_key=api_key)
    
    # Create the prompt
    prompt = f'''You are a myth-busting video producer creating YouTube Shorts (60 seconds).

Given the topic: "{topic}"

Generate suggestions in JSON format with this exact structure:
{{
    "myths": [
        {{
            "id": 1,
            "title": "Short catchy title (e.g., 'Odysseus Wasn't Lost for 10 Years')",
            "title_zh": "Chinese title",
            "hook": "Opening hook sentence that grabs attention",
            "summary": "One sentence summary of what we're debunking",
            "script": "Full 60-second voiceover script (about 150 words). Include: Hook (5s), Setup (10s), Twist/Revelation (20s), Evidence (15s), Conclusion (10s)"
        }},
        // Generate 4 different myth angles
    ],
    "arcs": [
        {{
            "id": 1,
            "name": "Arc name (e.g., 'Cosmic Mind-Blow')",
            "name_zh": "Chinese name",
            "description": "What this arc does",
            "structure": "Scene 1 -> Scene 2 -> Scene 3 -> Scene 4 -> Scene 5 -> Scene 6"
        }},
        // Generate 4 narrative arcs that FIT THIS SPECIFIC TOPIC
    ],
    "recommended_style": "one of: oil_painting_cartoon, watercolor_fantasy, cinematic, realistic, vintage_sepia, sci_fi_cinematic, pixar_3d, watercolor",
    "style_reasons": "Why this style fits the topic",
    "alternate_styles": ["style2", "style3"]
}}

IMPORTANT:
- Generate 4 different myth angles (different aspects to debunk about the topic)
- Generate 4 narrative arcs that specifically fit this topic type (not generic arcs)
- Scripts should be engaging, factual, and suitable for YouTube Shorts
- Each myth should be a genuinely interesting misconception worth debunking
- For space topics: use scientific wonder arcs
- For mythology: use hero deconstruction or divine revelation arcs
- For history: use hidden truth or conspiracy debunk arcs

Output ONLY valid JSON, no markdown or explanation.'''

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Parse the response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        result = json.loads(response_text.strip())
        return result
        
    except Exception as e:
        st.error(f"Gemini API error: {e}")
        # Return fallback suggestions
        return generate_fallback_suggestions(topic)


def generate_fallback_suggestions(topic: str) -> dict:
    """Generate fallback suggestions when Gemini is not available."""
    return {
        "myths": [
            {
                "id": 1,
                "title": f"The Truth About {topic}",
                "title_zh": f"關於 {topic} 的真相",
                "hook": f"Think you know the truth about {topic}?",
                "summary": f"What everyone gets wrong about {topic}",
                "script": f"""Think you know the truth about {topic}?

That's the story we've all been told. The legendary tale that's been passed down for generations.

But here's what they don't tell you. The evidence tells a completely different story.

Recent research has uncovered surprising facts that challenge everything we thought we knew.

The reality is far more interesting than the myth. Most people get this completely wrong.

So the next time someone mentions {topic}, you'll know the real story. The truth is often stranger than fiction."""
            },
            {
                "id": 2,
                "title": f"What They Never Told You About {topic}",
                "title_zh": f"他們從未告訴你的 {topic}",
                "hook": f"Everything you know about {topic} might be wrong.",
                "summary": f"The hidden truth behind {topic}",
                "script": f"""Everything you know about {topic} might be wrong.

We've been taught this story since childhood. But the original sources tell a very different tale.

Here's what actually happened. The historical records reveal surprising details.

This changes everything we thought we understood about {topic}.

Now you know the truth. Share this with someone who still believes the myth."""
            },
        ],
        "arcs": [
            {
                "id": 1,
                "name": "Myth Buster Classic",
                "name_zh": "經典破解迷思",
                "description": "Traditional myth debunking structure",
                "structure": "Hook -> Common Belief -> Evidence -> Revelation -> Impact -> Conclusion"
            },
            {
                "id": 2,
                "name": "Hidden Truth",
                "name_zh": "隱藏真相",
                "description": "Uncover what's been hidden",
                "structure": "Mystery -> Investigation -> Discovery -> Proof -> Revelation -> Resolution"
            },
        ],
        "recommended_style": detect_best_style_simple(topic),
        "style_reasons": "Based on topic keywords",
        "alternate_styles": ["cinematic", "realistic"]
    }


def detect_best_style_simple(topic: str) -> str:
    """Simple style detection based on keywords."""
    topic_lower = topic.lower()
    
    if any(word in topic_lower for word in ["greek", "odyssey", "zeus", "myth", "god", "mythology"]):
        return "watercolor_fantasy"
    elif any(word in topic_lower for word in ["space", "galaxy", "star", "planet", "cosmos", "universe"]):
        return "sci_fi_cinematic"
    elif any(word in topic_lower for word in ["edison", "tesla", "industrial", "inventor", "victorian"]):
        return "vintage_sepia"
    elif any(word in topic_lower for word in ["egypt", "pyramid", "sphinx", "ancient", "rome"]):
        return "cinematic"
    else:
        return "oil_painting_cartoon"


def get_all_styles():
    """Get all available visual styles with descriptions."""
    return {
        "oil_painting_cartoon": {
            "name": "Oil Painting Cartoon",
            "description": "Renaissance style, warm and scholarly",
            "best_for": "Historical figures, art topics"
        },
        "watercolor_fantasy": {
            "name": "Watercolor Fantasy",
            "description": "Dreamy, mythological storybook aesthetic",
            "best_for": "Mythology, legends, fantasy"
        },
        "cinematic": {
            "name": "Cinematic",
            "description": "Hollywood movie quality, dramatic lighting",
            "best_for": "Epic stories, ancient civilizations"
        },
        "realistic": {
            "name": "Realistic",
            "description": "Photorealistic documentary style",
            "best_for": "Science, nature, modern topics"
        },
        "vintage_sepia": {
            "name": "Vintage Sepia",
            "description": "Aged photography, historical feel",
            "best_for": "Industrial revolution, inventors"
        },
        "sci_fi_cinematic": {
            "name": "Sci-Fi Cinematic",
            "description": "Interstellar/Star Trek aesthetic",
            "best_for": "Space, cosmos, astronomy"
        },
        "pixar_3d": {
            "name": "Pixar 3D",
            "description": "Animated 3D cartoon style",
            "best_for": "Fun facts, kid-friendly content"
        },
        "watercolor": {
            "name": "Watercolor",
            "description": "Soft, dreamy artistic style",
            "best_for": "Gentle stories, artistic topics"
        },
    }


async def run_video_generation(
    topic: str,
    script: str,
    style: str,
    progress_callback,
    status_callback,
) -> Path:
    """Run the video generation pipeline."""
    from pipeline.generate_short import (
        ShortVideoGenerator,
        GenerationConfig,
        PreFlightCheck,
    )
    
    # Phase 0: Pre-flight checks
    status_callback("Phase 0: Running pre-flight checks...")
    progress_callback(0.05)
    
    preflight = PreFlightCheck()
    if not preflight.run_all_checks(verbose=False):
        errors = "\n".join(preflight.checks_failed)
        raise RuntimeError(f"Pre-flight checks failed:\n{errors}")
    
    progress_callback(0.10)
    
    # Create generator
    generator = ShortVideoGenerator()
    
    # Create config
    config = GenerationConfig(
        topic=topic,
        script=script,
        image_quality="high",
        subtitle_style="punch",
        auto_prompts=True,
    )
    
    # Run the actual generation
    status_callback("Generating video (this may take 5-10 minutes)...")
    progress_callback(0.15)
    
    result = await generator.generate(config)
    
    if not result.success:
        raise RuntimeError(f"Video generation failed: {result.error}")
    
    progress_callback(1.0)
    status_callback("Complete!")
    
    return result.final_video


# ============================================================================
# Streamlit App
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">🏛️ Myth Museum Video Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Create myth-busting YouTube Shorts with AI-powered suggestions</div>', unsafe_allow_html=True)
    
    # Sidebar with settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        image_quality = st.selectbox(
            "Image Quality",
            options=["high", "standard", "fallback"],
            index=0,
            help="high = Imagen 3, standard = Imagen @006, fallback = stock photos"
        )
        
        subtitle_style = st.selectbox(
            "Subtitle Style",
            options=["punch", "normal"],
            index=0,
            help="punch = animated first word + keyword highlighting"
        )
        
        st.divider()
        
        st.header("📊 Pre-flight Status")
        if st.button("Run Pre-flight Checks"):
            with st.spinner("Checking dependencies..."):
                from pipeline.generate_short import PreFlightCheck
                preflight = PreFlightCheck()
                preflight.run_all_checks(verbose=False)
                
                for check in preflight.checks_passed:
                    st.success(f"✓ {check}")
                for warning in preflight.checks_warned:
                    st.warning(f"⚠ {warning}")
                for error in preflight.checks_failed:
                    st.error(f"✗ {error}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # =====================================================================
        # Step 1: Enter Topic
        # =====================================================================
        st.header("1️⃣ Enter Topic")
        topic = st.text_input(
            "What topic do you want to explore?",
            placeholder="e.g., 'Galaxy', 'Odyssey', 'Edison', 'Sphinx'",
            key="topic_input"
        )
        
        # Generate suggestions button
        if st.button("🎯 Generate Smart Suggestions", type="primary", disabled=not topic):
            with st.spinner("🤖 AI is generating suggestions... (this may take 10-20 seconds)"):
                suggestions = generate_smart_suggestions(topic)
                st.session_state['suggestions'] = suggestions
                st.session_state['topic'] = topic
                st.session_state['selected_myth_id'] = 1
                st.session_state['selected_arc_id'] = 1
        
        # =====================================================================
        # Step 2: Select Myth
        # =====================================================================
        if 'suggestions' in st.session_state:
            suggestions = st.session_state['suggestions']
            
            st.divider()
            st.header("2️⃣ Select a Myth to Bust")
            
            myths = suggestions.get('myths', [])
            
            if myths:
                # Create radio options for myths
                myth_options = {}
                for myth in myths:
                    myth_id = myth.get('id', 1)
                    title = myth.get('title', 'Unknown')
                    title_zh = myth.get('title_zh', '')
                    summary = myth.get('summary', '')
                    
                    display_text = f"{title}"
                    if title_zh:
                        display_text += f" ({title_zh})"
                    
                    myth_options[myth_id] = {
                        'display': display_text,
                        'summary': summary,
                        'full': myth
                    }
                
                # Radio selection for myths
                selected_myth_id = st.radio(
                    "Choose which myth angle to explore:",
                    options=list(myth_options.keys()),
                    format_func=lambda x: myth_options[x]['display'],
                    key="myth_selection",
                    index=0,
                )
                
                # Show summary for selected myth
                if selected_myth_id in myth_options:
                    st.caption(f"📝 {myth_options[selected_myth_id]['summary']}")
                    st.session_state['selected_myth'] = myth_options[selected_myth_id]['full']
                
                # Show script preview in expander
                with st.expander("📜 View Full Script", expanded=False):
                    selected_myth = myth_options.get(selected_myth_id, {}).get('full', {})
                    script_text = selected_myth.get('script', '')
                    st.text_area(
                        "Script Preview (editable)",
                        value=script_text,
                        height=200,
                        key="script_preview"
                    )
            
            # =====================================================================
            # Step 3: Select Narrative Arc
            # =====================================================================
            st.divider()
            st.header("3️⃣ Select Narrative Arc")
            
            arcs = suggestions.get('arcs', [])
            
            if arcs:
                arc_options = {}
                for arc in arcs:
                    arc_id = arc.get('id', 1)
                    name = arc.get('name', 'Unknown')
                    name_zh = arc.get('name_zh', '')
                    description = arc.get('description', '')
                    structure = arc.get('structure', '')
                    
                    display_text = f"{name}"
                    if name_zh:
                        display_text += f" ({name_zh})"
                    
                    arc_options[arc_id] = {
                        'display': display_text,
                        'description': description,
                        'structure': structure
                    }
                
                selected_arc_id = st.radio(
                    "Choose the story structure:",
                    options=list(arc_options.keys()),
                    format_func=lambda x: arc_options[x]['display'],
                    key="arc_selection",
                    index=0,
                )
                
                # Show arc details
                if selected_arc_id in arc_options:
                    arc_info = arc_options[selected_arc_id]
                    st.caption(f"📖 {arc_info['description']}")
                    st.caption(f"🎬 {arc_info['structure']}")
            
            # =====================================================================
            # Step 4: Select Visual Style
            # =====================================================================
            st.divider()
            st.header("4️⃣ Select Visual Style")
            
            all_styles = get_all_styles()
            recommended_style = suggestions.get('recommended_style', 'cinematic')
            alternate_styles = suggestions.get('alternate_styles', [])
            style_reasons = suggestions.get('style_reasons', '')
            
            # Show recommendation
            if style_reasons:
                st.info(f"💡 **AI Recommendation:** {style_reasons}")
            
            # Style selection
            style_options = list(all_styles.keys())
            
            # Find index of recommended style
            default_index = style_options.index(recommended_style) if recommended_style in style_options else 0
            
            selected_style = st.radio(
                "Choose visual style:",
                options=style_options,
                format_func=lambda x: f"⭐ {all_styles[x]['name']} (Recommended)" if x == recommended_style else all_styles[x]['name'],
                index=default_index,
                key="style_selection"
            )
            
            # Show style details
            if selected_style in all_styles:
                style_info = all_styles[selected_style]
                st.caption(f"🎨 {style_info['description']}")
                st.caption(f"✨ Best for: {style_info['best_for']}")
            
            # =====================================================================
            # Step 5: Edit Script (Optional)
            # =====================================================================
            st.divider()
            st.header("5️⃣ Final Script (Editable)")
            
            # Get the script from selected myth
            default_script = ""
            if 'selected_myth' in st.session_state:
                default_script = st.session_state['selected_myth'].get('script', '')
            
            final_script = st.text_area(
                "Edit the voiceover script before generating:",
                value=default_script,
                height=250,
                key="final_script"
            )
            
            # =====================================================================
            # Step 6: Generate Video
            # =====================================================================
            st.divider()
            st.header("6️⃣ Generate Video")
            
            # Summary before generation
            col_summary1, col_summary2 = st.columns(2)
            with col_summary1:
                st.markdown(f"**Topic:** {st.session_state.get('topic', topic)}")
                if 'selected_myth' in st.session_state:
                    st.markdown(f"**Myth:** {st.session_state['selected_myth'].get('title', 'N/A')}")
            with col_summary2:
                st.markdown(f"**Style:** {all_styles.get(selected_style, {}).get('name', selected_style)}")
                st.markdown(f"**Quality:** {image_quality}")
            
            if st.button("🎬 Generate Video", type="primary", use_container_width=True):
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(value):
                    progress_bar.progress(value)
                
                def update_status(text):
                    status_text.text(text)
                
                try:
                    with st.spinner("Generating video... This may take 5-10 minutes."):
                        # Get the final script
                        script_to_use = final_script if final_script else default_script
                        
                        # Run async generation
                        video_path = asyncio.run(run_video_generation(
                            topic=st.session_state.get('topic', topic),
                            script=script_to_use,
                            style=selected_style,
                            progress_callback=update_progress,
                            status_callback=update_status,
                        ))
                        
                        st.session_state['video_path'] = video_path
                        st.success("✅ Video generated successfully!")
                        st.balloons()
                        
                except Exception as e:
                    st.error(f"❌ Generation failed: {str(e)}")
                    st.exception(e)
    
    # =====================================================================
    # Right Column: Video Preview
    # =====================================================================
    with col2:
        st.header("📺 Preview")
        
        if 'video_path' in st.session_state and st.session_state['video_path']:
            video_path = Path(st.session_state['video_path'])
            
            if video_path.exists():
                # Video player
                st.video(str(video_path))
                
                # Download button
                with open(video_path, "rb") as f:
                    video_bytes = f.read()
                    st.download_button(
                        label="⬇️ Download Video",
                        data=video_bytes,
                        file_name=f"myth_museum_{st.session_state.get('topic', 'video')[:20]}.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
                
                # Show output folder
                st.info(f"📁 Output folder:\n`{video_path.parent}`")
            else:
                st.warning("Video file not found.")
        else:
            st.info("Generated video will appear here after processing.")
            
            # Show workflow guide
            st.markdown("""
            **Workflow:**
            1. Enter a topic (e.g., "Galaxy")
            2. Click "Generate Smart Suggestions"
            3. Select a myth angle
            4. Choose narrative arc
            5. Pick visual style
            6. Edit script if needed
            7. Click "Generate Video"
            8. Preview and download!
            """)
            
            # Show example topics
            st.markdown("---")
            st.markdown("**💡 Example Topics:**")
            st.markdown("- Galaxy / Milky Way")
            st.markdown("- Odyssey / Greek Mythology")
            st.markdown("- Edison / Inventions")
            st.markdown("- Sphinx / Egypt")
            st.markdown("- Napoleon / History")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
        🏛️ Myth Museum • Powered by Gemini, Imagen 3, Google TTS, and FFmpeg<br>
        <a href="https://github.com/AlexBBno1/MetaGPT_Myth-Museum" target="_blank">GitHub Repository</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
