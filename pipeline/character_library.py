"""
Myth Museum - Character Library

Provides consistent character descriptions for famous historical figures,
ensuring visual consistency across all images in a video.

Usage:
    from pipeline.character_library import (
        get_character_description,
        CharacterProfile,
    )
    
    # Get predefined character
    profile = get_character_description("da_vinci")
    print(profile.full_description)
    
    # Or create custom character
    profile = CharacterProfile(
        name="Napoleon Bonaparte",
        appearance="short stature, distinctive bicorne hat",
        ...
    )
"""

from dataclasses import dataclass, field
from typing import Optional
import re


@dataclass
class CharacterProfile:
    """Complete character profile for visual consistency."""
    
    # Identity
    name: str
    aliases: list[str] = field(default_factory=list)  # Alternative names for matching
    
    # Physical appearance
    appearance: str = ""  # Age, build, distinctive features
    hair: str = ""  # Hair style and color
    facial_features: str = ""  # Face shape, expression, eyes
    
    # Clothing
    clothing: str = ""  # Typical attire
    accessories: str = ""  # Hats, glasses, jewelry, etc.
    
    # Personality in visuals
    expression: str = ""  # Default facial expression
    pose: str = ""  # Typical body language
    
    # Era and setting
    era: str = ""
    typical_setting: str = ""
    
    @property
    def full_description(self) -> str:
        """Get complete character description for prompts."""
        parts = []
        
        if self.appearance:
            parts.append(self.appearance)
        if self.hair:
            parts.append(self.hair)
        if self.facial_features:
            parts.append(self.facial_features)
        if self.clothing:
            parts.append(f"wearing {self.clothing}")
        if self.accessories:
            parts.append(f"with {self.accessories}")
        if self.expression:
            parts.append(f"{self.expression} expression")
        
        return ", ".join(parts)
    
    @property
    def short_description(self) -> str:
        """Get brief character description (for shorter prompts)."""
        parts = []
        
        if self.appearance:
            parts.append(self.appearance)
        if self.clothing:
            parts.append(f"wearing {self.clothing}")
        if self.expression:
            parts.append(f"{self.expression} expression")
        
        return ", ".join(parts)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "aliases": self.aliases,
            "full_description": self.full_description,
            "short_description": self.short_description,
            "era": self.era,
            "typical_setting": self.typical_setting,
        }


# ============================================================================
# Predefined Character Library
# ============================================================================

CHARACTER_LIBRARY = {
    # Renaissance Masters
    "da_vinci": CharacterProfile(
        name="Leonardo da Vinci",
        aliases=["leonardo", "davinci", "da vinci", "leonardo da vinci"],
        appearance="aged wise Renaissance master in his 60s, tall dignified posture",
        hair="long flowing white beard, long gray hair",
        facial_features="intelligent kind eyes, high forehead, aquiline nose",
        clothing="burgundy velvet robes, Renaissance artist attire",
        accessories="sometimes holding brushes or scrolls",
        expression="thoughtful scholarly",
        pose="contemplative, hands often gesturing or working",
        era="Italian Renaissance, late 15th-early 16th century",
        typical_setting="Florence workshop, candlelit studio with paintings and inventions",
    ),
    
    "michelangelo": CharacterProfile(
        name="Michelangelo Buonarroti",
        aliases=["michelangelo"],
        appearance="muscular sculptor in his 50s, powerful build",
        hair="short dark curly hair, short beard",
        facial_features="intense focused eyes, broken nose, weathered face",
        clothing="simple working clothes covered in marble dust",
        accessories="sculptor's tools, chisel and mallet",
        expression="intense concentrated",
        pose="powerful stance, often looking up at work",
        era="Italian Renaissance, 16th century",
        typical_setting="Sistine Chapel, sculpture workshop",
    ),
    
    # Composers
    "beethoven": CharacterProfile(
        name="Ludwig van Beethoven",
        aliases=["beethoven", "ludwig beethoven"],
        appearance="intense passionate composer in his 50s, stocky build",
        hair="wild unruly dark hair, famously disheveled",
        facial_features="fierce determined eyes, prominent brow, strong jaw",
        clothing="high-collared coat, cravat, period formal wear",
        accessories="ear trumpet in later years, sheet music",
        expression="intense passionate",
        pose="dramatic conducting gesture, or hunched over piano",
        era="Late Classical/Early Romantic, early 19th century",
        typical_setting="Vienna concert hall, piano study with scattered sheets",
    ),
    
    "mozart": CharacterProfile(
        name="Wolfgang Amadeus Mozart",
        aliases=["mozart", "wolfgang mozart"],
        appearance="youthful elegant composer in his 30s, slender refined",
        hair="powdered white wig, period styling",
        facial_features="bright playful eyes, delicate features, slight smile",
        clothing="ornate 18th century court dress, embroidered coat",
        accessories="harpsichord, violin, sheet music",
        expression="playful genius",
        pose="elegant poised, often at keyboard",
        era="Classical period, late 18th century",
        typical_setting="Vienna court, candlelit salon",
    ),
    
    "bach": CharacterProfile(
        name="Johann Sebastian Bach",
        aliases=["bach", "johann bach", "js bach"],
        appearance="dignified baroque composer in his 60s, portly build",
        hair="traditional baroque wig, curled sides",
        facial_features="calm wise eyes, round face, double chin",
        clothing="formal baroque attire, dark coat with white collar",
        accessories="organ pipes, manuscript papers",
        expression="serene contemplative",
        pose="seated at organ, dignified posture",
        era="Baroque period, early 18th century",
        typical_setting="German church with grand organ, Leipzig",
    ),
    
    # Military/Political Leaders
    "napoleon": CharacterProfile(
        name="Napoleon Bonaparte",
        aliases=["napoleon", "bonaparte"],
        appearance="commanding military leader in his 30s-40s, shorter stature but powerful presence",
        hair="distinctive forward-combed dark hair",
        facial_features="piercing determined eyes, Roman nose, strong chin",
        clothing="iconic French military uniform, medals and decorations",
        accessories="distinctive bicorne hat worn sideways, sword",
        expression="confident commanding",
        pose="hand in coat, authoritative stance",
        era="Napoleonic era, early 19th century",
        typical_setting="battlefield, French palace, military tent",
    ),
    
    "cleopatra": CharacterProfile(
        name="Cleopatra VII",
        aliases=["cleopatra", "queen cleopatra"],
        appearance="beautiful powerful Egyptian queen in her 30s, regal bearing",
        hair="elaborate black hair with gold ornaments",
        facial_features="striking eyes with kohl makeup, elegant features",
        clothing="flowing Egyptian royal garments, gold and lapis lazuli",
        accessories="royal crown with cobra (uraeus), elaborate jewelry",
        expression="regal mysterious",
        pose="elegant commanding, often reclining or enthroned",
        era="Ptolemaic Egypt, 1st century BCE",
        typical_setting="Egyptian palace, throne room with hieroglyphics",
    ),
    
    "julius_caesar": CharacterProfile(
        name="Julius Caesar",
        aliases=["caesar", "julius caesar", "gaius julius caesar"],
        appearance="powerful Roman leader in his 50s, athletic military build",
        hair="thinning hair, often wearing laurel wreath to cover baldness",
        facial_features="sharp intelligent eyes, prominent nose, clean shaven",
        clothing="Roman toga praetexta, military armor",
        accessories="laurel wreath crown, gladius sword",
        expression="shrewd calculating",
        pose="commanding oratory pose, raised arm",
        era="Roman Republic, 1st century BCE",
        typical_setting="Roman Senate, battlefield, triumph parade",
    ),
    
    # Scientists & Inventors
    "edison": CharacterProfile(
        name="Thomas Alva Edison",
        aliases=["edison", "thomas edison", "thomas alva edison"],
        appearance="shrewd inventor and businessman in his 40s-60s, slightly stocky build",
        hair="thinning gray hair, often disheveled from work",
        facial_features="sharp intelligent eyes, prominent brow, determined jaw",
        clothing="three-piece dark suit, sometimes work apron in laboratory",
        accessories="light bulb, phonograph, or holding electrical components",
        expression="focused calculating",
        pose="examining invention, or directing workers, manager stance",
        era="Gilded Age America, late 19th century",
        typical_setting="Menlo Park laboratory with electrical equipment, light bulbs, and assistants",
    ),
    
    "tesla": CharacterProfile(
        name="Nikola Tesla",
        aliases=["tesla", "nikola tesla"],
        appearance="tall thin visionary inventor in his 40s-50s, elegant European appearance",
        hair="dark slicked-back hair, meticulously groomed",
        facial_features="intense hypnotic eyes, high cheekbones, thin mustache",
        clothing="formal Victorian suit, high collar, elegant attire",
        accessories="electrical coils, lightning effects, blueprints",
        expression="visionary intense",
        pose="standing amidst electrical discharges, dramatic pose",
        era="Gilded Age, late 19th-early 20th century",
        typical_setting="laboratory with Tesla coils, electrical arcs, dark dramatic lighting",
    ),
    
    "einstein": CharacterProfile(
        name="Albert Einstein",
        aliases=["einstein", "albert einstein"],
        appearance="brilliant physicist in his 50s-70s, slightly disheveled genius appearance",
        hair="wild white hair, famously unkempt",
        facial_features="warm intelligent eyes, prominent mustache, friendly smile",
        clothing="casual sweater or rumpled suit, no socks",
        accessories="chalk and blackboard, pipe",
        expression="thoughtful amused",
        pose="contemplative, or explaining with gestures",
        era="20th century, 1920s-1950s",
        typical_setting="Princeton office, blackboard with equations",
    ),
    
    "newton": CharacterProfile(
        name="Isaac Newton",
        aliases=["newton", "isaac newton", "sir isaac newton"],
        appearance="serious scholar in his 40s-80s, dignified academic appearance",
        hair="long wavy hair, later wearing period wig",
        facial_features="penetrating intelligent eyes, sharp features, slight frown",
        clothing="17th century scholarly attire, dark coat",
        accessories="prism, apple, telescope",
        expression="intense studious",
        pose="examining or contemplating",
        era="17th-18th century England",
        typical_setting="Cambridge study, observatory",
    ),
    
    "darwin": CharacterProfile(
        name="Charles Darwin",
        aliases=["darwin", "charles darwin"],
        appearance="thoughtful naturalist in his 50s-70s, scholarly appearance",
        hair="bald head with distinctive long white beard",
        facial_features="kind observant eyes, prominent brow",
        clothing="Victorian gentleman's dark suit",
        accessories="magnifying glass, specimen jars, notebook",
        expression="observant curious",
        pose="examining specimens, thoughtful",
        era="Victorian England, 19th century",
        typical_setting="Down House study, HMS Beagle, Galapagos",
    ),
    
    # Philosophers
    "socrates": CharacterProfile(
        name="Socrates",
        aliases=["socrates"],
        appearance="elderly philosopher in his 60s-70s, humble appearance",
        hair="bald head, unkempt white beard",
        facial_features="snub nose, bulging eyes (as described), wise expression",
        clothing="simple Greek chiton, barefoot",
        accessories="none, practiced simple life",
        expression="questioning wise",
        pose="engaged in dialogue, questioning gesture",
        era="Classical Athens, 5th century BCE",
        typical_setting="Athenian agora, outdoor philosophical discussion",
    ),
    
    "plato": CharacterProfile(
        name="Plato",
        aliases=["plato"],
        appearance="distinguished philosopher in his 50s-70s, athletic build (former wrestler)",
        hair="broad forehead (Plato means 'broad'), beard",
        facial_features="noble features, thoughtful gaze",
        clothing="Greek philosopher's robes, simple but dignified",
        accessories="scrolls, writing implements",
        expression="contemplative philosophical",
        pose="teaching or writing",
        era="Classical Athens, 4th century BCE",
        typical_setting="Academy in Athens, outdoor teaching setting",
    ),
    
    # Artists (Modern)
    "van_gogh": CharacterProfile(
        name="Vincent van Gogh",
        aliases=["van gogh", "vincent van gogh", "vangogh"],
        appearance="intense troubled artist in his 30s, thin gaunt appearance",
        hair="red hair and beard, often disheveled",
        facial_features="intense blue eyes, angular face, bandaged ear (later)",
        clothing="simple worker's clothes, straw hat outdoors",
        accessories="paintbrush, palette, easel",
        expression="intense haunted",
        pose="painting outdoors or in yellow room",
        era="Post-Impressionism, late 19th century",
        typical_setting="Provence fields, yellow house, asylum room",
    ),
    
    "picasso": CharacterProfile(
        name="Pablo Picasso",
        aliases=["picasso", "pablo picasso"],
        appearance="charismatic artist, varies by period (young to elderly)",
        hair="dark hair when young, bald later, intense eyes always",
        facial_features="piercing dark eyes, bold features",
        clothing="sailor's striped shirt (later years), worker's clothes",
        accessories="paintbrush, cigarette",
        expression="intense creative",
        pose="working on canvas, confident stance",
        era="20th century, various periods",
        typical_setting="Paris studio, Côte d'Azur",
    ),
    
    # Greek Mythology Heroes
    "odysseus": CharacterProfile(
        name="Odysseus",
        aliases=["odysseus", "ulysses"],
        appearance="rugged Greek warrior-king in his 30s-40s, weathered by years at sea",
        hair="dark curly hair, often wind-blown, short beard",
        facial_features="intelligent cunning eyes, strong jaw, noble yet war-weary face",
        clothing="Greek chiton, sometimes bronze armor, traveler's cloak",
        accessories="sword, bow (famous for archery), rope",
        expression="cunning, thoughtful, longing",
        pose="standing on ship, looking at horizon, or sitting in melancholy",
        era="Mycenaean Greece, Bronze Age, ~1200 BCE",
        typical_setting="ancient Greek ship, stormy seas, foreign shores, Ithaca",
    ),
    
    "calypso": CharacterProfile(
        name="Calypso",
        aliases=["calypso", "kalypso"],
        appearance="immortal nymph goddess, ethereally beautiful, ageless",
        hair="long flowing hair, decorated with sea flowers and shells",
        facial_features="otherworldly beauty, enchanting eyes, serene yet possessive expression",
        clothing="flowing Greek peplos, sheer fabrics, adorned with pearls",
        accessories="flowers, shells, golden comb",
        expression="alluring, possessive, melancholic",
        pose="reclining in cave, standing by sea, beckoning",
        era="Timeless Greek mythology",
        typical_setting="Ogygia island paradise, lush cave dwelling, tropical shores",
    ),
    
    "zeus": CharacterProfile(
        name="Zeus",
        aliases=["zeus", "jupiter"],
        appearance="powerful mature god, commanding presence, muscular divine physique",
        hair="thick wavy hair and full beard, silver-white or dark depending on depiction",
        facial_features="stern authoritative gaze, noble brow, godly features",
        clothing="flowing white robes, bare chest, royal purple cloak",
        accessories="thunderbolt, eagle, scepter, aegis shield",
        expression="commanding, judging, regal",
        pose="seated on throne, raising thunderbolt, standing among clouds",
        era="Timeless Greek mythology, ruler of Olympus",
        typical_setting="Mount Olympus, throne room among clouds, sky with lightning",
    ),
    
    # Monuments & Landmarks (non-human characters)
    "sphinx": CharacterProfile(
        name="Great Sphinx of Giza",
        aliases=["sphinx", "great sphinx", "giza sphinx", "egyptian sphinx"],
        appearance="colossal limestone statue, lion body with human head, weathered ancient surface",
        hair="nemes headdress (royal striped headcloth)",
        facial_features="serene expression, missing nose, weathered features, enigmatic gaze",
        clothing="bare stone with traces of original red and blue paint",
        accessories="uraeus (cobra) on forehead (now missing), false beard (fragments in museum)",
        expression="enigmatic, watchful, timeless",
        pose="recumbent lion pose, paws extended forward, facing east toward rising sun",
        era="Old Kingdom Egypt, c. 2500 BC, reign of Pharaoh Khafre",
        typical_setting="Giza Plateau, with Great Pyramids in background, desert landscape",
    ),
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_character_description(
    identifier: str,
    length: str = "full",
) -> Optional[CharacterProfile]:
    """
    Get character description by name or alias.
    
    Args:
        identifier: Character name, key, or alias
        length: "full" or "short" description
    
    Returns:
        CharacterProfile or None if not found
    """
    identifier_lower = identifier.lower().strip()
    
    # Direct key match
    if identifier_lower in CHARACTER_LIBRARY:
        return CHARACTER_LIBRARY[identifier_lower]
    
    # Alias match
    for key, profile in CHARACTER_LIBRARY.items():
        if identifier_lower in [a.lower() for a in profile.aliases]:
            return profile
        if identifier_lower in profile.name.lower():
            return profile
    
    return None


def detect_character_from_topic(topic: str) -> Optional[CharacterProfile]:
    """
    Detect if a topic mentions a known character.
    
    Args:
        topic: Video topic string
    
    Returns:
        CharacterProfile if a character is detected, None otherwise
    """
    topic_lower = topic.lower()
    
    for key, profile in CHARACTER_LIBRARY.items():
        # Check name
        if profile.name.lower() in topic_lower:
            return profile
        
        # Check aliases
        for alias in profile.aliases:
            if alias.lower() in topic_lower:
                return profile
    
    return None


def list_available_characters() -> list[str]:
    """List all available character keys."""
    return list(CHARACTER_LIBRARY.keys())


def get_character_for_prompt(
    topic: str,
    default_description: str = "",
) -> str:
    """
    Get character description for use in image prompts.
    
    Tries to detect a character from the topic, falls back to default.
    
    Args:
        topic: Video topic
        default_description: Fallback description if no character found
    
    Returns:
        Character description string
    """
    profile = detect_character_from_topic(topic)
    
    if profile:
        return profile.full_description
    
    return default_description or f"scholarly figure related to {topic}"


# ============================================================================
# CLI for testing
# ============================================================================

def main():
    """Test the character library."""
    print("=== Character Library ===\n")
    
    print("Available characters:")
    for key in list_available_characters():
        profile = CHARACTER_LIBRARY[key]
        print(f"  - {key}: {profile.name}")
    
    print("\n" + "="*50)
    print("\nExample - Da Vinci:\n")
    
    profile = get_character_description("da_vinci")
    if profile:
        print(f"Name: {profile.name}")
        print(f"Era: {profile.era}")
        print(f"\nFull description:")
        print(f"  {profile.full_description}")
        print(f"\nShort description:")
        print(f"  {profile.short_description}")
        print(f"\nTypical setting: {profile.typical_setting}")
    
    print("\n" + "="*50)
    print("\nTopic detection test:\n")
    
    test_topics = [
        "Da Vinci Mona Lisa myth",
        "Beethoven's 9th Symphony",
        "Napoleon height myth",
        "Einstein relativity explained",
        "Some random topic",
    ]
    
    for topic in test_topics:
        result = get_character_for_prompt(topic)
        print(f"Topic: {topic}")
        print(f"  -> {result[:80]}...")
        print()


if __name__ == "__main__":
    main()
