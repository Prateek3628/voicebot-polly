"""
Audio transcription corrections for speech-to-text errors.
Fixes common misheard words and company names.
"""
import re
import logging

logger = logging.getLogger(__name__)

# Common misheard variations of "TechGropse"
TECHGROPSE_VARIATIONS = [
    # Direct variations
    'teke drops', 'tech drops', 'take cropse', 'tech crops', 'tech grobs',
    'tech grows', 'tech gross', 'take grows', 'tech craps', 'tech drops',
    'take drops', 'tech groups', 'tech grosse', 'tech grove', 'tech grape',
    'tech grab', 'tech grip', 'tech graph', 'tech grap', 'tech grep',
    'tek drops', 'tek crops', 'tek grows', 'tekdrops', 'tekcrops',
    'techdrops', 'techcrops', 'techgroups', 'tech grope', 'tech group',
    
    # Case variations
    'Teke drops', 'Tech drops', 'Take crops', 'Tech Drops', 'Tech Groups',
    'TECH DROPS', 'TECH CROPS', 'TAKE CROPS',
    
    # With punctuation/spacing issues
    'tech-drops', 'tech_drops', 'tech drops.', 'tech crops.',
    'tech,drops', 'tech crops,', 'tech drops!', 'tech crops!',
    
    # Phonetically similar
    'deck drops', 'deck crops', 'check drops', 'check crops',
    'tex drops', 'tex crops', 'tec drops', 'tec crops',
]

# Common misheard words and their corrections
WORD_CORRECTIONS = {
    # Company name variations (case insensitive)
    **{variation.lower(): 'TechGropse' for variation in TECHGROPSE_VARIATIONS},
    
    # Other common tech terms that might be misheard
    'react native': 'React Native',
    'react-native': 'React Native',
    'reactnative': 'React Native',
    'node js': 'Node.js',
    'nodejs': 'Node.js',
    'node-js': 'Node.js',
    'mongo db': 'MongoDB',
    'mongodb': 'MongoDB',
    'mongo-db': 'MongoDB',
    'my sql': 'MySQL',
    'mysql': 'MySQL',
    'my-sql': 'MySQL',
    'postgre sql': 'PostgreSQL',
    'postgresql': 'PostgreSQL',
    'postgre-sql': 'PostgreSQL',
    'java script': 'JavaScript',
    'javascript': 'JavaScript',
    'java-script': 'JavaScript',
    'type script': 'TypeScript',
    'typescript': 'TypeScript',
    'type-script': 'TypeScript',
}

def correct_audio_transcription(text: str) -> str:
    """
    Correct common audio transcription errors in text.
    
    Args:
        text: The transcribed text from speech-to-text
        
    Returns:
        Corrected text with proper company names and technical terms
    """
    if not text or not isinstance(text, str):
        return text
    
    original_text = text
    corrected_text = text
    corrections_made = []
    
    # Apply word corrections (case insensitive)
    for wrong_word, correct_word in WORD_CORRECTIONS.items():
        # Use regex for word boundary matching to avoid partial replacements
        pattern = r'\b' + re.escape(wrong_word) + r'\b'
        matches = re.finditer(pattern, corrected_text, re.IGNORECASE)
        
        for match in matches:
            corrected_text = corrected_text[:match.start()] + correct_word + corrected_text[match.end():]
            corrections_made.append(f"'{match.group()}' -> '{correct_word}'")
    
    # Special case: Handle "TechGropse" variations that might span multiple words
    # This catches cases like "tech drops" -> "TechGropse"
    for variation in TECHGROPSE_VARIATIONS:
        if variation.lower() in corrected_text.lower():
            # Use regex to replace whole word matches
            pattern = r'\b' + re.escape(variation) + r'\b'
            if re.search(pattern, corrected_text, re.IGNORECASE):
                corrected_text = re.sub(pattern, 'TechGropse', corrected_text, flags=re.IGNORECASE)
                corrections_made.append(f"'{variation}' -> 'TechGropse'")
    
    # Log corrections if any were made
    if corrections_made:
        logger.info(f"Audio corrections applied: {', '.join(corrections_made)}")
        logger.debug(f"Original: '{original_text}' -> Corrected: '{corrected_text}'")
    
    return corrected_text

def add_custom_correction(wrong_word: str, correct_word: str):
    """
    Add a custom word correction to the dictionary.
    
    Args:
        wrong_word: The incorrectly transcribed word
        correct_word: The correct word
    """
    WORD_CORRECTIONS[wrong_word.lower()] = correct_word
    logger.info(f"Added custom correction: '{wrong_word}' -> '{correct_word}'")

# Test function
def test_corrections():
    """Test the correction function with common examples."""
    test_cases = [
        "what does teke drops do?",
        "tell me about tech drops services",
        "how much does take cropse charge?",
        "what tech stack does tech crops use?",
        "I want to contact Tech Groups",
        "tech drops pricing information",
        "does tech grows do react native development?",
        "what technologies does deck drops use?",
    ]
    
    print("Testing audio corrections:")
    print("-" * 50)
    
    for test_text in test_cases:
        corrected = correct_audio_transcription(test_text)
        print(f"Original:  {test_text}")
        print(f"Corrected: {corrected}")
        print()

if __name__ == "__main__":
    test_corrections()