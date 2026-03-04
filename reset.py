#!/usr/bin/env python3
"""
Reset Script for Vivy Memory
This script resets the vivy_memory.json file to its default state.
"""

import json
import os

# Path to the memory file
MEMORY_FILE = "vivy_memory.json"

# Default memory structure
DEFAULT_MEMORY = {
    "name": None,
    "likes": [],
    "dislikes": [],
    "topics": {},
    "events": [],
    "summary": "",
    "style": {
        "humor": 0.6,
        "playful": 0.7
    },
    "tone": "neutral",
    "last_greeting": None,
    "last_user_time": None,
    "last_reply": "",
    "arc": {
        "topic": None,
        "stage": 0
    },
    "emotions": {
        "happiness": 0.5,
        "curiosity": 0.5,
        "affection": 0.3,
        "playfulness": 0.6
    },
    "relationship": {
        "affection_level": 0,
        "intimacy": 0,
        "trust": 0,
        "familiarity": 0,
        "stage": "stranger",
        "previous_topics": [],
        "teasing_memory": []
    },
    "emotional_memory": []
}

def reset_memory():
    """Reset the memory file to default values."""
    try:
        # Write the default memory structure to the file
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_MEMORY, f, indent=2, ensure_ascii=False)
        print("Memory successfully reset to default values.")
    except Exception as e:
        print(f"Error resetting memory: {e}")

def backup_memory():
    """Create a backup of the current memory file."""
    try:
        if os.path.exists(MEMORY_FILE):
            # Create backup with timestamp
            import time
            timestamp = int(time.time())
            backup_file = f"vivy_memory_backup_{timestamp}.json"
            with open(MEMORY_FILE, 'r', encoding='utf-8') as src:
                with open(backup_file, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            print(f"Memory backed up to {backup_file}")
            return backup_file
    except Exception as e:
        print(f"Warning: Could not create backup - {e}")
    return None

def main():
    """Main function to reset memory with optional backup."""
    print("Vivy Memory Reset Tool")
    print("=" * 30)
    
    # Ask user if they want to create a backup
    choice = input("Do you want to create a backup before resetting? (y/n): ").lower().strip()
    
    if choice == 'y' or choice == 'yes':
        backup_file = backup_memory()
    
    # Confirm reset
    print("\nThis will reset all memory to default values.")
    confirm = input("Are you sure you want to proceed? (type 'yes' to confirm): ").lower().strip()
    
    if confirm == 'yes':
        reset_memory()
        print("\nMemory reset complete!")
        print("Vivy will now start fresh with no conversation history.")
    else:
        print("Memory reset cancelled.")

if __name__ == "__main__":
    main()
