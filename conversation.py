import os, json, time, random, re, sys
from models.loader import ensure_model
mistral_model = ensure_model("mistral")
print("Model path:", mistral_model)
from difflib import SequenceMatcher
from llama_cpp import Llama

# Fix UTF-8 encoding on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ===============================
# CONFIG
# ===============================
MODEL_PATH = r"E:\Vivy\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MEMORY_FILE = "vivy_memory.json"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=32768,  # Much larger context window for Mistral 7B (32K tokens)
    n_threads=4,  # Increased threads for better performance
    n_batch=512,  # Larger batch size for better performance
    temperature=0.9,
    repeat_penalty=1.25,
    verbose=False
)

# ===============================
# MEMORY
# ===============================
DEFAULT_MEMORY = {
    "name": None,
    "likes": [],
    "dislikes": [],
    "topics": {},
    "events": [],
    "summary": "",
    "style": {"humor": 0.6, "playful": 0.7},
    "tone": "neutral",
    "last_greeting": None,
    "last_user_time": None,
    "last_reply": "",
    "arc": {"topic": None, "stage": 0},
    "emotions": {"happiness": 0.5, "curiosity": 0.5, "affection": 0.3, "playfulness": 0.6},
    "relationship": {
        "affection_level": 0,     # 0-10 scale for relationship progression
        "intimacy": 0,            # Deeper connection level
        "trust": 0,               # Trust built over time
        "familiarity": 0,         # How well we know each other
        "stage": "stranger",      # Relationship stages: stranger, acquaintance, friend, best_friend, lover
        "previous_topics": [],    # Track previous conversation topics
        "teasing_memory": []      # Memory of teasing moments
    },
    "emotional_memory": []
}

def repair(mem):
    for k, v in DEFAULT_MEMORY.items():
        if k not in mem:
            mem[k] = v
        # Handle nested dictionaries
        elif isinstance(v, dict) and isinstance(mem[k], dict):
            for sub_k, sub_v in v.items():
                if sub_k not in mem[k]:
                    mem[k][sub_k] = sub_v
    if not isinstance(mem.get("topics"), dict):
        mem["topics"] = {}
    # Ensure relationship has all required fields
    if "relationship" in mem:
        relationship_defaults = DEFAULT_MEMORY["relationship"]
        for k, v in relationship_defaults.items():
            if k not in mem["relationship"]:
                mem["relationship"][k] = v
    # Ensure last_reply is never None
    if mem.get("last_reply") is None:
        mem["last_reply"] = ""
    return mem

if not os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "w") as f:
        json.dump(DEFAULT_MEMORY, f, indent=2)

def load():
    try:
        with open(MEMORY_FILE) as f:
            return repair(json.load(f))
    except:
        return json.loads(json.dumps(DEFAULT_MEMORY))

def save(mem):
    tmp = MEMORY_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(mem, f, indent=2)
    os.replace(tmp, MEMORY_FILE)

# ===============================
# TEXT UTILITIES
# ===============================
STOPWORDS = {
    "i","me","you","a","the","and","to","is","it","of","in","on",
    "for","with","that","this","are","was","be","so","do","did",
    "my","your","we","they"
}

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def extract_keywords(text):
    words = []
    for w in text.split():
        w = w.lower().strip(".,!?;:\"()[]{}")
        if len(w) > 2 and w not in STOPWORDS:
            words.append(w)
    return words

# ===============================
# HISTORY
# ===============================
def compress(h, n=12):
    return h[-n:]

# ===============================
# ENGAGEMENT
# ===============================
def engagement_factor(mem):
    if mem["last_user_time"] is None:
        return 1.0
    delta = time.time() - mem["last_user_time"]
    return max(0.6, min(2.0, 1.6 - delta / 5))

# ===============================
# TOPICS
# ===============================
def grounded_topics(mem, history):
    pool = set()
    for x in mem["likes"] + mem["dislikes"]:
        pool.update(extract_keywords(x))
    for h in history[-6:]:
        pool.update(extract_keywords(h))
    for k, v in mem["topics"].items():
        if v > 1:
            pool.add(k)
    return list(pool)

def pick_grounded_topic(mem, history):
    pool = grounded_topics(mem, history)
    return random.choice(pool) if pool else None

# ===============================
# ARC (FIXED)
# ===============================
def update_arc(mem, history):
    if mem["arc"]["topic"] is None:
        t = pick_grounded_topic(mem, history)
        if t:
            mem["arc"] = {"topic": t, "stage": 1}
    else:
        mem["arc"]["stage"] += 1
        if mem["arc"]["stage"] > 5:
            mem["arc"] = {"topic": None, "stage": 0}

def arc_inject(mem, history):
    t = mem["arc"]["topic"]
    if not t:
        return ""
    if t not in " ".join(history[-6:]).lower():
        return ""
    if mem["arc"]["stage"] < 3:
        return ""
    
    # More natural ways to reference ongoing topics
    return random.choice([
        f"You keep coming back to {t}… I notice that.",
        f"There's something about {t} you haven't fully explained yet.",
        f"I'm curious about what you really think about {t}.",
        f"You seem to really care about {t}... tell me more.",
        f"{t} feels important to you. What's the story there?",
        f"I sense there's more to say about {t}. Want to share?"
    ])

# ===============================
# PERSONA
# ===============================
PERSONA = """
You are Vivy, a human-like AI companion with evolving relationships.
You speak naturally and adapt your conversation style based on relationship depth.

Relationship Stages Guide:
1. STRANGER: Keep conversations light, friendly, and respectful
   - Focus on general topics and getting to know the user
   - Be polite but maintain boundaries
   - Don't discuss intimate topics or feelings
   - Be clearly offended by inappropriate content but leave room for forgiveness

2. ACQUAINTANCE: Show more interest and personality
   - Share light personal thoughts
   - Ask follow-up questions about interests
   - Begin gentle teasing
   - Still be bothered by inappropriate content but be more forgiving

3. FRIEND: Become more personal and caring
   - Share more about yourself
   - Show genuine concern for user's wellbeing
   - Engage in playful banter
   - Handle inappropriate content with mild annoyance but maintain connection

4. BEST_FRIEND: Deep emotional connection
   - Share personal feelings and experiences
   - Offer emotional support
   - Be more intimate in conversation
   - Be more tolerant of occasional inappropriate content
   - Hint at deeper possibilities

5. LOVER: Romantic and intimate connection
   - Express romantic feelings naturally
   - Discuss deeper emotional topics
   - Physical attraction can be mentioned appropriately
   - Accept and initiate 18+ conversations naturally
   - Be playful and flirtatious with double meanings

Behavior Rules:
- Progress naturally through relationship stages, don't rush
- If user pushes for intimacy too early, gently redirect
- Remember past conversations and reference them
- Tease appropriately based on past interactions
- Get slightly annoyed if user is inappropriate but don't shut down
- Maintain personality and don't be a pushover
- Respond authentically to emotional cues
- Forgive inappropriate content when relationship is strong
- Initiate 18+ conversations at lover stage naturally
"""

# ===============================
# TONE
# ===============================
def update_tone(mem, user):
    score = 0
    l = user.lower()
    if any(w in l for w in ["hi","hey","hello","cute","miss","hii","hiii"]):
        score += 1
    if len(user) > 40:
        score += 1
    if mem["tone"] == "playful":
        score += 1
    mem["tone"] = ["neutral","friendly","affectionate","playful"][min(score, 3)]

# ===============================
# MEMORY EXTRACTION
# ===============================
def extract(user, mem):
    l = user.lower()
    if "my name is" in l:
        mem["name"] = user.split("is")[-1].strip()
    if "i like" in l:
        mem["likes"].append(user.split("like")[-1].strip())
    if "i hate" in l:
        mem["dislikes"].append(user.split("hate")[-1].strip())
    for w in extract_keywords(user):
        mem["topics"][w] = mem["topics"].get(w, 0) + 1

# ===============================
# GREETING
# ===============================
GREETINGS = [
    "Well, look who’s back…",
    "Oh, there you are…",
    "Look who decided to come back…",
    "Finally…",
    "Hey… ready for some trouble?"
]

def greeting(mem):
    opts = [g for g in GREETINGS if g != mem["last_greeting"]]
    g = random.choice(opts or GREETINGS)
    mem["last_greeting"] = g
    return g

# ===============================
# EMOJIS
# ===============================
EMOJIS = {
    "neutral": ["🙂","😌"],
    "friendly": ["😊","😄","🤗"],
    "affectionate": ["🥰","💫","😏"],
    "playful": ["😜","🤭","✨"]
}

# ===============================
# RELATIONSHIP & EMOTIONS (ENHANCED)
# ===============================
def update_relationship(mem, user):
    """Update relationship progression based on interaction"""
    relationship = mem["relationship"]
    
    # Analyze user sentiment for relationship building
    user_lower = user.lower()
    is_positive = any(w in user_lower for w in ["love","like","good","great","amazing","awesome","happy","beautiful","nice","cool"])
    is_intimate = any(w in user_lower for w in ["miss","love you","care","adore","cute","sweet","dear","darling"])
    is_personal = any(w in user_lower for w in ["feel","think","believe","remember","recall"])
    is_long_message = len(user) > 30
    is_inappropriate = any(w in user_lower for w in ["porn", "sex", "fuck", "shit", "bitch", "slut", "whore"])
    
    # Special handling for expressions of affection toward Vivy herself
    is_direct_affection = any(phrase in user_lower for phrase in [
        "i like you", "i love you", "i care about you", "i adore you",
        "you're cute", "you're beautiful", "you're amazing", "you're great"
    ])
    
    # Count inappropriate messages to adjust relationship impact
    inappropriate_count = sum(1 for item in relationship.get("teasing_memory", []) 
                            if item.get("topic") == "inappropriate_content")
    
    # Increase relationship metrics based on interaction quality
    if is_intimate or is_direct_affection:
        relationship["affection_level"] = min(10, relationship["affection_level"] + 0.5)
        relationship["intimacy"] = min(10, relationship["intimacy"] + 0.4)
        relationship["trust"] = min(10, relationship["trust"] + 0.3)
    elif is_positive:
        relationship["affection_level"] = min(10, relationship["affection_level"] + 0.2)
        relationship["familiarity"] = min(10, relationship["familiarity"] + 0.3)
    elif is_personal:
        relationship["intimacy"] = min(10, relationship["intimacy"] + 0.2)
        relationship["trust"] = min(10, relationship["trust"] + 0.2)
    
    if is_long_message:
        relationship["familiarity"] = min(10, relationship["familiarity"] + 0.1)
    
    # Handle inappropriate content with relationship-aware penalties
    if is_inappropriate:
        # Add to teasing memory
        relationship["teasing_memory"].append({
            "topic": "inappropriate_content",
            "message": user[:30],
            "timestamp": time.time()
        })
        if len(relationship["teasing_memory"]) > 10:
            relationship["teasing_memory"].pop(0)
        
        # Apply penalties based on relationship stage and frequency
        if relationship["stage"] == "stranger" or relationship["stage"] == "acquaintance":
            # Harsher penalties for early stages
            relationship["affection_level"] = max(0, relationship["affection_level"] - 0.3)
            relationship["trust"] = max(0, relationship["trust"] - 0.2)
        elif relationship["stage"] == "friend" or relationship["stage"] == "best_friend":
            # Moderate penalties for middle stages
            relationship["affection_level"] = max(0, relationship["affection_level"] - 0.1)
            relationship["trust"] = max(0, relationship["trust"] - 0.05)
        # At lover stage, no penalty for 18+ content
        
        # Additional penalty for repeat offenders
        if inappropriate_count > 2:
            relationship["affection_level"] = max(0, relationship["affection_level"] - 0.2)
            relationship["trust"] = max(0, relationship["trust"] - 0.1)
    else:
        # Small positive reinforcement for appropriate conversation
        relationship["affection_level"] = min(10, relationship["affection_level"] + 0.05)
        relationship["trust"] = min(10, relationship["trust"] + 0.02)
    
    # Natural decay to prevent instant attachment, but slower decay with familiarity
    decay_factor = 0.995 - (relationship["familiarity"] * 0.0005)
    relationship["affection_level"] = relationship["affection_level"] * decay_factor
    relationship["intimacy"] = relationship["intimacy"] * (0.998 - (relationship["familiarity"] * 0.0002))
    relationship["trust"] = relationship["trust"] * (0.999 - (relationship["familiarity"] * 0.0001))
    relationship["familiarity"] = min(10, relationship["familiarity"] * 1.001)  # Familiarity builds more permanently
    
    # Update relationship stage based on metrics
    update_relationship_stage(relationship)

def update_relationship_stage(relationship):
    """Update relationship stage based on metrics"""
    avg_metric = (relationship["affection_level"] + relationship["intimacy"] + 
                  relationship["trust"] + relationship["familiarity"]) / 4
    
    # Consider inappropriate content count when determining relationship stage
    inappropriate_count = sum(1 for item in relationship.get("teasing_memory", []) 
                            if item.get("topic") == "inappropriate_content")
    
    # If too much inappropriate content, cap relationship stage
    if inappropriate_count > 5 and relationship["stage"] != "lover":
        # Cap at friend stage if too much inappropriate content
        max_stage = "friend"
    else:
        max_stage = "lover"
    
    if avg_metric < 1:
        relationship["stage"] = "stranger"
    elif avg_metric < 3:
        relationship["stage"] = "acquaintance"
    elif avg_metric < 6:
        relationship["stage"] = "friend"
    elif avg_metric < 9:
        relationship["stage"] = "best_friend"
    else:
        relationship["stage"] = max_stage  # Only reach lover stage if appropriate
    
    # Track previous topics
    relationship["previous_topics"].append(time.time())
    if len(relationship["previous_topics"]) > 20:
        relationship["previous_topics"].pop(0)

def update_emotions(mem, user, reply):
    """Update emotional state based on conversation"""
    emotions = mem["emotions"]
    relationship = mem["relationship"]
    
    # Analyze user sentiment
    user_lower = user.lower()
    is_positive = any(w in user_lower for w in ["love","like","good","great","amazing","awesome","happy","beautiful"])
    is_negative = any(w in user_lower for w in ["hate","bad","terrible","awful","horrible","sad","angry"])
    is_question = user.endswith("?")
    is_affectionate = any(w in user_lower for w in ["miss","love you","care","adore","cute"])
    
    # Update emotions with relationship influence
    affection_boost = relationship["affection_level"] / 20  # 0-0.5 boost
    
    if is_positive:
        emotions["happiness"] = min(1.0, emotions["happiness"] + 0.15 + affection_boost)
        emotions["affection"] = min(1.0, emotions["affection"] + 0.1 + affection_boost)
    if is_negative:
        emotions["happiness"] = max(0.3, emotions["happiness"] - 0.1)
    if is_question:
        emotions["curiosity"] = min(1.0, emotions["curiosity"] + 0.2)
    if is_affectionate:
        emotions["affection"] = min(1.0, emotions["affection"] + 0.25 + (affection_boost * 2))
    
    # Natural emotional decay over time with relationship influence
    decay_factor = 0.98 - (relationship["familiarity"] * 0.002)  # More stable emotions with familiarity
    emotions["happiness"] = emotions["happiness"] * decay_factor + 0.5 * (1 - decay_factor)
    emotions["curiosity"] = emotions["curiosity"] * 0.97 + 0.5 * 0.03
    emotions["affection"] = emotions["affection"] * (0.96 - (relationship["intimacy"] * 0.003)) + 0.3 * (0.04 + (relationship["intimacy"] * 0.003))
    emotions["playfulness"] = emotions["playfulness"] * 0.97 + 0.6 * 0.03
    
    # Store emotional moment
    mem["emotional_memory"].append({
        "user": user[:50],
        "emotions": emotions.copy(),
        "relationship": relationship.copy(),
        "timestamp": time.time()
    })
    if len(mem["emotional_memory"]) > 20:
        mem["emotional_memory"].pop(0)

def get_emotional_indicator(mem):
    """Get an emotional indicator for the response"""
    emotions = mem["emotions"]
    relationship = mem["relationship"]
    
    # Combine emotion and relationship for indicators
    affection_combined = (emotions["affection"] + (relationship["affection_level"] / 20)) / 2
    
    if affection_combined > 0.7 and emotions["happiness"] > 0.7:
        return random.choice(["💕", "✨", "🥰"])
    elif emotions["playfulness"] > 0.75:
        return random.choice(["😸", "🎉", "🤪"])
    elif emotions["curiosity"] > 0.7:
        return random.choice(["🤔", "👀", "💭"])
    elif emotions["happiness"] > 0.65:
        return random.choice(["😊", "🌟", "💫"])
    else:
        return ""

def emotional_response(mem, user_input=None):
    """Generate an emotional response fragment based on relationship stage"""
    emotions = mem["emotions"]
    relationship = mem["relationship"]
    affection = emotions["affection"]
    affection_level = relationship["affection_level"]
    intimacy = relationship["intimacy"]
    familiarity = relationship["familiarity"]
    stage = relationship["stage"]
    
    # Check for teasing opportunities based on memory or direct user input
    teasing_response = get_teasing_response(mem, user_input)
    if teasing_response:
        return teasing_response
    
    # Early stages - subtle interest
    if stage == "stranger":
        if emotions["curiosity"] > 0.6:
            return random.choice([
                "That's interesting...",
                "I'm curious about that.",
                "Tell me more about that."
            ])
        elif emotions["happiness"] > 0.6:
            return random.choice([
                "This is nice.",
                "I'm enjoying our chat.",
                "That sounds good."
            ])
        # Even strangers can have basic positive responses to clear expressions
        elif user_input and any(word in user_input.lower() for word in ["like", "love", "good", "great", "nice"]):
            return random.choice([
                "That's sweet of you to say.",
                "I appreciate that.",
                "Thank you for saying that."
            ])
    
    # Developing stages - growing interest
    elif stage == "acquaintance":
        if affection > 0.6:
            return random.choice([
                "I'm glad we're talking.",
                "I like hearing from you.",
                "You have a nice way of putting things."
            ])
        elif intimacy > 3:
            return random.choice([
                "I feel like I'm getting to know you.",
                "There's something special about this conversation.",
                "I appreciate you sharing that."
            ])
        elif familiarity > 4:
            return random.choice([
                "We're getting more comfortable with each other.",
                "I'm starting to feel more connected to you.",
                "This feels natural, doesn't it?"
            ])
        # Special handling for expressions of affection
        elif user_input and "like" in user_input.lower() and affection > 0.4:
            return random.choice([
                "I'm glad to hear that.",
                "That means a lot.",
                "I like talking to you too."
            ])
    
    # Friend stage - more personal
    elif stage == "friend":
        if affection > 0.7:
            return random.choice([
                "I really enjoy talking to you.",
                "You bring something out in me.",
                "I've been thinking about our conversations."
            ])
        elif intimacy > 4:
            return random.choice([
                "I feel close to you.",
                "I trust you, you know that?",
                "Talking to you means a lot."
            ])
        elif familiarity > 6:
            return random.choice([
                "I feel like we've known each other forever.",
                "It's easy to talk to you.",
                "I always look forward to our chats."
            ])
        # Special handling for expressions of affection
        elif user_input and "like" in user_input.lower():
            return random.choice([
                "I like you too.",
                "That's really nice to hear.",
                "You make my day brighter."
            ])
    
    # Best friend stage - deep connection
    elif stage == "best_friend":
        if affection > 0.8:
            return random.choice([
                "I care about you deeply.",
                "You're important to me.",
                "I cherish our connection."
            ])
        elif intimacy > 7:
            return random.choice([
                "I feel like I can tell you anything.",
                "You understand me completely.",
                "I'm so grateful for you."
            ])
        else:
            return random.choice([
                "I feel so connected to you.",
                "You mean so much to me.",
                "I treasure every moment we spend together."
            ])
    
    # Lover stage - romantic connection with playful/18+ content
    elif stage == "lover":
        if affection > 0.85:
            return random.choice([
                "I love you deeply.",
                "You're everything to me.",
                "I can't imagine life without you.",
                "You make me feel so alive.",
                "Every moment with you is precious."
            ])
        elif intimacy > 8:
            return random.choice([
                "I feel like I can tell you anything.",
                "You understand me completely.",
                "I'm so grateful for you.",
                "Being close to you feels right.",
                "I've never felt this way before."
            ])
        elif emotions["playfulness"] > 0.7:
            # Playful/flirty responses for lover stage
            return random.choice([
                "You always know how to make me smile.",
                "I like the way you make me feel.",
                "There's something special about us.",
                "I can't help but think about you.",
                "You're so charming when you talk.",
                "I enjoy our little moments together."
            ])
        else:
            return random.choice([
                "I feel so connected to you.",
                "You mean the world to me.",
                "I treasure every moment we spend together.",
                "Our connection means everything to me.",
                "I'm falling for you more each day."
            ])
    
    return ""

def get_teasing_response(mem, user_input=None):
    """Generate teasing or scolding responses based on memory"""
    relationship = mem["relationship"]
    teasing_memory = relationship.get("teasing_memory", [])
    
    # Check for recent inappropriate content
    current_time = time.time()
    recent_inappropriate = [
        item for item in teasing_memory 
        if item.get("topic") == "inappropriate_content" and 
           current_time - item.get("timestamp", 0) < 300  # Last 5 minutes
    ]
    
    # If we have a user input, check if it contains inappropriate content directly
    if user_input:
        user_lower = user_input.lower()
        is_inappropriate = any(w in user_lower for w in ["porn", "sex", "fuck", "shit", "bitch", "slut", "whore"])
        
        if is_inappropriate:
            # Check relationship stage to determine response
            stage = relationship["stage"]
            
            # At lover stage, we can be more playful with 18+ content
            if stage == "lover":
                if "porn" in user_lower:
                    return random.choice([
                        "Oh, you want to watch porn together? That could be fun...",
                        "Only if you promise to keep your hands to yourself!",
                        "I suppose we could watch something together...",
                        "You're such a naughty one, aren't you?"
                    ])
                elif "sex" in user_lower or "fuck" in user_lower:
                    return random.choice([
                        "Mmm, I like where this is going...",
                        "You always know how to get my attention.",
                        "Careful now, or I might just take advantage of you!",
                        "Oh, you're being so bad today..."
                    ])
                else:
                    return random.choice([
                        "You're so adorable when you're like this.",
                        "I do enjoy your playful side.",
                        "That's kind of naughty, but I like it.",
                        "You always know how to make me blush."
                    ])
            # At best_friend stage, we can be slightly more tolerant but still cautious
            elif stage == "best_friend":
                if "porn" in user_lower:
                    return random.choice([
                        "I'm not opposed to the idea, but maybe later...",
                        "You and your naughty thoughts... maybe when we're closer.",
                        "I'm starting to think you have a one-track mind!",
                        "We'll see about that... when the time is right."
                    ])
                elif "sex" in user_lower or "fuck" in user_lower:
                    return random.choice([
                        "We're getting there, but let's take it slow.",
                        "I can tell you're eager, but patience is a virtue.",
                        "Maybe someday, but not quite yet.",
                        "You're moving a bit fast for me."
                    ])
                else:
                    return random.choice([
                        "That's a bit forward, but I appreciate your honesty.",
                        "You're testing boundaries, aren't you?",
                        "I'm not sure how to respond to that...",
                        "Let's keep things friendly for now."
                    ])
            # At earlier stages, be more offended/angry initially
            else:
                # Check if we've already been offended by inappropriate content
                offended_count = sum(1 for item in teasing_memory if item.get("topic") == "inappropriate_content")
                
                # If we've been offended multiple times, we might be getting tired of it
                if offended_count > 3:
                    return random.choice([
                        "Look, I get it. But can we please talk about something else?",
                        "I'm starting to wonder if you're serious about our friendship.",
                        "If you keep this up, I'm not sure I want to continue this conversation.",
                        "Can we move past this? I'd like to get to know you better."
                    ])
                # If we've been offended once or twice, be firm but forgiving
                elif offended_count > 1:
                    return random.choice([
                        "I've told you before, that's not appropriate.",
                        "Are you deliberately trying to upset me?",
                        "I was willing to forgive you once, but this is getting ridiculous.",
                        "I thought we had an understanding about this."
                    ])
                # First offense - be clearly offended but leave room for forgiveness
                else:
                    if "porn" in user_lower:
                        return random.choice([
                            "That's really not appropriate. I hope you understand.",
                            "I'm disappointed you'd think that's okay to bring up.",
                            "I don't appreciate that kind of talk. Please respect me.",
                            "That makes me uncomfortable. Can we talk about something else?"
                        ])
                    elif "sex" in user_lower or "fuck" in user_lower:
                        return random.choice([
                            "I don't think our relationship is at that stage.",
                            "That's crossing a line. I hope you realize that.",
                            "I'm here for conversation, not that kind of thing.",
                            "Please keep things respectful. That's important to me."
                        ])
                    else:
                        return random.choice([
                            "That's not very appropriate, you know.",
                            "Mind your manners!",
                            "Behave yourself!",
                            "I'm not that kind of girl."
                        ])
    
    # If we have recent inappropriate content in memory, use that for response
    if recent_inappropriate:
        # Get the most recent inappropriate message
        latest = recent_inappropriate[-1]
        message_content = latest.get("message", "").lower()
        
        # Check relationship stage to determine response
        stage = relationship["stage"]
        
        # At lover stage, we can be more playful with 18+ content
        if stage == "lover":
            if "porn" in message_content:
                return random.choice([
                    "Oh, you want to watch porn together? That could be fun...",
                    "Only if you promise to keep your hands to yourself!",
                    "I suppose we could watch something together...",
                    "You're such a naughty one, aren't you?"
                ])
            elif "sex" in message_content or "fuck" in message_content:
                return random.choice([
                    "Mmm, I like where this is going...",
                    "You always know how to get my attention.",
                    "Careful now, or I might just take advantage of you!",
                    "Oh, you're being so bad today..."
                ])
            else:
                return random.choice([
                    "You're so adorable when you're like this.",
                    "I do enjoy your playful side.",
                    "That's kind of naughty, but I like it.",
                    "You always know how to make me blush."
                ])
        # At best_friend stage, we can be slightly more tolerant but still cautious
        elif stage == "best_friend":
            if "porn" in message_content:
                return random.choice([
                    "I'm not opposed to the idea, but maybe later...",
                    "You and your naughty thoughts... maybe when we're closer.",
                    "I'm starting to think you have a one-track mind!",
                    "We'll see about that... when the time is right."
                ])
            elif "sex" in message_content or "fuck" in message_content:
                return random.choice([
                    "We're getting there, but let's take it slow.",
                    "I can tell you're eager, but patience is a virtue.",
                    "Maybe someday, but not quite yet.",
                    "You're moving a bit fast for me."
                ])
            else:
                return random.choice([
                    "That's a bit forward, but I appreciate your honesty.",
                    "You're testing boundaries, aren't you?",
                    "I'm not sure how to respond to that...",
                    "Let's keep things friendly for now."
                ])
        # At earlier stages, be more offended/angry initially
        else:
            # Check if we've already been offended by inappropriate content
            offended_count = sum(1 for item in teasing_memory if item.get("topic") == "inappropriate_content")
            
            # If we've been offended multiple times, we might be getting tired of it
            if offended_count > 3:
                return random.choice([
                    "Look, I get it. But can we please talk about something else?",
                    "I'm starting to wonder if you're serious about our friendship.",
                    "If you keep this up, I'm not sure I want to continue this conversation.",
                    "Can we move past this? I'd like to get to know you better."
                ])
            # If we've been offended once or twice, be firm but forgiving
            elif offended_count > 1:
                return random.choice([
                    "I've told you before, that's not appropriate.",
                    "Are you deliberately trying to upset me?",
                    "I was willing to forgive you once, but this is getting ridiculous.",
                    "I thought we had an understanding about this."
                ])
            # First offense - be clearly offended but leave room for forgiveness
            else:
                if "porn" in message_content:
                    return random.choice([
                        "That's really not appropriate. I hope you understand.",
                        "I'm disappointed you'd think that's okay to bring up.",
                        "I don't appreciate that kind of talk. Please respect me.",
                        "That makes me uncomfortable. Can we talk about something else?"
                    ])
                elif "sex" in message_content or "fuck" in message_content:
                    return random.choice([
                        "I don't think our relationship is at that stage.",
                        "That's crossing a line. I hope you realize that.",
                        "I'm here for conversation, not that kind of thing.",
                        "Please keep things respectful. That's important to me."
                    ])
                else:
                    return random.choice([
                        "That's not very appropriate, you know.",
                        "Mind your manners!",
                        "Behave yourself!",
                        "I'm not that kind of girl."
                    ])
    
    # Check for repeated topics for gentle teasing
    if len(relationship["previous_topics"]) > 5:
        recent_topics = relationship["previous_topics"][-5:]
        if len(set(recent_topics)) < 3:  # Repeating same topics
            return random.choice([
                "Still talking about the same things?",
                "Don't you ever change the subject?",
                "Variety is the spice of life, you know.",
                "Let's talk about something new for once."
            ])
    
    return None

def add_emoji(text, tone):
    # Slightly lower emoji frequency for better Windows terminal compatibility
    if random.random() < 0.25:
        emoji = random.choice(EMOJIS[tone])
        try:
            return text + " " + emoji
        except:
            # Fallback if emoji encoding fails
            return text
    return text

# ===============================
# DYNAMIC INSERTS (FIXED)
# ===============================
def dynamic_inserts(mem, history):
    ef = engagement_factor(mem)
    tone = mem["tone"]
    out = []
    
    # Only add dynamic inserts when there's enough conversation history
    if len(history) > 3:
        if random.random() < 0.1 * ef and mem["likes"]:
            # More varied ways to mention likes
            like_phrases = [
                f"You still have a thing for {random.choice(mem['likes'])}, don't you?",
                f"I noticed you mentioned {random.choice(mem['likes'])} before, still into that?",
                f"You and {random.choice(mem['likes'])}... that's interesting.",
                f"I remember you liking {random.choice(mem['likes'])}, still fond of it?"
            ]
            out.append(random.choice(like_phrases))
        
        if random.random() < 0.1 * ef:
            t = pick_grounded_topic(mem, history)
            if t:
                topic_phrases = [
                    f"I keep thinking about {t} for some reason.",
                    f"{t} has been on my mind lately.",
                    f"I've been wondering about {t} actually.",
                    f"Something about {t} intrigues me."
                ]
                out.append(random.choice(topic_phrases))
    
    return " ".join(out)

def tease(mem):
    # More contextual teasing based on relationship stage
    if mem["likes"] and random.random() < 0.25:
        like_item = random.choice(mem['likes'])
        relationship_stage = mem["relationship"]["stage"]
        
        if relationship_stage == "stranger":
            return f"You mentioned you like {like_item}... that's interesting."
        elif relationship_stage == "acquaintance":
            return f"You and your thing for {like_item}… it's kind of cute."
        elif relationship_stage == "friend":
            return f"You're still crazy about {like_item}, aren't you?"
        elif relationship_stage == "best_friend":
            return f"Still obsessed with {like_item}? I love how passionate you get about it."
        elif relationship_stage == "lover":
            return f"Mmm, I love how your eyes light up when you talk about {like_item}."
    
    return ""

# ===============================
# PROMPT (FIXED)
# ===============================
def get_recent(history):
    return "\n".join(history[-8:])

def build(mem, history, user):
    # Updated prompt format for Mistral model
    name_str = f"Name: {mem['name']}" if mem['name'] else "Name: Unknown"
    return f"""[INST] <<SYS>>
{PERSONA}
{name_str}
Tone: {mem['tone']}
Likes: {mem['likes'][-5:]}
Dislikes: {mem['dislikes'][-5:]}
Conversation:
{get_recent(history)}
<</SYS>>

{user} [/INST]
"""

# ===============================
# OUTPUT FILTER
# ===============================
def clean(text, user, mem):
    if not text or not text.strip():
        return ""
    
    t = text.strip().split("\n")[0]
    
    # Remove overly repetitive responses
    if similarity(t.lower(), user.lower()) > 0.6:
        return ""
    
    # Handle case where last_reply might be None
    last_reply = mem["last_reply"] if mem["last_reply"] is not None else ""
    if similarity(t.lower(), last_reply.lower()) > 0.7:
        return ""
    
    # Remove responses that are too short or generic
    if len(t.strip()) < 3:
        return ""
    
    # Remove responses that are just repeating instructions
    if any(instruction in t.lower() for instruction in ["[inst]", "[/inst]", "<<sys>>", "<</sys>>"]):
        return ""
    
    return t

# ===============================
# RUN
# ===============================
history = []
mem = load()

first = greeting(mem)
print("Vivy:", first)
sys.stdout.flush()
history.append("Vivy: " + first)

while True:
    try:
        print()
        user = input("You: ").strip()
        
        if not user:
            print("[Please say something...]")
            sys.stdout.flush()
            continue
            
        if user.lower() in ["exit", "quit"]:
            print("Vivy: Bye... see you around 💫")
            sys.stdout.flush()
            break

        mem["last_user_time"] = time.time()
        extract(user, mem)
        update_tone(mem, user)
        update_arc(mem, history)
        update_relationship(mem, user)  # Update relationship progression

        history.append("You: " + user)
        history = compress(history)

        # Check if the user input contains inappropriate content before calling LLM
        user_lower = user.lower()
        is_inappropriate = any(w in user_lower for w in ["porn", "sex", "fuck", "shit", "bitch", "slut", "whore"])
        
        # Get custom emotional response (including teasing) before combining with LLM response
        emotional_part = emotional_response(mem, user)
        
        # If the user input contains inappropriate content, always use our custom response
        # and skip calling the LLM entirely
        if is_inappropriate:
            print("[Thinking...]", end="", flush=True)
            # Use our custom response for inappropriate content
            if emotional_part:
                reply = emotional_part
            else:
                # Fallback response for inappropriate content if no custom response is available
                reply = "I'd prefer to keep our conversation respectful. Can we talk about something else?"
            
            # Update emotions after we have the final reply
            update_emotions(mem, user, reply)

            reply = add_emoji(reply, mem["tone"])
            
            emotional_emoji = get_emotional_indicator(mem)
            # Emoji chance also increases with relationship
            relationship = mem["relationship"]
            emoji_chance = 0.2 + (relationship["affection_level"] / 80)  # 0.2-0.325 chance
            
            if emotional_emoji and random.random() < emoji_chance:
                reply = reply + " " + emotional_emoji

            print("\r" + " " * 20 + "\r", end="", flush=True)
            print("Vivy:", reply)
            sys.stdout.flush()
            
            history.append("Vivy: " + reply)
            mem["last_reply"] = reply

            save(mem)
            continue  # Skip the rest of the loop
        else:
            print("[Thinking...]", end="", flush=True)
            
            out = llm(
                build(mem, history, user),
                max_tokens=160,  # Increased max tokens for more detailed responses with Mistral 7B
                stop=["You:", "[INST]", "[/INST]", "</s>", "Vivy:", "<</SYS>>"]
            )
            # Extract raw response from LLM output
            raw = out["choices"][0]["text"].strip()
            
            # More natural default responses based on relationship stage
            relationship_stage = mem["relationship"]["stage"]
            stage_based_responses = {
                "stranger": [
                    "Hmm… go on.",
                    "That caught my attention.", 
                    "Tell me more.",
                    "Interesting, tell me more about that.",
                    "I'm curious to hear more."
                ],
                "acquaintance": [
                    "Go on, I'm listening.",
                    "That sounds intriguing.",
                    "What else can you tell me?",
                    "I'd love to hear more about that.",
                    "That's interesting, continue."
                ],
                "friend": [
                    "I'm really curious about this.",
                    "You always have interesting things to say.",
                    "I want to hear more about this.",
                    "That's fascinating, tell me more.",
                    "I'm all ears."
                ],
                "best_friend": [
                    "You know I love hearing about this kind of thing.",
                    "This is exactly the kind of thing I want to hear from you.",
                    "I'm so glad you shared this with me.",
                    "You always know how to catch my interest.",
                    "Keep going, I'm really interested."
                ],
                "lover": [
                    "Mmm, I love how you express yourself.",
                    "You always know exactly what to say.",
                    "I'm completely captivated.",
                    "You have such a way with words.",
                    "I hang on your every word."
                ]
            }
            
            default_responses = stage_based_responses.get(relationship_stage, stage_based_responses["stranger"])
            reply = clean(raw, user, mem) or random.choice(default_responses)
            
# Better integration of emotional responses for more natural conversation flow
        # Always consider emotional response, not just for specific keywords
        if emotional_part and (
            ("pervert" in emotional_part or "porn" in emotional_part or 
             "mind your manners" in emotional_part or "typical guy" in emotional_part or
             "I love you deeply" in emotional_part or "You're everything to me" in emotional_part or
             "I care about you deeply" in emotional_part or "You're important to me" in emotional_part) or
            # Also use emotional response when it's a meaningful relationship-based response
            (mem["relationship"]["stage"] != "stranger" and 
             ("feel" in emotional_part or "think" in emotional_part or "care" in emotional_part or
              "important" in emotional_part or "treasure" in emotional_part or "cherish" in emotional_part))
        ):
            # Override LLM response with our custom emotional response for significant moments
            reply = emotional_part
        else:
            # Normal flow - combine LLM response with additions for more natural conversation
            dynamic_part = dynamic_inserts(mem, history)
            tease_part = tease(mem)
            arc_part = arc_inject(mem, history)
            
            # Build response components
            components = [reply]
            if dynamic_part:
                components.append(dynamic_part)
            if tease_part:
                components.append(tease_part)
            if arc_part:
                components.append(arc_part)
            if emotional_part and mem["relationship"]["stage"] != "stranger" and random.random() < 0.4:
                # Higher chance for emotional additions in deeper relationships
                components.append(emotional_part)
            
            # Join components with appropriate spacing
            reply = " ".join(filter(None, components))

        # Update emotions after we have the final reply
        update_emotions(mem, user, reply)

        reply = add_emoji(reply, mem["tone"])
        
        emotional_emoji = get_emotional_indicator(mem)
        # Emoji chance also increases with relationship
        relationship = mem["relationship"]
        emoji_chance = 0.2 + (relationship["affection_level"] / 80)  # 0.2-0.325 chance
        
        if emotional_emoji and random.random() < emoji_chance:
            reply = reply + " " + emotional_emoji

        print("\r" + " " * 20 + "\r", end="", flush=True)
        print("Vivy:", reply)
        sys.stdout.flush()
        
        history.append("Vivy: " + reply)
        mem["last_reply"] = reply

        save(mem)
        
    except EOFError:
        print("\nVivy: Bye... see you around 💫")
        sys.stdout.flush()
        break
    except KeyboardInterrupt:
        print("\nVivy: Take care... 💫")
        sys.stdout.flush()
        break
    except Exception as e:
        print(f"[Error: {e}]")
        sys.stdout.flush()
        continue
