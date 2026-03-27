"""
Generate synthetic word embeddings with realistic semantic clustering.
Creates ~1000 words across semantic categories, generates high-dim vectors
with intra-cluster similarity, then reduces to 3D via t-SNE.
"""
import numpy as np
from sklearn.manifold import TSNE
import json

np.random.seed(42)

# Define semantic categories with representative words
categories = {
    "names_male": [
        "john", "james", "william", "richard", "robert", "george", "david",
        "michael", "thomas", "charles", "peter", "paul", "edward", "henry",
        "joseph", "daniel", "matthew", "andrew", "simon", "martin", "stephen",
        "mark", "patrick", "christopher", "timothy", "benjamin", "samuel",
        "nicholas", "alexander", "jonathan", "anthony", "francis", "arthur",
        "raymond", "gerald", "eugene", "harold", "albert", "carl", "ralph",
        "howard", "lawrence", "ernest", "phillip", "walter", "bernard", "lee",
        "graham", "smith", "alan", "roger", "keith", "brian", "dennis", "jerry"
    ],
    "names_female": [
        "mary", "elizabeth", "patricia", "jennifer", "linda", "barbara",
        "susan", "jessica", "sarah", "margaret", "dorothy", "lisa", "nancy",
        "betty", "helen", "sandra", "donna", "carol", "ruth", "sharon",
        "michelle", "laura", "emily", "amanda", "melissa", "deborah", "anna",
        "rebecca", "virginia", "catherine", "christine", "kathleen", "janet",
        "diane", "alice", "julie", "heather", "teresa", "gloria", "evelyn",
        "joan", "victoria", "cheryl", "megan", "andrea", "hannah", "jacqueline"
    ],
    "numbers": [
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
        "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
        "hundred", "thousand", "million", "billion", "trillion", "dozen",
        "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
        "eighth", "ninth", "tenth", "twice", "triple", "double", "half",
        "quarter"
    ],
    "months_time": [
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december", "monday",
        "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "morning", "afternoon", "evening", "night", "midnight", "noon",
        "dawn", "dusk", "twilight", "today", "tomorrow", "yesterday",
        "week", "month", "year", "decade", "century", "spring", "summer",
        "autumn", "winter", "season", "daily", "weekly", "monthly", "annual"
    ],
    "countries": [
        "america", "england", "france", "germany", "china", "japan", "india",
        "russia", "brazil", "canada", "australia", "italy", "spain", "mexico",
        "korea", "turkey", "egypt", "iran", "iraq", "israel", "pakistan",
        "indonesia", "nigeria", "argentina", "colombia", "poland", "ukraine",
        "netherlands", "belgium", "sweden", "norway", "denmark", "finland",
        "switzerland", "austria", "portugal", "greece", "ireland", "scotland",
        "thailand", "vietnam", "philippines", "malaysia", "singapore"
    ],
    "cities": [
        "london", "paris", "tokyo", "york", "berlin", "rome", "moscow",
        "beijing", "washington", "chicago", "boston", "dallas", "seattle",
        "miami", "denver", "atlanta", "detroit", "phoenix", "toronto",
        "sydney", "melbourne", "mumbai", "delhi", "shanghai", "bangkok",
        "cairo", "istanbul", "madrid", "barcelona", "amsterdam", "brussels",
        "vienna", "prague", "warsaw", "budapest", "dublin", "edinburgh",
        "lisbon", "stockholm", "oslo", "copenhagen", "helsinki"
    ],
    "nature": [
        "river", "mountain", "ocean", "forest", "desert", "valley", "lake",
        "island", "beach", "cliff", "volcano", "glacier", "canyon", "prairie",
        "jungle", "swamp", "reef", "cave", "waterfall", "meadow", "hill",
        "stream", "pond", "bay", "harbor", "coast", "shore", "peak",
        "ridge", "plateau", "tundra", "marsh", "delta", "basin", "peninsula",
        "tree", "flower", "grass", "leaf", "root", "seed", "bloom", "branch"
    ],
    "animals": [
        "dog", "cat", "horse", "bird", "fish", "bear", "wolf", "lion",
        "tiger", "eagle", "shark", "whale", "dolphin", "elephant", "monkey",
        "snake", "rabbit", "deer", "fox", "owl", "hawk", "crow", "sparrow",
        "salmon", "trout", "turtle", "frog", "mouse", "rat", "pig",
        "cow", "sheep", "goat", "chicken", "duck", "goose", "swan",
        "penguin", "leopard", "cheetah", "gorilla", "panda", "zebra"
    ],
    "colors": [
        "red", "blue", "green", "yellow", "black", "white", "orange",
        "purple", "pink", "brown", "gray", "gold", "silver", "crimson",
        "scarlet", "azure", "emerald", "violet", "ivory", "coral",
        "turquoise", "indigo", "maroon", "navy", "olive", "tan", "beige",
        "amber", "bronze", "copper", "platinum", "ruby", "sapphire"
    ],
    "body": [
        "head", "eye", "hand", "heart", "brain", "blood", "bone", "skin",
        "face", "arm", "leg", "foot", "finger", "hair", "mouth", "nose",
        "ear", "neck", "chest", "shoulder", "knee", "teeth", "tongue",
        "lip", "thumb", "wrist", "elbow", "ankle", "spine", "skull",
        "muscle", "nerve", "lung", "liver", "stomach", "kidney"
    ],
    "emotions": [
        "love", "fear", "anger", "joy", "sadness", "hope", "pride",
        "shame", "guilt", "envy", "jealousy", "trust", "surprise",
        "disgust", "contempt", "anxiety", "grief", "happiness", "sorrow",
        "rage", "terror", "delight", "pleasure", "pain", "suffering",
        "compassion", "empathy", "sympathy", "gratitude", "regret",
        "loneliness", "excitement", "boredom", "confusion", "curiosity"
    ],
    "technology": [
        "computer", "software", "internet", "network", "data", "server",
        "code", "algorithm", "database", "program", "system", "digital",
        "cloud", "security", "encryption", "protocol", "bandwidth",
        "processor", "memory", "storage", "binary", "pixel", "interface",
        "robot", "automation", "artificial", "intelligence", "machine",
        "learning", "neural", "quantum", "cyber", "virtual", "blockchain"
    ],
    "science": [
        "atom", "molecule", "cell", "gene", "protein", "electron",
        "proton", "neutron", "photon", "gravity", "energy", "mass",
        "velocity", "force", "wave", "particle", "spectrum", "frequency",
        "radiation", "entropy", "catalyst", "isotope", "ion", "plasma",
        "nucleus", "orbit", "quark", "boson", "relativity", "evolution",
        "mutation", "genome", "chromosome", "enzyme", "antibody"
    ],
    "function_words": [
        "the", "and", "but", "or", "if", "then", "because", "although",
        "however", "therefore", "moreover", "furthermore", "nevertheless",
        "meanwhile", "consequently", "otherwise", "instead", "besides",
        "accordingly", "hence", "thus", "indeed", "perhaps", "probably",
        "certainly", "definitely", "apparently", "presumably", "supposedly",
        "regarding", "concerning", "according", "following", "including",
        "during", "between", "through", "within", "without", "against",
        "until", "since", "before", "after", "above", "below", "beyond"
    ],
    "academic": [
        "research", "study", "analysis", "theory", "method", "evidence",
        "hypothesis", "experiment", "observation", "conclusion", "abstract",
        "journal", "publication", "reference", "citation", "review",
        "thesis", "dissertation", "professor", "university", "academic",
        "scholar", "lecture", "seminar", "conference", "symposium",
        "chapter", "volume", "edition", "appendix", "bibliography", "index",
        "footnote", "manuscript", "peer", "methodology", "framework"
    ],
    "military": [
        "army", "navy", "soldier", "general", "captain", "colonel",
        "sergeant", "lieutenant", "battalion", "regiment", "brigade",
        "division", "corps", "fleet", "squadron", "platoon", "infantry",
        "artillery", "cavalry", "missile", "radar", "submarine", "aircraft",
        "tank", "weapon", "ammunition", "fortress", "siege", "combat",
        "reconnaissance", "intelligence", "strategy", "tactical", "deployment"
    ],
    "food": [
        "bread", "rice", "meat", "chicken", "beef", "pork", "cheese",
        "butter", "milk", "sugar", "salt", "pepper", "flour", "egg",
        "potato", "tomato", "onion", "garlic", "carrot", "corn", "wheat",
        "apple", "banana", "orange", "grape", "lemon", "strawberry",
        "chocolate", "coffee", "tea", "wine", "beer", "soup", "salad",
        "pasta", "pizza", "sauce", "oil", "vinegar", "honey", "cream"
    ],
    "music": [
        "song", "melody", "rhythm", "harmony", "chord", "note", "beat",
        "tempo", "pitch", "tone", "bass", "treble", "vocal", "guitar",
        "piano", "drum", "violin", "trumpet", "flute", "saxophone",
        "orchestra", "symphony", "concert", "album", "lyric", "chorus",
        "verse", "bridge", "solo", "acoustic", "jazz", "blues", "rock",
        "classical", "opera", "folk", "rap", "hip"
    ],
}

# Generate embeddings
DIM = 100  # high-dimensional space
all_words = []
all_vectors = []
all_categories = []

# Create a centroid for each category
centroids = {}
for i, cat in enumerate(categories.keys()):
    centroid = np.random.randn(DIM) * 0.3
    # Spread categories apart
    angle = 2 * np.pi * i / len(categories)
    centroid[0] += 3 * np.cos(angle)
    centroid[1] += 3 * np.sin(angle)
    centroid[int(i % DIM)] += 2.0  # additional separation dimension
    centroids[cat] = centroid

# Generate word vectors around centroids
for cat, words in categories.items():
    centroid = centroids[cat]
    for word in words:
        # Vector near centroid with some noise
        vec = centroid + np.random.randn(DIM) * 0.4
        all_words.append(word)
        all_vectors.append(vec)
        all_categories.append(cat)

vectors = np.array(all_vectors, dtype=np.float32)
print(f"Generated {len(all_words)} words across {len(categories)} categories")
print(f"Vector shape: {vectors.shape}")

# t-SNE reduction to 3D
print("Running t-SNE (3D)...")
tsne = TSNE(
    n_components=3,
    perplexity=30,
    max_iter=1000,
    random_state=42,
    learning_rate='auto',
    init='pca'
)
coords_3d = tsne.fit_transform(vectors)

# Normalize to [-50, 50] range for Three.js
for dim in range(3):
    mn, mx = coords_3d[:, dim].min(), coords_3d[:, dim].max()
    coords_3d[:, dim] = ((coords_3d[:, dim] - mn) / (mx - mn) - 0.5) * 100

# Build JSON output
# Category color map
color_map = {
    "names_male": "#4a7cff",
    "names_female": "#c77dff",
    "numbers": "#00d4aa",
    "months_time": "#ffb347",
    "countries": "#ff6b6b",
    "cities": "#ff9ecf",
    "nature": "#2ecc71",
    "animals": "#f39c12",
    "colors": "#e74c3c",
    "body": "#fd79a8",
    "emotions": "#e056fd",
    "technology": "#00cec9",
    "science": "#6c5ce7",
    "function_words": "#95a5a6",
    "academic": "#dfe6e9",
    "military": "#636e72",
    "food": "#fdcb6e",
    "music": "#ff7675",
}

data = {
    "words": [],
    "categories": list(categories.keys()),
    "color_map": color_map
}

for i, word in enumerate(all_words):
    data["words"].append({
        "word": word,
        "x": float(coords_3d[i, 0]),
        "y": float(coords_3d[i, 1]),
        "z": float(coords_3d[i, 2]),
        "category": all_categories[i],
        "color": color_map[all_categories[i]]
    })

with open("/home/claude/embedding-viz/data/embeddings_3d.json", "w") as f:
    json.dump(data, f)

print(f"Saved {len(data['words'])} word embeddings to embeddings_3d.json")
print("Sample:", data["words"][:3])
