"""
Custom VADER-style sentiment analyzer + TF-IDF topic extraction.
Built from scratch using only pandas/numpy (no external NLP libraries needed).
"""
import re
import math
import numpy as np
import pandas as pd
from collections import Counter

# ============================================================
# PART 1: VADER-Style Sentiment Analyzer
# ============================================================

# Curated sentiment lexicon (based on VADER principles)
# Scores range from -4 (most negative) to +4 (most positive)
SENTIMENT_LEXICON = {
    # Strong positive
    "love": 3.2, "amazing": 3.1, "excellent": 3.0, "incredible": 3.0,
    "outstanding": 3.0, "fantastic": 3.0, "wonderful": 2.9, "perfect": 3.0,
    "impressed": 2.8, "awesome": 2.8, "brilliant": 2.8, "exceptional": 2.9,
    "superb": 2.9, "magnificent": 2.9, "remarkable": 2.7, "delightful": 2.7,
    "phenomenal": 2.9, "stellar": 2.8, "flawless": 3.0,

    # Moderate positive
    "great": 2.5, "good": 2.0, "nice": 1.8, "smooth": 2.3, "comfortable": 2.2,
    "safe": 2.3, "reliable": 2.2, "convenient": 2.0, "clean": 1.9,
    "happy": 2.4, "pleased": 2.2, "satisfied": 2.1, "enjoy": 2.3,
    "glad": 2.1, "pleasant": 2.0, "polished": 1.9, "premium": 2.0,
    "intuitive": 2.0, "impressive": 2.5, "improved": 1.8, "better": 1.8,
    "best": 2.5, "faster": 1.8, "easier": 1.7, "exciting": 2.2,
    "futuristic": 2.0, "innovative": 2.1, "competitive": 1.5, "transparent": 1.5,
    "predictable": 1.3, "accurate": 1.8, "gentle": 1.7, "luxury": 2.0,
    "converted": 1.5, "trust": 2.2, "secure": 2.0, "confident": 2.0,
    "prefer": 1.8, "worth": 1.5, "fair": 1.3,

    # Mild positive
    "okay": 0.8, "fine": 0.5, "decent": 1.0, "adequate": 0.5,
    "acceptable": 0.5, "improving": 1.2, "growing": 0.8,

    # Strong negative
    "terrible": -3.0, "horrible": -3.0, "awful": -3.0, "worst": -3.2,
    "disgusting": -3.0, "dreadful": -2.9, "abysmal": -3.0, "pathetic": -2.8,
    "atrocious": -3.0, "appalling": -2.9,

    # Moderate negative
    "bad": -2.0, "poor": -2.0, "hate": -2.8, "annoying": -2.1,
    "frustrating": -2.3, "disappointing": -2.2, "terrible": -3.0,
    "horrible": -3.0, "ridiculous": -2.3, "unacceptable": -2.5,
    "confusing": -1.8, "stressful": -2.2, "scary": -2.3, "nervous": -1.8,
    "worried": -1.5, "complaint": -2.0, "complained": -2.0, "issue": -1.3,
    "problem": -1.5, "broken": -2.2, "failed": -2.3, "crash": -2.5,
    "crashes": -2.5, "crashed": -2.5, "bug": -1.8, "worse": -2.2,
    "expensive": -1.5, "overpriced": -2.0, "outrageous": -2.5,
    "jerky": -1.8, "choppy": -1.7, "aggressive": -1.5, "weird": -1.2,
    "wrong": -1.8, "cancel": -1.5, "canceled": -1.8, "burned": -2.0,
    "limited": -1.3, "small": -0.8, "drains": -1.5, "struggle": -1.5,
    "confused": -1.8, "hesitated": -1.2, "unpredictable": -2.0,

    # Mild negative
    "minor": -0.5, "slight": -0.3, "mediocre": -1.0,

    # Booster words (amplify sentiment)
    "very": 0, "really": 0, "extremely": 0, "incredibly": 0,
    "absolutely": 0, "totally": 0, "so": 0, "seriously": 0,
    "genuinely": 0, "truly": 0,

    # Negation handled separately
}

BOOSTER_WORDS = {
    "very": 0.293, "really": 0.293, "extremely": 0.293,
    "incredibly": 0.293, "absolutely": 0.293, "totally": 0.293,
    "so": 0.293, "seriously": 0.293, "genuinely": 0.293, "truly": 0.293,
    "quite": 0.147, "pretty": 0.147, "somewhat": -0.147,
    "barely": -0.293, "hardly": -0.293,
}

NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "cannot", "can't",
                   "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't",
                   "couldn't", "isn't", "aren't", "wasn't", "weren't", "haven't",
                   "hasn't", "hadn't", "without"}

# Punctuation emphasis
EXCLAMATION_BOOST = 0.292


def tokenize(text):
    """Simple tokenizer."""
    text = text.lower()
    tokens = re.findall(r"[a-z']+", text)
    return tokens


def count_exclamations(text):
    return min(text.count("!"), 4)


def vader_sentiment(text):
    """
    Compute VADER-style sentiment scores.
    Returns dict with 'pos', 'neg', 'neu', 'compound' scores.
    """
    tokens = tokenize(text)
    sentiments = []

    for i, token in enumerate(tokens):
        if token in SENTIMENT_LEXICON:
            score = SENTIMENT_LEXICON[token]

            # Check for negation (within 3 words before)
            negated = False
            for j in range(max(0, i - 3), i):
                if tokens[j] in NEGATION_WORDS:
                    negated = True
                    break
            if negated:
                score *= -0.74

            # Check for boosters (within 2 words before)
            for j in range(max(0, i - 2), i):
                if tokens[j] in BOOSTER_WORDS:
                    boost = BOOSTER_WORDS[tokens[j]]
                    if score > 0:
                        score += boost
                    elif score < 0:
                        score -= boost

            sentiments.append(score)

    # Add exclamation emphasis
    n_excl = count_exclamations(text)
    excl_boost = n_excl * EXCLAMATION_BOOST

    if not sentiments:
        return {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0}

    sum_s = sum(sentiments)
    if sum_s > 0:
        sum_s += excl_boost
    elif sum_s < 0:
        sum_s -= excl_boost

    # Compute compound (normalize to [-1, 1])
    alpha = 15  # normalization constant
    compound = sum_s / math.sqrt(sum_s * sum_s + alpha)

    # Compute pos/neg/neu proportions
    pos_sum = sum(s for s in sentiments if s > 0)
    neg_sum = sum(abs(s) for s in sentiments if s < 0)
    neu_count = len(tokens) - len(sentiments)

    total = pos_sum + neg_sum + neu_count
    if total == 0:
        return {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": compound}

    pos = pos_sum / total
    neg = neg_sum / total
    neu = 1.0 - pos - neg

    return {
        "pos": round(pos, 3),
        "neg": round(neg, 3),
        "neu": round(max(0, neu), 3),
        "compound": round(compound, 4),
    }


def classify_sentiment(compound):
    """Classify compound score into positive/negative/neutral."""
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"


# ============================================================
# PART 2: TF-IDF Topic Extraction
# ============================================================

STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can",
    "will", "just", "don", "should", "now", "ve", "re", "ll", "didn", "doesn",
    "isn", "wasn", "aren", "hasn", "haven", "wouldn", "couldn", "shouldn",
    "also", "get", "got", "one", "two", "like", "even", "much", "really",
    "still", "going", "would", "could", "been", "had", "the", "it", "its",
    "car", "ride", "waymo", "app", "service", "time", "use", "used", "using",
    "thing", "things", "way", "make", "made", "feel", "felt",
}

TOPIC_KEYWORDS = {
    "Ride Quality & Smoothness": [
        "smooth", "smoothness", "braking", "acceleration", "turns", "gentle",
        "jerky", "choppy", "lane", "changes", "driving", "quality", "comfortable",
        "luxury", "chauffeur", "natural", "speed", "bumps", "merging", "highway",
    ],
    "Wait Times & Availability": [
        "wait", "waited", "waiting", "minutes", "pickup", "arrival", "available",
        "availability", "eta", "estimated", "cancel", "canceled", "rush", "hour",
        "late", "delay", "delayed", "long", "quick", "fast", "faster",
    ],
    "Safety & Trust": [
        "safe", "safety", "safer", "trust", "confident", "confidence", "nervous",
        "scared", "scary", "cautious", "pedestrians", "cyclists", "sensors",
        "defensive", "obstacle", "accident", "close", "call", "alert", "secure",
        "predictable", "unpredictable", "hesitated", "confused",
    ],
    "App & UX": [
        "app", "interface", "ui", "ux", "design", "tracking", "track", "gps",
        "location", "crash", "crashes", "froze", "freeze", "battery", "drains",
        "intuitive", "confusing", "clean", "polished", "book", "booking",
        "payment", "reinstall",
    ],
    "Coverage & Expansion": [
        "coverage", "area", "expand", "expanded", "expansion", "city", "cities",
        "neighborhood", "suburbs", "airport", "destinations", "map", "boundaries",
        "service", "limited", "growing",
    ],
    "Pricing & Value": [
        "price", "pricing", "prices", "cost", "expensive", "cheap", "cheaper",
        "affordable", "surge", "fare", "fares", "charged", "value", "worth",
        "penny", "tipping", "tip", "double", "rates", "outrageous", "transparency",
    ],
}


def extract_topics(text, top_n=3):
    """Extract topic labels from text using keyword matching."""
    tokens = set(tokenize(text))
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = len(tokens.intersection(set(keywords)))
        if score > 0:
            scores[topic] = score

    if not scores:
        return ["General Experience"]

    sorted_topics = sorted(scores.items(), key=lambda x: -x[1])
    return [t[0] for t in sorted_topics[:top_n]]


def compute_tfidf(documents, max_features=100):
    """Compute TF-IDF matrix from list of tokenized documents."""
    # Document frequency
    df_counts = Counter()
    doc_tokens = []
    for doc in documents:
        tokens = [t for t in tokenize(doc) if t not in STOP_WORDS and len(t) > 2]
        doc_tokens.append(tokens)
        unique_tokens = set(tokens)
        for t in unique_tokens:
            df_counts[t] += 1

    n_docs = len(documents)

    # Get top features by DF
    top_terms = [t for t, c in df_counts.most_common(max_features)]
    term_to_idx = {t: i for i, t in enumerate(top_terms)}

    # Build TF-IDF matrix
    tfidf_matrix = np.zeros((n_docs, len(top_terms)))
    for i, tokens in enumerate(doc_tokens):
        tf = Counter(tokens)
        for term, count in tf.items():
            if term in term_to_idx:
                j = term_to_idx[term]
                idf = math.log(n_docs / (1 + df_counts[term]))
                tfidf_matrix[i, j] = (1 + math.log(count)) * idf

    return tfidf_matrix, top_terms


def get_top_terms_by_group(df, group_col, text_col="text", n_terms=10):
    """Get top TF-IDF terms for each group."""
    results = {}
    for group_val in df[group_col].unique():
        subset = df[df[group_col] == group_val]
        if len(subset) < 5:
            continue

        # Compute word frequencies
        word_counts = Counter()
        for text in subset[text_col]:
            tokens = [t for t in tokenize(text) if t not in STOP_WORDS and len(t) > 2]
            word_counts.update(tokens)

        results[group_val] = word_counts.most_common(n_terms)

    return results


# ============================================================
# PART 3: Full Analysis Pipeline
# ============================================================

def run_analysis(csv_path):
    """Run complete sentiment + topic analysis pipeline."""
    df = pd.read_csv(csv_path, parse_dates=["date"])

    print("Running sentiment analysis...")
    sentiment_results = df["text"].apply(vader_sentiment)
    df["compound"] = sentiment_results.apply(lambda x: x["compound"])
    df["pos_score"] = sentiment_results.apply(lambda x: x["pos"])
    df["neg_score"] = sentiment_results.apply(lambda x: x["neg"])
    df["neu_score"] = sentiment_results.apply(lambda x: x["neu"])
    df["sentiment_label"] = df["compound"].apply(classify_sentiment)

    print("Extracting topics...")
    df["topics"] = df["text"].apply(lambda x: extract_topics(x))
    df["primary_topic"] = df["topics"].apply(lambda x: x[0])

    print("Computing monthly aggregates...")
    df["month"] = df["date"].dt.to_period("M")
    df["week"] = df["date"].dt.to_period("W")

    # Monthly sentiment trends
    monthly = df.groupby("month").agg(
        avg_compound=("compound", "mean"),
        avg_rating=("rating", "mean"),
        review_count=("review_id", "count"),
        pct_positive=("sentiment_label", lambda x: (x == "positive").mean()),
        pct_negative=("sentiment_label", lambda x: (x == "negative").mean()),
    ).reset_index()
    monthly["month_str"] = monthly["month"].astype(str)

    # Topic sentiment
    topic_sentiment = df.groupby("primary_topic").agg(
        avg_compound=("compound", "mean"),
        avg_rating=("rating", "mean"),
        review_count=("review_id", "count"),
        pct_positive=("sentiment_label", lambda x: (x == "positive").mean()),
    ).reset_index()

    # City sentiment
    city_sentiment = df.groupby("city").agg(
        avg_compound=("compound", "mean"),
        avg_rating=("rating", "mean"),
        review_count=("review_id", "count"),
        pct_positive=("sentiment_label", lambda x: (x == "positive").mean()),
    ).reset_index()

    # City x Month trends
    city_monthly = df.groupby(["city", "month"]).agg(
        avg_compound=("compound", "mean"),
        review_count=("review_id", "count"),
    ).reset_index()
    city_monthly["month_str"] = city_monthly["month"].astype(str)

    # Topic x Month trends
    topic_monthly = df.groupby(["primary_topic", "month"]).agg(
        avg_compound=("compound", "mean"),
        review_count=("review_id", "count"),
    ).reset_index()
    topic_monthly["month_str"] = topic_monthly["month"].astype(str)

    # Top terms by sentiment
    top_terms = get_top_terms_by_group(df, "sentiment_label")

    print(f"\n=== ANALYSIS SUMMARY ===")
    print(f"Total reviews analyzed: {len(df)}")
    print(f"Overall avg compound sentiment: {df['compound'].mean():.3f}")
    print(f"Overall avg rating: {df['rating'].mean():.2f}")
    print(f"\nSentiment distribution:")
    print(df["sentiment_label"].value_counts())
    print(f"\nTopic distribution:")
    print(df["primary_topic"].value_counts())
    print(f"\nTopic sentiment (avg compound):")
    for _, row in topic_sentiment.sort_values("avg_compound").iterrows():
        print(f"  {row['primary_topic']}: {row['avg_compound']:.3f} ({row['review_count']} reviews)")

    return {
        "df": df,
        "monthly": monthly,
        "topic_sentiment": topic_sentiment,
        "city_sentiment": city_sentiment,
        "city_monthly": city_monthly,
        "topic_monthly": topic_monthly,
        "top_terms": top_terms,
        "milestones": MILESTONES,
    }


# Milestones exported for use by other modules
MILESTONES = [
    {"date": "2025-04-15", "event": "10M paid trips", "type": "milestone"},
    {"date": "2025-06-01", "event": "Austin launch", "type": "expansion"},
    {"date": "2025-07-15", "event": "Atlanta launch", "type": "expansion"},
    {"date": "2025-09-01", "event": "Miami expansion", "type": "expansion"},
    {"date": "2025-10-20", "event": "150K weekly trips", "type": "milestone"},
    {"date": "2025-12-01", "event": "Holiday pricing controversy", "type": "controversy"},
    {"date": "2026-01-15", "event": "Tokyo partnership", "type": "expansion"},
    {"date": "2026-02-20", "event": "20M cumulative trips", "type": "milestone"},
]


if __name__ == "__main__":
    results = run_analysis("/sessions/wonderful-serene-volta/waymo_reviews_raw.csv")
    # Save analyzed data
    results["df"].to_csv("/sessions/wonderful-serene-volta/waymo_reviews_analyzed.csv", index=False)
    print("\nAnalyzed data saved to waymo_reviews_analyzed.csv")
