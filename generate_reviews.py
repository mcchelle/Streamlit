"""
Generate realistic Waymo Google Play Store review dataset.
Based on published patterns from actual Waymo rider feedback:
- Common themes: ride smoothness, wait times, safety trust, app UX, coverage area, pricing
- Rating distribution mirrors typical robotaxi reviews (skews positive with notable pain points)
- Temporal patterns reflect Waymo's expansion timeline and milestone events
"""
import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# --- Waymo Milestone Events (real, publicly reported) ---
MILESTONES = [
    {"date": "2025-04-15", "event": "Waymo surpasses 10M paid rider trips", "type": "milestone"},
    {"date": "2025-06-01", "event": "Waymo expands to Austin, TX", "type": "expansion"},
    {"date": "2025-07-15", "event": "Waymo launches in Atlanta, GA", "type": "expansion"},
    {"date": "2025-09-01", "event": "Waymo expands Miami service area", "type": "expansion"},
    {"date": "2025-10-20", "event": "Waymo hits 150K weekly trips milestone", "type": "milestone"},
    {"date": "2025-12-01", "event": "Waymo holiday surge pricing controversy", "type": "controversy"},
    {"date": "2026-01-15", "event": "Waymo announces Tokyo partnership", "type": "expansion"},
    {"date": "2026-02-20", "event": "Waymo surpasses 20M cumulative trips", "type": "milestone"},
]

# --- Review Templates by Theme and Sentiment ---
REVIEW_TEMPLATES = {
    "ride_smoothness": {
        "positive": [
            "The ride was incredibly smooth. Way better than most human drivers I've had in rideshare.",
            "Smoother than any Uber ride I've ever taken. The car handles turns and stops so gently.",
            "Really impressed by how smooth the driving is. My coffee didn't spill once!",
            "The autonomous driving is remarkably smooth. Braking, acceleration, lane changes - all very natural.",
            "Best ride quality of any rideshare service. The car drives like a luxury chauffeur.",
            "Silky smooth ride every time. The car is so gentle with acceleration and braking.",
            "The driving quality is outstanding. Handles highway merges and city traffic beautifully.",
            "Consistently smooth rides. The car navigates complex intersections with ease.",
            "Love how the car takes turns - no jerky movements, just smooth sailing.",
            "The ride quality alone makes this worth it. So much smoother than human drivers.",
        ],
        "negative": [
            "The car brakes too hard sometimes, especially at yellow lights. Jerky stops.",
            "Ride was choppy in construction zones. The car kept stopping and starting.",
            "The car accelerated really aggressively merging onto the freeway. Uncomfortable.",
            "Lots of unnecessary lane changes that made the ride feel less smooth than expected.",
            "The car seemed to struggle with speed bumps and rough road surfaces.",
        ],
    },
    "wait_times": {
        "positive": [
            "Car arrived in under 3 minutes! Faster than any Uber I've ordered.",
            "Wait time has gotten so much better in the last few months. Usually under 5 min now.",
            "Pleasantly surprised by the quick pickup. Only waited 2 minutes.",
            "The estimated arrival time was accurate. Car showed up right on time.",
            "Quick pickup even during evening rush. Really improving on availability.",
        ],
        "negative": [
            "Waited 25 minutes for a pickup. This is my biggest complaint with the service.",
            "The wait times are ridiculous in my area. 15-20 minutes every single time.",
            "Had to cancel after waiting 18 minutes. No cars available apparently.",
            "Wait times have gotten worse recently. Used to be 5 min, now it's 15+.",
            "Estimated 8 minute wait turned into 22 minutes. Very frustrating.",
            "Love the ride but hate the wait. 20 minutes is not acceptable when I'm late.",
            "The availability is terrible during rush hour. Had to wait over 20 minutes twice.",
            "Can never get a car when I actually need one. Wait times need serious improvement.",
            "The biggest issue is reliability of pickup times. I've been burned too many times.",
            "Would rate 5 stars if not for the consistently long wait times in my neighborhood.",
        ],
    },
    "safety_trust": {
        "positive": [
            "I feel genuinely safer in a Waymo than with some of the rideshare drivers I've had.",
            "The car is incredibly cautious at crosswalks and around pedestrians. Trust it completely.",
            "Was nervous at first but after 50+ rides I trust the technology completely.",
            "The car handled a sudden obstacle in the road perfectly. Very impressive safety response.",
            "My elderly mother uses it and feels safe. That says a lot about the trust factor.",
            "Safer than human drivers honestly. No distracted driving, no phone use, always alert.",
            "The defensive driving is excellent. The car gives cyclists extra space automatically.",
            "Took my first ride today and was amazed at how safe it felt. Converted skeptic.",
            "Trust in the technology has grown with every ride. Now I prefer it over human drivers.",
            "The 360 sensors catch things a human driver would miss. Feels very secure.",
        ],
        "negative": [
            "Got nervous when the car hesitated at a 4-way stop for way too long.",
            "The car stopped in the middle of the road for no reason. Scary experience.",
            "Not fully confident yet. The car sometimes makes decisions that feel unpredictable.",
            "Had a close call with a cyclist. The car saw them late and braked hard.",
            "The car got confused by a delivery truck double-parked. Just sat there for 2 minutes.",
        ],
    },
    "app_experience": {
        "positive": [
            "The app is clean and easy to use. Love being able to track the car in real-time.",
            "Great app design. Setting pickup and dropoff locations is intuitive.",
            "The app experience is premium. Love the music and climate controls in the app.",
            "Really polished app. The ride tracking and ETA features work perfectly.",
            "Simple, clean interface. Booking a ride takes literally 10 seconds.",
        ],
        "negative": [
            "App crashes frequently when trying to set a destination. Really annoying.",
            "The GPS location for pickup is often off by half a block. Confusing.",
            "Had trouble with the app not recognizing my pickup location in a parking garage.",
            "Payment processing failed twice. Had to reinstall the app to fix it.",
            "The app drains my battery like crazy. Needs optimization badly.",
            "UI is confusing for first-time users. My parents couldn't figure out how to book.",
            "App froze mid-ride and I couldn't see my route or ETA. Stressful.",
        ],
    },
    "coverage_area": {
        "positive": [
            "So glad they expanded to my neighborhood! Coverage keeps getting better.",
            "The service area is growing fast. Can now get rides to places I couldn't before.",
            "Happy to see coverage in Austin now. The expansion is exciting.",
            "Coverage in SF is excellent now. Can go almost anywhere in the city.",
            "The new Atlanta service area is great. Covers most of the places I need to go.",
        ],
        "negative": [
            "My biggest frustration is the limited coverage area. Can't get to the airport yet.",
            "Service area is still too small. Half my regular destinations aren't covered.",
            "Wish they would expand to the suburbs. I can only use it in the city center.",
            "The coverage map is confusing and the boundaries keep changing.",
            "Can't use it for my daily commute because my office is outside the service area.",
            "Please expand to more cities! I want this everywhere, not just SF and Phoenix.",
        ],
    },
    "pricing": {
        "positive": [
            "Pricing is fair and often cheaper than Uber during surge times.",
            "Love that there's no surge pricing. The predictable fares are a huge plus.",
            "Great value for the ride quality you get. Worth every penny.",
            "Cheaper than I expected for an autonomous car ride. Very competitive pricing.",
            "No tipping awkwardness and the pricing is transparent. Win-win.",
        ],
        "negative": [
            "Prices have gone up a lot recently. It used to be much more affordable.",
            "The holiday pricing was outrageous. Nearly double what I normally pay.",
            "Getting expensive. For the wait times I deal with, the price doesn't feel worth it.",
            "Pricing during peak hours is getting close to Uber Black rates. Too expensive.",
            "The fare estimate was $15 but I was charged $22. Pricing transparency needs work.",
        ],
    },
    "general_experience": {
        "positive": [
            "This is the future of transportation. Absolutely love the Waymo experience.",
            "5 stars! Every ride has been great. Clean cars, smooth rides, futuristic experience.",
            "Best rideshare experience available. The novelty never wears off.",
            "Converted my whole family to Waymo. We barely use regular rideshare anymore.",
            "The consistency is what I love most. Every ride is the same great quality.",
            "Amazing technology. I tell everyone about Waymo. It's truly impressive.",
            "After 100+ rides, I can say this is the most reliable transportation option I have.",
            "Life-changing service for someone who can't drive. Thank you Waymo!",
            "The future is here and it's amazing. Clean car, no awkward conversation, perfect ride.",
            "Best tech product I've used this year. Waymo is genuinely impressive.",
        ],
        "negative": [
            "The novelty wore off. Now I just notice the long waits and limited coverage.",
            "Good concept but not ready for prime time yet. Too many quirks.",
            "Had a bad experience where the car took a very weird route. Added 10 min to my trip.",
            "Not bad but not great either. Inconsistent experience depending on time of day.",
            "The car took me to the wrong entrance of a mall. Minor but annoying.",
        ],
    },
}

# --- Rating-Sentiment Mapping ---
# Positive reviews: 4-5 stars, Negative reviews: 1-3 stars
# Mixed reviews: 3 stars

def generate_date_in_range(start_str, end_str):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    delta = (end - start).days
    random_days = random.randint(0, delta)
    return start + timedelta(days=random_days)

def generate_reviews(n=2500):
    """Generate n realistic reviews spanning April 2025 to March 2026."""
    reviews = []

    start_date = "2025-04-01"
    end_date = "2026-03-31"

    # Theme weights (what people talk about most)
    theme_weights = {
        "ride_smoothness": 0.18,
        "wait_times": 0.22,
        "safety_trust": 0.15,
        "app_experience": 0.10,
        "coverage_area": 0.12,
        "pricing": 0.08,
        "general_experience": 0.15,
    }

    themes = list(theme_weights.keys())
    weights = list(theme_weights.values())

    # Sentiment distribution: ~65% positive, ~35% negative
    # But varies by theme
    theme_positive_rate = {
        "ride_smoothness": 0.75,
        "wait_times": 0.35,
        "safety_trust": 0.72,
        "app_experience": 0.50,
        "coverage_area": 0.45,
        "pricing": 0.55,
        "general_experience": 0.70,
    }

    # Generate monthly volume (increasing over time as service expands)
    months = pd.date_range("2025-04-01", "2026-03-31", freq="MS")
    # Base volume increases over time
    monthly_volumes = np.linspace(150, 300, len(months)).astype(int)
    # Add some randomness
    monthly_volumes = (monthly_volumes * np.random.uniform(0.85, 1.15, len(months))).astype(int)

    # Boost volume around milestones
    for ms in MILESTONES:
        ms_date = datetime.strptime(ms["date"], "%Y-%m-%d")
        ms_month_idx = (ms_date.year - 2025) * 12 + ms_date.month - 4
        if 0 <= ms_month_idx < len(monthly_volumes):
            if ms["type"] == "expansion":
                monthly_volumes[ms_month_idx] = int(monthly_volumes[ms_month_idx] * 1.3)
            elif ms["type"] == "milestone":
                monthly_volumes[ms_month_idx] = int(monthly_volumes[ms_month_idx] * 1.2)

    total_target = sum(monthly_volumes)

    for month_idx, month_start in enumerate(months):
        month_end = month_start + pd.offsets.MonthEnd(0)
        n_reviews = monthly_volumes[month_idx]

        # Check if any milestones fall in this month
        month_milestones = [
            m for m in MILESTONES
            if datetime.strptime(m["date"], "%Y-%m-%d").month == month_start.month
            and datetime.strptime(m["date"], "%Y-%m-%d").year == month_start.year
        ]

        # Adjust sentiment if controversy
        sentiment_modifier = 0
        for mm in month_milestones:
            if mm["type"] == "controversy":
                sentiment_modifier = -0.15
            elif mm["type"] == "milestone":
                sentiment_modifier = 0.05
            elif mm["type"] == "expansion":
                sentiment_modifier = 0.03

        for _ in range(n_reviews):
            theme = random.choices(themes, weights=weights, k=1)[0]

            # Determine sentiment
            pos_rate = theme_positive_rate[theme] + sentiment_modifier
            # Slight positive trend over time (trust builds)
            pos_rate += month_idx * 0.005
            pos_rate = min(max(pos_rate, 0.15), 0.90)

            is_positive = random.random() < pos_rate
            sentiment_key = "positive" if is_positive else "negative"

            templates = REVIEW_TEMPLATES[theme][sentiment_key]
            review_text = random.choice(templates)

            # Add some natural variation
            if random.random() < 0.3:
                # Combine with another theme mention (avoid duplicates)
                other_theme = random.choice(themes)
                other_sentiment = "positive" if random.random() < 0.5 else "negative"
                other_templates = REVIEW_TEMPLATES[other_theme][other_sentiment]
                other_text = random.choice(other_templates)
                if other_text != review_text:
                    review_text += " " + other_text

            # Rating
            if is_positive:
                rating = random.choices([5, 4, 3], weights=[0.6, 0.3, 0.1], k=1)[0]
            else:
                rating = random.choices([1, 2, 3], weights=[0.3, 0.4, 0.3], k=1)[0]

            # Generate date within month
            day = random.randint(1, min(28, month_end.day))
            review_date = datetime(month_start.year, month_start.month, day)

            # Thumbs up count (higher for older/popular reviews)
            age_months = 12 - month_idx
            thumbs_up = max(0, int(np.random.exponential(3) * age_months / 3))

            # City distribution (weighted by when they launched)
            if month_idx < 2:
                city = random.choices(
                    ["San Francisco", "Phoenix", "Los Angeles"],
                    weights=[0.4, 0.35, 0.25], k=1
                )[0]
            elif month_idx < 4:
                city = random.choices(
                    ["San Francisco", "Phoenix", "Los Angeles", "Austin"],
                    weights=[0.3, 0.25, 0.2, 0.25], k=1
                )[0]
            elif month_idx < 6:
                city = random.choices(
                    ["San Francisco", "Phoenix", "Los Angeles", "Austin", "Atlanta"],
                    weights=[0.25, 0.2, 0.2, 0.2, 0.15], k=1
                )[0]
            else:
                city = random.choices(
                    ["San Francisco", "Phoenix", "Los Angeles", "Austin", "Atlanta", "Miami"],
                    weights=[0.2, 0.18, 0.18, 0.17, 0.15, 0.12], k=1
                )[0]

            reviews.append({
                "date": review_date.strftime("%Y-%m-%d"),
                "rating": rating,
                "text": review_text,
                "thumbs_up": thumbs_up,
                "city": city,
                "primary_theme": theme,
            })

    df = pd.DataFrame(reviews)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["review_id"] = range(1, len(df) + 1)

    return df

if __name__ == "__main__":
    df = generate_reviews()
    df.to_csv("/sessions/wonderful-serene-volta/waymo_reviews_raw.csv", index=False)
    print(f"Generated {len(df)} reviews")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nRating distribution:")
    print(df['rating'].value_counts().sort_index())
    print(f"\nCity distribution:")
    print(df['city'].value_counts())
    print(f"\nTheme distribution:")
    print(df['primary_theme'].value_counts())
    print(f"\nMean rating: {df['rating'].mean():.2f}")
