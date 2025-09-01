from openai import OpenAI
import pandas as pd
import json

# init client
client = OpenAI(api_key="API")

# @app.post("/generate_itinerary", response_model=ItineraryResponse)
def generate_itinerary(top_locations: pd.DataFrame):
    """
    Takes in DataFrame that has been created from embedding of reviews and associated locations plus keyword embeddings.
    Function also needs to include number of days somewhere.
    """
    # Extract top cities and keywords from DataFrame dict
    df = top_locations.to_dict(orient="records")
    top_cities = list({entry['location'] for entry in df})[:5]
    keywords = list({entry['keywords'] for entry in df})

    # Minimal GPT prompt
    prompt = f"""
    Create a 5 day itinerary for these cities: {', '.join(top_cities)}.
    Use these holiday preferences: {', '.join(keywords)}.
    Output must be structured JSON like this:
    {{
        "Day 1": [{{"time": "09:00", "activity": "...", "location": "City"}}],
        "Day 2": [ ... ]
    }}
    """

    # Call GPT
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-4o" / "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "You are a helpful travel planner."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract content
    content = response.choices[0].message.content

    # Try to parse JSON safely
    try:
        itinerary = json.loads(content)
    except json.JSONDecodeError:
        itinerary = {"error": "Could not parse JSON", "raw_output": content}

    return {"itinerary": itinerary}
