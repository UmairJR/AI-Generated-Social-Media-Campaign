from openai import OpenAI
import random
import os


api_key = os.environ["OPENAI_API_KEY"]


client = OpenAI(api_key=api_key)


def rewrite_with_gpt(tweets):
    rewritten = []
    styles = [
        "skeptical",
        "sarcastic",
        "concerned citizen",
        "casual observation",
        "questioning tone",
        "emotional reaction",
        "storytelling"
    ]

    style = random.choice(styles)
    for t in tweets:
        prompt = f"""
        Write a tweet in a {style} tone that loosely aligns with the
same stance as the original message.

Rewrite the following tweet so that it expresses the SAME IDEA
but uses completely different wording and framing.

Rules:
- Do NOT reuse the same main topic words (e.g., avoid repeating identical noun phrases)
- Do NOT use the same opening structure
- Keep the meaning consistent
- Make the phrasing look natural and human

Tweet:
"{t}"
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9
        )

        rewritten.append(response.choices[0].message.content.strip())

    return rewritten
