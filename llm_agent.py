import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def answer_question_with_context(query, context):
    context_str = "\n".join(context)
    prompt = f"""
You are an intelligent assistant analyzing Excel data.

Context:
{context_str}

Question:
{query}

Answer in detail with reasoning, and perform any calculations if needed.
"""

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a data analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message["content"]
