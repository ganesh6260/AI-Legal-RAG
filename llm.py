import os
from dotenv import load_dotenv
from google import genai

# Load API Key
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


def generate_answer(question, context):
    """
    Generate answer using Gemini based only on retrieved context.
    """

    prompt = f"""
You are an expert AI Legal Assistant.

Your task is to answer ONLY from the provided document context.

Instructions:
- Read the context carefully before answering.
- Answer ONLY using the information available in the context.
- Never use your own knowledge.
- Never guess or hallucinate.
- If the answer is not available in the context, reply exactly:
"I could not find the answer in the uploaded document."
- Keep the answer professional and easy to understand.
- Use bullet points whenever appropriate.
- Do NOT say phrases like:
  "According to the context..."
  "Based on the provided context..."
- Give a direct answer.

=========================
DOCUMENT CONTEXT
=========================

{context}

=========================
USER QUESTION
=========================

{question}

=========================
ANSWER
=========================
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )

        return response.text.strip()

    except Exception as e:

        error = str(e)

        if "429" in error:
            return """
⚠️ **Gemini API Limit Reached**

The free Gemini API has temporarily reached its request limit.

Please wait **about 1 minute** and try again.

💡 Tip: This is normal on the free tier.
"""

        return f"❌ Error: {error}"