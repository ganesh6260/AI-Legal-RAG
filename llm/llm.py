import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from rag.prompt_template import prompt_template

# Load API Key
load_dotenv()

# Create Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

def generate_answer(question, context):
    """
    Generate answer using LangChain + Gemini.
    """

    prompt = prompt_template.format(
        context=context,
        question=question
    )

    try:

        response = llm.invoke(prompt)

        # Most reliable way
        if hasattr(response, "text"):
            return response.text

        # Fallback
        if isinstance(response.content, str):
            return response.content

        if isinstance(response.content, list):
            answer = ""

            for item in response.content:

                if isinstance(item, dict):
                    answer += item.get("text", "")

            return answer.strip()

        return str(response.content)

    except Exception as e:

        error = str(e)

        if "429" in error:
            return """
⚠️ Gemini API limit reached.

Please wait one minute and try again.
"""

        if "503" in error:
            return """
⚠️ Gemini servers are busy.

Please try again after a few moments.
"""

        return f"❌ Error: {error}"