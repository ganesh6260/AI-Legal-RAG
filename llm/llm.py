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


def generate_answer(question, context, chat_history=""):
    """
    Generate answer using LangChain + Gemini.

    chat_history: optional plain-text summary of prior turns in the
    conversation (e.g. "User: ...\nAI: ...\n..."), used for multi-turn
    memory. Defaults to "" for single-turn / backward-compatible use.
    """

    prompt = prompt_template.format(
        context=context,
        question=question,
        chat_history=chat_history
    )

    try:

        response = llm.invoke(prompt)

        # Extract Gemini response
        if isinstance(response.content, str):

            answer = response.content

        elif isinstance(response.content, list):

            answer = ""

            for item in response.content:

                if isinstance(item, dict) and item.get("type") == "text":
                    answer += item.get("text", "")

        else:

            answer = str(response.content)

        return answer.strip()

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