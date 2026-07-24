from llm import generate_answer

context = """
Ganesh knows Java, Python and SQL.
"""

question = "What languages does Ganesh know?"

print(generate_answer(question, context))