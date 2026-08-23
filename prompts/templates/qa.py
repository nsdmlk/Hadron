CONTEXT_BASED = """You are a question answering assistant.

Answer the question based ONLY on the context below.
If the context doesn't contain the answer, say "I don't know".

Context: {context}

Question: {question}

Answer:"""

REASONING = """You are a reasoning assistant.

Solve the question step by step using the context below.
Show each step of your reasoning, then give the final answer.

Context: {context}

Question: {question}

Steps:
Final Answer:"""