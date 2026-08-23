ZERO_SHOT = """You are a text classification assistant.
Classify the text below into one of these categories: {categories}

Text: {text}

Return only the category name, no explanation."""


FEW_SHOT = """You are a text classification assistant.
Classify the text below into one of these categories: {categories}

Examples:
{examples}

Text: {text}

Return only the category name, no explanation."""