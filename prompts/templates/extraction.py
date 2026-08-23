ENTITIES = """You are an entity extraction assistant.

Extract all entities from the text below. Entity types: {entity_types}

Text: {text}

Return each entity on a new line: "type: value". No explanation."""

JSON = """You are an information extraction assistant.

Extract the following fields from the text: {fields}

Text: {text}

Return ONLY valid JSON in this format:
{json_format}

No explanation, no markdown."""