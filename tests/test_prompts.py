from prompts.templates.base import PromptTemplate
from prompts.templates.classification import ZERO_SHOT, FEW_SHOT
from prompts.templates.qa import CONTEXT_BASED, REASONING
from prompts.templates.extraction import ENTITIES, JSON
from prompts.templates.summarization import BASIC, DETAILED

def test_format():
    template = PromptTemplate("Hello, {name}!")
    result = template.format(name="World")
    assert result == "Hello, World!"

def test_batch_format():
    template = PromptTemplate("Hello, {name}!")
    results = template.batch_format([{"name": "Alice"}, {"name": "Bob"}])
    assert results == ["Hello, Alice!", "Hello, Bob!"]

def test_classification_zero_shot():
    prompt = ZERO_SHOT.format(categories="positive, negative", text="I love this")
    assert "positive, negative" in prompt
    assert "I love this" in prompt
    assert "category" in prompt.lower()

def test_classification_few_shot():
    prompt = FEW_SHOT.format(categories="spam, not spam", 
                             examples="Example: 'Buy now' -> spam",
                             text="Hello friend")
    assert "spam, not spam" in prompt
    assert "Example" in prompt
    assert "Hello friend" in prompt

def test_qa_context_based():
    prompt = CONTEXT_BASED.format(context="The sky is blue.", question="What color is the sky?")
    assert "The sky is blue." in prompt
    assert "What color is the sky?" in prompt
    assert "I don't know" in prompt

def test_qa_reasoning():
    prompt = REASONING.format(context="Water boils at 100°C.", question="At what temperature does water boil?")
    assert "Water boils at 100°C." in prompt
    assert "At what temperature does water boil?" in prompt
    assert "Steps" in prompt

def test_extraction_entities():
    prompt = ENTITIES.format(entity_types="people, locations", text="John and Mary went to Paris.")
    assert "John" in prompt
    assert "Mary" in prompt
    assert "Paris" in prompt

def test_extraction_json():
    prompt = JSON.format(
        fields="date",
        json_format='{"date": ""}',
        text="The event is on 2024-06-15."
    )
    assert "2024-06-15" in prompt
    assert "JSON" in prompt
    
def test_summarization_basic():
    prompt = BASIC.format(text="This is a long article about AI.", n_sentences=3)
    assert "long article" in prompt
    assert "AI" in prompt
    
def test_summarization_detailed():
    prompt = DETAILED.format(text="This is a long article about AI.")
    assert "long article" in prompt
    assert "AI" in prompt
    assert "Key points" in prompt