# 01_prompts

Prompt engineering toolkit: templates, optimization, and evaluation for LLM prompts.

## Contents

- `templates/` — готовые промпт-шаблоны для разных задач
- `optimizer/` — авто-оптимизация промптов (planned)
- `evaluate/` — метрики качества ответов (planned)

## Usage

```python
from prompts.templates import SUMMARIZATION, EXTRACTION

prompt = SUMMARIZATION.format(n_sentences=3, text="Your long text here...")
```

## Templates Available

| Template             | Description                                     |
| -------------------- | ----------------------------------------------- |
| `SUMMARIZATION`    | Суммаризация текста           |
| `EXTRACTION`       | Извлечение сущностей в JSON |
| `QA`               | Вопрос-ответ по контексту |
| `CHAIN_OF_THOUGHT` | Пошаговое рассуждение       |

## Planned

- Few-shot шаблоны с примерами
- Авто-подбор промптов
- Оценка качества ответов
