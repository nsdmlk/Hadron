class PromptTemplate:
    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

    def batch_format(self, batch_kwargs: list[dict]) -> list[str]:
        return [self.format(**kwargs) for kwargs in batch_kwargs]