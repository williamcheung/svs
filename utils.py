def load_prompt(file_name: str) -> str:
    with open(f'prompts/{file_name}', 'r', encoding='utf-8') as f:
        return f.read()
