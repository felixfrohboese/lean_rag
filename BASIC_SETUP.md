# Basic Setup

## Accounts

- OpenAI for API key


## Git Workflow 

Commands

```bash

git init

git add .
git commit -m "commit message"
git push origin main
```


## Virtual Environment

### Commands

```bash
python3 -m venv rag-pilot
source rag-pilot/bin/activate
deactivate
```

## Dependencies

- Python 3.13 (through Homebrew)
- pip package manager



### Commands

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Upon changes to dependencies:

```bash
pip freeze > requirements.txt
```