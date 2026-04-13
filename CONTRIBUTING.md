# Contributing

Thanks for your interest in contributing to the LangChain Cookbook!

## Adding a New Recipe

1. Create a new numbered folder: `XX_recipe_name/`
2. Include:
   - `main.py` — self-contained, runnable script
   - `README.md` — problem statement, solution overview, how to run
   - `requirements.txt` (optional) — only if extra deps beyond the root are needed
3. Use environment variables for API keys (never hardcode)
4. Add your recipe to the root `README.md` table of contents
5. Open a PR with a clear description

## Guidelines

- Keep recipes focused on one concept
- Include clear comments in the code
- Handle missing API keys gracefully with helpful error messages
- Test that your recipe runs end-to-end before submitting
