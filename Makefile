.PHONY: help install lint check run-all

RECIPES := $(sort $(wildcard [0-9][0-9]_*/))

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install: ## Install all dependencies (pip)
	pip install -r requirements.txt

uv-install: ## Install all dependencies (uv)
	uv pip install -r requirements.txt

lint: ## Check all recipes for syntax errors
	@echo "Checking syntax..."
	@for dir in $(RECIPES); do \
		python -m py_compile $${dir}main.py && echo "  OK  $${dir}main.py" || echo "  FAIL $${dir}main.py"; \
	done

check: ## Verify each recipe has main.py and README.md
	@echo "Checking recipe structure..."
	@for dir in $(RECIPES); do \
		ok=true; \
		[ -f "$${dir}main.py" ]   || { echo "  MISSING $${dir}main.py";   ok=false; }; \
		[ -f "$${dir}README.md" ] || { echo "  MISSING $${dir}README.md"; ok=false; }; \
		$$ok && echo "  OK  $${dir}"; \
	done

run-%: ## Run a specific recipe, e.g. make run-01_basic_rag
	cd $* && python main.py

run-all: ## Run every recipe sequentially (requires API keys)
	@for dir in $(RECIPES); do \
		echo "\n========== $${dir} =========="; \
		cd $${dir} && python main.py && cd .. || { echo "FAILED: $${dir}"; cd ..; }; \
	done
