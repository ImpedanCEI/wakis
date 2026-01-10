# wakis Makefile


# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

.PHONY: help install install-gpu clean docs

# Based on https://gist.github.com/prwhite/8168133?permalink_comment_id=4718682#gistcomment-4718682

## (Default) Print listing of key targets with their descriptions
help:
	@printf "\n\033[1;33mwakis Makefile\033[0m\n"
	@printf "\n\033[1;34mUsage:\033[0m \033[1;33mmake <target>\033[0m\n\n"
	@awk '\
		/^## / { \
			desc = substr($$0, 4); \
			getline nextline; \
			if (match(nextline, /^([a-zA-Z0-9._-]+):/)) { \
				target = substr(nextline, RSTART, RLENGTH-1); \
				printf "  \033[34m%-20s\033[0m %s\n", target, desc; \
			} \
		} \
	' $(MAKEFILE_LIST)

## Install Python dependencies
install:
	@echo -e "Installing Python dependencies..."
	@pip install -e  .[full]

## Install Python dependencies
install-gpu:
	@echo -e "Installing Python dependencies with GPU support..."
	@pip install -e  .[full-gpu]

## Remove temporary and generated files
clean:
	@echo -e "Cleaning up directory..."
	@rm -rf __pycache__ .ruff_cache .pytest_cache *.egg-info .html *.html coverage.xml .coverage .mypy_cache results
	@ cd docs && make -s clean


## Run Python tests
test:
	pytest -s

## Build documentation
docs:
	cd docs && make html
