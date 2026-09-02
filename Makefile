# ==============================================================================
# CONFIGURATION
# ==============================================================================
MAIN_SCRIPT = call-me-maybe

GOINFRE       = $(HOME)/goinfre
CACHE         = .cache
GOINFRE_VENV  = $(GOINFRE)/venvs/$(shell basename $(CURDIR))
GOINFRE_CACHE = $(GOINFRE)/$(CACHE)/uv

VENV          = .venv
UV            = uv
PYTHON       = $(VENV)/bin/python

HF_HOME               = $(GOINFRE)/.cache/huggingface
UV_CACHE_DIR          = $(GOINFRE_CACHE)
UV_PYTHON_PREFERENCE  = managed

export HF_HOME
export UV_CACHE_DIR
export UV_PYTHON_PREFERENCE

MYPY_FLAGS  := --exclude=$(VENV) --exclude='llm_sdk' --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

GREEN       := \033[0;32m
INFO_COLOR  := \033[0;36m
INFO_BOLD   := \033[1;36m
WARN_COLOR  := \033[0;93m
WARN_BOLD   := \033[1;93m
ERROR_COLOR := \033[0;31m
ERROR_BOLD  := \033[1;31m
RESET       := \033[0m

# Avoids conflicts if a file has the same name as a rule
.PHONY: all install run debug clean lint lint-strict lock sync add

# ==============================================================================
# MANDATORY RULES
# ==============================================================================

all: install lint

install:
	@echo "$(INFO_BOLD)🔧 Configuring the development environment...$(RESET)"
	@if [ ! -L $(VENV) ]; then \
		rm -rf $(VENV); \
		mkdir -p $(GOINFRE_VENV); \
		ln -s $(GOINFRE_VENV) $(VENV); \
	fi

	@echo "$(INFO_BOLD)📦 Added default development tools (flake8, mypy)...$(RESET)"
	@if [ -f pyproject.toml ]; then \
		echo "$(INFO_BOLD)ℹ️  The pyproject.toml file already exists. uv will update the dependencies.$(RESET)"; \
	else \
		$(UV) init --package; \
	fi
	$(UV) add --dev flake8 mypy

	$(UV) add ./llm_sdk

	@echo "$(INFO_BOLD) Synchronizing dependencies with UV sync...$(RESET)"
	$(UV) sync

# ==============================================================================
# UV
# ==============================================================================
lock:
	$(UV) lock

sync:
	$(UV) sync

# ==============================================================================
# ADDING PACKAGES
# ==============================================================================

add:
	@$(eval PKGS := $(filter-out $@,$(MAKECMDGOALS)))
	@if [ -z "$(PKGS)" ]; then \
		echo "$(ERROR_BOLD)❌ Error:$(RESET) $(ERROR_COLOR)Specify at least one package.$(RESET)"; \
		echo "   Exemple: make add mypy flake8"; \
		exit 1; \
	fi
	$(UV) add $(PKGS)

# Allows make to accept package names after "add"
%:
	@:


run:
	@$(UV) run $(MAIN_SCRIPT)

debug:
	@$(UV) run $(PYTHON) -m pdb $(MAIN_SCRIPT)

# ==============================================================================
# CLEAN - Complete clearing of caches and temporary files
# ==============================================================================
clean:
	@echo "$(WARN_BOLD)🧹 Clearing caches and temporary files...$(RESET)"
	rm -rf .mypy_cache .pytest_cache .uv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

fclean: clean
	@echo "$(WARN_BOLD)🧹 Cleaning caches, venvs, uv.lock, .python-version...$(RESET)"
	rm -rf $(GOINFRE)/$(CACHE)
	rm -rf $(VENV)
	rm -rf $(GOINFRE)/venvs
	rm -rf .python-version
	rm -rf pyproject.toml
	rm -rf uv.lock
	rm -rf main.py
	rm -rf src

# ==============================================================================
# lint - Code standards verification (Flake8 and Mypy)
# ==============================================================================
lint:
	@if [ ! -f $(VENV)/bin/flake8 ]; then echo "$(ERROR_BOLD)❌ ERROR:$(RESET) $(ERROR_COLOR)Run 'make install' first.$(RESET)"; exit 1; fi
	@echo "$(INFO_BOLD)🔍 [Flake8] Standard coding verification...$(RESET)"
# 	$(VENV)/bin/flake8 . --exclude=$(VENV)
	$(VENV)/bin/flake8 .
	@echo "$(INFO_BOLD)🔍 [Mypy] Static type analysis (Standard mode)...$(RESET)"
	$(VENV)/bin/mypy . $(MYPY_FLAGS)

# ==============================================================================
# lint-strict - Enhanced verification method with (Flake8 and Mypy --strict)
# ==============================================================================
lint-strict:
	@if [ ! -f $(VENV)/bin/flake8 ]; then echo "$(ERROR_BOLD)❌ ERROR:$(RESET) $(ERROR_COLOR)Run 'make install' first.$(RESET)"; exit 1; fi
	@echo "$(INFO_BOLD)🔍 [Flake8] Standard coding verification...$(RESET)"
# 	$(VENV)/bin/flake8 . --exclude=$(VENV)
	$(VENV)/bin/flake8 .
	@echo "$(INFO_BOLD)🔍 [Mypy] Static type analysis (Strict mode)...$(RESET)"
	$(VENV)/bin/mypy . --strict $(MYPY_FLAGS)
