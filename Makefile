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

# Variables d'environnement
HF_HOME               = $(GOINFRE)/.cache/huggingface
UV_CACHE_DIR          = $(GOINFRE_CACHE)
UV_PYTHON_PREFERENCE  = managed

export HF_HOME
export UV_CACHE_DIR
export UV_PYTHON_PREFERENCE

# Évite les conflits si un fichier porte le même nom qu'une règle
.PHONY: all install run debug clean lint lint-strict lock sync add

# ==============================================================================
# RÈGLES MANDATAIRES
# ==============================================================================

all: install lint

# IV.2 : install - Configuration du venv dans le goinfre et installation des dépendances
install:
	@echo "🔧 Configuration de l'environnement de développement..."
	 if [ ! -L $(VENV) ]; then \
		rm -rf $(VENV); \
		mkdir -p $(GOINFRE_VENV); \
		ln -s $(GOINFRE_VENV) $(VENV); \
	 fi

	@echo "📦 Ajout des outils de dev par défaut (flake8, mypy)..."
	$(UV) init --package
	$(UV) add --dev flake8 mypy
	$(UV) add ./llm_sdk

	@echo "📦 Synchronisation des dépendances avec uv sync..."
	$(UV) sync

# ==============================================================================
# UV
# ==============================================================================
lock:
	$(UV) lock

sync:
	$(UV) sync

# ==============================================================================
# AJOUT DE PACKAGES
# ==============================================================================

add:
	@$(eval PKGS := $(filter-out $@,$(MAKECMDGOALS)))
	@if [ -z "$(PKGS)" ]; then \
		echo "❌ Erreur: Spécifiez au moins un package."; \
		echo "   Exemple: make add mypy flake8"; \
		exit 1; \
	fi
	$(UV) add $(PKGS)

# Permet à make d'accepter les noms de packages après "add"
%:
	@:


run:
	@$(UV) run $(MAIN_SCRIPT)

debug:
	@$(UV) run $(PYTHON) -m pdb $(MAIN_SCRIPT)

# ==============================================================================
# CLEAN - Nettoyage complet des caches et fichiers temporaires
# ==============================================================================
clean:
	@echo "🧹 Nettoyage des caches et fichiers temporaires..."
	rm -rf .mypy_cache .pytest_cache .uv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

fclean: clean
	@echo "🧹 Nettoyage des caches, venvs, uv.lock, .python-version..."
	rm -rf $(GOINFRE)/$(CACHE)
	rm -rf $(VENV)
	rm -rf $(GOINFRE)/venvs
	rm -rf .python-version
	rm -rf pyproject.toml
	rm -rf uv.lock
	rm -rf main.py
	rm -rf src

# ==============================================================================
# lint - Vérification des normes de code (Flake8 et Mypy)
# ==============================================================================
lint:
	@if [ ! -f $(VENV)/bin/flake8 ]; then echo "❌ Erreur: Lancez 'make install' d'abord."; exit 1; fi
	@echo "🔍 [Flake8] Vérification du codage standard..."
	$(VENV)/bin/flake8 .
	@echo "🔍 [Mypy] Analyse statique des types (Mode Standard)..."
	$(VENV)/bin/mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

# ==============================================================================
# lint-strict - Mode de vérification renforcé
# ==============================================================================
lint-strict:
	@if [ ! -f $(VENV)/bin/flake8 ]; then echo "❌ Erreur: Lancez 'make install' d'abord."; exit 1; fi
	@echo "🔍 [Flake8] Vérification du codage standard..."
	$(VENV)/bin/flake8 .
	@echo "🔍 [Mypy] Analyse statique des types (Mode Strict)..."
	$(VENV)/bin/mypy . --strict
