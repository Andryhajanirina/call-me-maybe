# export UV_CACHE_DIR = "$(HOME)/goinfre/.cache/uv"

# export XDG_DATA_HOME = "$(HOME)/goinfre/.local/share"

# go:
# 	uv cache dir

# init:
# 	uv python install ">=3.10"
# 	uv init --python 3.10

# sync:
# 	uv sync

# add:
# 	uv add flake8
# # 	uv add mypy

# to:
# 	uv python pin 3.10

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Nom du script principal à exécuter
# MAIN_SCRIPT = main.py
MAIN_SCRIPT = src/test_llm

# Chemins pour le quota 42 (déportation dans goinfre)
GOINFRE      = $(HOME)/goinfre/
CACHE        = .cache
GOINFRE_VENV = $(GOINFRE)/venvs/$(shell basename $(CURDIR))
GOINFRE_CACHE = $(GOINFRE)/$(CACHE)/uv
VENV         = .venv
UV           = uv

# Détection de l'exécutable python dans le venv
PYTHON       = $(VENV)/bin/python

# Évite les conflits si un fichier porte le même nom qu'une règle
.PHONY: all install run debug clean lint lint-strict

# ==============================================================================
# RÈGLES MANDATAIRES
# ==============================================================================

all: install lint

# IV.2 : install - Configuration du venv dans le goinfre et installation des dépendances
install:
	@echo "🔧 Configuration de l'environnement de développement..."
	@export UV_CACHE_DIR="$(GOINFRE_CACHE)/.cache/uv" && \
	 export UV_PYTHON_PREFERENCE="managed" && \
	 if [ ! -L $(VENV) ]; then \
		rm -rf $(VENV); \
		mkdir -p $(GOINFRE_VENV); \
		ln -s $(GOINFRE_VENV) $(VENV); \
	 fi

	@echo "📦 Ajout des outils de dev par défaut (flake8, mypy)..."
	@export UV_CACHE_DIR="$(GOINFRE_CACHE)/.cache/uv" && \
	 export UV_PYTHON_PREFERENCE="managed" && \
	 $(UV) init
	 $(UV) add --dev flake8 mypy

	@echo "📦 Synchronisation des dépendances avec uv sync..."
	@export UV_CACHE_DIR="$(GOINFRE_CACHE)/.cache/uv" && \
	 export UV_PYTHON_PREFERENCE="managed" && \
	 $(UV) sync

# sync - Permet de forcer manuellement la resynchronisation du pyproject.toml
sync:
	@export UV_CACHE_DIR="$(GOINFRE_CACHE)/.cache/uv" && \
	 export UV_PYTHON_PREFERENCE="managed" && \
	 $(UV) sync

init:
	@if [ -f pyproject.toml ]; then \
		echo "💡 Le fichier pyproject.toml existe déjà. Étape d'initialisation ignorée."; \
	else \
		echo "🚀 Initialisation du projet avec uv init..."; \
		export UV_PYTHON_PREFERENCE="managed" && $(UV) init --python 3.10; \
	fi

# add - Règle magique pour ajouter un ou plusieurs packages (Ex: make add mypy flake8)
add:
	@# Extraction de tous les arguments après "add"
	$(eval PKGS := $(filter-out $@,$(MAKECMDGOALS)))
	@if [ -z "$(PKGS)" ]; then \
		echo "❌ Erreur: Spécifiez au moins un package. Exemple: make add mypy flake8"; \
		exit 1; \
	 fi
	@export UV_CACHE_DIR="$(HOME)/goinfre/.cache/uv" && \
	 export UV_PYTHON_PREFERENCE="managed" && \
	 $(UV) add $(PKGS)

# Astuce indispensable pour éviter que Make ne cherche à exécuter les arguments comme des cibles distinctes
%:
	@:

# IV.2 : run - Exécution du script principal via l'interpréteur du venv
run:
	@if [ ! -f $(PYTHON) ]; then echo "❌ Erreur: Lancez 'make install' d'abord."; exit 1; fi
	@if [ ! -f $(MAIN_SCRIPT) ]; then echo "❌ Erreur: $(MAIN_SCRIPT) introuvable."; exit 1; fi
	$(PYTHON) -m $(MAIN_SCRIPT)

# IV.2 : debug - Exécution en mode debug avec le module pdb natif
debug:
	@if [ ! -f $(PYTHON) ]; then echo "❌ Erreur: Lancez 'make install' d'abord."; exit 1; fi
	@if [ ! -f $(MAIN_SCRIPT) ]; then echo "❌ Erreur: $(MAIN_SCRIPT) introuvable."; exit 1; fi
	$(PYTHON) -m pdb $(MAIN_SCRIPT)

# IV.2 : clean - Nettoyage complet des caches et fichiers temporaires
clean:
	@echo "🧹 Nettoyage des caches et fichiers temporaires..."
	rm -rf .mypy_cache .pytest_cache .uv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

fclean: clean
	rm -rf $(GOINFRE)/$(CACHE)
	rm -rf $(VENV)
	rm -rf $(GOINFRE)/venvs
	rm -rf .python-version

# IV.2 : lint - Vérification stricte des normes de code (Flake8 et Mypy demandés)
lint:
	@if [ ! -f $(VENV)/bin/flake8 ]; then echo "❌ Erreur: Lancez 'make install' d'abord."; exit 1; fi
	@echo "🔍 [Flake8] Vérification du codage standard..."
	$(VENV)/bin/flake8 .
	@echo "🔍 [Mypy] Analyse statique des types (Mode Standard)..."
	$(VENV)/bin/mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

# IV.2 : lint-strict (optional) - Mode de vérification renforcé
lint-strict:
	@if [ ! -f $(VENV)/bin/flake8 ]; then echo "❌ Erreur: Lancez 'make install' d'abord."; exit 1; fi
	@echo "🔍 [Flake8] Vérification du codage standard..."
	$(VENV)/bin/flake8 .
	@echo "🔍 [Mypy] Analyse statique des types (Mode Strict)..."
	$(VENV)/bin/mypy . --strict
