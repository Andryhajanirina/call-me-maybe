1. Diagnostic de ton travail actuel  
D'après le code et les fichiers de ton projet :
    * **Architecture de base en place** : Tu as créé la structure dans ```src/call_me_maybe/``` avec plusieurs modules (```models.py, function_schema.py, json_decoder.py, token_constraints.py, json_state.py, json_context.py, parsing.py, main.py```).  
    * **Structure des schémas Pydantic & validation** : ```function_schema.py``` et ```models.py``` gèrent la conversion entre ```functions_definition.json``` et la validation Pydantic exigée par le sujet.  
    * **Moteur de contraintes (Constrained Decoding)** : Tu as commencé l'implémentation de la restriction des tokens pour garantir du JSON 100% valide et conforme au schéma.  
    
2. Rappel des points clés et pièges du sujet  
    1. Règle absolue du Constrained Decoding :  
        * Le modèle ```Qwen/Qwen3-0.6B``` ne doit jamais être laissé libre de générer ce qu'il veut.  
        * À chaque étape de génération (token par token), tu obtiens la liste des ***logits*** via ```get_logits_from_input_ids()```.  
        * Tu dois appliquer **```-inf```** (**```float('-inf')```**) à tous les tokens dont l'ajout briserait la syntaxe JSON ou le schéma de la fonction ciblée.  

    2. Identification des fonctions par le LLM :  
        * Le sujet interdit explicitement d'utiliser des heuristiques ("magic") pour choisir la fonction.  
        * La sélection doit se faire via le LLM ou être guidée par l'espace des tokens autorisés.  

    3. Format du fichier de sortie :  
        * Le résultat doit être écrit dans **```data/output/function_calling_results.json```** (ou la destination passée en argument).  
        Format exact par élément :  
        ```JSON
        {
            "prompt": "What is the sum of 2 and 3?",
            "name": "fn_add_numbers",
            "parameters": {"a": 2.0, "b": 3.0}
        }
        ```

    4. Conformité du code et Qualité :  
        * Python 3.10+.  
        * Linters requis : **```flake8```** et **```mypy```**.  
        * Gestion stricte des types (**```typing```**), docstrings (PEP 257), et gestion propre des ressources (**```try/except```**, context managers).  
        * Makefile contenant : **```install, run, debug, clean, lint```**.  

3. Étapes pour vérifier et finaliser ton projet  
**Étape A : Tester et valider les contraintes de typage & linting**  
Exécute la commande de linting exigée par le sujet pour détecter tout problème d'annotations de type ou de style :  
```Bash
make lint
```
(Ceci lance **```flake8```** et **```mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs```**).  
**Étape B : Lancement du programmeVérifie que l'exécution via uv fonctionne sans erreur :**  

```Bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

**Étape C : Vérification du README.md (Mandatoire pour le barème)**  
Le sujet est très strict sur les exigences du ```README.md``` :  
1. Première ligne obligatoire (en italics) :  
```Markdown
*This project has been created as part of the 42 curriculum by <ton_login>.*
```

2. Langue : Doit être entièrement rédigé en anglais.  
3. Sections requises :  
    * Description  
    * Instructions  
    * Resources (détailler aussi l'usage fait de l'IA pour le projet)  
    * Algorithm explanation (explication détaillée de ton approche de Constrained Decoding)  
    * Design decisions  
    * Performance analysis  
    * Challenges faced  
    * Testing strategy  
    * Example usage  

4. Pistes pour les Bonus (si souhaités)  
Si tu souhaites viser les points bonus :  
    1. Recoder le Tokenizer : Implémenter soi-même la logique **```encode```** / **```decode```** à partir du fichier de vocabulaire (**```get_path_to_vocab_file()```**) et des logits (**```get_logits_from_input_ids()```**), sans appeler directement **```encode()```**/**```decode()```** du SDK.  
    2. Gestion de types/objets complexes imbriqués dans les arguments.  
    3. Suite de tests automatisés complète (ex: **```pytest```**).  
    4. Visualisation du processus de génération pas-à-pas / filtrage de tokens.  
    