Par défaut, l'outil uv stocke son cache global et ses données dans votre dossier personnel ``(~/.cache/uv ou ~/.local/share/uv)``  
situé sur votre partition Home, qui est saturée (300 Mo), et non sur votre espace goinfre (30 Go).  

### Pourquoi cela bloque  
**Cache par défaut :** uv télécharge et extrait les paquets dans le répertoire de cache global de votre Home avant de créer des liens vers votre projet.  
**Espace insuffisant :** Vos 300 Mo de Home se remplissent instantanément, provoquant une erreur d'espace disque, alors que votre goinfre dispose de 30 Go libres.  
### Comment régler le problème  
Vous devez indiquer à uv d'utiliser votre espace goinfre pour stocker son cache et ses données en définissant les variables d'environnement adaptées:  
* Définissez le dossier de cache sur goinfre en ajoutant ceci dans votre fichier de configuration du shell ``(~/.bashrc ou ~/.zshrc)``:
```Bash
echo 'export UV_CACHE_DIR="/chemin/vers/votre/goinfre/.cache/uv"' >> ~/.zshrc
```
```Bash
echo 'export UV_CACHE_DIR="/home/andry-ha/goinfre/.cache/uv"' >> ~/.zshrc
```

* Redirigez aussi le dossier de données si uv y installe des versions de Python:  
```Bash
echo 'export XDG_DATA_HOME="/home/andry-ha/goinfre/.local/share"' >> ~/.zshrc
```
* Appliquez les changements dans votre terminal ouvert :
```Bash
source ~/.bashrc
```


### COMMAND POUR INSTALLER mypy et flake8 avec la version python 3.10
```Bash
uv add mypy --python 3.10


# GOINFRE
echo "HF_HOME=$HF_HOME"
echo "UV_CACHE_DIR=$UV_CACHE_DIR"

df -h "$HOME" /goinfre

du -sh /goinfre/andry-ha/.cache/huggingface 2>/dev/null
du -sh /home/andry-ha/.cache/huggingface 2>/dev/null

export HF_HOME="/goinfre/andry-ha/.cache/huggingface"
export UV_CACHE_DIR="/goinfre/andry-ha/.cache/uv"

mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

echo "HF_HOME=$HF_HOME"
df -h "$HF_HOME"


# LIST ARBORESCENCE OF DIRECTORY
pwd
find . -maxdepth 3 -type f \
    -not -path './.git/*' \
    -not -path './.venv/*' \
    -not -path './.mypy_cache/*' \
    | sort