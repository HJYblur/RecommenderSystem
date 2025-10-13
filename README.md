# Recommender System

### Git Usage
Since we want to assure the reproducibility of the notebook, we try to store as much intermediate data as we could, which also includes some huge data. Thus, we use `git lfs` for safe transfer.
To set your computer ready for git lfs, you should:

1. Install git‑lfs
- macOS (Homebrew):
  `brew install git-lfs`
- Debian/Ubuntu:
  `sudo apt install git-lfs`
- Windows (Chocolatey):
  `choco install git-lfs`
- Or follow official instructions: https://git-lfs.github.com/

2. Initialize git‑lfs in your repo
  `git lfs install`

3. Track common large file types used in this project:
   git lfs track "*.pkl"
   git lfs track "*.joblib"