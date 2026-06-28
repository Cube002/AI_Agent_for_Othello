# Git Workflow for This Repo

## Remotes

| Name     | URL                                                |
| -------- | -------------------------------------------------- |
| `origin` | `git@github.com:Cube002/AI_Agent_for_Othello.git`  |
| `upstream` | `git@github.com:barbora-besedova/AI_Agent_for_Othello.git` |

---

## Before You Start Work

```bash
git fetch --all --prune
git log --oneline origin/main -5
git log --oneline upstream/main -5
```

This shows what's new on both remotes before you touch anything.

---

## Pulling Barbora's Changes

Her changes live on the `upstream` remote. To get them:

```bash
# Fetch her latest
git fetch upstream

# Merge into your local main
git merge upstream/main
```

If she created a **Pull Request** on her fork (not uncommon), fetch that specifically:

```bash
git fetch upstream refs/pull/2/head:refs/remotes/upstream/pr2
git merge upstream/pr2
```

---

## Committing Your Changes

```bash
# 1. Stage your files (avoid __pycache__/*.pyc!)
git add <your-files>

# 2. Double-check what's staged
git status

# 3. Commit
git commit -m "Short descriptive message"

# 4. Check if remote has new changes
git fetch origin

# 5. If remote has diverged, rebase (cleaner than merge)
git pull --rebase origin main
# OR: git rebase origin/main

# 6. Push
git push origin main
```

### Important: never force-push (`git push --force`)

If force-push is needed, it means your history diverged badly — ask first.

### Never commit `__pycache__/` or `.pyc` files

They will be rejected or cause noise. Check `.gitignore` first before staging.

---

## If You Get Rejected (like we did)

```
! [rejected] main -> main (fetch first)
```

**Don't panic.** The fix is:

```bash
git fetch origin
git pull --rebase origin main   # replays your commits on top of remote
git push origin main
```

No force push needed.

---

## Summary of What Went Wrong Last Time

1. We staged `__pycache__/` files — unnecessary noise
2. Remote had been force-pushed with barbora's merge, so our push was rejected
3. We had to `stash` untracked files, `pull --rebase`, then `stash pop`
4. Large `.pth` model files made `stash` and `push` very slow

**Next time:**

- `git fetch --all` first, always
- If push is rejected, `git pull --rebase origin main` (never --force)
- Keep `__pycache__` out of commits
- Be patient with slow pushes (big model files)
