# PyPI Release Guide

Concrete release steps for future uploads of `pdslasso`.

## 1. Work On `dev`

Make your changes on a feature branch or directly on `dev`.

```bash
git checkout dev
git pull origin dev
```

Run your local checks before committing:

```bash
python -m pytest
```

Commit the release-ready changes:

```bash
git add pyproject.toml pdslasso/__init__.py pdslasso/core.py README.md
git commit -m "describe change"
git push origin dev
```

## 2. Merge To `main`

Merge the finished `dev` branch into `main` after review.

```bash
git checkout main
git pull origin main
git merge dev
git push origin main
```

If you use pull requests, open a PR from `dev` to `main` and merge it there instead of using `git merge` locally.

## 3. Bump Version

Update the version in both places before building a release:

- [`pyproject.toml`](pyproject.toml)
- [`pdslasso/__init__.py`](pdslasso/__init__.py)

Use a new version number every release, for example `0.1.2`.

## 4. Build Distribution Files

Clean any old artifacts first:

```bash
rm -f dist/*
```

Build the source distribution and wheel:

```bash
python -m build --no-isolation
```

After a successful build, `dist/` should contain exactly:

- `pdslasso-<version>.tar.gz`
- `pdslasso-<version>-py3-none-any.whl`

## 5. Validate The Artifacts

Check that the files render correctly on package indexes:

```bash
twine check dist/*
```

If this fails, fix the metadata before uploading.

## 6. Upload To TestPyPI

Use TestPyPI first when you want a dry run.

Make sure `~/.pypirc` has a `[testpypi]` section with:

```ini
username = __token__
password = <your TestPyPI token>
```

Upload:

```bash
twine upload --repository testpypi dist/*
```

Install from TestPyPI in a fresh environment to verify the release:

```bash
uv venv .venv-test
source .venv-test/bin/activate
uv pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  pdslasso
```

Smoke test the install:

```bash
python - <<'PY'
from pdslasso import PDSLasso, simulate_pds_data

df, _ = simulate_pds_data(n=100, p=10, true_effect=2.0, random_seed=0)
model = PDSLasso(
    data=df,
    y="y",
    d="d",
    control_cols=[c for c in df.columns if c.startswith("x")],
)
res = model.fit()
print("OK", float(res.params["d"]))
PY
```

## 7. Upload To Real PyPI

After TestPyPI looks good, upload the same built files to real PyPI.

Make sure `~/.pypirc` has a `[pypi]` section with:

```ini
username = __token__
password = <your PyPI token>
```

Upload:

```bash
twine upload dist/*
```

Or explicitly:

```bash
twine upload --repository pypi dist/*
```

## 8. Tag The Release

Create and push a git tag after the upload succeeds:

```bash
git tag v0.1.2
git push origin v0.1.2
```

## 9. Create A GitHub Release

Create a GitHub Release for the same tag and include:

- the version number
- a short summary of changes
- any notable warnings or compatibility notes

## 10. Release Checklist

Before every public release:

- version bumped in `pyproject.toml`
- version bumped in `pdslasso/__init__.py`
- tests passed
- build succeeded
- `twine check` passed
- TestPyPI upload verified, if used
- `dist/` contains only the current release files
- git tag created and pushed

