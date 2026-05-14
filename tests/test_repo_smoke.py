from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_core_repo_files_exist():
    for path in [
        "README.md",
        "LICENSE",
        "AUTHORS.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "docs/final-audit.md",
        ".github/workflows/python-ci.yml",
    ]:
        assert (ROOT / path).exists(), path


def test_readme_mentions_search_terms():
    content = (ROOT / "README.md").read_text(encoding="utf-8", errors="ignore").lower()
    assert "research" in content
    assert "qualitative" in content or "quantitative" in content
