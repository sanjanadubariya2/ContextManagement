"""Provider selection: keys come from .env or the shell; Gemini is preferred;
there is no silent fallback to the fake provider."""

import os

import pytest

from mpc import config
from mpc.llm import LLMConfigError, get_adapter, resolve_provider

KEYS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "MPC_LLM_PROVIDER")


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    # Record the originals so anything load_dotenv sets is undone afterwards.
    for var in KEYS:
        monkeypatch.setenv(var, "x")
        monkeypatch.delenv(var)
    monkeypatch.setenv("HOME", str(tmp_path))  # no `ant auth login` profile
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path / "no-repo")
    monkeypatch.chdir(tmp_path)
    config.get_settings.cache_clear()
    yield tmp_path
    config.get_settings.cache_clear()


def test_gemini_key_in_dotenv_selects_gemini(clean_env):
    (clean_env / ".env").write_text("GEMINI_API_KEY=test-gemini-key\n", encoding="utf-8")
    provider, why = resolve_provider()
    assert provider == "gemini" and "GEMINI_API_KEY" in why
    assert os.environ["GEMINI_API_KEY"] == "test-gemini-key"  # visible to the SDK
    a = get_adapter()
    assert a.provider == "gemini" and a.model == "gemini-3.8-flash"


def test_gemini_preferred_when_both_keys_exist(clean_env):
    (clean_env / ".env").write_text("GEMINI_API_KEY=g\nANTHROPIC_API_KEY=a\n", encoding="utf-8")
    assert resolve_provider()[0] == "gemini"
    assert resolve_provider("anthropic")[0] == "anthropic"


def test_anthropic_key_alone_selects_anthropic(clean_env):
    (clean_env / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-test\n", encoding="utf-8")
    assert resolve_provider()[0] == "anthropic"


def test_shell_wins_over_dotenv(clean_env, monkeypatch):
    (clean_env / ".env").write_text("GEMINI_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "from-shell")
    resolve_provider()
    assert os.environ["GEMINI_API_KEY"] == "from-shell"


def test_no_key_is_an_error_not_a_fake(clean_env):
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        get_adapter()


def test_forced_gemini_without_key_fails(clean_env, monkeypatch):
    (clean_env / ".env").write_text("ANTHROPIC_API_KEY=a\n", encoding="utf-8")
    monkeypatch.setenv("MPC_LLM_PROVIDER", "gemini")
    with pytest.raises(LLMConfigError, match="no Gemini key"):
        get_adapter()


def test_unsaved_empty_dotenv_is_named(clean_env, monkeypatch):
    monkeypatch.setattr(config, "REPO_ROOT", clean_env)
    (clean_env / ".env").write_text("", encoding="utf-8")
    with pytest.raises(LLMConfigError, match="empty on disk"):
        get_adapter()


def test_fake_only_when_asked_for(clean_env, monkeypatch):
    monkeypatch.setenv("MPC_LLM_PROVIDER", "fake")
    assert get_adapter().provider == "fake"


def test_unknown_provider_rejected(clean_env, monkeypatch):
    monkeypatch.setenv("MPC_LLM_PROVIDER", "gpt")
    with pytest.raises(LLMConfigError, match="one of"):
        get_adapter()
