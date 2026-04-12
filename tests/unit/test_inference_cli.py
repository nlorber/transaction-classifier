"""Tests for the serving CLI entry point."""

from unittest.mock import patch

from click.testing import CliRunner

from transaction_classifier.inference.cli import serve


class TestServeCLI:
    @patch("transaction_classifier.inference.cli.uvicorn")
    def test_serve_defaults(self, mock_uvicorn):
        runner = CliRunner()
        result = runner.invoke(serve, [])
        assert result.exit_code == 0
        mock_uvicorn.run.assert_called_once()
        call_kwargs = mock_uvicorn.run.call_args
        assert call_kwargs[1]["factory"] is True
        assert call_kwargs[1]["log_level"] == "info"

    @patch("transaction_classifier.inference.cli.uvicorn")
    def test_serve_with_overrides(self, mock_uvicorn):
        runner = CliRunner()
        result = runner.invoke(serve, ["--host", "0.0.0.0", "--port", "9000", "-v"])
        assert result.exit_code == 0
        call_kwargs = mock_uvicorn.run.call_args
        assert call_kwargs[1]["host"] == "0.0.0.0"
        assert call_kwargs[1]["port"] == 9000
        assert call_kwargs[1]["log_level"] == "debug"

    @patch("transaction_classifier.inference.cli.uvicorn")
    def test_serve_with_reload(self, mock_uvicorn):
        runner = CliRunner()
        result = runner.invoke(serve, ["--reload"])
        assert result.exit_code == 0
        call_kwargs = mock_uvicorn.run.call_args
        assert call_kwargs[1]["reload"] is True
