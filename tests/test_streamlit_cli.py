import subprocess
import sys
import pytest

def test_streamlit_cli_available():
    """Test that the Streamlit CLI is available and can be run."""
    result = subprocess.run([sys.executable, '-m', 'streamlit', '--version'], capture_output=True, text=True)
    assert result.returncode == 0, f"Streamlit CLI not available: {result.stderr or result.stdout}"
    assert 'Streamlit' in result.stdout or 'streamlit' in result.stdout 