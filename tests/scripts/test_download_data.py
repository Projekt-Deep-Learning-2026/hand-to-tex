import io
import tarfile
from unittest.mock import MagicMock, patch

import pytest

from hand_to_tex.scripts.download_data import download_data


class TestDownloadData:
    @pytest.fixture
    def mock_tgz(self):
        """Creates a mock .tgz file in memory with a simple structure."""
        inner_file_content = b"content"

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo(name="test_data/file.txt")
            info.size = len(inner_file_content)
            tar.addfile(info, io.BytesIO(inner_file_content))

        return buf.getvalue()

    def test_download_data_skips_if_exists(self, tmp_path, capsys):
        """If directory already exists, download_data should skip downloading."""
        data_dir = tmp_path / "data"
        (data_dir / "mathwriting-2024-excerpt").mkdir(parents=True)

        url = "https://example.com/mathwriting-2024-excerpt.tgz"
        download_data(url, dir_name=str(data_dir))

        captured = capsys.readouterr()
        assert "Skipping" in captured.out

    @patch("urllib.request.urlopen")
    def test_download_data_success(self, mock_urlopen, tmp_path, mock_tgz):
        """Tests successful download and extraction of data."""
        mock_response = MagicMock()
        mock_response.getheader.return_value = len(mock_tgz)
        mock_response.read.side_effect = [mock_tgz, b""]
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        data_dir = tmp_path / "test_download"
        url = "https://example.com/test_data.tgz"

        download_data(url, dir_name=str(data_dir))

        assert (data_dir / "test_data").exists()
        assert (data_dir / "test_data" / "file.txt").exists()

        assert not (data_dir / "test_data.tgz").exists()
