"""
Unit tests for HybridRecommender.__init__ in recommendor.py

Covers:
- Happy paths: normal file, custom save_dir, empty file
- Edge cases: missing file, unreadable file, invalid CSV, None as path, non-string path
"""

import os
import tempfile
import pandas as pd
import pytest

from recommendor import HybridRecommender

class TestHybridRecommenderInit:
    @pytest.fixture(autouse=True)
    def patch_methods(self, monkeypatch):
        """
        Patch _prepare_data, _build_model, and save to prevent side effects and focus on __init__ logic.
        """
        monkeypatch.setattr(HybridRecommender, "_prepare_data", lambda self: None)
        monkeypatch.setattr(HybridRecommender, "_build_model", lambda self: None)
        monkeypatch.setattr(HybridRecommender, "save", lambda self: None)

    @pytest.mark.happy
    def test_init_with_valid_csv_default_save_dir(self, tmp_path):
        """
        Test that __init__ correctly loads a valid CSV and sets default save_dir.
        """
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        rec = HybridRecommender(str(csv_path))
        assert rec.save_dir == "saved_model/"
        assert isinstance(rec.df, pd.DataFrame)
        pd.testing.assert_frame_equal(rec.df, df)

    @pytest.mark.happy
    def test_init_with_valid_csv_custom_save_dir(self, tmp_path):
        """
        Test that __init__ sets a custom save_dir when provided.
        """
        df = pd.DataFrame({"x": [10], "y": [20]})
        csv_path = tmp_path / "data2.csv"
        df.to_csv(csv_path, index=False)
        custom_dir = str(tmp_path / "custom_dir")

        rec = HybridRecommender(str(csv_path), save_dir=custom_dir)
        assert rec.save_dir == custom_dir
        pd.testing.assert_frame_equal(rec.df, df)

    @pytest.mark.happy
    def test_init_with_empty_csv(self, tmp_path):
        """
        Test that __init__ loads an empty CSV (with headers, no rows) without error.
        """
        df = pd.DataFrame(columns=["foo", "bar"])
        csv_path = tmp_path / "empty.csv"
        df.to_csv(csv_path, index=False)

        rec = HybridRecommender(str(csv_path))
        assert list(rec.df.columns) == ["foo", "bar"]
        assert rec.df.empty

    @pytest.mark.edge
    def test_init_with_missing_file_raises(self, tmp_path):
        """
        Test that __init__ raises FileNotFoundError when the CSV file does not exist.
        """
        missing_path = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError):
            HybridRecommender(str(missing_path))

    @pytest.mark.edge
    def test_init_with_unreadable_file_raises(self, tmp_path):
        """
        Test that __init__ raises a PermissionError (or OSError) if the file is not readable.
        """
        csv_path = tmp_path / "unreadable.csv"
        with open(csv_path, "w") as f:
            f.write("a,b\n1,2\n")
        os.chmod(csv_path, 0)  # Remove all permissions

        try:
            with pytest.raises((PermissionError, OSError)):
                HybridRecommender(str(csv_path))
        finally:
            # Restore permissions so tmp_path can be cleaned up
            os.chmod(csv_path, 0o600)

    @pytest.mark.edge
    def test_init_with_invalid_csv_raises(self, tmp_path):
        """
        Test that __init__ raises a ParserError or ValueError if the CSV is malformed.
        """
        csv_path = tmp_path / "bad.csv"
        with open(csv_path, "w") as f:
            f.write("not,a,csv\n\"unterminated string\n")
        with pytest.raises(Exception) as excinfo:
            HybridRecommender(str(csv_path))
        # Accept any pandas error for malformed CSV
        assert "Error tokenizing" in str(excinfo.value) or "ParserError" in str(type(excinfo.value))

    @pytest.mark.edge
    def test_init_with_none_path_raises(self):
        """
        Test that __init__ raises a TypeError if path is None.
        """
        with pytest.raises(TypeError):
            HybridRecommender(None)

    @pytest.mark.edge
    def test_init_with_non_string_path_raises(self):
        """
        Test that __init__ raises a TypeError if path is not a string (e.g., int).
        """
        with pytest.raises(TypeError):
            HybridRecommender(12345)