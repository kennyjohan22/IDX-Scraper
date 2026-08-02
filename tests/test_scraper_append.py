import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import scraper
import scraper_browser


class AppendToMasterTests(unittest.TestCase):
    def _assert_recent_duplicate_is_not_appended(self, module):
        with tempfile.TemporaryDirectory() as temp_dir:
            master_path = Path(temp_dir) / "master.csv"
            log_path = Path(temp_dir) / "scraper.log"
            target_date = "2026-07-31"

            existing = pd.DataFrame(
                {
                    "DATE": ["2025-01-02"] * 10_000 + [target_date],
                    "Kode Saham": ["TEST"] * 10_001,
                }
            )
            existing.to_csv(master_path, index=False)
            incoming = pd.DataFrame({"Kode Saham": ["TEST"]})

            with patch.object(module, "MASTER_CSV", str(master_path)), patch.object(
                module, "LOG_FILE", str(log_path)
            ):
                module.append_to_master(incoming, target_date)

            persisted = pd.read_csv(master_path)
            self.assertEqual(len(persisted), 10_001)
            self.assertEqual((persisted["DATE"] == target_date).sum(), 1)

    def test_requests_scraper_checks_dates_beyond_first_10000_rows(self):
        self._assert_recent_duplicate_is_not_appended(scraper)

    def test_browser_scraper_checks_dates_beyond_first_10000_rows(self):
        self._assert_recent_duplicate_is_not_appended(scraper_browser)


if __name__ == "__main__":
    unittest.main()
