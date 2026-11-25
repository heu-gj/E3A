"""Read-only full ZIP/TAR versus extracted HOI4D hand-file reconciliation."""
import argparse
import json
from pathlib import Path

from e3a.data.handpose_archive import reconcile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--archive', type=Path, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = reconcile(args.root, args.archive, args.output_dir)
    # Full path lists remain in local evidence, not terminal output.
    print(json.dumps({k: result[k] for k in ['status', 'ARCHIVE_READABILITY',
        'ARCHIVE_CRC_INTEGRITY', 'EXTRACTION_EQUIVALENCE', 'DOWNLOAD_COMPLETENESS', 'sides']}, sort_keys=True))


if __name__ == '__main__':
    main()
