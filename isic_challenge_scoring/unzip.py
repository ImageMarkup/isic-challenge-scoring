import os
import pathlib
import shutil
import tempfile
from typing import Tuple
import zipfile

from isic_challenge_scoring.types import ScoreException


def extract_zip(zip_path: pathlib.Path, output_path: pathlib.Path, flatten: bool = True) -> None:
    """Extract a zip file, optionally flattening it into a single directory."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            if flatten:
                for member_info in zf.infolist():
                    member_name = member_info.filename
                    if member_name.startswith('__MACOSX'):
                        # Ignore Mac OS X metadata
                        continue

                    member_base_name = os.path.basename(member_name)
                    if not member_base_name:
                        # Skip directories
                        continue

                    member_output_path = output_path / member_base_name

                    with zf.open(member_info) as input_stream, member_output_path.open(
                        'wb'
                    ) as output_stream:
                        shutil.copyfileobj(input_stream, output_stream)
            else:
                zf.extractall(output_path)
    except zipfile.BadZipfile as e:
        raise ScoreException(f'Could not read ZIP file "{zip_path.name}": {str(e)}.')


def unzip_all(input_file: pathlib.Path) -> Tuple[pathlib.Path, tempfile.TemporaryDirectory]:
    """Extract a ZIP file to a temporary directory."""
    output_temp_dir = tempfile.TemporaryDirectory()
    output_path = pathlib.Path(output_temp_dir.name)

    extract_zip(input_file, output_path)

    return output_path, output_temp_dir
