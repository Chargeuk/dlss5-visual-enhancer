"""Install only the retired v7 worker files needed by legacy VTS requests."""
import argparse
import shutil
import zipfile
from pathlib import Path


def install(archive_path, root):
    destination = (root / 'bin/runtime/legacy-v7').resolve()
    prefixes = ('bin/runtime/host/', 'bin/runtime/dlss/', 'bin/runtime/dlssnr/')
    required = {'bin/runtime/host/nvngx.dll', 'bin/runtime/host/dxgi.dll',
                'bin/runtime/dlss/nvngx_dlss.dll', 'bin/runtime/dlssnr/renodx-dlss5.addon64',
                'bin/runtime/dlssnr/nvngx_dlssnr.dll'}
    with zipfile.ZipFile(archive_path) as archive:
        if not required.issubset(archive.namelist()):
            raise ValueError('Expected the complete official v7.0 portable release archive.')
        targets = []
        for info in archive.infolist():
            if not info.filename.startswith(prefixes) or info.is_dir():
                continue
            target = (destination / info.filename.removeprefix('bin/runtime/')).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Unsafe runtime archive path.')
            targets.append((info, target))
        for info, target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output)
    print(f'Installed {len(targets)} legacy runtime files in {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path, help='DLSS.5.Visual.Enhancer.v7.0.zip')
    args = parser.parse_args()
    install(args.archive, Path(__file__).resolve().parents[1])
