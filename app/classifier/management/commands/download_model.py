"""Fetch the fine-tuned weights published as a GitHub release asset.

The weights (~240 MB in fp16) are too large for a regular git commit, so the notebook's output is
attached to a release instead. Re-running the notebook produces the same folder locally.
"""
import io
import shutil
import urllib.request
import zipfile

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

RELEASE_URL = 'https://github.com/Aguado4/spamlens-hacking-ai-Aguado-Cruz-Ramirez/releases/download/v1.0/finetuned.zip'


class Command(BaseCommand):
    help = 'Download the SMS fine-tuned model into model/finetuned.'

    def add_arguments(self, parser):
        parser.add_argument('--url', default=RELEASE_URL)
        parser.add_argument('--force', action='store_true', help='Replace an existing model folder.')

    def handle(self, *args, url, force, **options):
        target = settings.SPAMLENS_FINETUNED_DIR
        if (target / 'config.json').exists() and not force:
            self.stdout.write(self.style.SUCCESS(f'Model already present at {target} (use --force to replace).'))
            return

        self.stdout.write(f'Downloading {url} ...')
        try:
            with urllib.request.urlopen(url) as response:
                payload = response.read()
        except OSError as exc:
            raise CommandError(f'Download failed: {exc}') from exc

        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            if not any(n.endswith('config.json') for n in names):
                raise CommandError('The archive does not contain a model (config.json missing).')
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True)
            for name in names:
                if name.endswith('/'):
                    continue
                # Flatten paths so a hostile archive cannot write outside the target folder.
                (target / name.rsplit('/', 1)[-1]).write_bytes(archive.read(name))

        self.stdout.write(self.style.SUCCESS(f'Model saved to {target}. Restart the server to use it.'))
