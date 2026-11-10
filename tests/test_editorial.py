import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from talk_journal import cli


def sha(data):
    return hashlib.sha256(data).hexdigest()


class EditorialTests(unittest.TestCase):
    def test_rejects_normalized_nested_destinations_without_mutating_base(self):
        from talk_journal.editorial import ingest
        import os
        fixtures = Path(__file__).parent / 'fixtures' / 'editorial'

        def inventory(directory):
            return {p.relative_to(directory): (p.is_dir(), p.stat().st_mtime_ns,
                                               None if p.is_dir() else p.read_bytes())
                    for p in (directory, *directory.rglob('*'))}

        for variant in ('direct', 'deep', 'relative', 'normalized-base', 'replay'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                base = root / 'base'
                self.assertEqual(ingest(fixtures / 'source-1.txt', fixtures / 'packet-1.json', base), 'CREATED')
                before = inventory(base)
                inputs = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                          for p in fixtures.iterdir() if p.is_file()}
                destination = base / ('next/deep' if variant == 'deep' else 'next')
                base_argument = base
                if variant == 'relative':
                    destination = Path(os.path.relpath(root, Path.cwd())) / 'base' / '..' / 'base' / 'next'
                elif variant == 'normalized-base':
                    base_argument = root / 'base' / '..' / 'base'
                number = '1' if variant == 'replay' else '2'
                with self.assertRaisesRegex(ValueError, 'Destination must not be inside base'):
                    ingest(fixtures / f'source-{number}.txt', fixtures / f'packet-{number}.json',
                           destination, base=base_argument)
                self.assertFalse(destination.exists())
                self.assertEqual(inventory(base), before)
                self.assertEqual(ingest(fixtures / 'source-1.txt', fixtures / 'packet-1.json', base), 'NOOP')
                self.assertEqual(ingest(fixtures / 'source-1.txt', fixtures / 'packet-1.json',
                                        root / 'base' / '..' / 'base', base=base), 'NOOP')
                self.assertEqual(inventory(base), before)
                self.assertEqual(inputs, {p: (p.read_bytes(), p.stat().st_mtime_ns)
                                          for p in fixtures.iterdir() if p.is_file()})

    def test_authored_fictional_packets_navigation_and_conflict_guards(self):
        from talk_journal.editorial import ingest
        import re
        fixtures = Path(__file__).parent / 'fixtures' / 'editorial'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            one, two = root / 'one', root / 'two'
            ingest(fixtures / 'source-1.txt', fixtures / 'packet-1.json', one)
            ingest(fixtures / 'source-2.txt', fixtures / 'packet-2.json', two, base=one)
            for page in two.rglob('*.md'):
                for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', page.read_text()):
                    self.assertTrue((page.parent / target).is_file(), (page, target))
            page = (two / 'rooftop-trial.md').read_text()
            self.assertIn('if weather permits', page)
            self.assertIn('budget and currency remain unsettled', page)
            self.assertIn('not a greenhouse', page)
            self.assertEqual(len(json.loads((two / 'manifest.json').read_text())['sources']), 2)
            self.assertIn('Bo initially withheld approval', (two / 'trial-decisions.md').read_text())
            packet = json.loads((fixtures / 'packet-2.json').read_text())
            packet['pages'][0]['before_sha256'] = '0' * 64
            stale = root / 'stale.json'
            stale.write_text(json.dumps(packet))
            with self.assertRaisesRegex(ValueError, 'before-image'):
                ingest(fixtures / 'source-2.txt', stale, root / 'stale', base=one)
            self.assertFalse((root / 'stale').exists())
            # Changed prose for an already consumed source is a conflict, not a rewrite.
            with self.assertRaisesRegex(ValueError, 'different editorial packet'):
                ingest(fixtures / 'source-2.txt', stale, root / 'conflict', base=two)
            (one / 'rooftop-trial.md').write_text('unexpected human edit')
            with self.assertRaisesRegex(ValueError, 'modified'):
                ingest(fixtures / 'source-2.txt', fixtures / 'packet-2.json', root / 'corrupt', base=one)

    def test_interrupted_write_has_no_completion_and_cannot_be_used_as_base(self):
        from talk_journal.editorial import ingest
        from unittest.mock import patch
        fixtures = Path(__file__).parent / 'fixtures' / 'editorial'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            partial = root / 'partial'
            with patch('talk_journal.editorial.atomic_text', side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):
                    ingest(fixtures / 'source-1.txt', fixtures / 'packet-1.json', partial)
            self.assertFalse((partial / 'manifest.json').exists())
            with self.assertRaises(OSError):
                ingest(fixtures / 'source-1.txt', fixtures / 'packet-1.json', root / 'next', base=partial)

    def test_rejects_untrusted_base_and_symlink_outputs(self):
        from talk_journal.editorial import ingest
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, packet, one, two = [root / item for item in ('raw.txt', 'packet.json', 'one', 'two')]
            raw = b'Ada: Maybe beans.\n'
            source.write_bytes(raw)
            data = {'schema_version': 1, 'source_sha256': sha(raw),
                    'author': 'openai-codex/gpt-6.1-sol-900k',
                    'dossier': 'Ada considers beans.', 'uncertainty': ['Undecided.'],
                    'quotes': ['Maybe beans.'],
                    'pages': [{'path': 'garden.md', 'before_sha256': None,
                               'markdown': '# Garden\n\nUndecided.\n'}]}
            packet.write_text(json.dumps(data))
            ingest(source, packet, one)
            marker = one / 'manifest.json'
            original = marker.read_bytes()
            manifest = json.loads(original)
            # A hostile manifest must not even read paths outside its snapshot.
            manifest['artifacts']['../absent-private-file'] = '0' * 64
            marker.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                ingest(source, packet, two, base=one)
            self.assertFalse(two.exists())
            marker.write_bytes(original)
            page = one / 'garden.md'
            saved = page.read_bytes()
            page.unlink()
            outside = root / 'outside.md'
            outside.write_bytes(saved)
            page.symlink_to(outside)
            with self.assertRaises(ValueError):
                ingest(source, packet, two, base=one)
            page.unlink()
            page.write_bytes(saved)
            (one / 'untracked.md').write_text('unreviewed')
            with self.assertRaises(ValueError):
                ingest(source, packet, two, base=one)
            (one / 'untracked.md').unlink()
            alias = root / 'alias'
            alias.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                ingest(source, packet, alias / 'unexpected')
            self.assertFalse((root / 'unexpected').exists())

    def test_rejects_malformed_unsafe_or_mismatched_packets_before_writes(self):
        from talk_journal.editorial import ingest
        import copy
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, packet, destination = root / 'raw.txt', root / 'packet.json', root / 'wiki'
            raw = b'Ada: Maybe beans.\n'
            source.write_bytes(raw)
            good = {'schema_version': 1, 'source_sha256': sha(raw),
                    'author': 'openai-codex/gpt-6.1-sol-900k',
                    'dossier': 'Ada considers beans.', 'uncertainty': ['Undecided.'],
                    'quotes': ['Maybe beans.'],
                    'pages': [{'path': 'garden.md', 'before_sha256': None,
                               'markdown': '# Garden\n\nUndecided.\n'}]}
            variants = []
            for key, value in [('schema_version', True), ('source_sha256', '0' * 64),
                               ('quotes', ['invented']), ('uncertainty', []),
                               ('dossier', '<script>bad</script>'), ('pages', []),
                               ('author', 'unverified-other-model')]:
                bad = copy.deepcopy(good)
                bad[key] = value
                variants.append(bad)
            for name in ['../escaped.md', '/absolute.md', 'sources/raw.md', 'index.md',
                         'review.md', 'manifest.json', 'C:\\evil.md', '.hidden.md', 'Garden.md']:
                bad = copy.deepcopy(good)
                bad['pages'][0]['path'] = name
                variants.append(bad)
            for body in ['[escape](../outside.md)', '[remote](https://example.com)',
                         '![[secret]]', '![image](garden.md)', '<img src=x>',
                         '[missing](missing.md)', '[[unknown]]', '\x00bad',
                         '[ref][x]\n\n[x]: javascript:bad', '\\[escape](../outside.md)']:
                bad = copy.deepcopy(good)
                bad['pages'][0]['markdown'] = body
                variants.append(bad)
            bad = copy.deepcopy(good)
            bad['pages'] *= 2
            variants.append(bad)
            variants.extend([[], None, {'schema_version': 1}, {**good, 'extra': True}])
            for bad in variants:
                with self.subTest(packet=bad):
                    packet.write_text(json.dumps(bad))
                    with self.assertRaises(ValueError):
                        ingest(source, packet, destination)
                    self.assertFalse(destination.exists())
                    self.assertEqual(source.read_bytes(), raw)

    def test_replay_and_overlapping_update_preserve_history_and_review_full_diff(self):
        from talk_journal.editorial import ingest
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, packet = root / 'raw.txt', root / 'packet.json'
            first = b'Ada: I might plant beans.\n'
            source.write_bytes(first)
            data = {
                'schema_version': 1, 'source_sha256': sha(first),
                'author': 'openai-codex/gpt-6.1-sol-900k',
                'dossier': 'Ada is considering beans.',
                'uncertainty': ['No commitment.'], 'quotes': ['I might plant beans.'],
                'pages': [{'path': 'garden.md', 'before_sha256': None,
                           'markdown': '# Garden\n\nAda may plant beans.\n'}]
            }
            packet.write_text(json.dumps(data))
            one, two = root / 'one', root / 'two'
            self.assertEqual(ingest(source, packet, one), 'CREATED')
            frozen = {p.relative_to(one): (p.read_bytes(), p.stat().st_mtime_ns)
                      for p in one.rglob('*') if p.is_file()}
            self.assertEqual(ingest(source, packet, one, base=one), 'NOOP')
            self.assertEqual(frozen, {p.relative_to(one): (p.read_bytes(), p.stat().st_mtime_ns)
                                     for p in one.rglob('*') if p.is_file()})
            second = b'Ada: I chose peas instead, but the date is uncertain.\n'
            source.write_bytes(second)
            data.update(source_sha256=sha(second), dossier='Ada now chooses peas instead of beans.',
                        uncertainty=['Planting date uncertain.'], quotes=['I chose peas instead, but the date is uncertain.'])
            data['pages'] = [{'path': 'garden.md', 'before_sha256': sha((one / 'garden.md').read_bytes()),
                              'markdown': '# Garden\n\nAda considered beans, then chose peas. Date uncertain.\n'}]
            packet.write_text(json.dumps(data))
            self.assertEqual(ingest(source, packet, two, base=one), 'CREATED')
            self.assertIn('chose peas', (two / 'garden.md').read_text())
            self.assertIn(sha(first), (two / 'garden.md').read_text())
            self.assertIn(sha(second), (two / 'garden.md').read_text())
            self.assertEqual((two / 'sources' / (sha(first) + '.txt')).read_bytes(), first)
            review = (two / 'review.md').read_text()
            self.assertIn('-Ada may plant beans.', review)
            self.assertIn('+Ada considered beans, then chose peas.', review)
            self.assertIn('Before page', review)
            self.assertEqual(frozen, {p.relative_to(one): (p.read_bytes(), p.stat().st_mtime_ns)
                                     for p in one.rglob('*') if p.is_file()})

    def test_cli_creates_linked_reviewable_snapshot_without_touching_raw(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'transcript.txt'
            raw = b'Ada: I might plant beans. I have not decided.\n'
            source.write_bytes(raw)
            packet = root / 'packet.json'
            packet.write_text(json.dumps({
                'schema_version': 1, 'source_sha256': sha(raw),
                'author': 'openai-codex/gpt-6.1-sol-900k',
                'dossier': 'Ada considers beans; this is not a commitment.',
                'uncertainty': ['Planting remains undecided.'],
                'quotes': ['I might plant beans.', 'I have not decided.'],
                'pages': [{'path': 'garden.md', 'before_sha256': None,
                           'markdown': '# Garden\n\nAda may plant beans; undecided.\n'}]
            }))
            wiki = root / 'wiki'
            self.assertEqual(cli.main(['editorial', str(source), str(packet), str(wiki)]), 0)
            self.assertEqual(cli.main(['editorial', str(source), str(packet), str(wiki)]), 0)
            self.assertEqual(source.read_bytes(), raw)
            self.assertIn('garden.md', (wiki / 'index.md').read_text())
            self.assertIn('undecided', (wiki / 'garden.md').read_text())
            self.assertEqual((wiki / 'sources' / (sha(raw) + '.txt')).read_bytes(), raw)
            review = (wiki / 'review.md').read_text()
            self.assertIn('Full page', review)
            self.assertIn('+++ garden.md', review)
            self.assertIn('Planting remains undecided.', review)
            self.assertTrue((wiki / 'manifest.json').is_file())


if __name__ == '__main__':
    unittest.main()
