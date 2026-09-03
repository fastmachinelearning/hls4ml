#!/usr/bin/env python3
"""Generate assistant-specific views of the documents in .agents/.

The documents in .agents/ are the single source of truth. Assistants disagree about where such files live and
what metadata they carry, so this script writes copies in the layout each one expects. Generated output is
git-ignored; edit .agents/ and regenerate.

    python .agents/agent_adapters.py --list
    python .agents/agent_adapters.py claude cursor
    python .agents/agent_adapters.py --all --clean

Adding an adapter is a matter of writing one function and adding it to ADAPTERS. Please do not add
assistant-specific content to the source documents themselves.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parent
SKIP = {'README.md', 'local-setup.template.md'}


def read_docs() -> list[dict]:
    """Return the parsed front matter and body of every source document."""
    docs = []
    for path in sorted(SOURCE.glob('*.md')):
        if path.name in SKIP:
            continue
        text = path.read_text()
        match = re.match(r'^---\n(.*?)\n---\n(.*)$', text, re.S)
        if not match:
            raise SystemExit(f'{path}: missing front matter')
        front, body = match.group(1), match.group(2)

        name = re.search(r'^name:\s*(.+)$', front, re.M)
        description = re.search(r'^description:\s*>-\n((?:\s{2,}.*\n?)+)', front, re.M)
        globs = re.findall(r'^\s+-\s*"(.+)"$', front, re.M)
        if not name or not description:
            raise SystemExit(f'{path}: front matter needs a name and a description')

        docs.append(
            {
                'stem': path.stem,
                'name': name.group(1).strip(),
                'description': ' '.join(line.strip() for line in description.group(1).splitlines()),
                'globs': globs,
                'body': body.lstrip('\n'),
                'path': path,
            }
        )
    return docs


def adapt_claude(docs: list[dict]) -> Path:
    """Claude Code: .claude/skills/<name>/SKILL.md, dispatched by description."""
    out = REPO / '.claude' / 'skills'
    for doc in docs:
        directory = out / doc['name']
        directory.mkdir(parents=True, exist_ok=True)
        body = doc['body'].replace('](', '](../../../.agents/')
        front = f'---\nname: {doc["name"]}\ndescription: {doc["description"]}\n---\n\n'
        (directory / 'SKILL.md').write_text(front + body)
    return out


def adapt_cursor(docs: list[dict]) -> Path:
    """Cursor: .cursor/rules/<name>.mdc, activated by glob."""
    out = REPO / '.cursor' / 'rules'
    out.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        globs = ', '.join(doc['globs'])
        front = f'---\ndescription: {doc["description"]}\nglobs: {globs}\nalwaysApply: false\n---\n\n'
        (out / f'{doc["stem"]}.mdc').write_text(front + doc['body'])
    return out


def adapt_copilot(docs: list[dict]) -> Path:
    """GitHub Copilot: .github/instructions/<name>.instructions.md, applied by path."""
    out = REPO / '.github' / 'instructions'
    out.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        apply_to = ','.join(doc['globs']) or '**'
        front = f"---\napplyTo: '{apply_to}'\n---\n\n"
        (out / f'{doc["stem"]}.instructions.md').write_text(front + f'<!-- {doc["description"]} -->\n\n' + doc['body'])
    return out


def adapt_plain(docs: list[dict]) -> Path:
    """A single concatenated file, for assistants that take one document."""
    out = REPO / 'agent-docs.md'
    parts = ['# hls4ml agent documentation\n', '<!-- Generated from .agents/ by .agents/agent_adapters.py -->\n']
    for doc in docs:
        parts.append(f'\n\n---\n\n<!-- when: {doc["description"]} -->\n\n{doc["body"]}')
    out.write_text(''.join(parts))
    return out


ADAPTERS = {
    'claude': adapt_claude,
    'cursor': adapt_cursor,
    'copilot': adapt_copilot,
    'plain': adapt_plain,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('adapters', nargs='*', metavar='ADAPTER', help=f'any of: {", ".join(ADAPTERS)}')
    parser.add_argument('--all', action='store_true', help='generate every view')
    parser.add_argument('--list', action='store_true', help='list the available views and exit')
    parser.add_argument('--clean', action='store_true', help='remove generated output first')
    args = parser.parse_args()

    if args.list:
        for name, func in ADAPTERS.items():
            print(f'{name:10s} {func.__doc__.splitlines()[0]}')
        return

    selected = list(ADAPTERS) if args.all else args.adapters
    if not selected:
        parser.error('name at least one adapter, or pass --all (see --list)')
    unknown = [name for name in selected if name not in ADAPTERS]
    if unknown:
        parser.error(f'unknown adapter(s): {", ".join(unknown)} (see --list)')

    docs = read_docs()
    for name in selected:
        if args.clean:
            target = (
                REPO
                / {
                    'claude': '.claude/skills',
                    'cursor': '.cursor/rules',
                    'copilot': '.github/instructions',
                    'plain': 'agent-docs.md',
                }[name]
            )
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        written = ADAPTERS[name](docs)
        print(f'{name}: wrote {len(docs)} documents to {written.relative_to(REPO)}')


if __name__ == '__main__':
    main()
