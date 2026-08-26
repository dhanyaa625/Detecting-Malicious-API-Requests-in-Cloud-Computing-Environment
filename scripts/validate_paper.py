r"""
Static validation of paper/agentic_pacx.tex without a LaTeX compiler.

Checks the failure modes that silently break a build: an undefined macro, a
\ref with no \label, a \cite with no \bibitem, unbalanced environments,
braces, math delimiters, or algorithmic IF/FOR blocks, and a raw % inside a
macro value (which would comment out the rest of the line).

Exits non-zero on any failure, so it can gate a commit.

Run:  python scripts/validate_paper.py
"""
import io
import os
import re
import sys
from collections import Counter

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'paper', 'agentic_pacx.tex')
src = io.open(P, encoding='utf-8').read()

# Strip comments so they don't create phantom refs.
lines = []
for ln in src.split('\n'):
    out, esc = [], False
    for ch in ln:
        if ch == '\\':
            esc = not esc
            out.append(ch)
            continue
        if ch == '%' and not esc:
            break
        esc = False
        out.append(ch)
    lines.append(''.join(out))
body = '\n'.join(lines)

fail = 0


def check(name, bad, detail=''):
    global fail
    if bad:
        fail += 1
        print(f'  [FAIL] {name}: {bad} {detail}')
    else:
        print(f'  [ok]   {name}')


print('=== LaTeX static validation ===\n')

# 1. Every macro used is defined
defined = set(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', body))
defined |= set(re.findall(r'\\def\\([A-Za-z]+)', body))
# Remove "\\" line breaks first, else "\\Foo" reads as a macro named Foo.
scan = body.replace('\\\\', ' ')
used = Counter(re.findall(r'\\([A-Z][A-Za-z]*)(?![a-zA-Z])', scan))
KNOWN_LATEX = {
    'IEEEoverridecommandlockouts', 'IEEEauthorblockN', 'IEEEauthorblockA',
    'IEEEkeywords', 'REQUIRE', 'ENSURE', 'IF', 'ELSE', 'ENDIF', 'STATE',
    'FOR', 'ENDFOR', 'RETURN', 'ELSIF', 'WHILE', 'ENDWHILE', 'RGB', 'T',
    'BibTeX', 'AND', 'OR', 'NOT', 'TRUE', 'FALSE', 'COMMENT', 'REPEAT',
    'UNTIL', 'LOOP', 'ENDLOOP', 'PRINT', 'Huge', 'Large', 'LARGE',
    # standard math/base commands
    'Big', 'Bigg', 'Delta', 'Psi', 'Vert', 'Gamma', 'Lambda', 'Omega',
    'Sigma', 'Theta', 'Phi', 'Pi', 'Leftarrow', 'Rightarrow', 'TO',
}
undefined = sorted(m for m in used if m not in defined and m not in KNOWN_LATEX)
check('all macros defined', undefined)

# 2. Defined-but-unused macros (harmless, but flag)
unused = sorted(d for d in defined if used.get(d, 0) == 0 and d != 'BibTeX')
if unused:
    print(f'  [warn] defined but never used: {unused}')

# 3. Every \ref has a \label
labels = set(re.findall(r'\\label\{([^}]+)\}', body))
refs = set(re.findall(r'\\(?:eq)?ref\{([^}]+)\}', body))
check('all \\ref targets exist', sorted(refs - labels))

dupes = [k for k, v in Counter(re.findall(r'\\label\{([^}]+)\}', body)).items() if v > 1]
check('no duplicate labels', dupes)

# 4. Every \cite key has a \bibitem
bibitems = set(re.findall(r'\\bibitem\{([^}]+)\}', body))
cites = set()
for grp in re.findall(r'\\cite\{([^}]+)\}', body):
    cites |= {c.strip() for c in grp.split(',')}
check('all \\cite keys have \\bibitem', sorted(cites - bibitems))
uncited = sorted(bibitems - cites)
if uncited:
    print(f'  [warn] bibitems never cited: {uncited}')

# 5. Environments balanced
begins = Counter(re.findall(r'\\begin\{([^}]+)\}', body))
ends = Counter(re.findall(r'\\end\{([^}]+)\}', body))
unbalanced = {k: (begins[k], ends[k]) for k in set(begins) | set(ends)
              if begins[k] != ends[k]}
check('environments balanced', unbalanced)

# 6. Braces balanced
depth = 0
esc = False
for ch in body:
    if esc:
        esc = False
        continue
    if ch == '\\':
        esc = True
    elif ch == '{':
        depth += 1
    elif ch == '}':
        depth -= 1
check('braces balanced', depth if depth != 0 else None, f'(final depth {depth})')

# 7. Math-mode $ parity
dollars = len(re.findall(r'(?<!\\)\$', body.replace('$$', '')))
check('inline math $ parity', 'odd count' if dollars % 2 else None)

# 8. Literal % must be escaped inside macro VALUES (a raw % would comment out)
raw_pct = []
for m in re.finditer(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}', src):
    val = m.group(2)
    if re.search(r'(?<!\\)%', val):
        raw_pct.append(m.group(1))
check('macro values escape %', raw_pct)


# 9. algorithmic IF/FOR/WHILE blocks balanced
def count(cmd):
    return len(re.findall(r'\\' + cmd + r'(?![A-Za-z])', body))


algo_pairs = {
    'IF/ENDIF': (count('IF') - count('ELSIF'), count('ENDIF')),
    'FOR/ENDFOR': (count('FOR'), count('ENDFOR')),
    'WHILE/ENDWHILE': (count('WHILE'), count('ENDWHILE')),
}
algo_bad = {k: v for k, v in algo_pairs.items() if v[0] != v[1]}
check('algorithmic blocks balanced', algo_bad)

print()
print('RESULT:', 'ALL CHECKS PASSED' if fail == 0 else f'{fail} CHECK(S) FAILED')
sys.exit(1 if fail else 0)
