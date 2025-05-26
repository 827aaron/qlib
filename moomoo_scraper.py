import os
import sys
from urllib.parse import urljoin, urlparse
from collections import deque

# Ensure required packages are installed
required = ['requests', 'beautifulsoup4', 'html2text']
for pkg in required:
    try:
        __import__(pkg.split('==')[0])
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])

import requests
from bs4 import BeautifulSoup
import html2text

BASE_URL = 'https://openapi.moomoo.com'
START_PAGE = BASE_URL + '/moomoo-api-doc/intro/intro.html'
DOC_PREFIX = '/moomoo-api-doc/'
LOCAL_ROOT = 'moomoo-api-doc'

# Only crawl Chinese Python documentation
LANG_MARKERS = ['zh', 'cn']
PYTHON_SEG = '/python/'
EXCLUDE_SEGMENTS = ['example', 'examples', 'demo', 'sample']


def is_chinese_python_doc(path: str) -> bool:
    """Return True if path represents a Chinese Python doc page."""
    if PYTHON_SEG not in path:
        return False
    lowered = path.lower()
    if not any(marker in lowered for marker in LANG_MARKERS):
        return False
    if any(seg in lowered for seg in EXCLUDE_SEGMENTS):
        return False
    return True


def save_markdown(url: str, html: str) -> None:
"""Convert html content to Markdown and save to local path mirroring the URL.

Only Chinese Python documentation pages are saved."""
    parsed = urlparse(url)
    path = parsed.path
    if not path.startswith(DOC_PREFIX) or not is_chinese_python_doc(path):
        return
    local_path = os.path.join(LOCAL_ROOT, path[len(DOC_PREFIX):])
    if local_path.endswith('.html') or local_path.endswith('.htm'):
        local_path = local_path.rsplit('.', 1)[0] + '.md'
    else:
        local_path += '.md'
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    md = html2text.html2text(html)
    with open(local_path, 'w', encoding='utf-8') as f:
        f.write(md)


def crawl(start_url: str):
    visited = set()
    queue = deque([start_url])
    while queue:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        try:
            resp = requests.get(url)
            resp.raise_for_status()
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            continue

        html = resp.text
        save_markdown(url, html)

        soup = BeautifulSoup(html, 'html.parser')
        for a in soup.find_all('a', href=True):
            link = a['href']
            if link.startswith('#'):
                continue
            abs_link = urljoin(url, link)
            parsed = urlparse(abs_link)
            if parsed.netloc != urlparse(BASE_URL).netloc:
                continue
            if not parsed.path.startswith(DOC_PREFIX):
                continue
            if not is_chinese_python_doc(parsed.path):
                continue
            normalized = BASE_URL + parsed.path
            if normalized not in visited:
                queue.append(normalized)


if __name__ == '__main__':
    crawl(START_PAGE)
