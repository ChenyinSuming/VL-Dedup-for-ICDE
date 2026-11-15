"""Requests to the two AI auditors through an OpenAI-compatible chat completions endpoint.

Set AUDIT_API_BASE_URL (for example https://api.openai.com/v1) and AUDIT_API_SECRET before
running an audit step. Auditor 1 is gpt-6.1-sol and auditor 2 is gpt-5.6-terra.
"""
import base64
import json
import os
import time
import urllib.request
from pathlib import Path

AUDITOR_MODELS = {1: 'gpt-6.1-sol', 2: 'gpt-5.6-terra'}


def image_part(path):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,' + data, detail='high'))


def chat(auditor, messages, token_limit, token_field='max_tokens', timeout=900, attempts=3):
    """Send one request and return the decoded JSON response."""
    base = os.environ['AUDIT_API_BASE_URL'].rstrip('/')
    url = base if base.endswith('/chat/completions') else base + '/chat/completions'
    payload = {'model': AUDITOR_MODELS[auditor], 'messages': messages, token_field: token_limit}
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method='POST', headers={'Authorization': 'Bearer ' + os.environ['AUDIT_API_SECRET'], 'Content-Type': 'application/json'})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(10 * attempt)


def content(response):
    return response['choices'][0]['message']['content']
