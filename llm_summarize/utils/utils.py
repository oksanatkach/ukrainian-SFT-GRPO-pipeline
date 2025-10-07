import transformers
import torch
import random
import numpy as np

def set_seed(seed: int) -> None:
    transformers.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

import requests
from bs4 import BeautifulSoup
import re


def extract_text_from_html(url):
    """
    Extract plain text content from the HTML page
    """
    try:
        # Fetch the HTML content
        response = requests.get(url)
        response.raise_for_status()
        response.encoding = response.apparent_encoding

        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script, style, nav, header, footer
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()

        # Get the main content - try to find the body or main content area
        main_content = soup.find('body') or soup
        ALL = ''
        for element in main_content.find_all('p'):
            text = element.text
            text = text.replace('\n', '').strip()
            text = re.sub(r'\s+', ' ', text)

            if 'my-ch-stage' in element.attrs.get('class', []):
                text = f"({text})"
            if 'my-ch' in element.attrs.get('class', []):
                text = f"{text}:"

            ALL += '\n' + text

        return ALL

    except Exception as e:
        return f"Error: {str(e)}"
