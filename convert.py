# Author: Alexander Van de Kleut
# Use this file to convert jupyter notebooks to markdown format.
# Notebooks should be in ~/assets/notebooks/<foldername>/

import os
import sys
import nbformat
from traitlets.config import Config
from nbconvert import HTMLExporter

from urllib.request import urlopen

url = 'http://jakevdp.github.com/downloads/notebooks/XKCD_plots.ipynb'
response = urlopen(url).read().decode()
jake_notebook = nbformat.reads(response, as_version=4)
html_exporter = HTMLExporter()
html_exporter.template_file = 'basic'
(body, resources) = html_exporter.from_notebook_node(jake_notebook)


# html_exporter = HTMLExporter()
# html_exporter.template_file = 'basic'
#
# with open(sys.argv[1], 'r') as ipynb:
#     contents = ipynb.read()
#     print(contents)
#     (body, resources) = html_exporter.from_notebook_node(contents)
#     print(body)
