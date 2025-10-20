from ingest import data_ingest_and_index
from generator import generate

pdf_paths = ['../data/test_docs/test01.pdf',
             '../data/test_docs/test02.pdf',
             '../data/test_docs/test03.pdf']


data_ingest_and_index(pdf_paths)
generate()
