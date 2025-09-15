import inspect
import json
import os
from datetime import datetime

def custom_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def recursive_object_seralizer(obj, visited):
    seralized_dict = {}
    keys = list(obj.__dict__)
    for k in keys:
        if id(obj.__dict__[k]) in visited:
            seralized_dict[k] = "<RECURSIVE {}>".format(obj.__dict__[k])
            continue
        if isinstance(obj.__dict__[k], (float, int, str, bool, type(None))):
            seralized_dict[k] = obj.__dict__[k]
        elif isinstance(obj.__dict__[k], tuple):
            ## handle tuple
            seralized_dict[k] = obj.__dict__[k]
        elif isinstance(obj.__dict__[k], set):
            ## handle set
            seralized_dict[k] = obj.__dict__[k]
        elif isinstance(obj.__dict__[k], list):
            ## handle list
            seralized_dict[k] = obj.__dict__[k]
        elif hasattr(obj.__dict__[k], '__dict__'):
            ## handle object
            visited.append(id(obj.__dict__[k]))
            seralized_dict[k] = obj.__dict__[k]
        elif isinstance(obj.__dict__[k], dict):
            visited.append(id(obj.__dict__[k]))
            seralized_dict[k] = obj.__dict__[k]
        elif callable(obj.__dict__[k]):
            ## handle function
            if hasattr(obj.__dict__[k], '__name__'):
                seralized_dict[k] = "<function {}>".format(obj.__dict__[k].__name__)
        else:
            seralized_dict[k] = str(obj.__dict__[k])
    return seralized_dict

def inspect_code(func):
   def wrapper(*args, **kwargs):
       visited = []
       json_base = "/home/changshu/ClassEval/data/benchmark_solution_code/input-output/"
       if not os.path.exists(json_base):
           os.mkdir(json_base)
       jsonl_path = json_base + "/PDFHandler.jsonl"
       para_dict = {"name": func.__name__}
       args_names = inspect.getfullargspec(func).args
       if len(args) > 0 and hasattr(args[0], '__dict__') and args_names[0] == 'self':
           ## 'self'
           self_args = args[0]
           para_dict['self'] = recursive_object_seralizer(self_args, [id(self_args)])
       else:
           para_dict['self'] = {}
       if len(args) > 0 :
           if args_names[0] == 'self':
               other_args = {}
               for m,n in zip(args_names[1:], args[1:]):
                   other_args[m] = n
           else:
               other_args = {}
               for m,n in zip(args_names, args):
                   other_args[m] = n
           
           para_dict['args'] = other_args
       else:
           para_dict['args'] = {}
       if kwargs:
           para_dict['kwargs'] = kwargs
       else:
           para_dict['kwargs'] = {}
          
       result = func(*args, **kwargs)
       para_dict["return"] = result
       with open(jsonl_path, 'a') as f:
           f.write(json.dumps(para_dict, default=custom_serializer) + "\n")
       return result
   return wrapper


'''
# The class allows merging multiple PDF files into one and extracting text from PDFs using PyPDF2 library.

import PyPDF2

class PDFHandler:
    def __init__(self, filepaths):
        """
        takes a list of file paths filepaths as a parameter.
        It creates a list named readers using PyPDF2, where each reader opens a file from the given paths.
        """
        self.filepaths = filepaths
        self.readers = [PyPDF2.PdfFileReader(fp) for fp in filepaths]

    def merge_pdfs(self, output_filepath):
        """
        Read files in self.readers which stores handles to multiple PDF files.
        Merge them to one pdf and update the page number, then save in disk.
        :param output_filepath: str, ouput file path to save to
        :return: str, "Merged PDFs saved at {output_filepath}" if successfully merged
        >>> handler = PDFHandler(['a.pdf', 'b.pdf'])
        >>> handler.merge_pdfs('out.pdf')
        Merged PDFs saved at out.pdf
        """

    def extract_text_from_pdfs(self):
        """
        Extract text from pdf files in self.readers
        :return pdf_texts: list of str, each element is the text of one pdf file
        >>> handler = PDFHandler(['a.pdf', 'b.pdf'])
        >>> handler.extract_text_from_pdfs()
        ['Test a.pdf', 'Test b.pdf']
        """
'''

import PyPDF2


class PDFHandler:
    def __init__(self, filepaths):
        self.filepaths = filepaths
        # PdfFileReader is deprecated and was removed in PyPDF2 3.0.0. Use PdfReader instead.
        self.readers = [PyPDF2.PdfReader(fp) for fp in filepaths]

    @inspect_code
    def merge_pdfs(self, output_filepath):
        pdf_writer = PyPDF2.PdfWriter()

        for reader in self.readers:
            # reader.getNumPages is deprecated and was removed in PyPDF2 3.0.0. Use len(reader.pages) instead.
            for page_num in range(len(reader.pages)):
                # reader.getPage(pageNumber) is deprecated and was removed in PyPDF2 3.0.0. Use reader.pages[page_number] instead.
                page = reader.pages[page_num]
                # addPage is deprecated and was removed in PyPDF2 3.0.0. Use add_page instead.
                pdf_writer.add_page(page)

        with open(output_filepath, 'wb') as out:
            pdf_writer.write(out)
        return f"Merged PDFs saved at {output_filepath}"

    @inspect_code
    def extract_text_from_pdfs(self):
        pdf_texts = []
        for reader in self.readers:
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                pdf_texts.append(page.extract_text())
        return pdf_texts



import os
import unittest
from PyPDF2 import PdfFileReader
from reportlab.pdfgen import canvas


class TestPDFHandler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_files = ["test1.pdf", "test2.pdf"]
        cls.test_text = ["This is a test1.", "This is a test2."]
        for i in range(2):
            c = canvas.Canvas(cls.test_files[i])
            c.drawString(100, 100, cls.test_text[i])
            c.showPage()
            c.save()

    @classmethod
    def tearDownClass(cls):
        for filename in cls.test_files:
            os.remove(filename)
        os.remove("merged.pdf")



class PDFHandlerTestMergePdfs(unittest.TestCase):
    def setUp(self) -> None:
        TestPDFHandler.setUpClass()

    def tearDown(self) -> None:
        TestPDFHandler.tearDownClass()

    def test_merge_pdfs(self):
        TestPDFHandler.setUpClass()
        handler = PDFHandler(TestPDFHandler.test_files)
        result = handler.merge_pdfs("merged.pdf")
        self.assertEqual("Merged PDFs saved at merged.pdf", result)
        self.assertTrue(os.path.exists("merged.pdf"))



class PDFHandlerTestExtractTextFromPdfs(unittest.TestCase):
    def setUp(self) -> None:
        TestPDFHandler.setUpClass()

    def test_extract_text_from_pdfs(self):
        TestPDFHandler.setUpClass()
        handler = PDFHandler(TestPDFHandler.test_files)
        result = handler.extract_text_from_pdfs()
        self.assertEqual(result, ["This is a test1.\n", "This is a test2.\n"])


class PDFHandlerTestMain(unittest.TestCase):
    def setUp(self) -> None:
        TestPDFHandler.setUpClass()

    def tearDown(self) -> None:
        TestPDFHandler.tearDownClass()

    def test_main(self):
        TestPDFHandler.setUpClass()
        handler = PDFHandler(TestPDFHandler.test_files)
        result = handler.merge_pdfs("merged.pdf")
        self.assertEqual("Merged PDFs saved at merged.pdf", result)
        self.assertTrue(os.path.exists("merged.pdf"))

        result = handler.extract_text_from_pdfs()
        self.assertEqual(result, ["This is a test1.\n", "This is a test2.\n"])
