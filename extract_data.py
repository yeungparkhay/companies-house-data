import numpy as np
import pandas as pd
import os
from os.path import isfile, join
import re
import importlib
import xlsxwriter
from styleframe import StyleFrame

import xbrl_image_parser as xip

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.path import Path
from matplotlib.patches import PathPatch

from imageio import imread

read_path = 'downloads/'
write_path = 'output/'

## Download and process files ###############################


def extract_PDF(read_path, write_path):
    files = [f for f in os.listdir(read_path) if isfile(join(read_path, f))]

    importlib.reload(xip)

    for file in files:
        results = xip.process_PDF(read_path + file)

        # Save results
        results.to_pickle(write_path + file.replace('pdf', 'pkl'))
        results = pd.read_pickle(write_path + file.replace('pdf', 'pkl'))

        # Reformat numbers and drop unnecessary rows and columns
        results['value'] = results['value'].apply(
            lambda x: float(x.replace('-', '0').replace(')', '').replace('(', '-')))
        results = results[['label', 'statement', 'year', 'unit', 'value']]
        results['year'] = results['year'].astype("int64")
        results.drop_duplicates(['label', 'year'], keep='last', inplace=True)

        # Unstack dataframe
        results = results.groupby(['label', 'year', 'unit', 'statement'], sort=False)[
            'value'].sum().unstack('year')

        # Split dataframe into balance sheet and income statement
        balance_sheet = results.query("statement == 'Balance sheet'")
        income_statement = results.query("statement == 'Income statement'")

        for i in (balance_sheet, income_statement):
            i.reset_index(inplace=True)
            i.drop(['statement'], axis=1, inplace=True)

        # Write to Excel
        writer = StyleFrame.ExcelWriter(
            write_path + file.replace('pdf', 'xlsx'))
        balance_sheet.to_excel(excel_writer=writer,
                               sheet_name='Balance sheet')
        income_statement.to_excel(
            excel_writer=writer, sheet_name='Income statement')

        writer.save()


extract_PDF(read_path, write_path)
