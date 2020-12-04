'''
Downloads all available financial statements for a company listed on Companies House.
Takes a single Companies House 'company number' as input.
'''

import requests
import json
import pandas as pd
import numpy as np

key = 'C59-GL9Hnn2LpII0CRuSN9X8DrT4x5llA6G3mlzP'  # Companies House API key
company_no = '08065100'  # Companies House company number

profile_url = "https://api.companieshouse.gov.uk/company/{company_no}"
accounts_url = "https://api.companieshouse.gov.uk/company/{company_no}/filing-history?category=accounts"


def retrieve_statements(company_no):
    """
    Retrieves financial statements based on company number
    """
    r = requests.get(profile_url.format(company_no=company_no), auth=(key, ""))
    company_name = r.json()['company_name']

    r = requests.get(accounts_url.format(
        company_no=company_no), auth=(key, ""))

    # Find number of pages
    pages = int(np.ceil(r.json()['total_count'] / r.json()['items_per_page']))

    # Retrieve annual accounts
    doc_ids = {'description': [],
               'date': [],
               'url': []
               }

    for entry in r.json()['items']:
        if entry['type'] == "AA":
            doc_ids['description'].append(entry['description'])
            doc_ids['date'].append(entry['action_date'])
            doc_ids['url'].append(entry['links']['document_metadata'])

    df = pd.DataFrame(doc_ids, columns=['description', 'date', 'url'])

    for index, row in df.iterrows():
        doc = requests.get(row['url'] + "/content", auth=(key, ""))
        with open("downloads/" + company_name + " " + row['date'] + ".pdf", "wb") as f:
            f.write(doc.content)


retrieve_statements(company_no)
