'''
Functions to download appeals data and corresponding orignal case information

April 2020
'''
import requests
from bs4 import BeautifulSoup
import re
import urllib.request
import PyPDF2
import io
# for testing
import pandas as pd

STARTING_PG = 'https://www.hhs.gov/about/agencies/dab/decisions/' + \
              'board-decisions/board-decisions-by-year/index.html'


def get_orig_case_info(txt):
    '''
    Input: the text of the APPEAL decision, as a string.

    Returns: the case number and case year of the ORIGINAL ALJ decision.
             Use this to construct the URL for the original ALJ decision.
    '''
    case_match = re.search(r"[A-Z][A-Z]\d\d\d\d \(\d\d\d\d\)", txt)
    i = 0
    if not case_match:
        case_match = re.search(r"DAB No.*?\d\)", txt)
        i = 2
    orig = case_match.group().split()
    casenum = orig[i].lower()
    caseyr = re.search(r"(?<=\()\d\d\d\d", case_match.group()).group()
    return casenum, caseyr


def get_pdf_txt(pdf_url):
    '''
    Input: the URL of any pdf file.

    Returns: the text contents of the pdf file as a string.
    '''
    response = requests.get(pdf_url)
    pdfReader = PyPDF2.PdfFileReader(io.BytesIO(response.content))
    docstring = ''
    for page in pdfReader.pages:
        txt = page.extractText()
        docstring += txt
    return clean_text(docstring)


def clean_text(pdf_docstring):
    '''
    Takes docstring and cleans out the \n chars
    
    Inputs: docstring representing pdf conversion

    Output: a cleaner docstring representation of same pdf
    '''
    return ' '.join(pdf_docstring.replace("\n", '').split())


def make_soup(url):
    '''
    Input: any URL.

    Returns: delicious soup.
    '''
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    return soup


def get_urls_all_years(url):
    '''
    written while falling asleep, will docstring later
    '''
    soup = make_soup(url)
    yr_lst = []
    for yr in soup.find("div", {"class": "syndicate"}).findAll('a'):
        yr_lst.append((yr.getText(), f"https://hhs.gov{yr.get('href')}"))
    yr_url_dict = {}
    for yrnum, yrurl in yr_lst[:21]:
      yr_url_dict[yrnum] = get_urls_one_year(yrurl)
    return yr_url_dict


def get_urls_one_year(url):
    '''
    written while falling asleep, will docstring later
    '''
    yrsoup = make_soup(url)
    yrurllst = []
    for case in yrsoup.find("div", {"class": "syndicate"}).findAll('a'):
      if ('adobe' not in case.get('href')) and ('mailto' not in case.get('href')):
        yrurllst.append((f"https://hhs.gov{case.get('href')}", case.getText()))
    return yrurllst


def convert_decision_binary(decision_txt):
    '''
    THIS NEEDS TO BE COMPLETED
    Converts the text re the case decision to binary
      0 if appelate court affirms
      1 if overturns original ruling
    '''
    outcome = None
    overturn_lst = ['vacate']
    affirm_lst = ['affirm', 'uphold', 'sustain']
    return outcome


def get_html_info(appeal_url):
    '''
    Takes the URL of an appeal decision (which is a HTML page),
    gets the appeal outcome,
    goes back to the original ALJ decision (which is also a HTML page),
    and gets the full text of the original ALJ decision.

    Input:
    the URL of the appeal decision (a HTML page)

    Returns:
    - the appeal outcome
    - the full text of the appeal decision, as a string
    - the full text of the original ALJ decision, as a string
    '''
    # soup = make_soup(appeal_url)
    # all_appeal_text = clean_text(soup.find("div", {"class": "field-name-body"}).getText())
    print('appeal url', appeal_url)
    all_appeal_text, soup = initialize_get_full_text_html(appeal_url)
    orig_info = soup.find("div", {"class": "field-item even"}).find('p').getText()
    try:
        casenum, caseyr = get_orig_case_info(orig_info)
        outcome = soup.find("div", {"class": "legal-decision-judge"}).find_previous().getText()
        orig_url = f'https://www.hhs.gov/about/agencies/dab/decisions/alj-decisions/{caseyr}/alj-{casenum}/index.html'
        all_orig_text = get_original_text(casenum, caseyr)

        return outcome, all_appeal_text, all_orig_text, casenum, orig_url, appeal_url
    except:
        # these babies got problems to be fixed later
        print(appeal_url)
        pass


def initialize_get_full_text_html(url):
    '''
    '''
    soup = make_soup(url)
    all_text = clean_text(soup.find("div", {"class": "field-name-body"}).getText())
    return all_text, soup


def get_pdf_info(appeal_url):
    '''
    Takes the URL of an appeal decision (which is a pdf html page),
    gets the appeal outcome,
    goes back to the original ALJ decision (which is also a HTML page),
    and gets the full text of the original ALJ decision.

    Input:
    the URL of the appeal decision (a HTML page)

    Returns:
    - the appeal outcome
    - the full text of the appeal decision, as a string
    - the full text of the original ALJ decision, as a string
    '''
    # regex in next line to have text start at same spot as html text
    all_appeal_text = re.search(r'DECISION.*', get_pdf_txt(appeal_url)).group()
    print('appeal url', appeal_url)
    casenum, caseyr = get_orig_case_info(all_appeal_text)
    conclusion = re.search("(?<=Conclusion ).*?\.", all_appeal_text)
    if conclusion:
        outcome = conclusion.group()
    else:
        outcome = 'by hand'
    orig_url = f'https://www.hhs.gov/sites/default/files/alj-{casenum}.pdf'
    all_orig_text = get_original_text(casenum, caseyr)
    return outcome, all_appeal_text, all_orig_text, casenum, orig_url, appeal_url

def get_original_text(casenum, caseyr):
    '''
    '''
    try:
        orig_url = f'https://www.hhs.gov/sites/default/files/alj-{casenum}.pdf'
        all_orig_text = get_pdf_txt(orig_url)
    except:
        orig_url = f'https://www.hhs.gov/about/agencies/dab/decisions/alj-decisions/{caseyr}/alj-{casenum}/index.html'
        all_orig_text, _ = initialize_get_full_text_html(orig_url)
    return all_orig_text


def combining(initial_url=STARTING_PG):
    '''
    '''
    full_urls = get_urls_all_years(initial_url)

    for year in full_urls.keys():
        # indexing list below for testing
        test_cases = full_urls[year][:3]
        for case in test_cases:
            dab_url = case[0]
            print('dab_url', dab_url)
            dab_id = re.search(r'(?<=[dab|dab-])\d*\d', dab_url.lower()).group()
            print('dab_id', dab_id)
            if dab_url[-4:] in ['html', '.htm']:
                try:
                    outcome, dab_text, alj_text, alj_id, alj_url, \
                    dab_url = get_html_info(dab_url)
                except:
                    pass
            else:
                outcome, dab_text, alj_text, alj_id, alj_url, \
                dab_url = get_pdf_info(dab_url)
            print('outcome', outcome)

