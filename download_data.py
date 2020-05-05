'''
Functions to download appeals data and corresponding orignal case information

April 2020

architecture:

every year has one page of appeals (appeals in 2020, 2019, etc) - get URLs for those year pages (get_urls_all_years SHOULD do this but again i was half-asleep)
2. for every year's page of appeals:
get the URLs for all the appeals linked for that year (get_urls_one_year should do this)
3. for each single appeal URL in a year:
at this point, pull the appeal number from the link text
pull the full text of the appeal (using a different function depending if it's HTML or pdf)
pull the appeal decision (just the sentence for now, we'll worry about binary conversion later)
get the case number and year of the original decision (a more robust version of get_orig_case_info)
construct URL of original decision using case number and year. have both HTML and pdf
go to the URL of original decision. try both HTML and pdf, if the HTML URL doesn't have a response it's a pdf so do that instead
pull the full text of the original decision (using a different function depending if it's HTML or pdf)
4. i'm thinking we write this straight into the database using psycopg2? obviously we can open/close cursors, etc, but this might be better than having intermediate csvs?
'''
import re
import PyPDF2 # Food for thought: I think pdfminer.six may be a stronger library here?
import io
import requests
from urllib.parse import urlparse, urljoin, urlunparse
from bs4 import BeautifulSoup
from urllib.error import HTTPError
import psycopg2
import csv
from configparser import ConfigParser
import argparse
# for testing
import pandas as pd
import logging
from collections import namedtuple

ALJ_START_PAGE = 'https://www.hhs.gov/about/agencies/dab/decisions/alj-decisions/' +\
                'alj-decisions-by-year/index.html'
DAB_START_PAGE = 'https://www.hhs.gov/about/agencies/dab/decisions/board-decisions/' +\
                 'board-decisions-by-year/index.html'

OLD_URL_PATTERN = r'files/static/dab/decisions/'

PDF = 0
OLD_HTML = 1
NEW_HTML = 2

TBL_WHITELIST = ['raw_data', 'raw_data_test']

logging.basicConfig(filename='scraper.log', level=logging.DEBUG)

APPEAL_RECORD_FIELDS = ['dab_id', 'alj_id', 'dab_text', 'alj_text', 'dab_url', 'alj_url',
                        'overturned']
AppealRecord = namedtuple("AppealRecord", APPEAL_RECORD_FIELDS)

class Appeal:

    def __init__(self, dab_url, dab_case_info, alj_catalog):
        '''
        Generate a new Appeal object. Likely use case is to only call with dab_id.

        Inputs:
        dab_url (str): url of the DAB case for this appeal
        dab_case_info (str): DAB case info string
        alj_catalog (dict): dictionary with ALJ case numbers as keys and the associated
                            case's URL as value (generated by get_alj_catalog function)
        '''
        self.__init_all_vars()
        self.dab_url = urlparse(dab_url)

        self.__extract_dab_id(dab_case_info)
        if not self.dab_id:
            logging.warning(f'The following appears not to be a DAB case: {dab_case_info} ' +
                        f'\n{dab_url}')
            return None

        self.dab_text = scrape_decision_text(self.dab_url)
        if not self.dab_text:
            logging.warning("Couldn't extract DAB text for the following case: DAB ID " +
                        f"{self.dab_id}, DAB URL: {self.dab_url}")
            return None

        self.__extract_dab_outcome()
        if not self.dab_outcome:
                logging.warning("Couldn't extract outcome for the following case:" +
                                f"DAB ID: {self.dab_id}")
        else:
            self.__convert_dab_outcome()
            if not self.dab_outcome_binary:
                logging.warning("Couldn't convert outcome to binary for the following case: " +
                            f"DAB ID: {self.dab_id}\nOutcome text:\n{self.dab_outcome}")

        self.__extract_alj_id()
        if not self.alj_id:
            logging.warning("Couldn't find ALJ ID for the following case: DAB ID: " +
                        f"{self.dab_id}")
            return None

        self.alj_url = alj_catalog.get(self.alj_id, None)
        if not self.alj_url:
            logging.warning("Couldn't find ALJ URL for the following case: DAB ID: " +
                        f"{self.dab_id}, ALJ ID: {self.alj_id}")
            return None

        self.alj_text = scrape_decision_text(self.alj_url)
        if not self.alj_text:
            logging.warning("Couldn't scrape ALJ text for the following case: DAB ID: " +
                            f"{self.dab_id}, ALJ ID: {self.alj_id}" +
                            f"ALJ URL: {urlunparse(self.alj_url)}")
            return None

    def __init_all_vars(self):
        '''
        Initialize all expected attributes in a Appeal object to None.

        Updates: self.dab_url, self.dab_id, self.dab_text, self.dab_outcome,
                 self.dab_outcome_binary, self.alj_id, self.alj_url, self.alj_text
        '''
        self.dab_url = None
        self.dab_id = None
        self.dab_text = None
        self.dab_outcome = None
        self.dab_outcome_binary = None
        self.alj_id = None
        self.alj_url = None
        self.alj_text = None

    def __extract_dab_id(self, case_info_str):
        '''
        Extract the DAB case number from a DAB case info string generated by the
        get_dab_decisions function.

        Updates: self.dab_id, if a DAB case number is successfully found
        '''
        dab_str = re.search(r'(?<=[dab|dab-|dab ])\d*\d', case_info_str.lower())
        if dab_str:
            dab_id = dab_str.group()

        if not dab_str or len(dab_id) != 4:
            dab_str = re.search(r'(?<=a-)\d.*(?=;)', case_info_str.lower())

        if dab_str:
            self.dab_id = ''.join(re.findall(r'\d+', dab_str.group()))


    def __extract_dab_outcome(self):
        '''
        Extract the text outcome section from a DAB decision.

        Updates: self.dab_outcome, if a text outcome is successfully found
        '''
        decision_format = get_decision_format(self.dab_url)
        if decision_format == PDF:
            conclusion = re.findall("(?<=Conclusion).*?\.", self.dab_text)
            if conclusion:
                self.dab_outcome = conclusion[-1]
        elif decision_format == OLD_HTML:
            conclusion = re.findall("(?<=Conclusion).*?\.", self.dab_text)
            if conclusion:
                self.dab_outcome = conclusion[-1]
        else:
            soup = make_soup(urlunparse(self.dab_url)) # I don't love hitting the website again here but...
            if not soup:
                logging.warning("Couldn't download the following DAB URL to extract" +
                                f"outcome {urlunparse(self.dab_url)}")
                return None

            conclusion = soup.find("div", {"class": "legal-decision-judge"})\
                             .find_previous()\
                             .getText()
            if conclusion:
                self.dab_outcome = conclusion

    def __convert_dab_outcome(self):
        '''
        Converts the outcome text from a DAB case to a binary outcome variable (0 if the DAB
        affirms the ALJ's ruling, 1 if the DAB overturns the ALJ's ruling).

        Updates: self.dab_outcome_binary, if the outcome text is successfully converted
        '''
        overturned = re.search(r'(vacate)', self.dab_outcome)
        if overturned:
            self.dab_outcome_binary = 1
        affirmed = re.search(r'(affirm)|(uphold)|(sustain)', self.dab_outcome)
        if affirmed:
            self.dab_outcome_binary = 0

    def __extract_alj_id(self):
        '''
        Find the case number of the ALJ case being appealed in a DAB decision.

        Updates: self.alj_id, if an ALJ case is successfully found
        '''
        pattern = r'[CT][RB]\D{0,2}(\d\W{0,1}){1,4}(?=[\W_])'
        match = re.search(pattern, self.dab_text)
        if match:
            whitespace = re.compile('[^A-Za-z0-9]')
            self.alj_id = whitespace.sub('', match.group(0))

    def to_tuple(self):
        '''
        Return an Appeal object as a named tuple.

        Returns: AppealRecord (a named tuple)
        '''
        dab_url = urlunparse(self.dab_url) if self.dab_url is not None else None
        alj_url = urlunparse(self.alj_url) if self.alj_url is not None else None
        return AppealRecord(self.dab_id, self.alj_id, self.dab_text, self.alj_text,
                            dab_url, alj_url, self.dab_outcome_binary)

    def to_postgres(self, cur, table):
        '''
        Write the appeal to a PostgreSQL database. Does not commit the change.

        Inputs:
        cur (psycopg2 cursor): cursors to execute the insert with
        table (str): table name to insert the appeal to (must be in TBL_WHITELIST)
        '''
        if table in TBL_WHITELIST:
            insert_statement = f"""
                                INSERT INTO {table}
                                VALUES (%s, %s, %s, %s, %s, %s, %s);
                                """
            cur.execute(insert_statement, self.to_tuple())

    def __repr__(self):
        '''
        Generate a string representation of an Appeal object.

        Returns: string
        '''
        return f'DAB ID: {self.dab_id}, DAB URL: {urlunparse(self.dab_url) if self.dab_url is not None else None}\n' +\
               f'DAB text: {self.dab_text[:100] if self.dab_text is not None else None}...\n' +\
               f'Binary outcome: {"Overturned" if self.dab_outcome_binary else "Upheld"}' +\
               f'\nOutcome text: {self.dab_outcome}\n' +\
               f'ALJ ID: {self.alj_id}, ' +\
               f'ALJ URL: {urlunparse(self.alj_url) if self.alj_url is not None else None}' +\
               f'\nALJ text: {self.alj_text[:100] if self.alj_text is not None else None}...'

def get_decision_format(url):
    '''
    Detect wheter a decision is in PDF, OLD_HTML, or NEW_HTML format.

    Inputs:
    url (urllib.parse.ParseResult): the decision URL

    Returns: int
    '''
    if url.path.endswith('.pdf'):
        return PDF
    elif OLD_URL_PATTERN in url.path:
        return OLD_HTML
    else:
        return NEW_HTML

def clean_text(raw_text):
    '''
    Takes string and clean out the \n characters.

    Inputs:
    raw_text (str): text to clean

    Returns: string
    '''
    return raw_text.replace("\n", ' ')

def make_soup(url):
    '''
    Obtains a bs4.BeautifulSoup object from a given URL.

    Inputs:
    url (str): the URL to get

    Returns: bs4.BeautifulSoup
    '''
    try:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        return soup
    except Exception as e:
        logging.warning(f"Couldn't access the following URL: {url}\nError message: {e}")
        return None


def scrape_decision_text(url):
    '''
    Obtain the text of a DAB or ALJ decision from the decision's URL.

    Inputs:
    url (str): the URL to get

    Returns: string
    '''
    url_type = get_decision_format(url)
    url = urlunparse(url)
    if url_type == PDF:
        try:
            response = requests.get(url)
        except Exception as e:
            logging.warning(f"Couldn't access the following URL: {url}\nError message: {e}")
            return None
        pdfReader = PyPDF2.PdfFileReader(io.BytesIO(response.content))
        raw_text = ''
        for page in pdfReader.pages:
            pg_text = page.extractText()
            raw_text += pg_text
        if not raw_text:
            return None
    elif url_type == OLD_HTML:
        soup = make_soup(url)
        if not soup:
            return None
        tables = soup.find_all("td", {"colspan": "2"})[1:]
        raw_text = ''
        for td in tables:
            paragraphs = td.find_all('p')
            for paragraph in paragraphs:
                raw_text += paragraph.getText()
        if not raw_text:
            return None
    else:
        soup = make_soup(url)
        if not soup:
            return None
        text_section = soup.find("div", {"class": "field-name-body"})
        if text_section:
            raw_text = text_section.getText()
        else:
            return None

    return clean_text(raw_text)

def gen_alj_catalog_one_yr(url):
    '''
    Catalog all the ALJ decisions from a single year.

    Inputs:
    url (str): webpage lising all the ALJ decisions from a single year

    Ouptputs: dictionary with ALJ case numbers as keys and URLs as values
    '''
    soup = make_soup(url)
    if not soup:
        logging.warning(f"Couldn't generate ALJ catalog from the following page: {url}")
        return {}
    decision_dict = {}
    syndicate_div = soup.find('div', {'class': 'syndicate'})
    if not syndicate_div:
        logging.warning(f"Couldn't parse ALJ catalog from the following page: {url}")
        return {}
    for case_link in syndicate_div.findAll('a'):
        case_url = urlparse(urljoin(url, case_link['href']))
        case_title = case_link.get_text()
        case_id = re.search(r'[CT][RB][^ ]+', case_title)
        if case_id:
            decision_dict[case_id.group(0)] = case_url
        else:
            logging.warning(f"Couldn't catalog the following ALJ case: URL {case_url} " +
                            f"Title: {case_title}")

    return decision_dict

def gen_alj_catalog(start_url):
    '''
    Catalog all ALJ decisions.

    Inputs:
    start_url (str): root webpage to find all the ALJ decisions

    Returns: dictionary with ALJ case numbers as keys and URLs as values
    '''
    soup = make_soup(start_url)
    if not soup:
        raise Exception("Couldn't access starting page for generating ALJ catalog")
    decision_dict = {}
    syndicate_div = soup.find('div', {'class': 'syndicate'})
    if not syndicate_div:
        raise Exception("Couldn't parse ALJ catalog start page")
    for year_link in syndicate_div.findAll('a'):
        link_url = urljoin(start_url, year_link['href'])
        year_dict = gen_alj_catalog_one_yr(link_url)
        decision_dict = {**decision_dict, **year_dict}

    return decision_dict

def get_dab_decisions(url):
    '''
    Find all DAB decisions.

    Inputs:
    start_url (str): root webpage to find all the DAB decisions

    Returns: dictionary with years (str) as keys and list of tuples like
             (case url (str), case info (str)), where each tuple represents a case with
             that year
    '''
    soup = make_soup(url)
    if not soup:
        raise Exception("Couldn't access starting page for generating DAB catalog")
    yr_lst = []
    syndicate_div = soup.find("div", {"class": "syndicate"})
    if not syndicate_div:
        raise Exception("Couldn't parse DAB catalog start page")
    for yr in syndicate_div.findAll('a'):
        yr_lst.append((yr.getText(), f"https://hhs.gov{yr.get('href')}")) # my eyes bleed so this if for later but maybe should do this with urljoin
    yr_url_dict = {}
    for yrnum, yrurl in yr_lst[:21]: # a thought, why not just grab all and then deal with consequences later?
      yr_url_dict[yrnum] = get_dab_decisions_one_year(yrurl)
    return yr_url_dict


def get_dab_decisions_one_year(url):
    '''
    Find all the DAB decisions from a single year.

    Inputs:
    url (str): webpage lising all the DAB decisions from a single year

    Returns: list of tuples like (case url (str), case info (str))
    '''
    yrsoup = make_soup(url)
    if not yrsoup:
        logging.warning(f"Couldn't generate DAB catalog from the following page: {url}")
        return []
    yrurllst = []
    syndicate_div = yrsoup.find('div', {'class': 'syndicate'})
    if not syndicate_div:
        logging.warning(f"Couldn't parse DAB catalog from the following page: {url}")
        return {}
    for case in syndicate_div.findAll('a'):
      if ('adobe' not in case.get('href')) and ('mailto' not in case.get('href')): # my eyes bleed so this if for later but maybe should do this with urljoin
        yrurllst.append((f"https://hhs.gov{case.get('href')}", case.getText()))
    return yrurllst


## MAIN FLOW
def go(alj_start=ALJ_START_PAGE, dab_start=DAB_START_PAGE, credentials='secrets.ini',
       load_table='raw_data_test', save_failed=None, limit=None):
    '''
    Generate all appeals records and load them to a PostgreSQL database.

    Inputs:
    alj_start (str): root webpage to find all the ALJ decisions
    dab_start (str): root webpage to find all the DAB decisions
    credentials (str): filepath to DB credentials
    load_table (str): name of table to load records to
    save_failed (str or None): filepath to save CSV of appeals thaqt couldn't be uploaded to
                               the DB, if None, failed uploads list is not stored
    limit (int): limit the number of records looked at in a given year
    '''
    config = ConfigParser()
    config.read(credentials)

    alj_catalog = gen_alj_catalog(alj_start)
    dab_decisions = get_dab_decisions(dab_start)
    total_dab_cases = sum(len(val) for key, val in dab_decisions.items())

    conn = psycopg2.connect(**config['Advanced ML Database'])
    cur = conn.cursor()
    if save_failed:
        failed_csv = open(save_failed, 'w')
        failed_writer = csv.writer(failed_csv)
        failed_writer.writerow(APPEAL_RECORD_FIELDS + ['reason'])
    cases_uploaded = 0
    cases_failed = 0
    print('-' * 10 + '\nStarting case processing.\n' + '-' * 10)
    for year in dab_decisions.keys():
        cases = dab_decisions[year]
        if limit:
            cases = cases[:limit]
        for url, case_info in cases:
            appeal = Appeal(url, case_info, alj_catalog)
            try:
                appeal.to_postgres(cur, load_table)
                conn.commit() # this is the error
                cases_uploaded += 1
            except Exception as e:
                conn.rollback()
                logging.warning(f'Failed to upload the following record:' +
                                f'DAB ID: {appeal.dab_id}, ALJ ID: {appeal.alj_id}\n' +
                                f'Error message: {e}')
                cases_failed += 1
                if save_failed:
                    record = appeal.to_tuple() + (e,)
                    failed_writer.writerow(record)
            cases_processed = cases_uploaded + cases_failed
            if not cases_processed % 50:
                print(f'{cases_processed}/{total_dab_cases} cases processed ' +
                    f'({cases_uploaded} successful, {cases_failed} failed)')
    print('-' * 10 + '\nCase processing complete.\n' + '-' * 10)
    if save_failed:
        failed_csv.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scrape DAB and associated ALJ cases.")
    parser.add_argument('--alj', dest='alj_start', default=ALJ_START_PAGE, type=str,
                        help='root webpage to find all the ALJ decisions')
    parser.add_argument('--dab', dest='dab_start', default=DAB_START_PAGE, type=str,
                        help='root webpage to find all the DAB decisions')
    parser.add_argument('--creds', dest='credentials', default='secrets.ini', type=str,
                        help='filepath to DB credentials')
    parser.add_argument('--table', dest='load_table', default='raw_data_test', type=str,
                        help='table to load appeals data to')
    parser.add_argument('--failedcsv', dest='save_failed', default=None, type=str,
                        help='filepath to save CSV of failed uploads')
    parser.add_argument('--limit', dest='limit', default=None, type=int,
                        help='limit the number of records looked at in a given year')
    args = parser.parse_args()
    go(**args.__dict__)
