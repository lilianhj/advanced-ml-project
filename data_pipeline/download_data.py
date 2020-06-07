'''
Functions to scrape and parse HHS DAB appeals and corresponding original case information,
then load these appeals to a PostgreSQL database.

April 2020  
'''

import argparse
from collections import namedtuple
from configparser import ConfigParser
import csv
import io
import logging
import os
import re
from urllib.parse import urlparse, urljoin, urlunparse

from bs4 import BeautifulSoup
import psycopg2 as pg
import psycopg2.sql as sql
import requests
import textract


ALJ_START_PAGE = 'https://www.hhs.gov/about/agencies/dab/decisions/alj-decisions/' +\
                'alj-decisions-by-year/index.html'
DAB_START_PAGE = 'https://www.hhs.gov/about/agencies/dab/decisions/board-decisions/' +\
                 'board-decisions-by-year/index.html'
TEMP_FILE_LOC = os.path.join(os.path.dirname(__file__), 'temporary.pdf')
OLD_URL_PATTERN = r'files/static/dab/decisions/'

PDF = 0
OLD_HTML = 1
NEW_HTML = 2

TBL_WHITELIST = ['raw_data', 'raw_data_2', 'raw_data_exp', 'raw_data_new']

logging.basicConfig(filename='scraper.log', level=logging.DEBUG)

APPEAL_RECORD_FIELDS = ['dab_id', 'alj_id', 'dab_text', 'alj_text', 'dab_url', 'alj_url',
                        'outcome_text', 'overturned', 'dab_year', 'alj_year']
AppealRecord = namedtuple("AppealRecord", APPEAL_RECORD_FIELDS)

class Appeal:
    '''
    A DAB appeal, containing informaiton both on the DAB cases and, if applicable, the
    associated ALJ decision
    '''

    def __init__(self, dab_year, dab_url, dab_case_info, alj_catalog):
        '''
        Generate a new Appeal object. Likely use case is to only call with dab_id.

        Inputs:
        dab_url (str): url of the DAB case for this appeal
        dab_case_info (str): DAB case info string
        alj_catalog (dict): dictionary with ALJ case numbers as keys and the associated
                            case's URL as value (generated by get_alj_catalog function)
        '''
        self.__init_all_vars()
        self.dab_year = dab_year
        self.dab_url = urlparse(dab_url)

        self.__extract_dab_id(dab_case_info)
        if self.dab_id is None:
            logging.warning(f'The following appears not to be a DAB case: {dab_case_info} ' +
                            f'\n{dab_url}')
            return None

        self.dab_text, self.dab_soup = scrape_decision_text(self.dab_url)
        if self.dab_text is None:
            logging.warning("Couldn't extract DAB text for the following case: DAB ID " +
                            f"{self.dab_id}, DAB URL: {self.dab_url}")
            return None

        self.__extract_dab_outcome()
        if self.dab_outcome is None:
            logging.warning("Couldn't extract outcome for the following case:" +
                            f"DAB ID: {self.dab_id}")
        else:
            self.__convert_dab_outcome()
            if self.dab_outcome_binary is None:
                logging.warning("Couldn't convert outcome to binary for the following case: " +
                                f"DAB ID: {self.dab_id}\nOutcome text:\n{self.dab_outcome}")

        self.__extract_alj_id()
        if self.alj_id is None:
            logging.warning("Couldn't find ALJ ID for the following case: DAB ID: " +
                            f"{self.dab_id}")
            return None

        self.alj_url = alj_catalog.get(self.alj_id, None)
        if self.alj_url is None:
            logging.warning("Couldn't find ALJ URL for the following case: DAB ID: " +
                            f"{self.dab_id}, ALJ ID: {self.alj_id}")
            return None
        else:
            alj_year = re.search(r'/(\d\d\d\d)/', self.alj_url.path)
            if alj_year:
                self.alj_year = alj_year.group(1)

        self.alj_text, _ = scrape_decision_text(self.alj_url)
        if self.alj_text is None:
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
        self.dab_soup = None
        self.dab_outcome = None
        self.dab_outcome_binary = None
        self.alj_id = None
        self.alj_url = None
        self.alj_text = None
        self.dab_year = None
        self.alj_year = None

    def __extract_dab_id(self, case_info_str):
        '''
        Extract the DAB case number from a DAB case info string generated by the
        get_dab_decisions function.

        Updates: self.dab_id, if a DAB case number is successfully found
        '''
        dab_str = re.search(r'(((DAB)|(RUL)|(ER))[- ]{0,1}[\d-]*\d)|(A-[\d-]*\d)', case_info_str)
        if dab_str:
            self.dab_id = dab_str.group(0)

    def __extract_dab_outcome(self):
        '''
        Extract the text outcome section from a DAB decision.

        Updates: self.dab_outcome, if a text outcome is successfully found
        '''
        decision_format = get_decision_format(self.dab_url) 
        if decision_format == PDF or decision_format == OLD_HTML:
            outcome_match = r'(?<=Conclusion|CONCLUSION)(.*?)(\/s\/|JUDGE|__|$)'
            conclusion = re.findall(outcome_match, self.dab_text)
            text_outcome = None
            while conclusion: # handle overlapping matches
                text_outcome = conclusion[-1]
                conclusion = re.findall(outcome_match, ''.join(text_outcome))
            if text_outcome is not None:
                self.dab_outcome = text_outcome[0]            
            else:
                logging.warning("Unable to extract the outcome for the folloowing DAB " +
                                f"URL: {urlunparse(self.dab_url)}\n Defaulting to the " +
                                "first and last 1000 character.")
                self.dab_outcome = self.dab_text[:1000] + ' ' + self.dab_text[-1000:]
        else:
            if not self.dab_soup:
                logging.warning("Soup is unavailable to extract the outcome for the " +
                                f"following DAB URL: {urlunparse(self.dab_url)}\n")
                return None

            conclusion = self.dab_soup.find("div", {"class": "legal-decision-judge"})\
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
        overturned_kwrd = r'(v\W{0,1}a\W{0,1}c\W{0,1}a\W{0,1}t\W{0,1}e)|' +\
                          r'(r\W{0,1}e\W{0,1}v\W{0,1}e\W{0,1}r\W{0,1}s\W{0,1}e)|' +\
                          r'(r\W{0,1}e\W{0,1}m\W{0,1}a\W{0,1}n\W{0,1}d)|' +\
                          r'(e\W{0,1}r\W{0,1}r\W{0,1}e\W{0,1}d)|' +\
                          r'(m\W{0,1}o\W{0,1}d\W{0,1}i\W{0,1}f\W{0,1}y)|' +\
                          r'(m\W{0,1}o\W{0,1}d\W{0,1}i\W{0,1}f\W{0,1}i\W{0,1}e\W{0,1}s)|' +\
                          r'(g\W{0,1}r\W{0,1}a\W{0,1}n\W{0,1}t\W{0,1}e\W{0,1}d)' +\
                          r'(o\W{0,1}v\W{0,1}e\W{0,1}r\W{0,1}t\W{0,1}u\W{0,1}r\W{0,1}n)'

        overturned = re.search(overturned_kwrd, self.dab_outcome, re.I)
        affirmed_kwrds = r'(a\W{0,1}f\W{0,1}f\W{0,1}i\W{0,1}r\W{0,1}m)|' +\
                         r'(u\W{0,1}p\W{0,1}h\W{0,1}o\W{0,1}l\W{0,1}d)|' +\
                         r'(u\W{0,1}p\W{0,1}h\W{0,1}e\W{0,1}l\W{0,1}d)|' +\
                         r'(s\W{0,1}u\W{0,1}s\W{0,1}t\W{0,1}a\W{0,1}i\W{0,1}n)|' +\
                         r'(d\W{0,1}e\W{0,1}n\W{0,1}y)|' +\
                         r'(d\W{0,1}e\W{0,1}n\W{0,1}i\W{0,1}e\W{0,1}s)|' +\
                         r'(d\W{0,1}e\W{0,1}n\W{0,1}i\W{0,1}e\W{0,1}d)|' +\
                         r'(c\W{0,1}o\W{0,1}r\W{0,1}r\W{0,1}e\W{0,1}c\W{0,1}t\W{0,1}l\W{0,1}y)|' +\
                         r'(l\W{0,1}e\W{0,1}g\W{0,1}a\W{0,1}l\W{0,1}l\W{0,1}y\W{0,2}s\W{0,1}o\W{0,1}u\W{0,1}n\W{0,1}d)|' +\
                         r'(f\W{0,1}r\W{0,1}e\W{0,1}e\W{0,2}f\W{0,1}r\W{0,1}o\W{0,1}m\W{0,2}l\W{0,1}e\W{0,1}g\W{0,1}a\W{0,1}l\W{0,2}e\W{0,1}r\W{0,1}r\W{0,1}o\W{0,1}r)|' +\
                         r'(d\W{0,1}e\W{0,1}c\W{0,1}l\W{0,1}i\W{0,1}n)|' +\
                         r'(d\W{0,1}i\W{0,1}d\W{0,2}n\W{0,1}o\W{0,1}t\W{0,2}e\W{0,1}r\W{0,1}r)|' +\
                         r'(n\W{0,1}o\W{0,2}n\W{0,1}e\W{0,1}e\W{0,1}d\W{0,2}t\W{0,1}o)|' +\
                         r'(n\W{0,1}o\W{0,2}g\W{0,1}e\W{0,1}n\W{0,1}u\W{0,1}i\W{0,1}n\W{0,1}e\W{0,2}d\W{0,1}i\W{0,1}s\W{0,1}p\W{0,1}u\W{0,1}t\W{0,1}e)|' +\
                         r'(n\W{0,1}o\W{0,2}d\W{0,1}i\W{0,1}s\W{0,1}p\W{0,1}u\W{0,1}t\W{0,1}e)|' +\
                         r'(n\W{0,1}o\W{0,1}\W{0,2}e\W{0,1}r\W{0,1}r\W{0,2}o\W{0,1}r)|' +\
                         r'(r\W{0,1}e\W{0,1}j\W{0,1}e\W{0,1}c\W{0,1}t\W{0,1}e\W{0,1}d)|' +\
                         r'(n\W{0,1}o\W{0,2}n\W{0,1}e\W{0,1}w\W{0,2}e\W{0,1}v\W{0,1}i\W{0,1}d\W{0,1}e\W{0,1}n\W{0,1}c\W{0,1}e)|' +\
                         r'(n\W{0,1}o\W{0,2}a\W{0,1}d\W{0,1}j\W{0,2}u\W{0,1}s\W{0,1}t\W{0,1}m\W{0,1}e\W{0,1}n\W{0,1}t)|' +\
                         r'(p\W{0,1}r\W{0,1}o\W{0,1}p\W{0,1}e\W{0,1}r)'

        affirmed = re.search(affirmed_kwrds, self.dab_outcome, re.I)
        if affirmed and not overturned:
            self.dab_outcome_binary = 0
        if overturned: # overturned or partially overturned
            self.dab_outcome_binary = 1

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
                            dab_url, alj_url, self.dab_outcome, self.dab_outcome_binary, self.dab_year, self.alj_year)

    def to_postgres(self, cur, table):
        '''
        Write the appeal to a PostgreSQL database. Does not commit the change.

        Inputs:
        cur (psycopg2 cursor): cursors to execute the insert with
        table (str): table name to insert the appeal to (must be in TBL_WHITELIST)
        '''
        if table in TBL_WHITELIST: # check table name safety
            insert_statement = sql.SQL("""
                                        INSERT INTO {}
                                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                                                %s, %s);
                                """).format(sql.Identifier(table))
            cur.execute(insert_statement, self.to_tuple())
        else:
            raise Exception('Specified upload table not whitelisted.')

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
    if url.path.lower().endswith('.pdf'):
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
    return re.sub(r'(\n)|(\d{0,3}\x0c\d{0,3})|(Page \d{0,3})', ' ', raw_text)

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
    url (urllib.parse.ParseResult): the URL to get

    Returns: tuple of string, bs4.BeautifulSoup
    '''
    url_type = get_decision_format(url)
    url = urlunparse(url)
    soup = None # ensures return_soup doesn't throw an error for PDFs
    if url_type == PDF:
        try:
            response = requests.get(url)
        except Exception as e:
            logging.warning(f"Couldn't access the following URL: {url}\nError message: {e}")
            return (None, None)
        try:
            with open(TEMP_FILE_LOC, 'wb') as pdf_f:
                pdf_f.write(response.content)
            raw_text = textract.process(TEMP_FILE_LOC).decode('utf-8')
            os.remove(TEMP_FILE_LOC)
        except Exception as e:
            logging.warning(f"Couldn't read this pdf: {url}\nError message: {e}")
            return (None, None)
    elif url_type == OLD_HTML:
        soup = make_soup(url)
        if not soup:
            return (None, None)
        tables = soup.find_all("td")[1:]
        raw_text = ''
        if tables:
            for td in tables:
                paragraphs = td.find_all('p')
                paragraphs += td.find_all('a', {'name': 'JUDGE'}) # helpful for getting conclusion
                for paragraph in paragraphs:
                    raw_text += paragraph.getText()
        else:
            paras = soup.find("body").find_all("p")
            for para in paras:
                raw_text += para.getText()
    else:
        soup = make_soup(url)
        if not soup:
            return (None, None)
        raw_text = None # allows simpler check for raw_text
        text_section = soup.find("div", {"class": "field-name-body"})
        if text_section:
            raw_text = text_section.getText()

    if raw_text is None:
        return raw_text, soup
    else:
        return clean_text(raw_text), soup

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

def get_dab_decisions(url, year_lb=float('-inf'), year_ub=float('inf')):
    '''
    Find all DAB decisions.

    Inputs:
    start_url (str): root webpage to find all the DAB decisions
    year_lb (numeric): first year to grab DAB decisions from
    year_ub (numeric): last year to grab DAB decisions from

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
        yr_lst.append((yr.getText(), urljoin('https://hhs.gov', yr.get('href'))))
    yr_url_dict = {}
    for yrnum, yrurl in yr_lst:
        if year_lb <= int(yrnum) <= year_ub:
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
        if ('adobe' not in case.get('href')) and ('mailto' not in case.get('href')):
            yrurllst.append((urljoin('https://www.hhs.gov', case.get('href')),
                             case.getText()))
    return yrurllst


## MAIN FLOW
def go(alj_start=ALJ_START_PAGE, dab_start=DAB_START_PAGE, credentials='secrets.ini',
       load_table='raw_data', save_failed=None, limit_recs=None,
       year_lb=float('-inf'), year_ub=float('inf')):
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
    year_lb (numeric): first year to grab DAB decisions from
    year_ub (numeric): last year to grab DAB decisions from
    '''
    config = ConfigParser()
    config.read(credentials)

    alj_catalog = gen_alj_catalog(alj_start)
    dab_decisions = get_dab_decisions(dab_start, year_lb, year_ub)
    total_dab_cases = sum(len(val) for key, val in dab_decisions.items())

    conn = pg.connect(**config['Advanced ML Database'])
    cur = conn.cursor()
    if save_failed:
        failed_csv = open(save_failed, 'w')
        failed_writer = csv.writer(failed_csv)
        failed_writer.writerow(APPEAL_RECORD_FIELDS + ['reason'])
    cases_uploaded = 0
    cases_failed = 0
    print('-' * 10 + '\nStarting case processing.\n' + '-' * 10)
    for year, cases in dab_decisions.items():
        if limit_recs:
            cases = cases[:limit_recs]
        for url, case_info in cases:
            appeal = Appeal(year, url, case_info, alj_catalog)
            try:
                appeal.to_postgres(cur, load_table)
                conn.commit()
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
                      f'({cases_uploaded} successfully uploaded, ' +
                      f'{cases_failed} failed)')
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
    parser.add_argument('--table', dest='load_table', default='raw_data', type=str,
                        help='table to load appeals data to (note: table must be whitelisted)')
    parser.add_argument('--failedcsv', dest='save_failed', default=None, type=str,
                        help='filepath to save CSV of failed uploads')
    parser.add_argument('--limitrecs', dest='limit_recs', default=None, type=int,
                        help='limit the number of records looked at in a given year (debugging setting)')
    parser.add_argument('--yearlb', dest='year_lb', default=float('-inf'), type=float,
                        help='last year to grab DAB decisions from')
    parser.add_argument('--yearub', dest='year_ub', default=float('inf'), type=float,
                        help='last year to grab DAB decisions from')
    args = parser.parse_args()
    go(**args.__dict__)