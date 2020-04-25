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
    case_match = re.search(r"[A-Z][A-Z](\s)?\d\d\d\d \(\d\d\d\d\)", txt)
    if case_match:
        caseyr = re.search(r"(?<=\()\d\d\d\d", case_match.group()).group()
    i = 0
    if not case_match:
        case_match = re.search(r"DAB [A-Z][A-Z]\d\d\d\d", txt)
        case_yr_match = re.search(r"the \w+ \d(\d)?, \d\d\d\d", txt)
        caseyr = case_yr_match.group().split(', ')[1]
        i = 1
    print("case matched:", case_match.group())
    if case_match.group().count(' ') == 1:
        orig = case_match.group().split()
        casenum = orig[i].lower()
    else:
        orig = case_match.group().split()
        casenum = orig[0].lower() + orig[1]
    print(f"casenum {casenum} caseyear {caseyr}")
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


def get_html_info(appeal_url, reject_lst, problem_lst):
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
    all_appeal_text, soup = initialize_get_full_text_html(appeal_url)
    
    orig_info = soup.find("div", {"class": "field-item even"}).find('p').getText()
    alj_check = re.search("ALJ", all_appeal_text)
    if not alj_check:
        print("not an ALJ thing")
        reject_lst.append(appeal_url)
        # need to exit function but ughhh checking for a None return is so annoying
    try:
        casenum, caseyr = get_orig_case_info(orig_info)
        outcome = soup.find("div", {"class": "legal-decision-judge"}).find_previous().getText()
        # orig_url = f'https://www.hhs.gov/about/agencies/dab/decisions/alj-decisions/{caseyr}/alj-{casenum}/index.html'
        all_orig_text, orig_url = get_original_text(casenum, caseyr, reject_lst)
        return outcome, all_appeal_text, all_orig_text, casenum, orig_url, appeal_url
    except:
        # these babies got problems to be fixed later
        print('this has problems: not in df', appeal_url)
        print("adding to problem list")
        problem_lst.append(appeal_url)
        pass


def initialize_get_full_text_html(url):
    '''

    Input:

    Output:

    '''
    soup = make_soup(url)
    all_text = clean_text(soup.find("div", {"class": "field-name-body"}).getText())
    return all_text, soup


def get_pdf_info(appeal_url, reject_lst):
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
    alj_check = re.search("ALJ", all_appeal_text)
    if not alj_check:
        print("not an ALJ thing")
        reject_lst.append(appeal_url)
        return
    #strict_dab_check = re.search("^(?!DAB No.)DAB.*?\d\d\d\d", all_appeal_text)
    strict_dab_check = [x for x in re.findall("DAB.*?\d\)", all_appeal_text) if "No." not in x]
    dab_no_check = re.search("DAB No.*?\d\)", all_appeal_text)
    if dab_no_check and not strict_dab_check:
        print("no original decision")
        reject_lst.append(appeal_url)
        return
    casenum, caseyr = get_orig_case_info(all_appeal_text)
    conclusion = re.search("(?<=Conclusion ).*?\.", all_appeal_text)
    if conclusion:
        outcome = conclusion.group()
    else:
        outcome = 'by hand'
    # orig_url = f'https://www.hhs.gov/sites/default/files/alj-{casenum}.pdf'
    all_orig_text, orig_url = get_original_text(casenum, caseyr, reject_lst)
    return outcome, all_appeal_text, all_orig_text, casenum, orig_url, appeal_url


def get_original_text(casenum, caseyr, reject_lst):
    '''

    Inputs:

    Output:
    '''
    try:
        orig_url = f'https://www.hhs.gov/sites/default/files/alj-{casenum}.pdf'
        #print(orig_url)
        all_orig_text = get_pdf_txt(orig_url)
    except:
        print("pdf doesn't exist")
        try:
            orig_url = f'https://www.hhs.gov/sites/default/files/static/dab/decisions/alj-decisions/{caseyr}/{casenum}.pdf'
            #print(orig_url)
            all_orig_text = get_pdf_txt(orig_url) 
        except:
            print("pdf still doesn't exist")
            try:
                orig_url = f'https://www.hhs.gov/sites/default/files/static/dab/decisions/alj-decisions/{caseyr}/{casenum.upper()}.pdf'
                #print(orig_url)
                all_orig_text = get_pdf_txt(orig_url) 
            except:
                print("wow this pdf")
                try:
                    mod_casenum = casenum[2:]
                    orig_url = f'https://www.hhs.gov/sites/default/files/static/dab/decisions/alj-decisions/{caseyr}/cr{mod_casenum}.pdf'
                    #print(orig_url)
                    all_orig_text = get_pdf_txt(orig_url)
                except:
                    print("holy shit")
                    try:
                        mod_casenum = casenum[2:]
                        orig_url = f'https://www.hhs.gov/sites/default/files/alj-dab{mod_casenum}.pdf'
                        #print(orig_url)
                        all_orig_text = get_pdf_txt(orig_url)
                    except:
                        print("pdf really doesn't exist")
                        orig_url = f'https://www.hhs.gov/about/agencies/dab/decisions/alj-decisions/{caseyr}/alj-{casenum}/index.html'
                        #print(orig_url)
                        all_orig_text, _ = initialize_get_full_text_html(orig_url)
                        if all_orig_text.startswith("We're sorry"):
                            print("wow now the HTML is acting up too")
                            orig_url = f'https://www.hhs.gov/sites/default/files/static/dab/decisions/alj-decisions/{caseyr}/{casenum.upper()}.htm'
                            print("have to figure out how to scrape from the old HTM pages")
                            pass
    print("preview:", all_orig_text[:50])
    print("original decision URL:", orig_url)
    return all_orig_text, orig_url


def combining(initial_url=STARTING_PG):
    '''

    Inputs:

    Outputs:
        a test df as a proxy for what would be inputted into the db


    '''
    full_urls = get_urls_all_years(initial_url)

    #initial df -- just for testing then remove
    test_df = pd.DataFrame(columns=['dab_id', 'alj_id', 'dab_text', 'alj_text',
                                    'dab_url', 'alj_url', 'decision'])
    problem_lst = []
    reject_lst = []
    for year in full_urls.keys():
        # indexing list below for testing
        test_cases = full_urls[year][:3]
        for case in test_cases:
            dab_url = case[0]
            dab_id = re.search(r'(?<=[dab|dab-])\d*\d', dab_url.lower()).group()
            print("appeal id:", dab_url)
            if dab_url[-4:] in ['html', '.htm']:
                try:
                    outcome, dab_text, alj_text, alj_id, alj_url, \
                    dab_url = get_html_info(dab_url, reject_lst, problem_lst)
                    if alj_text.startswith("We're sorry"):
                        print("this is a problem, adding to list")
                        problem_lst.append(dab_url)
                    row = [dab_id, alj_id, dab_text, alj_text, dab_url, alj_url, outcome]
                    print("adding HTML appeal to df")
                    test_df.loc[len(test_df)] = row
                except:
                    pass
            else:
                if get_pdf_info(dab_url, reject_lst): #this is what causes that weird double loop, because checking that it's not a None being returned
                    outcome, dab_text, alj_text, alj_id, alj_url, \
                    dab_url = get_pdf_info(dab_url, reject_lst)
                    if alj_text.startswith("We're sorry"):
                        print("this is a problem, adding to list")
                        problem_lst.append(dab_url)
                    row = [dab_id, alj_id, dab_text, alj_text, dab_url, alj_url, outcome]
                    print("adding pdf appeal to df")
                    test_df.loc[len(test_df)] = row
    return test_df, problem_lst, reject_lst



    '''
    Webpages from test above with problems

['https://hhs.gov/about/agencies/dab/decisions/board-decisions/2020/board-dab-2987/index.html',
 'https://hhs.gov/about/agencies/dab/decisions/board-decisions/2019/board-dab-2982/index.html',
 'https://hhs.gov/about/agencies/dab/decisions/board-decisions/2019/board-dab-2981/index.html',
 'https://hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2005/dab2007.htm',
 'https://hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2005/dab2006.htm',
 'https://hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2004/dab1956.htm',
 'https://hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2004/dab1955.htm',
 'https://hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2004/dab1954.htm',
 'https://hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2003/dab1903.html',
 'https://hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2003/dab1902.html',
 'https://hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2003/dab1901.html',
 'https://hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2003/dab1861.html',
 'https://hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2002/dab1860.html',
 'https://hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2002/dab1859.html',
 'https://hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2001/dab1805.html',
 'https://hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2001/dab1804.html',
 'https://hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2001/dab1803.html',
 'https://hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2000/dab1758.html',
 'https://hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2000/dab1757.html',
 'https://hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2000/dab1756.html']
    '''
