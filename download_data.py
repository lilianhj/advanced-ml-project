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
import PyPDF2
import io
import requests
from bs4 import BeautifulSoup
from urllib.error import HTTPError
# for testing
import pandas as pd
import logging

STARTING_PG = 'https://www.hhs.gov/about/agencies/dab/decisions/' + \
              'board-decisions/board-decisions-by-year/index.html'

OLD_URL_PATTERN = r'files/static/dab/decisions/.*/\d\d\d\d.*htm'
logging.basicConfig(filename='scraper.log', level=logging.DEBUG)

def get_orig_case_info(txt, old=False):
    '''
    Input: the text of the APPEAL decision, as a string.

    Returns: the case number and case year of the ORIGINAL ALJ decision.
             Use this to construct the URL for the original ALJ decision.
    '''
    if old:
        casenum = re.search('CR\d\d\d\d', old).group()  
        caseyr = re.search('(?<=, )\d\d\d\d', old).group()
        return casenum, caseyr


    case_match = re.search(r"[A-Z][A-Z](\s)?\d\w\d(\d)?(\s)?\((\d\d\d\d|\w+ \d(\d)?, \d\d\d\d)\)", txt)
    i = 0
    if case_match:
        caseyr = re.search(r"(?<=\()(.*)?\d\d\d\d", case_match.group()).group()
        if ',' in caseyr:
            caseyr = caseyr[-4:]
            i = 3
    if not case_match:
        case_match = re.search(r"DAB [A-Z][A-Z]\d\d\d(\d)?", txt)
        case_yr_match = re.search(r"(the|a|an) \w+ \d(\d)?, \d\d\d\d", txt)
        if case_yr_match:
            caseyr = case_yr_match.group().split(', ')[1]
        i = 1
    if not case_match:
        case_match = re.search(r"DAB No. CR\d\d\d(\d)?", txt)
        caseyr = re.search(r"(?<=\()\d\d\d\d", case_match.group()).group()
        i = 2
    print("case matched:", case_match.group())
    if case_match.group().count(' ') == 0:
        orig = case_match.group().split('(')
        casenum = orig[0].lower()
    else:
        if case_match.group().count(' ') == 1 or (i == 2):
            orig = case_match.group().split()
            casenum = orig[i].lower()
        else:
            orig = case_match.group().split()
            casenum = orig[0].lower() + orig[1]
            if i == 3:
                orig = case_match.group().split()
                casenum = orig[0].lower()
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
    Converts the decision text from an appeals case to a binary outcome variable.
    - 0 if appelate court affirms lower court's ruling
    - 1 if appelate court overturns overturns lower court's ruling

    Inputs:
    decision_txt (str): the sentence containg the conclusion of the appelate court's
        findings
    
    Returns: int
    '''
    outcome = None
    overturn_lst = ['vacate']
    affirm_lst = ['affirm', 'uphold', 'sustain']
    overturned = re.search(r'(vacate)', decision_txt)
    if overturned:
        outcome = 1
    else:
         # nesting this to reduce # of re searches needed
        affirmed = re.search(r'(affirm)|(uphold)|(sustain)', decision_txt)
        if affirmed:
            outcome = 0
        else:
            logging.info("Couldn't find outcome in the following decision text:\n",
                         decision_txt)

    return outcome


def get_html_info(appeal_url, reject_lst, problem_lst):
    '''
    Takes the URL of an appeal decision (which is a HTML page),
    gets the appeal outcome,
    goes back to the original ALJ decision,
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
        all_orig_text, orig_url = get_original_text_revamp(casenum, caseyr)
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


def get_html_info_old_format(appeal_url):
    '''
    Input:

    Output:
    '''
    # https://hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2005/dab2007.htm
    # https://hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2003/dab1861.html
    # https://www.hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2001/dab1804.html
    all_appeal_text, soup = get_full_text_old_html(appeal_url)
    alj_check = re.search("ALJ", all_appeal_text)
    if not alj_check:
        print("not an ALJ thing")
    try:
        casenum, caseyr = get_orig_case_info(all_appeal_text, soup.text)
        # need to fine-tune to get the last occurrence of Conclusion in the text. negative lookahead? https://frightanic.com/software-development/regex-match-last-occurrence/
        # think this is fixed w/ the switch to findall
        conclusion = re.findall("(?<=\.Conclusion).*?\.", all_appeal_text)
        if conclusion:
            outcome = conclusion[-1]
        else:
            outcome = 'by hand'
        all_orig_text, orig_url = get_original_text_revamp(casenum, caseyr)
        return outcome, all_appeal_text, all_orig_text, casenum, orig_url, appeal_url
    except Exception as e:
        print(e)
        print("man idk it's probably not an ALJ thing")
        pass


def get_full_text_old_html(url):
    '''
    Input:

    Output:
    '''
    soup = make_soup(url)
    tables = soup.find_all("td", {"colspan": "2"})[1:]
    docstring = ''
    for td in tables:
        paras = td.find_all('p')
        for para in paras:
            docstring += para.getText()
    cleaned_text = clean_text(docstring)
    return cleaned_text, soup


def get_pdf_info(appeal_url, reject_lst):
    '''
    Takes the URL of an appeal decision (which is a pdf),
    gets the appeal outcome,
    goes back to the original ALJ decision,
    and gets the full text of the original ALJ decision.

    Input:
    the URL of the appeal decision (a pdf)

    Returns:
    - the appeal outcome
    - the full text of the appeal decision, as a string
    - the full text of the original ALJ decision, as a string
    '''
    all_appeal_text = get_pdf_txt(appeal_url)
    alj_check = re.search("ALJ", all_appeal_text)
    if not alj_check:
        print("not an ALJ thing")
        reject_lst.append(appeal_url)
        return
    #strict_dab_check = re.search("^(?!DAB No.)DAB.*?\d\d\d\d", all_appeal_text)
    strict_dab_check = [x for x in re.findall("DAB.*?\d\)", all_appeal_text) if "No." not in x]
    if re.search("DAB No. CR\d\d\d(\d)?", all_appeal_text):
        strict_dab_check.append(re.search("DAB No. CR\d\d\d(\d)?", all_appeal_text))
    dab_no_check = re.search("DAB No.*?\d\)", all_appeal_text)
    if dab_no_check and (not strict_dab_check):
        print("no original decision")
        reject_lst.append(appeal_url)
        return
    # regex in next line to have text start at same spot as html text
    all_appeal_text = re.search(r'DECISION.*', all_appeal_text).group()
    casenum, caseyr = get_orig_case_info(all_appeal_text)
    conclusion = re.search("(?<=Conclusion ).*?\.", all_appeal_text)
    if conclusion:
        outcome = conclusion.group()
    else:
        outcome = 'by hand'
    # orig_url = f'https://www.hhs.gov/sites/default/files/alj-{casenum}.pdf'
    all_orig_text, orig_url = get_original_text_revamp(casenum, caseyr)
    return outcome, all_appeal_text, all_orig_text, casenum, orig_url, appeal_url


def get_alj_url(casenum, caseyr):
    options = [
        f'https://www.hhs.gov/sites/default/files/alj-{casenum}.pdf',
        f'https://www.hhs.gov/sites/default/files/static/dab/decisions/alj-decisions/{caseyr}/{casenum}.pdf',
        f'https://www.hhs.gov/sites/default/files/static/dab/decisions/alj-decisions/{caseyr}/{casenum.upper()}.pdf',
        f'https://www.hhs.gov/sites/default/files/static/dab/decisions/alj-decisions/{caseyr}/cr{casenum[2:]}.pdf',
        f'https://www.hhs.gov/sites/default/files/alj-dab{casenum[2:]}.pdf',
        f'https://www.hhs.gov/about/agencies/dab/decisions/alj-decisions/{caseyr}/alj-{casenum}/index.html',
        f'https://www.hhs.gov/sites/default/files/static/dab/decisions/alj-decisions/{caseyr}/{casenum.upper()}.htm',
        f'https://www.hhs.gov/sites/default/files/static/dab/decisions/alj-decisions/{caseyr}/{casenum}.htm',
        f'https://www.hhs.gov/sites/default/files/static/dab/decisions/alj-decisions/{caseyr}/{casenum.upper()}.html',
        f'https://www.hhs.gov/sites/default/files/static/dab/decisions/alj-decisions/{caseyr}/{casenum}.html',
        f'https://www.hhs.gov/sites/default/files/static/dab/decisions/alj-decisions/{caseyr}/{casenum[:2]}d{casenum[2:]}.pdf'
    ]
    for potential_url in options:
        request = requests.get(potential_url)
        if request.status_code == 200:
            return potential_url
        else:
            int_case_yr = int(caseyr) - 1
            case_yr = str(int_case_yr)
            for potential_url in options:
                request = requests.get(potential_url)
                if request.status_code == 200:
                    return potential_url
    return


def get_original_text_revamp(casenum, caseyr):
    '''
    this is the new function for looping over and trying all possible URLs for the original decision
    '''
    fail_count = 0
    potential_alj_url = get_alj_url(casenum, caseyr)
    if potential_alj_url:
        orig_url = potential_alj_url
        if orig_url.endswith("pdf"):
            all_orig_text = get_pdf_txt(orig_url)
        else:
            if re.search(OLD_URL_PATTERN, orig_url):
                all_orig_text, _ = get_full_text_old_html(orig_url)
            else:
                all_orig_text, _ = initialize_get_full_text_html(orig_url)
            print("preview:", all_orig_text[:75])
        return all_orig_text, orig_url
    else:
        fail_count +=1
        print(fail_count)
        return '', ''


def get_dab_id(case_info_str):
    dab_str = re.search(r'(?<=[dab|dab-|dab ])\d*\d', case_info_str.lower())
    if dab_str:
        dab_id = dab_str.group()
    if not dab_str or len(dab_id) != 4:
        dab_str = re.search(r'(?<=a-)\d.*(?=;)', case_info_str.lower())
        if dab_str:
            dab_id = ''.join(re.findall(r'\d+', dab_str.group()))
        else:
            dab_id = None
    return dab_id


def combine_for_ind(case, problem_lst, reject_lst):
    '''
    '''
    row = []
    dab_url, quick_case_info = case
    dab_id = get_dab_id(quick_case_info)
    if not dab_id:
        reject_lst.append(dab_url)
        return
    elif dab_url[-4:] in ['html', '.htm']:
        if re.search(OLD_URL_PATTERN, dab_url):
            try:
                outcome, dab_text, alj_text, alj_id, alj_url, \
                dab_url = get_html_info_old_format(dab_url)
                row = [dab_id, alj_id, dab_text, alj_text, dab_url, alj_url, outcome]
                print("adding old-style HTML appeal to df")
            except:
                pass
        else:
            try:
                outcome, dab_text, alj_text, alj_id, alj_url, \
                dab_url = get_html_info(dab_url, reject_lst, problem_lst)
                if alj_text.startswith("We're sorry"):
                    print("this is a problem, adding to list")
                    problem_lst.append(dab_url)
                row = [dab_id, alj_id, dab_text, alj_text, dab_url, alj_url, outcome]
                print("adding HTML appeal to df")
            except:
                pass
    else:
        pdf_outcome = get_pdf_info(dab_url, reject_lst)
        if pdf_outcome:
            outcome, dab_text, alj_text, alj_id, alj_url, \
            dab_url = pdf_outcome
            if alj_text.startswith("We're sorry"):
                print("this is a problem, adding to list")
                problem_lst.append(dab_url)
            row = [dab_id, alj_id, dab_text, alj_text, dab_url, alj_url, outcome]
            print("adding pdf appeal to df")
    return row


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
        #indexing list below for testing
        test_cases = full_urls[year][:10]
        for case in test_cases:
            print("appeal id:", case)
            row = combine_for_ind(case, problem_lst, reject_lst)
            if row:
                test_df.loc[len(test_df)] = row
            else:
                print('num probs', len(problem_lst))
                print('num reject', len(reject_lst))
    return test_df, problem_lst, reject_lst


'''
I want to note that the reason why this function can't get https://www.hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2011/dab2434.pdf is because the original decision is actually DAB CR2409 (2011), not DAB CR2409 (2010) like the appeal states. this is just bad data entry on their part and idk how we can correct for it except by hand. this is the actual decision file https://www.hhs.gov/sites/default/files/static/dab/decisions/alj-decisions/2011/cr2409.pdf 

other problems:
https://www.hhs.gov/sites/default/files/board-dab2755.pdf ("DAB No." format is generally to be avoided but necessary here, fixed by allowing specifically "DAB No. CR")
https://www.hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2015/dab2667.pdf (some date formatting bs, fixed horribly in get_orig_case_info)
https://www.hhs.gov/sites/default/files/static/dab/decisions/alj-decisions/2013/crd2838.pdf apparently there's yet another possible URL format! i hate it! fixed in the get_original_text_revamp function
https://www.hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2010/dab2355rev.pdf this is just bad OCR rendering a number as a letter, fixed in get_orig_case_info but this regex is getting worse and worse, we really need to make get_orig_case_info more robust to exceptions
https://www.hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2006/dab2054.pdf adding an optional space between case number and date because OCR is crap again
https://www.hhs.gov/sites/default/files/static/dab/decisions/board-decisions/2002/dab1852.pdf apparently original case number can be 3 digits rather than 4, fixing regex in get_orig_case_info
'''


# def get_original_text(casenum, caseyr):
#     '''

#     SUDDENLY WONDERING if we should just download all the original ALJ decisions and fuzzy-match their IDs to the case numbers in the appeals, rather than constructing and trying out all the different possible URL permutations. fml
#     ok nah I stopped being lazy and made the loop of all possibilities above. fuck this function

#     Inputs:

#     Output:
#     '''
#     try:
#         orig_url = f'https://www.hhs.gov/sites/default/files/alj-{casenum}.pdf'
#         #print(orig_url)
#         all_orig_text = get_pdf_txt(orig_url)
#     except:
#         print("pdf doesn't exist")
#         try:
#             orig_url = f'https://www.hhs.gov/sites/default/files/static/dab/decisions/alj-decisions/{caseyr}/{casenum}.pdf'
#             #print(orig_url)
#             all_orig_text = get_pdf_txt(orig_url) 
#         except:
#             print("pdf still doesn't exist")
#             try:
#                 orig_url = f'https://www.hhs.gov/sites/default/files/static/dab/decisions/alj-decisions/{caseyr}/{casenum.upper()}.pdf'
#                 #print(orig_url)
#                 all_orig_text = get_pdf_txt(orig_url) 
#             except:
#                 print("wow this pdf")
#                 try:
#                     mod_casenum = casenum[2:]
#                     orig_url = f'https://www.hhs.gov/sites/default/files/static/dab/decisions/alj-decisions/{caseyr}/cr{mod_casenum}.pdf'
#                     #print(orig_url)
#                     all_orig_text = get_pdf_txt(orig_url)
#                 except:
#                     print("holy shit")
#                     try:
#                         mod_casenum = casenum[2:]
#                         orig_url = f'https://www.hhs.gov/sites/default/files/alj-dab{mod_casenum}.pdf'
#                         #print(orig_url)
#                         all_orig_text = get_pdf_txt(orig_url)
#                     except:
#                         print("pdf really doesn't exist")
#                         orig_url = f'https://www.hhs.gov/about/agencies/dab/decisions/alj-decisions/{caseyr}/alj-{casenum}/index.html'
#                         #print(orig_url)
#                         all_orig_text, _ = initialize_get_full_text_html(orig_url)
#                         if all_orig_text.startswith("We're sorry"):
#                             print("wow now the HTML is acting up too, scraping old HTML")
#                             orig_url = f'https://www.hhs.gov/sites/default/files/static/dab/decisions/alj-decisions/{caseyr}/{casenum.upper()}.htm'
#                             # have to build in additional conditions for whether the casenum is capitalised or not, and whether it ends with htm or html. I'm tired of this
#                             # something about checking that urllib.request.urlopen(orig_url).code == 200? maybe create all possibilities and loop through to check status code, rather than doing a million try-except blocks... https://gist.github.com/fedir/5883651
#                             all_orig_text, _ = get_full_text_old_html(orig_url)
#                             if not all_orig_text:
#                                 print("this file really doesn't exist period")
#     print("preview:", all_orig_text[:50])
#     print("original decision URL:", orig_url)
#     return all_orig_text, orig_url

