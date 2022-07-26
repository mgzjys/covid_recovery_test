#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import json
from time import sleep
import zipfile
import io
from datetime import *
from dateutil import tz
import sys
import os
import pandas as pd
import warnings
import numpy as np


# In[2]:


warnings.simplefilter(action="ignore")


# In[61]:


BASE_URL = "https://az1.qualtrics.com/API"
TIMEZONE = "America/Los_Angeles"
TZ = tz.gettz(TIMEZONE)

# PULL_SURVEYS = {
#     #wave 11
#     "COVID19_symptom_survey_intl_V11_noneu_transfer",
#     "COVID19_symptom_survey_intl_V11_eu_transfer",
# }
PULL_SURVEY_IDS = {
    #"SV_78muCKs44y7XMsC", #V11_eu
    #"SV_bO3Eu4gw1quormC", #V11_noneu
    # "SV_djrS4eUaNXY1EfY", #EU wave 12
    # "SV_7P87apT4hAGJJPg", #ROW Wave 12
    "SV_cTLORaCk1qxfixU", #EU wave 13
    "SV_6DnpVXXm2aYdnSe", #ROW Wave 13
}
PROJECT_PATH = "./"


#TOKEN = "zXEecjMVhZXNi1w5hZtaA05JjBU93jo9VV5z8Ftd"
TOKEN = "VFNU3fktNc4sL4GkM7mB3ikfidcw6BL9aZu1h9ND"


# In[65]:


def progress(t):
    n = 2 * (t + 1)
    return "." * min(10, n), pow(2, max(0, t - 4))


def dicta(*dcts):
    ret = dict()
    for d in dcts:
        ret.update(d)
    return ret


def make_fetchers(tokenfile="qualtrics_api_token.txt"):

    header = {"X-API-TOKEN": TOKEN}

    def g(endpoint, **kw):
        url = f"{BASE_URL}/v3/{endpoint}"
        r = requests.get(url, params=kw, headers=header)
        return r

    def p(endpoint, data):
        url = f"{BASE_URL}/v3/{endpoint}"
        r = requests.post(url, json=data, headers=header)
        return r

    return g, p


def getSurveyFileList(start, end,pull_surveys=PULL_SURVEY_IDS):
    fetch, post = make_fetchers()

    
    ### Get newly exported filenames
    survey_info = fetch("surveys")
    name_list = []
    for surv in survey_info.json()["result"]["elements"]:
        if surv["isActive"] and surv["id"] in pull_surveys:
            name_list.append(
                f"{end.date()}.{start.date()}.{end.date()}.{surv['id']}.{surv['name'].replace(' ','_')}.csv"
            )
    return name_list

def fetchSurveyResponse(start, end, savepath=PROJECT_PATH):
    fetch, post = make_fetchers()

    ### Get newly exported filenames
    name_list = getSurveyFileList(start, end)

    EXPORT_SUCCESS = True
    # check if already exported
    print(savepath)
    for fname in name_list:
        exportfile = os.path.join(savepath, fname)
#         print(exportfile)
        if not os.path.exists(exportfile):
            EXPORT_SUCCESS = False
    print("export success",EXPORT_SUCCESS)
    if not EXPORT_SUCCESS:
        print("fetching qualtrics survey data...")
        print("from:", start.isoformat())
        print("to:", end.isoformat())
        resp = do_main(fetch, post, start, end, savepath)

def download_by_fileid(surveyid,fileid,fname='download_by_fileid',savepath=PROJECT_PATH):
    fetch, post = make_fetchers()
    base = f"surveys/{surveyid}/export-responses/"
    r = fetch(f"{base}{fileid}/file")
    if not r.ok:
        return r
    z = zipfile.ZipFile(io.BytesIO(r.content))
    print(z.namelist)
    for n in z.namelist():
        with open(
            os.path.join(
                savepath,
                f"{fname}.csv",
            ),
            "wb",
        ) as out:
            out.write(z.read(n))
        break


# In[50]:


def do_main(fetch, post, start, end, savepath="."):
    resp = fetch("whoami")
    if not resp.ok:
        return resp
    resp = fetch("surveys")
    if not resp.ok:
        return resp
    results = []
    for surv in resp.json()["result"]["elements"]:
        if not surv["isActive"]:
            continue
        if not surv["id"] in PULL_SURVEY_IDS:
            continue
        print(json.dumps(surv, sort_keys=True, indent=3))
        base = f"surveys/{surv['id']}/export-responses/"
        load = {
                "format": "csv",
                "timeZone": TIMEZONE,
                "startDate": start.isoformat(),
                "endDate": end.isoformat(),
                "breakoutSets": "true",
                "includeDisplayOrder":"true",
                "seenUnansweredRecode": -77,
                "multiselectSeenUnansweredRecode": 0
            }
        r = post(
            base,
            load,
        )
        if not r.ok:
            return r
        progressId = r.json()["result"]["progressId"]
        print(r.text)
        progressStatus = "inProgress"
        t = 0
        wait, waitt = progress(t)
        while progressStatus != "complete" and progressStatus != "failed":
            t += 1
            r = fetch(f"{base}{progressId}")
            if not r.ok:
                return r
            progressStatus = r.json()["result"]["status"]
            pct = r.json()["result"]["percentComplete"]
            print(f"{progressStatus}: {pct}")
            if pct < 100:
                for i in wait:
                    sleep(waitt)
                    print(i, end="", flush=True)
                sleep(waitt)
                print()
            wait, waitt = progress(t)

        if progressStatus == "failed":
            print('progressStatus failed')
            return r
        fileId = r.json()["result"]["fileId"]
        r = fetch(f"{base}{fileId}/file")
        if not r.ok:
            return r
        z = zipfile.ZipFile(io.BytesIO(r.content))
        for n in z.namelist():
            with open(
                os.path.join(
                    savepath,
                    f"{end.date()}.{start.date()}.{end.date()}.{surv['id']}.{surv['name'].replace(' ','_')}.csv",
                ),
                "wb",
            ) as out:
                out.write(z.read(n))
            break
        results.append(r)
    return results


# In[51]:


def generateCID_newlogic(fnames, startdate, enddate, fp=PROJECT_PATH):
    # check if fnames all exist
    for fname in fnames:
        exportfile = os.path.join(fp, fname)
        if not os.path.exists(exportfile):
            print(exportfile)
            print(
                f"""
                Response file is missing -- Please check the getQualitrics script
                has been run successfully run. 
                """
            )
            exit(2)
    
    # read over files in fp regardless fnames
    keep_pd_dic = read_qualtrics_response.readSurveyResponseFile_lite(startdate, enddate,fp,filenames=fnames)
    
    intl_parta_token = pd.concat(
        [p[p.B1_Flag.notna()]["token"] for p in keep_pd_dic.values() if p is not None]
    ).drop_duplicates()


    intl_full_token = pd.concat(
        [p[p.FR_CNT >= 2]['token'] for p in keep_pd_dic.values() if p is not None]
    ).drop_duplicates()
    merge_dup = pd.concat([p for p in keep_pd_dic.values() if p is not None])
    merge_dup = merge_dup.sort_values('StartDate',ascending=True)
    merge_p = merge_dup.drop_duplicates(subset="token",keep='first')

    intl_part_a_token = merge_p[merge_p.B1_Flag.notna()]["token"]
    intl_partial_token = merge_p[merge_p.FR_CNT >= 2]["token"]
    intl_full_mA_token = merge_p[(merge_p.full_flag > 0)&(merge_p.module=='A')]['token']
    intl_full_mB_token = merge_p[(merge_p.full_flag > 0)&(merge_p.module=='B')]['token']
    
    # intl_full_mA_token =  pd.concat(
    #     [p[(p.full_flag > 0)&(p.module=='A')]['token'] for p in keep_pd_dic.values() if p is not None]
    # ).drop_duplicates()
    # intl_full_mB_token =  pd.concat(
    #     [p[(p.full_flag > 0)&(p.module=='B')]['token'] for p in keep_pd_dic.values() if p is not None]
    # ).drop_duplicates()
    print("parta shape", intl_parta_token.shape)
    print("full shape", intl_full_token.shape)

    true_enddate = datetime.combine(
        enddate - timedelta(days=1), time(23, 59, 59), tzinfo=TZ
    )
    stdt_str = startdate.date().strftime("%Y-%m-%d")
    eddt_str = true_enddate.date().strftime("%Y-%m-%d")
    stime = "00_00_" + stdt_str
    etime = "23_59_" + eddt_str
    partial_fname = fp + f"cvid_cids_part_a_response_{stime}_-_{etime}.csv"
    full_fname = fp + f"cvid_cids_full_response_{stime}_-_{etime}.csv"
    print(partial_fname)
    print(full_fname)
    ww_parta_fname = fp + f"umd_respondent_ids/cvid_cids_part_a_response_{stime}_-_{etime}.csv"
    ww_partial_fname = fp + f"umd_respondent_ids/cvid_cids_partial_response_{stime}_-_{etime}.csv"
    ww_full_ma_fname = fp + f"umd_respondent_ids/cvid_cids_full_response_modul_a_{stime}_-_{etime}.csv"
    ww_full_mb_fname = fp + f"umd_respondent_ids/cvid_cids_full_response_modul_b_{stime}_-_{etime}.csv"

    intl_parta_token.to_csv(partial_fname, index=False, header=False)
    intl_full_token.to_csv(full_fname, index=False, header=False)
    intl_part_a_token.to_csv(ww_parta_fname, index=False, header=False)
    intl_partial_token.to_csv(ww_partial_fname, index=False, header=False)
    
    intl_full_mA_token.to_csv(ww_full_ma_fname, index=False, header=False)
    intl_full_mB_token.to_csv(ww_full_mb_fname, index=False, header=False)


# In[45]:





# In[ ]:





# In[52]:


if __name__ == "__main__":
    #download_by_fileid('SV_bryl2pXQhCalqtw','07f01ee1-a761-42ec-9c2c-d4c83c987b43-def')
    if len(sys.argv) == 1:
        start = datetime.combine(date.today() - timedelta(days=1), time(0, 0, 0), tzinfo=TZ)
        end = datetime.combine(date.today() - timedelta(days=0), time(4, 0, 0), tzinfo=TZ)
    
    # debugging usage: input specific date and day range back from the date
    else:
        date_str = str(sys.argv[1])
    #do_main_lite(all_start_date_str='2021-04-01', delta=7)
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        start = datetime.combine(date_obj - timedelta(days=1), time(0, 0, 0), tzinfo=TZ)
        end = datetime.combine(date_obj - timedelta(days=0), time(4, 0, 0), tzinfo=TZ)
        print(start,end)
    
    fetchSurveyResponse(start, end)
    name_list = getSurveyFileList(start, end)
    print("\nNAME LIST:")
    print(name_list)

    print("Renaming country and region A columns...")
    #rename_a_country_region(name_list)
    rename_a_country_region_lite(name_list)
    eu_row_names = merge_split_eu_row_surveys(name_list)
    print(eu_row_names)
    print("Generating CIDs...")
    generateCID_newlogic(eu_row_names, start, end)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


####################
###### just testing from here
####################


# In[22]:


# date_str = '2021-04-20'
# print(f'get customized survey {date_str}')
# input_start_date_obj = datetime.strptime(date_str, '%Y-%m-%d')
# start = datetime.combine(input_start_date_obj + timedelta(days=0), time(0, 0, 0),
#                                         tzinfo=TZ)
# end = datetime.combine(input_start_date_obj + timedelta(days=1), time(4, 0, 0),
#                                         tzinfo=TZ)


# In[67]:


# fetchSurveyResponse(start, end)


# In[68]:


# name_list = getSurveyFileList(start, end)
# print("\nNAME LIST:")
# print(name_list)


# In[ ]:




