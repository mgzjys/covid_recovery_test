# %load /local/data/fanj/COVID_Project/FB_Qualtrics_Code/read_qualtrics_response.py
#!/bin/python3
#import requests

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
from dateutil.parser import parse
warnings.simplefilter(action="ignore")

TIMEZONE = "America/Los_Angeles"
TZ = tz.gettz(TIMEZONE)

PROJECT_PATH = "/gpfs/data1/cgis1gp/covid_survey_data_warehouse/old_fs/api/qualtrics_raw_data_recoded/"
DATAROOT_PATH = "/gpfs/data1/cgis1gp/covid_survey_data_warehouse/"

Mapping_change = False
Current_day = ""
#after "2021-12-21"
country_code_lookup = pd.read_csv(DATAROOT_PATH+'metadata/CTIS_survey_country_region_map_table_ver1.122821.csv')
#after "2020-04-20"
# country_code_lookup = pd.read_csv(DATAROOT_PATH+'metadata/CTIS_survey_country_region_map_table_ver1.083021.csv')


#taken off AFG
weighted_list = pd.read_csv(DATAROOT_PATH+'metadata/FB_Symptom_Survey_Weighted_Country_List_20220415.csv',header=None,names=['country','iso2','iso3'])
#weighted_list = pd.read_csv(DATAROOT_PATH+'metadata/FB_Symptom_Survey_Weighted_Country_List_114.csv',header=None,names=['country','iso2','iso3'])
allow_unw_list = pd.read_csv(DATAROOT_PATH+'metadata/unweighted_countries_allowlist_20220415.csv',header=0)
#allow_unw_list = pd.read_csv(DATAROOT_PATH+'metadata/unweighted_countries_allowlist_revised.csv',header=0)
allow_iso_3 = allow_unw_list.iso3.to_list() + weighted_list.iso3.to_list()
allow_iso_v12exp = [
    'DEU',
    'FRA',
    'MEX',
    'IND',
    'IDN',
    'JPN',
    'AUS',
    'ZAF',
    'VNM',
]

#######################################################
# read survey response file  and process for survey
# version 5
#######################################################
def readSurveyResponseFile_v6(startdate, enddate,fp=PROJECT_PATH):

    startdate_str = startdate.strftime("%Y-%m-%d %H:%M:%S")
    enddate_str = enddate.strftime("%Y-%m-%d %H:%M:%S")

    true_enddate = datetime.combine(
        enddate - timedelta(days=1), time(23, 59, 59), tzinfo=TZ
    )
    true_enddate_str = true_enddate.strftime("%Y-%m-%d %H:%M:%S")
    print("processing data range:", startdate_str, true_enddate_str)

    eu_fname = ""
    neu_fname = ""
    eu_fname2 = ""
    neu_fname2 = ""
    eu_fname3 = ""
    neu_fname3 = ""
    eu_fname4 = ""
    neu_fname4 = ""
    eu_fname5 = ""
    neu_fname5 = ""
    for fname in os.listdir(fp):
        rng_name = f"{startdate.date()}.{enddate.date()}"
        if rng_name in fname and "_EU_wave1" in fname:
            eu_fname = fp + fname

        elif rng_name in fname and "_nonEU_wave1" in fname:
            neu_fname = fp + fname

        elif rng_name in fname and "_EU_V2" in fname:
            eu_fname2 = fp + fname

        elif rng_name in fname and "_nonEU_V2" in fname:
            neu_fname2 = fp + fname
        elif rng_name in fname and "V3_EU" in fname:
            eu_fname3 = fp + fname

        elif rng_name in fname and "V3_nonEU" in fname:
            neu_fname3 = fp + fname
        elif rng_name in fname and "V4_EU" in fname:
            eu_fname4 = fp + fname

        elif rng_name in fname and "V4_nonEU" in fname:
            neu_fname4 = fp + fname

        elif rng_name in fname and "V5_EU" in fname:
            eu_fname5 = fp + fname

        elif rng_name in fname and "V5_nonEU" in fname:
            neu_fname5 = fp + fname

    # print(eu_fname)
    # print(eu_fname2)
    # print(eu_fname3)
    # print(eu_fname4)
    # print(eu_fname5)

    ### function to join eu non_eu responses
    def join_eu_neu_weight(eu_pd, neu_pd, survey_version=4):

        #     startdate_str = "2020-05-01 00:00:00"
        #     true_enddate_str = "2020-05-02 23:59:59"

        ####################################################
        # From 05/01 to 05/15 survey v1, v2 has this issue
        ###################################################
        if ("B1_13" in eu_pd.columns) and ("B1_12" not in eu_pd.columns):
            eu_pd.rename(columns={"B1_13": "B1_12"}, inplace=True)

        intl_pd = pd.concat([eu_pd.loc[2:, :], neu_pd.loc[2:, :]], ignore_index=True)

        intl_pd["intro1"] = intl_pd.apply(
            lambda x: x.intro1_eu if pd.isna(x.intro1_noneu) else x.intro1_noneu, axis=1
        )

        intl_pd["intro2"] = intl_pd.apply(
            lambda x: x.intro2_eu if pd.isna(x.intro2_noneu) else x.intro2_noneu, axis=1
        )

        ## change datatype of columns for later filtering
        intl_pd["intro1"] = intl_pd["intro1"].astype("float")
        intl_pd["intro2"] = intl_pd["intro2"].astype("float")
        intl_pd["A1"] = intl_pd["A1"].astype("float")

        ## Count na in responses for B1
        B1_cols = []
        for col_name in intl_pd.columns:
            if col_name.startswith("B1_") and 'matrix' not in col_name:
                B1_cols.append(col_name)

        # print(B1_cols)

        ## Count na in responses for A2
        A2_cols = []
        for col_name in intl_pd.columns:
            if col_name.startswith("A2"):
                A2_cols.append(col_name)
        # A2_cols=['A2','A2_2_1', 'A2_2_2']
        intl_pd["A2NA"] = intl_pd.loc[:, A2_cols].isna().sum(axis=1)
        intl_pd["A2NA"] = intl_pd["A2NA"] + (intl_pd.loc[:, A2_cols] == -77).sum(axis=1)

        intl_pd["A2_Flag"] = intl_pd.A2NA.apply(
            lambda x: 1 if x < len(A2_cols) else np.nan
        )

        print("raw response shape", eu_pd.shape, neu_pd.shape, intl_pd.shape)

        intl_pd["B1NA"] = intl_pd.loc[:, B1_cols].isna().sum(axis=1)
        intl_pd["B1_Flag"] = intl_pd.B1NA.apply(
            lambda x: 1 if x < len(B1_cols) else np.nan
        )

        intl_keep_pd = intl_pd[
            (intl_pd.token.notna())
            & (intl_pd.DistributionChannel == "anonymous")
            & (intl_pd.intro1 == 4.0)
            & (intl_pd.intro2 == 4.0)
            & (intl_pd.A1 == 23.0)
            & (intl_pd.A2_Flag.notna())
            & (intl_pd.StartDate >= startdate_str)
            & (intl_pd.StartDate <= true_enddate_str)
        ]

        intl_keep_pd.reset_index(inplace=True)

        return intl_keep_pd

    def readEU_NEUResponse(eu_fname, neu_fname, survey_version=4):
        print("reading...")
        print(eu_fname)
        eu_pd = pd.read_csv(eu_fname)
        print("reading...")
        print(neu_fname)
        neu_pd = pd.read_csv(neu_fname)

        eu_pd["survey_region"] = "EU"
        neu_pd["survey_region"] = "ROW"

        #     sel_cols=neu_pd.columns
        # sel_cols = list(set(eu_pd.columns).intersection(set(neu_pd.columns)))
        if eu_pd.shape[0] < 10 and neu_pd.shape[0] < 10:
            return None
        else:
            intl_keep_pd = join_eu_neu_weight(
                eu_pd, neu_pd, survey_version=survey_version
            )
            return intl_keep_pd

    def merge_pd(eu_fname, neu_fname, additional_cols, survey_version=4):

        intl_keep_pd = readEU_NEUResponse(
            eu_fname, neu_fname, survey_version=survey_version
        )
        if intl_keep_pd is not None:
            print('number of response for survey version',survey_version,": ",intl_keep_pd.shape)
        
            print(set(additional_cols).difference(set(intl_keep_pd.columns)))

            intl_keep_pd["FR_CNT"] = intl_keep_pd.loc[:, additional_cols].notna().sum(axis=1)

            return intl_keep_pd
        else:
            return None

    ##########################################################
    # generate and write token id files
    ##########################################################
    additional_cols_v1 = [
        "B2a",
        "B2b",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7a",
        "B7b",
        "B8",
        "C0_1",
        "C0_2",
        "C0_3",
        "C0_4",
        "C0_5",
        "C0_6",
        "C1_m",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "D1",
        "D2",
        "D3",
        "D4",
        "E2",
        "E3",
        "E4",
        "E5",
        "B1_Flag",
    ]
    additional_cols_v2 = [
        "B2a",
        "B2b",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7a",
        "B7b",
        "B8",
        "C0_1",
        "C0_2",
        "C0_3",
        "C0_4",
        "C0_5",
        "C0_6",
        "C1_m",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "E2",
        "E3",
        "E4",
        "E5",
        "B1_Flag",
    ]
    additional_cols_v3_v4 = [
        "B2b",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7a",
        "B7b",
        "B7c",
        "B8",
        "C0_1",
        "C0_2",
        "C0_3",
        "C0_4",
        "C0_5",
        "C0_6",
        "C1_m",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "E2",
        "E3",
        "E4",
        "E5",
        "F1",
        "F2",
        "F3",
        "B1_Flag",
    ]

    additional_cols_v5 = [
        "B2b",
        "B1b_x1",
        "B1b_x2",
        "B1b_x3",
        "B1b_x4",
        "B1b_x5",
        "B1b_x6",
        "B1b_x7",
        "B1b_x8",
        "B1b_x9",
        "B1b_x10",
        "B1b_x11",
        "B1b_x12",
        "B1b_x13",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7a",
        "B7b",
        "B7c",
        "B8",
        "B9",
        "B10",
        "B11a",
        "B11b",
        "B11c",
        "B12a_1",
        "B12a_2",
        "B12a_3",
        "B12a_4",
        "B12a_5",
        "B12a_6",
        "B12b_1",
        "B12b_2",
        "B12b_3",
        "B12b_4",
        "B12b_5",
        "B12b_6",
        "B12c_1",
        "B12c_2",
        "B12c_3",
        "B12c_4",
        "B12c_5",
        "B12c_6",
        "B13_1",
        "B13_2",
        "B13_3",
        "B13_4",
        "B13_5",
        "B13_6",
        "B13_7",
        "B14_1",
        "B14_2",
        "B14_3",
        "B14_4",
        "B14_5",
        "C0_1",
        "C0_2",
        "C0_3",
        "C0_4",
        "C0_5",
        "C0_6",
        "C1_m",
        "C2",
        "C7",
        "C8",
        "C3",
        "C5",
        "C6",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6_1",
        "D6_2",
        "D6_3",
        "D7",
        "D8",
        "D9",
        "D10a",
        "D10b",
        "E3",
        "E4",
        "E6",
        "E2",
        "E5",
        "E7",
        "F1",
        "F2_1",
        "F2_2",
        "F3_au",
        "F3_de",
        "B1_Flag",
    ]

    keep_pd_dic = dict()
    if len(eu_fname) > 0 and len(neu_fname) > 0:
        keep_pd_dic[1] = merge_pd(
            eu_fname, neu_fname, additional_cols_v1, survey_version=1
        )
    if len(eu_fname2) > 0 and len(neu_fname2) > 0:
        keep_pd_dic[2] = merge_pd(
            eu_fname2, neu_fname2, additional_cols_v2, survey_version=2
        )
    if len(eu_fname3) > 0 and len(neu_fname3) > 0:
        keep_pd_dic[3] = merge_pd(
            eu_fname3, neu_fname3, additional_cols_v3_v4, survey_version=3
        )
    if len(eu_fname4) > 0 and len(neu_fname4) > 0:
        keep_pd_dic[4] = merge_pd(
            eu_fname4, neu_fname4, additional_cols_v3_v4, survey_version=4
        )
    if len(eu_fname5) > 0 and len(neu_fname5) > 0:
        keep_pd_dic[5] = merge_pd(
            eu_fname5, neu_fname5, additional_cols_v5, survey_version=5
        )

    return keep_pd_dic
list_versions = ['v1','v2','v3','v4','v5','v6','v6b','v7','v8','v9',
                'v10',
                'v10b','v10bc', 'v11','v12','v13']
version_region_file_suffixs = {
    #version id:[[labels as part of the file name],[labels not in the file name]]
    'v1_EU':[['_EU_wave1'],[]],
    'v1_nonEU':[['_nonEU_wave1'],[]],
    'v2_EU':[['_EU_V2'],[]],
    'v2_nonEU':[['_nonEU_V2'],[]],
    'v3_EU':[['V3_EU'],[]],
    'v3_nonEU':[['V3_nonEU'],[]],
    'v4_EU':[['V4_EU'],[]],
    'v4_nonEU':[['V4_nonEU'],[]],
    'v5_EU':[['V5_EU'],[]],
    'v5_nonEU':[['V5_nonEU'],[]],
    'v6_EU':[['V6_EU'],['1119','part']],
    'v6_nonEU':[['V6_nonEU'],['1119','part']],
    'v6b_EU':[['_V6_eu_1119'],[]],
    'v6b_nonEU':[['_V6_noneu_1119'],[]],
    'v7_EU':[['_V7_eu'],[]],
    'v7_nonEU':[['_V7_noneu'],[]],
    'v8_EU':[['_V8_eu'],[]],
    'v8_nonEU':[['_V8_noneu'],[]],
    'v9_EU':[['_V9_eu'],[]],
    'v9_nonEU':[['_V9_noneu'],[]],
    'v10_EU':[['_V10_eu'],[]],
    'v10_nonEU':[['_V10_noneu'],[]],
    'v10b_EU':[['_V10b_eu'],['Control']],
    'v10b_nonEU':[['_V10b_noneu'],['Control']],
    'v10bc_EU':[['_V10b_eu_-_Control'],[]],
    'v10bc_nonEU':[['_V10b_noneu_-_Control'],[]],
    'v11_EU':[['_V11_eu'],[]],
    'v11_nonEU':[['_V11_noneu'],[]],
    'v12_EU':[['SV_djrS4eUaNXY1EfY'],[]],
    'v12_nonEU':[['SV_7P87apT4hAGJJPg'],[]],
    'v13_EU':[['SV_cTLORaCk1qxfixU'],[]],
    'v13_nonEU':[['SV_6DnpVXXm2aYdnSe'],[]],
}
additional_cols_v1 = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7a",
    "B7b",
    "B8",
    "C0_1",
    "C0_2",
    "C0_3",
    "C0_4",
    "C0_5",
    "C0_6",
    "C1_m",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "D1",
    "D2",
    "D3",
    "D4",
    "E2",
    "E3",
    "E4",
    "E5",
    "B1_Flag",
]
additional_cols_v2 = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7a",
    "B7b",
    "B8",
    "C0_1",
    "C0_2",
    "C0_3",
    "C0_4",
    "C0_5",
    "C0_6",
    "C1_m",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "E2",
    "E3",
    "E4",
    "E5",
    "B1_Flag",
]
additional_cols_v3_v4 = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7a",
    "B7b",
    "B7c",
    "B8",
    "C0_1",
    "C0_2",
    "C0_3",
    "C0_4",
    "C0_5",
    "C0_6",
    "C1_m",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "E2",
    "E3",
    "E4",
    "E5",
    "F1",
    "F2",
    "F3",
    "B1_Flag",
]

additional_cols_v5 = [
    "B2b",
    "B1b_x1",
    "B1b_x2",
    "B1b_x3",
    "B1b_x4",
    "B1b_x5",
    "B1b_x6",
    "B1b_x7",
    "B1b_x8",
    "B1b_x9",
    "B1b_x10",
    "B1b_x11",
    "B1b_x12",
    "B1b_x13",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7a",
    "B7b",
    "B7c",
    "B8",
    "B9",
    "B10",
    "B11a",
    "B11b",
    "B11c",
    "B12a_1",
    "B12a_2",
    "B12a_3",
    "B12a_4",
    "B12a_5",
    "B12a_6",
    "B12b_1",
    "B12b_2",
    "B12b_3",
    "B12b_4",
    "B12b_5",
    "B12b_6",
    "B12c_1",
    "B12c_2",
    "B12c_3",
    "B12c_4",
    "B12c_5",
    "B12c_6",
    "B13_1",
    "B13_2",
    "B13_3",
    "B13_4",
    "B13_5",
    "B13_6",
    "B13_7",
    "B14_1",
    "B14_2",
    "B14_3",
    "B14_4",
    "B14_5",
    "C0_1",
    "C0_2",
    "C0_3",
    "C0_4",
    "C0_5",
    "C0_6",
    "C1_m",
    "C2",
    "C7",
    "C8",
    "C3",
    "C5",
    "C6",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6_1",
    "D6_2",
    "D6_3",
    "D7",
    "D8",
    "D9",
    "D10a",
    "D10b",
    "E3",
    "E4",
    "E6",
    "E2",
    "E5",
    "E7",
    "F1",
    "F2_1",
    "F2_2",
    "F3_au",
    "F3_de",
    "B1_Flag",
]

additional_cols_v6 = [
    "B2b",
    "B1b_x1",
    "B1b_x2",
    "B1b_x3",
    "B1b_x4",
    "B1b_x5",
    "B1b_x6",
    "B1b_x7",
    "B1b_x8",
    "B1b_x9",
    "B1b_x10",
    "B1b_x11",
    "B1b_x12",
    "B1b_x13",
    "B1b_x14",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7a",
    "B7b",
    "B8",
    "B9",
    "B10",
    "B11a",
    "B11b",
    "B12a_1",
    "B12a_2",
    "B12a_3",
    "B12a_4",
    "B12a_5",
    "B12a_6",
    "B12b_1",
    "B12b_2",
    "B12b_3",
    "B12b_4",
    "B12b_5",
    "B12b_6",
    "B13_1",
    "B13_2",
    "B13_3",
    "B13_4",
    "B13_5",
    "B13_6",
    "B13_7",
    "B14_1",
    "B14_2",
    "B14_3",
    "B14_4",
    "B14_5",
    "C0_matrix_1",
    "C0_matrix_2",
    "C0_matrix_3",
    "C0_matrix_4",
    "C0_matrix_5",
    "C0_matrix_6",
    "C1_m",
    "C2",
    "C7",
    "C8",
    "C3",
    "C5",
    "C6",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6_1",
    "D6_2",
    "D6_3",
    "D7",
    "D8",
    "D9",
    "D10a",
    "D10b",
    "E3",
    "E4",
    "E6",
    "E2",
    "E5",
    "E7",
    "F1",
    "F2_1",
    "F2_2",
    "F3_au",
    "F3_de",
    "B1_Flag"
]

additional_cols_v6_1119 = [
    "B2b",
    "B1b_x1",
    "B1b_x2",
    "B1b_x3",
    "B1b_x4",
    "B1b_x5",
    "B1b_x6",
    "B1b_x7",
    "B1b_x8",
    "B1b_x9",
    "B1b_x10",
    "B1b_x11",
    "B1b_x12",
    "B1b_x13",
    "B1b_x14",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7a",
    "B7c",
    "B8",
    "B9",
    "B10",
    "B11b",
    "B12b_1",
    "B12b_2",
    "B12b_3",
    "B12b_4",
    "B12b_5",
    "B12b_6",
    "B13_1",
    "B13_2",
    "B13_3",
    "B13_4",
    "B13_5",
    "B13_6",
    "B13_7",
    "B14_1",
    "B14_2",
    "B14_3",
    "B14_4",
    "B14_5",
    "C13_1",
    "C13_2",
    "C13_3",
    "C13_4",
    "C13_5",
    "C13_6",
    "C0_1",
    "C0_2",
    "C0_3",
    "C0_4",
    "C0_5",
    "C0_6",
    "C1_m",
    "C2",
    "C7",
    "C8",
    "C3",
    "C5",
    "C6",
    "C14",
    "C9",
    "C10",
    "C11_no_1",
    "C11_no_2",
    "C11_no_3",
    "C11_no_4",
    "C11_no_5",
    "C11_no_6",
    "C11_no_7",
    "C11_no_7_TEXT",
    "C11_unsure_1",
    "C11_unsure_2",
    "C11_unsure_3",
    "C11_unsure_4",
    "C11_unsure_5",
    "C11_unsure_6",
    "C11_unsure_7",
    "C11_unsure_7_TEXT",
    "C12",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6_1",
    "D6_2",
    "D6_3",
    "D7",
    "D8",
    "D9",
    "D10a",
    "D10b",
    "E3",
    "E4",
    "E6",
    "E2",
    "E5",
    "E7",
    "F1",
    "F2_1",
    "F2_2",
    "F3_au",
    "F3_de",
    "B1_Flag"
]

additional_cols_v7 = [
    "A3",
    "B2b",
    "B1b_x1",
    "B1b_x2",
    "B1b_x3",
    "B1b_x4",
    "B1b_x5",
    "B1b_x6",
    "B1b_x7",
    "B1b_x8",
    "B1b_x9",
    "B1b_x10",
    "B1b_x11",
    "B1b_x12",
    "B1b_x13",
    "B1b_x14",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7c",
    "B8",
    "B9",
    "B10",
    "B11b",
    "B12b_1",
    "B12b_2",
    "B12b_3",
    "B12b_4",
    "B12b_5",
    "B12b_6",
    "V1",
    "V2",
    "V3",
    "V4_1",
    "V4_2",
    "V4_3",
    "V4_4",
    "V4_5",
    "B13_1",
    "B13_2",
    "B13_3",
    "B13_4",
    "B13_5",
    "B13_6",
    "B13_7",
    "B14_1",
    "B14_2",
    "B14_3",
    "B14_4",
    "B14_5",
    "C13_1",
    "C13_2",
    "C13_3",
    "C13_4",
    "C13_5",
    "C13_6",
    "C0_1",
    "C0_2",
    "C0_3",
    "C0_4",
    "C0_5",
    "C0_6",
    "C1_m",
    "C2",
    "C7",
    "C8",
    "C3",
    "C5",
    "C6",
    "C14",
    "C9",
    "C10",
    "C11_no_1",
    "C11_no_2",
    "C11_no_3",
    "C11_no_4",
    "C11_no_5",
    "C11_no_6",
    "C11_no_7",
    "C11_no_7_TEXT",
    "C11_unsure_1",
    "C11_unsure_2",
    "C11_unsure_3",
    "C11_unsure_4",
    "C11_unsure_5",
    "C11_unsure_6",
    "C11_unsure_7",
    "C11_unsure_7_TEXT",
    "C12",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6_1",
    "D6_2",
    "D6_3",
    "D7",
    "D8",
    "D9",
    "D10a",
    "D10b",
    "E3",
    "E4",
    "E6",
    "E2",
    "E5",
    "E7",
    "F1",
    "F2_1",
    "F2_2",
    "F3_au",
    "F3_de",
    "B1_Flag"
]

additional_cols_v8 = [
    "A3",
    "B2b",
    "B1b_x1",
    "B1b_x2",
    "B1b_x3",
    "B1b_x4",
    "B1b_x5",
    "B1b_x6",
    "B1b_x7",
    "B1b_x8",
    "B1b_x9",
    "B1b_x10",
    "B1b_x11",
    "B1b_x12",
    "B1b_x13",
    "B1b_x14",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7c",
    "B8",
    "B9",
    "B10",
    "B11b",
    "B12b_1",
    "B12b_2",
    "B12b_3",
    "B12b_4",
    "B12b_5",
    "B12b_6",
    "V1",
    "V2",
    "V3",
    "V4_1",
    "V4_2",
    "V4_3",
    "V4_4",
    "V4_5",
    "V9",
    "B13_1",
    "B13_2",
    "B13_3",
    "B13_4",
    "B13_5",
    "B13_6",
    "B13_7",
    "B14_1",
    "B14_2",
    "B14_3",
    "B14_4",
    "B14_5",
    "C13_1",
    "C13_2",
    "C13_3",
    "C13_4",
    "C13_5",
    "C13_6",
    "C0_1",
    "C0_2",
    "C0_3",
    "C0_4",
    "C0_5",
    "C0_6",
    "C1_m",
    "C2",
    "C7",
    "C8",
    "C3",
    "C5",
    "C6",
    "C14",
    "C9",
    "C10",
    "C11_no_1",
    "C11_no_2",
    "C11_no_3",
    "C11_no_4",
    "C11_no_5",
    "C11_no_6",
    "C11_no_7",
    "C11_no_7_TEXT",
    "C11_unsure_1",
    "C11_unsure_2",
    "C11_unsure_3",
    "C11_unsure_4",
    "C11_unsure_5",
    "C11_unsure_6",
    "C11_unsure_7",
    "C11_unsure_7_TEXT",
    "C12",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6_1",
    "D6_2",
    "D6_3",
    "D7",
    "D8",
    "D9",
    "D10a",
    "D10b",
    "E3",
    "E4",
    "E6",
    "E2",
    "E5",
    "E7",
    "F1",
    "F2_1",
    "F2_2",
    "F3_au",
    "F3_de",
    "B1_Flag"
]

additional_cols_v9 = [
    "A3",
    "B2b",
    "B1b_x1",
    "B1b_x2",
    "B1b_x3",
    "B1b_x4",
    "B1b_x5",
    "B1b_x6",
    "B1b_x7",
    "B1b_x8",
    "B1b_x9",
    "B1b_x10",
    "B1b_x11",
    "B1b_x12",
    "B1b_x13",
    "B1b_x14",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7c",
    "B8",
    "B9",
    "B10",
    "B11b",
    "B12b_1",
    "B12b_2",
    "B12b_3",
    "B12b_4",
    "B12b_5",
    "B12b_6",
    "V1",
    "V2",
    "V2a",
    "V3",
    "V4_1",
    "V4_6",
    "V4_3",
    "V4_4",
    "V4_5",
    "V5a_1",
    "V5a_2",
    "V5a_3",
    "V5a_4",
    "V5a_5",
    "V5a_6",
    "V5a_7",
    "V5a_8",
    "V5a_9",
    "V5b_1",
    "V5b_2",
    "V5b_3",
    "V5b_4",
    "V5b_5",
    "V5b_6",
    "V5b_7",
    "V5b_8",
    "V5b_9",
    "V5c_1",
    "V5c_2",
    "V5c_3",
    "V5c_4",
    "V5c_5",
    "V5c_6",
    "V5c_7",
    "V5c_8",
    "V5c_9",
    "V5d_1",
    "V5d_2",
    "V5d_3",
    "V5d_4",
    "V5d_5",
    "V5d_6",
    "V5d_7",
    "V5d_8",
    "V5d_9",
    "V6_1",
    "V6_2",
    "V6_3",
    "V6_4",
    "V6_5",
    "V6_6",
    "V6_7",
    "V9",
    "V10_1",
    "V10_2",
    "V10_3",
    "V10_4",
    "V10_5",
    "V10_6",
    "V10_7",
    "V10_8",
    "V10_9",
    "V10_10",
    "V11",
    "V12",
    "B13_1",
    "B13_2",
    "B13_3",
    "B13_4",
    "B13_5",
    "B13_6",
    "B13_7",
    "B14_1",
    "B14_2",
    "B14_3",
    "B14_4",
    "B14_5",
    "C13_1",
    "C13_2",
    "C13_3",
    "C13_4",
    "C13_5",
    "C13_6",
    "C0_1",
    "C0_2",
    "C0_3",
    "C0_4",
    "C0_5",
    "C0_6",
    "C1_m",
    "C2",
    "C7",
    "C8",
    "C3",
    "C5",
    "C6",
    "C14",
    "C9"
    "C9a",
    "C10",
    "C12",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6_1",
    "D6_2",
    "D6_3",
    "D7",
    "D8",
    "D9",
    "D10a",
    "D10b",
    "E3",
    "E4",
    "E6",
    "E2",
    "E5",
    "E7",
    "F1",
    "F2_1",
    "F2_2",
    "F3_au",
    "F3_de",
    "B1_Flag"
]

additional_cols_v10 = [
    "A3",
    "B2b",
    "B1b_x1",
    "B1b_x2",
    "B1b_x3",
    "B1b_x4",
    "B1b_x5",
    "B1b_x6",
    "B1b_x7",
    "B1b_x8",
    "B1b_x9",
    "B1b_x10",
    "B1b_x11",
    "B1b_x12",
    "B1b_x13",
    "B1b_x14",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7c",
    "B8",
    "B9",
    "B10",
    "B11b",
    "B12b_1",
    "B12b_2",
    "B12b_3",
    "B12b_4",
    "B12b_5",
    "B12b_6",
    "V1",
    "V2",
    "V2a",
    "V3",
    "V4_1",
    "V4_6",
    "V4_3",
    "V4_4",
    "V4_5",
    "V5a_1",
    "V5a_2",
    "V5a_3",
    "V5a_4",
    "V5a_5",
    "V5a_6",
    "V5a_7",
    "V5a_8",
    "V5a_9",
    "V5a_10",
    "V5b_1",
    "V5b_2",
    "V5b_3",
    "V5b_4",
    "V5b_5",
    "V5b_6",
    "V5b_7",
    "V5b_8",
    "V5b_9",
    "V5b_10",
    "V5c_1",
    "V5c_2",
    "V5c_3",
    "V5c_4",
    "V5c_5",
    "V5c_6",
    "V5c_7",
    "V5c_8",
    "V5c_9",
    "V5c_10",
    "V5d_1",
    "V5d_2",
    "V5d_3",
    "V5d_4",
    "V5d_5",
    "V5d_6",
    "V5d_7",
    "V5d_8",
    "V5d_9",
    "V5d_10",
    "V6_1",
    "V6_2",
    "V6_3",
    "V6_4",
    "V6_5",
    "V6_6",
    "V6_7",
    "V9",
    "V10_1",
    "V10_2",
    "V10_3",
    "V10_4",
    "V10_5",
    "V10_6",
    "V10_7",
    "V10_8",
    "V10_9",
    "V10_10",
    "V11",
    "V12",
    "V13",
    "V15",
    "V16",
    "B13_1",
    "B13_2",
    "B13_3",
    "B13_4",
    "B13_5",
    "B13_6",
    "B13_7",
    "B14_1",
    "B14_2",
    "B14_3",
    "B14_4",
    "B14_5",
    "C13_1",
    "C13_2",
    "C13_3",
    "C13_4",
    "C13_5",
    "C13_6",
    "C0_1",
    "C0_2",
    "C0_3",
    "C0_4",
    "C0_5",
    "C0_6",
    "C1_m",
    "C2",
    "C7",
    "C8",
    "C3",
    "C5",
    "C6",
    "C14",
    "C9"
    "C9a",
    "C10",
    "C12",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6_1",
    "D6_2",
    "D6_3",
    "D7",
    "D8",
    "D9",
    "D10a",
    "D10b",
    "E3",
    "E4",
    "E6",
    "E2",
    "E5",
    "E7",
    "B1_Flag"
]

additional_cols_v11 = [
    "A3_5",
    "A3_6",
    "A3_7",
    "B0",
    "G1",
    "G2",
    "G3",
    "H1",
    "H2",
    "H3",
    "I1",
    "I2",
    "I5",
    "I6_1",
    "I6_2",
    "I6_3",
    "I6_4",
    "I6_5",
    "I6_6",
    "I6_7",
    "I6_8",
    "I7",
    "I8",
    "I9_noneu",
    "I10_noneu_1",
    "I10_noneu_2",
    "I10_noneu_3",
    "I10_noneu_4",
    "I10_noneu_5",
    "J2",
    "J2",
    "K1",
    "V18a",
    "V18b",
    "V19",
    "B8a",
    "B12d",
    "D7a",
    "E7a",
    "E8",
    "V3a",
    "V15a",
    "V16a",
    "B2b",
    "B1b_x2",
    "B1b_x3",
    "B1b_x4",
    "B1b_x5",
    "B1b_x6",
    "B1b_x7",
    "B1b_x8",
    "B1b_x9",
    "B1b_x11",
    "B3",
    "B4",
    "B7c",
    "B8",
    "B11b",
    "B12b_1",
    "B12b_2",
    "B12b_3",
    "B12b_4",
    "B12b_5",
    "B12b_6",
    "V1",
    "V2",
    "V3",
    "V5a_1",
    "V5a_2",
    "V5a_3",
    "V5a_4",
    "V5a_5",
    "V5a_6",
    "V5a_7",
    "V5a_8",
    "V5a_9",
    "V5a_10",
    "V5b_1",
    "V5b_2",
    "V5b_3",
    "V5b_4",
    "V5b_5",
    "V5b_6",
    "V5b_7",
    "V5b_8",
    "V5b_9",
    "V5b_10",
    "V5c_1",
    "V5c_2",
    "V5c_3",
    "V5c_4",
    "V5c_5",
    "V5c_6",
    "V5c_7",
    "V5c_8",
    "V5c_9",
    "V5c_10",
    "V5d_1",
    "V5d_2",
    "V5d_3",
    "V5d_4",
    "V5d_5",
    "V5d_6",
    "V5d_7",
    "V5d_8",
    "V5d_9",
    "V5d_10",
    "V6_1",
    "V6_2",
    "V6_3",
    "V6_4",
    "V6_5",
    "V6_6",
    "V6_7",
    "V9",
    "V10_1",
    "V10_2",
    "V10_3",
    "V10_4",
    "V10_5",
    "V10_6",
    "V10_7",
    "V10_8",
    "V10_9",
    "V10_10",
    "V11",
    "V12",
    "V15",
    "V16",
    "B13_1",
    "B13_2",
    "B13_3",
    "B13_4",
    "B13_5",
    "B13_6",
    "B13_7",
    "B14_1",
    "B14_2",
    "B14_3",
    "B14_4",
    "B14_5",
    "C13_1",
    "C13_2",
    "C13_3",
    "C13_4",
    "C13_5",
    "C13_6",
    "C0_1",
    "C0_2",
    "C0_3",
    "C0_4",
    "C0_5",
    "C0_6",
    "C5",
    "C14",
    "C9"
    "C10",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D7",
    "D10a",
    "E3",
    "E4",
    "E6",
    "E2",
    "E5",
    "E7",
    "B1_Flag"
]

additional_cols_v12 = pd.read_csv(DATAROOT_PATH+'metadata/addcols_V12_work.csv',header=None,index_col = False,names=['col']).col.tolist()
additional_cols_v13 = pd.read_csv(DATAROOT_PATH+'metadata/addcols_V13_work.csv',header=None,index_col = False,names=['col']).col.tolist()

dict_additional_cols_by_version={
    list_versions[0]:additional_cols_v1,
    list_versions[1]:additional_cols_v2,
    list_versions[2]:additional_cols_v3_v4,
    list_versions[3]:additional_cols_v3_v4,
    list_versions[4]:additional_cols_v5,
    list_versions[5]:additional_cols_v6,
    list_versions[6]:additional_cols_v6_1119,
    list_versions[7]:additional_cols_v7,
    list_versions[8]:additional_cols_v8,
    list_versions[9]:additional_cols_v9,
    list_versions[10]:additional_cols_v10,
    'v10b':additional_cols_v10,
    'v10bc':additional_cols_v10,
    'v11':additional_cols_v11,
    'v12':additional_cols_v12,
    'v13':additional_cols_v13,
}

list_version_region_ids = list(version_region_file_suffixs.keys())

def find_version_region(version_region_file_suffixs,filename):
    '''
    find version region for the given filename with a predefined dict version_region_file_suffixs
    return key value or False when no key found
    '''
    for k,v in version_region_file_suffixs.items():
        keys_with = v[0]
        keys_without = v[1]
        bool_with = all(key in filename for key in keys_with)
        bool_without = all(key not in filename for key in keys_without)
        if bool_with and bool_without:
            return k
    #if all matching failed return false
    return False
def readSurveyResponseFile_lite(startdate,enddate,fp=PROJECT_PATH,filenames=None):
    startdate_str = startdate.strftime("%Y-%m-%d %H:%M:%S")
    enddate_str = enddate.strftime("%Y-%m-%d %H:%M:%S")
    if startdate.strftime("%Y-%m-%d") == "2021-12-21":
        Mapping_change=True
        Current_day = "2021-12-21"
    #elif 
    else:
        Mapping_change=False
        Current_day = startdate.strftime("%Y-%m-%d")
    true_enddate = datetime.combine(
        enddate - timedelta(days=1), time(23, 59, 59), tzinfo=TZ
    )
    true_enddate_str = true_enddate.strftime("%Y-%m-%d %H:%M:%S")
    print("processing data range:", startdate_str, true_enddate_str)
    #print(f"search {startdate.date()}.{enddate.date()} in: ",os.listdir(fp))
    #store key-value pair (version, filenames) in the dict
    res_filenames_by_version_id = {}
    # create dict for version:filename
    check_filenames=False
    if filenames is not None:
        check_filenames = True
        print('files checklist',filenames)
    for fname in os.listdir(fp):
        rng_name = f"{startdate.date()}.{enddate.date()}"
        if rng_name in fname:
            matched_version = find_version_region(version_region_file_suffixs,fname)
            if matched_version:
                if check_filenames:
                    if fname in filenames:
                        res_filenames_by_version_id[matched_version] = fp + fname
                    else:
                        print(f'{fname} is not in checklists')
                else:
                    res_filenames_by_version_id[matched_version] = fp + fname

    def join_eu_neu_weight(eu_pd, neu_pd, survey_version):

        #     startdate_str = "2020-05-01 00:00:00"
        #     true_enddate_str = "2020-05-02 23:59:59"

        ####################################################
        # From 05/01 to 05/15 survey v1, v2 has this issue
        ###################################################
        if ("B1_13" in eu_pd.columns) and ("B1_12" not in eu_pd.columns):
            eu_pd.rename(columns={"B1_13": "B1_12"}, inplace=True)
        if survey_version in ['v10b']:
            intl_pd = pd.concat([eu_pd, neu_pd], ignore_index=True)
        else:
            intl_pd = pd.concat([eu_pd.loc[2:, :], neu_pd.loc[2:, :]], ignore_index=True)

        intl_pd["intro1"] = intl_pd.apply(
            lambda x: x.intro1_eu if pd.isna(x.intro1_noneu) else x.intro1_noneu, axis=1
        )

        intl_pd["intro2"] = intl_pd.apply(
            lambda x: x.intro2_eu if pd.isna(x.intro2_noneu) else x.intro2_noneu, axis=1
        )

        ## change datatype of columns for later filtering
        intl_pd["intro1"] = intl_pd["intro1"].astype("float")
        intl_pd["intro2"] = intl_pd["intro2"].astype("float")
        intl_pd["A1"] = intl_pd["A1"].astype("float")
        intl_pd["A2_2_1"] = intl_pd["A2_2_1"].astype("float")
        intl_pd["A2_2_2"] = intl_pd["A2_2_2"].astype("float")

        ## Count na in responses for B1
        B1_cols = []
        for col_name in intl_pd.columns:
            if col_name.startswith("B1_") and 'matrix' not in col_name:
                B1_cols.append(col_name)

        # print(B1_cols)

        ## Count na in responses for A2
        A2_cols = []
        for col_name in intl_pd.columns:
            if col_name.startswith("A2"):
                A2_cols.append(col_name)
        # A2_cols=['A2','A2_2_1', 'A2_2_2']
        intl_pd["A2NA"] = intl_pd.loc[:, A2_cols].isna().sum(axis=1)
        intl_pd["A2NA"] = intl_pd["A2NA"] + (intl_pd.loc[:, A2_cols] == -77).sum(axis=1)

        intl_pd["A2_Flag"] = intl_pd.A2NA.apply(
            lambda x: 1 if x < len(A2_cols) else np.nan
        )
        if survey_version in ['v13','v14'] and 'V2a' in intl_pd.columns:
            intl_pd.rename(columns={'V2a':'V2a_2'},inplace=True)
        if 'V20_year ' in intl_pd.columns:
            intl_pd.rename(columns={'V20_year ':'V20_year'},inplace=True)
        print("raw response shape", eu_pd.shape, neu_pd.shape, intl_pd.shape)

        ##############################################
        # B1 response is valid if :
        # - responded yes/no for at least one symptom
        # - did NOT respond yes for ALL symptoms
        #
        #############################################
        def countB1_yes(r):
            s = 0
            for e in r:
                if pd.notna(e) and int(float(e)) == 1:
                    s += 1
            return s

        intl_pd["B1_yes"] = intl_pd.loc[:, B1_cols].apply(
            lambda x: countB1_yes(x), axis=1
        )

        intl_pd["B1NA"] = intl_pd.loc[:, B1_cols].isna().sum(axis=1)
        intl_pd["B1NA"] = intl_pd["B1NA"] + (intl_pd.loc[:, B1_cols] == -77).sum(axis=1)
        intl_pd["B1_NAFlag"] = intl_pd.B1NA.apply(
            lambda x: 1 if x < len(B1_cols) else np.nan
        )

        def B1_valid(r):
            if pd.notna(r.B1_NAFlag) and r.B1_yes < len(B1_cols):
                return 1
            else:
                return np.nan

        intl_pd["B1_Flag"] = intl_pd.apply(lambda x: B1_valid(x), axis=1)

        #############################################
        # for B2 (days sick)
        # not a decimal if greater than 14
        # not negative
        # not greater than 1000
        #############################################

        def B2_valid(x):
            if x > 0 and x < 1000.0 and x > 14.0:
                return float(round(x)) == x

            elif x > 0 and x < 1000.0 and x < 14.0:
                return True
            else:
                return False

        if "B2a" in intl_pd.columns:
            intl_pd["B2"] = np.where(
                intl_pd["B2a"].isnull(), intl_pd["B2b"], intl_pd["B2a"]
            )
        else:
            intl_pd["B2"] = intl_pd["B2b"]

        intl_pd["B2"] = pd.to_numeric(intl_pd["B2"], errors="coerce")
        intl_pd["B2"] = intl_pd["B2"].astype("float")
        intl_pd["B2_Flag"] = intl_pd.B2.apply(lambda x: B2_valid(x))

        ##################################################
        # for B4 or E5 (number of people known sick; number of people in household)
        # not a decimal
        # not negative
        # not greater than 100 (B4) or 20 (E5)
        ##############################################

        def B4_valid(x):
            if x > 0 and x <= 100.0:
                return float(round(x)) == x

            else:
                return False

        intl_pd['B4'] = intl_pd.B4.apply(lambda x: np.nan if x == "" else x)
        intl_pd["B4"] = intl_pd["B4"].astype("float")
        intl_pd["B4_Flag"] = intl_pd.B4.apply(lambda x: B4_valid(x))

        def E5_valid(x):
            if x > 0 and x <= 20.0:
                return float(round(x)) == x

            else:
                return False

        intl_pd["E5"] = pd.to_numeric(intl_pd["E5"], errors="coerce")
        intl_pd["E5"] = intl_pd["E5"].astype("float")
        intl_pd["E5_Flag"] = intl_pd.E5.apply(lambda x: E5_valid(x))

        ##################################################################
        # Did NOT provide an INVALID response to two or more of B2, B4, or E5
        #
        ##################################################################
        def countInvalidAdditional(r):
            s = 0
            if pd.notna(r.B2) and not r.B2_Flag:
                s = s + 1
            if pd.notna(r.B4) and not r.B4_Flag:
                s = s + 1
            if pd.notna(r.E5) and not r.E5_Flag:
                s = s + 1
            return s

        intl_pd["invalid_num"] = intl_pd.apply(
            lambda x: countInvalidAdditional(x), axis=1
        )

        # eval_cols = ["B2_Flag", "B4_Flag", "E5_Flag"]
        # intl_pd["valid_num"] = intl_pd.loc[:, eval_cols].notna().sum(axis=1)
        intl_keep_pd = intl_pd
        print('deploy version intl_pd', intl_pd.shape)

        intl_keep_pd = intl_pd[
            (intl_pd.token.notna())
            & (intl_pd.DistributionChannel == "anonymous")
            & ((intl_pd.intro1 == 4.0) | (intl_pd.intro1 == 1.0))
            & ((intl_pd.intro2 == 4.0) | (intl_pd.intro2 == 1.0))
            & ((intl_pd.A1 == 23.0) | (intl_pd.A1 == 1.0))
            & (intl_pd.A2_Flag.notna())
            & (intl_pd.StartDate >= startdate_str)
            & (intl_pd.StartDate <= true_enddate_str)
            ]
        print('intl_keep_pd after filter', intl_keep_pd.shape)
        intl_keep_pd.reset_index(inplace=True)

        return intl_keep_pd

    def readEU_NEUResponse(eu_fname, neu_fname, survey_version):
        print("reading...")
        print(eu_fname)
        eu_pd = pd.read_csv(eu_fname)
        print("reading...")
        print(neu_fname)
        neu_pd = pd.read_csv(neu_fname)

        eu_pd["survey_region"] = "EU"
        neu_pd["survey_region"] = "ROW"

        if eu_pd.shape[0] < 3 and neu_pd.shape[0] < 3:
            print(eu_fname,neu_fname,'has no (few) records')
            return None
        else:
            intl_keep_pd = join_eu_neu_weight(
                eu_pd, neu_pd, survey_version=survey_version
            )
            return intl_keep_pd
    def full_completion_flag(df):
        if "E3" in df.columns and "E4" in df.columns:
            df["full_flag"] = (df["FR_CNT"]>= 2) & (df["Finished"].astype(float)>0) & (df["E3"].astype(float) > 0) & (df["E4"].astype(float) > 0)
            df["full_flag_int"] = df["full_flag"].astype(int)
        else:
            df["full_flag_int"] = 0
        if 'FL_31_DO_ModuleA' in df.columns:
            df['module'] = np.where(df['FL_31_DO_ModuleA'].fillna(-99).astype(int) > 0,'A',np.where(df['FL_31_DO_ModuleB'].fillna(-99).astype(int) > 0,'B',-99))
        else:
            df['module'] = -99
        return df
    def merge_pd(eu_fname, neu_fname, additional_cols, survey_version):

        intl_keep_pd = readEU_NEUResponse(
            eu_fname, neu_fname, survey_version=survey_version
        )
        if intl_keep_pd is not None:
            print('number of response for survey version', survey_version, ": ", intl_keep_pd.shape)

            intl_keep_pd["add_count"] = (
                # intl_keep_pd.loc[:, additional_cols].notna().sum(axis=1)
                intl_keep_pd.reindex(columns=additional_cols).notna().sum(axis=1)
            )
            intl_keep_pd["FR_CNT"] = intl_keep_pd.add_count - intl_keep_pd.invalid_num
            intl_keep_pd = full_completion_flag(intl_keep_pd)
            def join_country(df,ccl):
                rc = df[['A2_2_1','country']].fillna(-99).astype({'A2_2_1': 'int32','country':'str'})
                rci = rc.merge(ccl[['country_region_numeric','region_id']],left_on='A2_2_1',right_on='country_region_numeric',how='left')
                keep_index = rci.query('A2_2_1 > 0 and region_id.isin(@allow_iso_3)',engine='python').index
                # if str(survey_version) == 'v12':
                #     keep_index = response_country_ISO.query('A2_2_1 > 0 and region_id.isin(@allow_iso_v12exp)',engine='python').index
                dff = df.iloc[keep_index]
                return dff
            if Mapping_change and Current_day == "2021-12-21":
                print("Current day is change day 12212021")
                change_str = "2021-12-21 12:13:00"
                country_code_lookup1 = pd.read_csv(DATAROOT_PATH+'metadata/CTIS_survey_country_region_map_table_ver1.083021.csv')
                country_code_lookup2 = pd.read_csv(DATAROOT_PATH+'metadata/CTIS_survey_country_region_map_table_ver1.122821.csv')
                intl_keep_pd['start_time'] = pd.to_datetime(intl_keep_pd.StartDate)
                bc = intl_keep_pd[intl_keep_pd.start_time <= change_str]
                ac = intl_keep_pd[intl_keep_pd.start_time > change_str]
                intl_keep_pd_country_filtered = pd.concat([join_country(bc,country_code_lookup1),join_country(ac,country_code_lookup2)])


            elif not Mapping_change:
                if parse(Current_day) < parse("2021-12-21"):
                    country_code_lookup = pd.read_csv(DATAROOT_PATH+'metadata/CTIS_survey_country_region_map_table_ver1.083021.csv')
                elif parse(Current_day) < parse("2021-12-30"):
                    country_code_lookup = pd.read_csv(DATAROOT_PATH+'metadata/CTIS_survey_country_region_map_table_ver1.122821.csv')
                elif parse(Current_day) >= parse("2021-12-30"):
                    country_code_lookup = pd.read_csv(DATAROOT_PATH+'metadata/CTIS_survey_country_region_map_table_ver1.123021.csv')
                response_country = intl_keep_pd[['A2_2_1','country']].fillna(-99).astype({'A2_2_1': 'int32','country':'str'})
                response_country_ISO = response_country.merge(country_code_lookup[['country_region_numeric','region_id']],left_on='A2_2_1',right_on='country_region_numeric',how='left')
                keep_index = response_country_ISO.query('A2_2_1 > 0 and region_id.isin(@allow_iso_3)',engine='python').index
                # if str(survey_version) == 'v12':
                #     keep_index = response_country_ISO.query('A2_2_1 > 0 and region_id.isin(@allow_iso_v12exp)',engine='python').index
                intl_keep_pd_country_filtered = intl_keep_pd.loc[keep_index]
            print('number of response after country filter', survey_version, ": ", intl_keep_pd_country_filtered.shape)
            return intl_keep_pd_country_filtered
            # return intl_keep_pd
        else:
            return None

    keep_pd_dic = dict()
    for version in list_versions:
        eu_key = version+'_EU'
        neu_key = version+'_nonEU'
        if eu_key in res_filenames_by_version_id and neu_key in res_filenames_by_version_id:
            keep_pd_dic[version] = merge_pd(
                res_filenames_by_version_id[eu_key], res_filenames_by_version_id[neu_key], dict_additional_cols_by_version[version], survey_version=version
            )
    return keep_pd_dic


################################################################
# read survey response files and join different files together
# new logic
###############################################################
def readSurveyResponseFile_newlogic(startdate, enddate,fp=PROJECT_PATH):

    startdate_str = startdate.strftime("%Y-%m-%d %H:%M:%S")
    enddate_str = enddate.strftime("%Y-%m-%d %H:%M:%S")

    true_enddate = datetime.combine(
        enddate - timedelta(days=1), time(23, 59, 59), tzinfo=TZ
    )
    true_enddate_str = true_enddate.strftime("%Y-%m-%d %H:%M:%S")
    print("processing data range:", startdate_str, true_enddate_str)

    eu_fname = ""
    neu_fname = ""
    eu_fname2 = ""
    neu_fname2 = ""
    eu_fname3 = ""
    neu_fname3 = ""
    eu_fname4 = ""
    neu_fname4 = ""
    eu_fname5 = ""
    neu_fname5 = ""
    eu_fname6 = ""
    neu_fname6 = ""
    eu_fname6_1119 = ""
    neu_fname6_1119 = ""
    eu_fname7 = ""
    neu_fname7 = ""
    eu_fname8 = ""
    neu_fname8 = ""
    eu_fname9 = ""
    neu_fname9 = ""
    eu_fname10 = ""
    neu_fname10 = ""
    eu_fname11 = ""
    neu_fname11 = ""
    
    for fname in os.listdir(fp):
        rng_name = f"{startdate.date()}.{enddate.date()}"
        if rng_name in fname and "_EU_wave1" in fname:
            eu_fname = fp + fname

        elif rng_name in fname and "_nonEU_wave1" in fname:
            neu_fname = fp + fname

        elif rng_name in fname and "_EU_V2" in fname:
            eu_fname2 = fp + fname

        elif rng_name in fname and "_nonEU_V2" in fname:
            neu_fname2 = fp + fname
        elif rng_name in fname and "V3_EU" in fname:
            eu_fname3 = fp + fname

        elif rng_name in fname and "V3_nonEU" in fname:
            neu_fname3 = fp + fname
        elif rng_name in fname and "V4_EU" in fname:
            eu_fname4 = fp + fname

        elif rng_name in fname and "V4_nonEU" in fname:
            neu_fname4 = fp + fname

        elif rng_name in fname and "V5_EU" in fname:
            eu_fname5 = fp + fname

        elif rng_name in fname and "V5_nonEU" in fname:
            neu_fname5 = fp + fname
        
        elif rng_name in fname and "V6_EU" in fname and "1119" not in fname and "part" not in fname:
            eu_fname6 = fp + fname

        elif rng_name in fname and "V6_nonEU" in fname and "1119" not in fname and "part" not in fname:
            neu_fname6 = fp + fname

        elif rng_name in fname and "_V6_eu_1119" in fname: 
            eu_fname6_1119 = fp + fname
            
        elif rng_name in fname and "_V6_noneu_1119" in fname:
            neu_fname6_1119 = fp + fname

        elif rng_name in fname and "_V7_eu" in fname: 
            eu_fname7 = fp + fname
            
        elif rng_name in fname and "_V7_noneu" in fname:
            neu_fname7 = fp + fname
        
        elif rng_name in fname and "_V8_eu" in fname: 
            eu_fname8 = fp + fname
            
        elif rng_name in fname and "_V8_noneu" in fname:
            neu_fname8 = fp + fname
        
        elif rng_name in fname and "_V9_eu" in fname: 
            eu_fname9 = fp + fname
            
        elif rng_name in fname and "_V9_noneu" in fname:
            neu_fname9 = fp + fname
        
        elif rng_name in fname and "_V10_eu" in fname: 
            eu_fname10 = fp + fname
            
        elif rng_name in fname and "_V10_noneu" in fname:
            neu_fname10 = fp + fname
        
        elif rng_name in fname and "_V11_eu" in fname: 
            eu_fname11 = fp + fname

        elif rng_name in fname and "_V11_noneu" in fname:
            neu_fname11 = fp + fname

    # print(eu_fname10)
    # print(neu_fname10)

    ### function to join eu non_eu responses
    def join_eu_neu_weight(eu_pd, neu_pd, survey_version=4):

        #     startdate_str = "2020-05-01 00:00:00"
        #     true_enddate_str = "2020-05-02 23:59:59"

        ####################################################
        # From 05/01 to 05/15 survey v1, v2 has this issue
        ###################################################
        if ("B1_13" in eu_pd.columns) and ("B1_12" not in eu_pd.columns):
            eu_pd.rename(columns={"B1_13": "B1_12"}, inplace=True)

        intl_pd = pd.concat([eu_pd.loc[2:, :], neu_pd.loc[2:, :]], ignore_index=True)

        intl_pd["intro1"] = intl_pd.apply(
            lambda x: x.intro1_eu if pd.isna(x.intro1_noneu) else x.intro1_noneu, axis=1
        )

        intl_pd["intro2"] = intl_pd.apply(
            lambda x: x.intro2_eu if pd.isna(x.intro2_noneu) else x.intro2_noneu, axis=1
        )

        ## change datatype of columns for later filtering
        intl_pd["intro1"] = intl_pd["intro1"].astype("float")
        intl_pd["intro2"] = intl_pd["intro2"].astype("float")
        intl_pd["A1"] = intl_pd["A1"].astype("float")
        intl_pd["A2_2_1"] = intl_pd["A2_2_1"].astype("float")
        intl_pd["A2_2_2"] = intl_pd["A2_2_2"].astype("float")
        

        ## Count na in responses for B1
        B1_cols = []
        for col_name in intl_pd.columns:
            if col_name.startswith("B1_") and 'matrix' not in col_name:
                B1_cols.append(col_name)

        # print(B1_cols)

        ## Count na in responses for A2
        A2_cols = []
        for col_name in intl_pd.columns:
            if col_name.startswith("A2"):
                A2_cols.append(col_name)
        # A2_cols=['A2','A2_2_1', 'A2_2_2']
        intl_pd["A2NA"] = intl_pd.loc[:, A2_cols].isna().sum(axis=1)
        intl_pd["A2NA"] = intl_pd["A2NA"] + (intl_pd.loc[:, A2_cols] == -77).sum(axis=1)

        intl_pd["A2_Flag"] = intl_pd.A2NA.apply(
            lambda x: 1 if x < len(A2_cols) else np.nan
        )

        print("raw response shape", eu_pd.shape, neu_pd.shape, intl_pd.shape)

        ##############################################
        # B1 response is valid if :
        # - responded yes/no for at least one symptom
        # - did NOT respond yes for ALL symptoms
        #
        #############################################
        def countB1_yes(r):
            s = 0
            for e in r:
                if pd.notna(e) and int(float(e)) == 1:
                    s += 1
            return s

        intl_pd["B1_yes"] = intl_pd.loc[:, B1_cols].apply(
            lambda x: countB1_yes(x), axis=1
        )

        intl_pd["B1NA"] = intl_pd.loc[:, B1_cols].isna().sum(axis=1)
        intl_pd["B1NA"] = intl_pd["B1NA"] + (intl_pd.loc[:, B1_cols] == -77).sum(axis=1)
        intl_pd["B1_NAFlag"] = intl_pd.B1NA.apply(
            lambda x: 1 if x < len(B1_cols) else np.nan
        )

        def B1_valid(r):
            if pd.notna(r.B1_NAFlag) and r.B1_yes < len(B1_cols):
                return 1
            else:
                return np.nan

        intl_pd["B1_Flag"] = intl_pd.apply(lambda x: B1_valid(x), axis=1)

        #############################################
        # for B2 (days sick)
        # not a decimal if greater than 14
        # not negative
        # not greater than 1000
        #############################################

        def B2_valid(x):
            if x > 0 and x < 1000.0 and x > 14.0:
                return float(round(x)) == x

            elif x > 0 and x < 1000.0 and x < 14.0:
                return True
            else:
                return False

        if "B2a" in intl_pd.columns:
            intl_pd["B2"] = np.where(
                intl_pd["B2a"].isnull(), intl_pd["B2b"], intl_pd["B2a"]
            )
        else:
            intl_pd["B2"] = intl_pd["B2b"]

        intl_pd["B2"] = pd.to_numeric(intl_pd["B2"], errors="coerce")
        intl_pd["B2"] = intl_pd["B2"].astype("float")
        intl_pd["B2_Flag"] = intl_pd.B2.apply(lambda x: B2_valid(x))

        ##################################################
        # for B4 or E5 (number of people known sick; number of people in household)
        # not a decimal
        # not negative
        # not greater than 100 (B4) or 20 (E5)
        ##############################################

        def B4_valid(x):
            if x > 0 and x <= 100.0:
                return float(round(x)) == x

            else:
                return False
                
        intl_pd['B4']=intl_pd.B4.apply(lambda x: np.nan if x=="" else x)
        intl_pd["B4"] = intl_pd["B4"].astype("float")
        intl_pd["B4_Flag"] = intl_pd.B4.apply(lambda x: B4_valid(x))

        def E5_valid(x):
            if x > 0 and x <= 20.0:
                return float(round(x)) == x

            else:
                return False

        intl_pd["E5"] = pd.to_numeric(intl_pd["E5"], errors="coerce")
        intl_pd["E5"] = intl_pd["E5"].astype("float")
        intl_pd["E5_Flag"] = intl_pd.E5.apply(lambda x: E5_valid(x))

        ##################################################################
        # Did NOT provide an INVALID response to two or more of B2, B4, or E5
        #
        ##################################################################
        def countInvalidAdditional(r):
            s = 0
            if pd.notna(r.B2) and not r.B2_Flag:
                s = s + 1
            if pd.notna(r.B4) and not r.B4_Flag:
                s = s + 1
            if pd.notna(r.E5) and not r.E5_Flag:
                s = s + 1
            return s

        intl_pd["invalid_num"] = intl_pd.apply(
            lambda x: countInvalidAdditional(x), axis=1
        )

        # eval_cols = ["B2_Flag", "B4_Flag", "E5_Flag"]
        # intl_pd["valid_num"] = intl_pd.loc[:, eval_cols].notna().sum(axis=1)

        intl_keep_pd = intl_pd
        intl_keep_pd = intl_pd[
            (intl_pd.token.notna())
            & (intl_pd.DistributionChannel == "anonymous")
            & ((intl_pd.intro1 == 4.0) | (intl_pd.intro1 == 1.0))
            & ((intl_pd.intro2 == 4.0) | (intl_pd.intro2 == 1.0))
            & ((intl_pd.A1 == 23.0) | (intl_pd.A1 == 1.0))
            & (intl_pd.A2_Flag.notna())
            & (intl_pd.StartDate >= startdate_str)
            & (intl_pd.StartDate <= true_enddate_str)
        ]
        print('intl_keep_pd_old', intl_keep_pd.shape)
        intl_keep_pd.reset_index(inplace=True)

        return intl_keep_pd

    def readEU_NEUResponse(eu_fname, neu_fname, survey_version=4):
        print("reading...")
        print(eu_fname)
        eu_pd = pd.read_csv(eu_fname)
        print("reading...")
        print(neu_fname)
        neu_pd = pd.read_csv(neu_fname)

        eu_pd["survey_region"] = "EU"
        neu_pd["survey_region"] = "ROW"

        if eu_pd.shape[0] < 10 and neu_pd.shape[0] < 10:
            return None
        else:
            intl_keep_pd = join_eu_neu_weight(
                eu_pd, neu_pd, survey_version=survey_version
            )
            return intl_keep_pd

    def merge_pd(eu_fname, neu_fname, additional_cols, survey_version=4):

        intl_keep_pd = readEU_NEUResponse(
            eu_fname, neu_fname, survey_version=survey_version
        )
        

        if intl_keep_pd is not None:
            print('number of response for survey version',survey_version,": ",intl_keep_pd.shape)

            intl_keep_pd["add_count"] = (
                # intl_keep_pd.loc[:, additional_cols].notna().sum(axis=1)
                intl_keep_pd.reindex(columns = additional_cols).notna().sum(axis=1)
            )
            intl_keep_pd["FR_CNT"] = intl_keep_pd.add_count - intl_keep_pd.invalid_num

            return intl_keep_pd
        else:
            return None

    ##########################################################
    # generate and write token id files
    ##########################################################
    additional_cols_v1 = [
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7a",
        "B7b",
        "B8",
        "C0_1",
        "C0_2",
        "C0_3",
        "C0_4",
        "C0_5",
        "C0_6",
        "C1_m",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "D1",
        "D2",
        "D3",
        "D4",
        "E2",
        "E3",
        "E4",
        "E5",
        "B1_Flag",
    ]
    additional_cols_v2 = [
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7a",
        "B7b",
        "B8",
        "C0_1",
        "C0_2",
        "C0_3",
        "C0_4",
        "C0_5",
        "C0_6",
        "C1_m",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "E2",
        "E3",
        "E4",
        "E5",
        "B1_Flag",
    ]
    additional_cols_v3_v4 = [
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7a",
        "B7b",
        "B7c",
        "B8",
        "C0_1",
        "C0_2",
        "C0_3",
        "C0_4",
        "C0_5",
        "C0_6",
        "C1_m",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "E2",
        "E3",
        "E4",
        "E5",
        "F1",
        "F2",
        "F3",
        "B1_Flag",
    ]

    additional_cols_v5 = [
        "B2b",
        "B1b_x1",
        "B1b_x2",
        "B1b_x3",
        "B1b_x4",
        "B1b_x5",
        "B1b_x6",
        "B1b_x7",
        "B1b_x8",
        "B1b_x9",
        "B1b_x10",
        "B1b_x11",
        "B1b_x12",
        "B1b_x13",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7a",
        "B7b",
        "B7c",
        "B8",
        "B9",
        "B10",
        "B11a",
        "B11b",
        "B11c",
        "B12a_1",
        "B12a_2",
        "B12a_3",
        "B12a_4",
        "B12a_5",
        "B12a_6",
        "B12b_1",
        "B12b_2",
        "B12b_3",
        "B12b_4",
        "B12b_5",
        "B12b_6",
        "B12c_1",
        "B12c_2",
        "B12c_3",
        "B12c_4",
        "B12c_5",
        "B12c_6",
        "B13_1",
        "B13_2",
        "B13_3",
        "B13_4",
        "B13_5",
        "B13_6",
        "B13_7",
        "B14_1",
        "B14_2",
        "B14_3",
        "B14_4",
        "B14_5",
        "C0_1",
        "C0_2",
        "C0_3",
        "C0_4",
        "C0_5",
        "C0_6",
        "C1_m",
        "C2",
        "C7",
        "C8",
        "C3",
        "C5",
        "C6",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6_1",
        "D6_2",
        "D6_3",
        "D7",
        "D8",
        "D9",
        "D10a",
        "D10b",
        "E3",
        "E4",
        "E6",
        "E2",
        "E5",
        "E7",
        "F1",
        "F2_1",
        "F2_2",
        "F3_au",
        "F3_de",
        "B1_Flag",
    ]

    additional_cols_v6 = [ 
        "B2b",
        "B1b_x1",
        "B1b_x2",
        "B1b_x3",
        "B1b_x4",
        "B1b_x5",
        "B1b_x6",
        "B1b_x7",
        "B1b_x8",
        "B1b_x9",
        "B1b_x10",
        "B1b_x11",
        "B1b_x12",
        "B1b_x13",
        "B1b_x14",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7a",
        "B7b",
        "B8",
        "B9",
        "B10",
        "B11a",
        "B11b",
        "B12a_1",
        "B12a_2",
        "B12a_3",
        "B12a_4",
        "B12a_5",
        "B12a_6",
        "B12b_1",
        "B12b_2",
        "B12b_3",
        "B12b_4",
        "B12b_5",
        "B12b_6",
        "B13_1",
        "B13_2",
        "B13_3",
        "B13_4",
        "B13_5",
        "B13_6",
        "B13_7",
        "B14_1",
        "B14_2",
        "B14_3",
        "B14_4",
        "B14_5",
        "C0_matrix_1",
        "C0_matrix_2",
        "C0_matrix_3",
        "C0_matrix_4",
        "C0_matrix_5",
        "C0_matrix_6",
        "C1_m",
        "C2",
        "C7",
        "C8",
        "C3",
        "C5",
        "C6",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6_1",
        "D6_2",
        "D6_3",
        "D7",
        "D8",
        "D9",
        "D10a",
        "D10b",
        "E3",
        "E4",
        "E6",
        "E2",
        "E5",
        "E7",
        "F1",
        "F2_1",
        "F2_2",
        "F3_au",
        "F3_de",
        "B1_Flag"
    ]

    additional_cols_v6_1119 = [
        "B2b",
        "B1b_x1",
        "B1b_x2",
        "B1b_x3",
        "B1b_x4",
        "B1b_x5",
        "B1b_x6",
        "B1b_x7",
        "B1b_x8",
        "B1b_x9",
        "B1b_x10",
        "B1b_x11",
        "B1b_x12",
        "B1b_x13",
        "B1b_x14",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7a",
        "B7c",
        "B8",
        "B9",
        "B10",
        "B11b",
        "B12b_1",
        "B12b_2",
        "B12b_3",
        "B12b_4",
        "B12b_5",
        "B12b_6",
        "B13_1",
        "B13_2",
        "B13_3",
        "B13_4",
        "B13_5",
        "B13_6",
        "B13_7",
        "B14_1",
        "B14_2",
        "B14_3",
        "B14_4",
        "B14_5",
        "C13_1",
        "C13_2",
        "C13_3",
        "C13_4",
        "C13_5",
        "C13_6",
        "C0_1",
        "C0_2",
        "C0_3",
        "C0_4",
        "C0_5",
        "C0_6",
        "C1_m",
        "C2",
        "C7",
        "C8",
        "C3",
        "C5",
        "C6",
        "C14",
        "C9",
        "C10",
        "C11_no_1",
        "C11_no_2",
        "C11_no_3",
        "C11_no_4",
        "C11_no_5",
        "C11_no_6",
        "C11_no_7",
        "C11_no_7_TEXT",
        "C11_unsure_1",
        "C11_unsure_2",
        "C11_unsure_3",
        "C11_unsure_4",
        "C11_unsure_5",
        "C11_unsure_6",
        "C11_unsure_7",
        "C11_unsure_7_TEXT",
        "C12",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6_1",
        "D6_2",
        "D6_3",
        "D7",
        "D8",
        "D9",
        "D10a",
        "D10b",
        "E3",
        "E4",
        "E6",
        "E2",
        "E5",
        "E7",
        "F1",
        "F2_1",
        "F2_2",
        "F3_au",
        "F3_de",
        "B1_Flag"
    ]

    additional_cols_v7 = [
        "A3",
        "B2b",
        "B1b_x1",
        "B1b_x2",
        "B1b_x3",
        "B1b_x4",
        "B1b_x5",
        "B1b_x6",
        "B1b_x7",
        "B1b_x8",
        "B1b_x9",
        "B1b_x10",
        "B1b_x11",
        "B1b_x12",
        "B1b_x13",
        "B1b_x14",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7c",
        "B8",
        "B9",
        "B10",
        "B11b",
        "B12b_1",
        "B12b_2",
        "B12b_3",
        "B12b_4",
        "B12b_5",
        "B12b_6",
        "V1",
        "V2",
        "V3",
        "V4_1",
        "V4_2",
        "V4_3",
        "V4_4",
        "V4_5",
        "B13_1",
        "B13_2",
        "B13_3",
        "B13_4",
        "B13_5",
        "B13_6",
        "B13_7",
        "B14_1",
        "B14_2",
        "B14_3",
        "B14_4",
        "B14_5",
        "C13_1",
        "C13_2",
        "C13_3",
        "C13_4",
        "C13_5",
        "C13_6",
        "C0_1",
        "C0_2",
        "C0_3",
        "C0_4",
        "C0_5",
        "C0_6",
        "C1_m",
        "C2",
        "C7",
        "C8",
        "C3",
        "C5",
        "C6",
        "C14",
        "C9",
        "C10",
        "C11_no_1",
        "C11_no_2",
        "C11_no_3",
        "C11_no_4",
        "C11_no_5",
        "C11_no_6",
        "C11_no_7",
        "C11_no_7_TEXT",
        "C11_unsure_1",
        "C11_unsure_2",
        "C11_unsure_3",
        "C11_unsure_4",
        "C11_unsure_5",
        "C11_unsure_6",
        "C11_unsure_7",
        "C11_unsure_7_TEXT",
        "C12",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6_1",
        "D6_2",
        "D6_3",
        "D7",
        "D8",
        "D9",
        "D10a",
        "D10b",
        "E3",
        "E4",
        "E6",
        "E2",
        "E5",
        "E7",
        "F1",
        "F2_1",
        "F2_2",
        "F3_au",
        "F3_de",
        "B1_Flag"
    ]

    additional_cols_v8 = [
        "A3",
        "B2b",
        "B1b_x1",
        "B1b_x2",
        "B1b_x3",
        "B1b_x4",
        "B1b_x5",
        "B1b_x6",
        "B1b_x7",
        "B1b_x8",
        "B1b_x9",
        "B1b_x10",
        "B1b_x11",
        "B1b_x12",
        "B1b_x13",
        "B1b_x14",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7c",
        "B8",
        "B9",
        "B10",
        "B11b",
        "B12b_1",
        "B12b_2",
        "B12b_3",
        "B12b_4",
        "B12b_5",
        "B12b_6",
        "V1",
        "V2",
        "V3",
        "V4_1",
        "V4_2",
        "V4_3",
        "V4_4",
        "V4_5",
        "V9",
        "B13_1",
        "B13_2",
        "B13_3",
        "B13_4",
        "B13_5",
        "B13_6",
        "B13_7",
        "B14_1",
        "B14_2",
        "B14_3",
        "B14_4",
        "B14_5",
        "C13_1",
        "C13_2",
        "C13_3",
        "C13_4",
        "C13_5",
        "C13_6",
        "C0_1",
        "C0_2",
        "C0_3",
        "C0_4",
        "C0_5",
        "C0_6",
        "C1_m",
        "C2",
        "C7",
        "C8",
        "C3",
        "C5",
        "C6",
        "C14",
        "C9",
        "C10",
        "C11_no_1",
        "C11_no_2",
        "C11_no_3",
        "C11_no_4",
        "C11_no_5",
        "C11_no_6",
        "C11_no_7",
        "C11_no_7_TEXT",
        "C11_unsure_1",
        "C11_unsure_2",
        "C11_unsure_3",
        "C11_unsure_4",
        "C11_unsure_5",
        "C11_unsure_6",
        "C11_unsure_7",
        "C11_unsure_7_TEXT",
        "C12",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6_1",
        "D6_2",
        "D6_3",
        "D7",
        "D8",
        "D9",
        "D10a",
        "D10b",
        "E3",
        "E4",
        "E6",
        "E2",
        "E5",
        "E7",
        "F1",
        "F2_1",
        "F2_2",
        "F3_au",
        "F3_de",
        "B1_Flag"
    ]

    additional_cols_v9 = [
        "A3",
        "B2b",
        "B1b_x1",
        "B1b_x2",
        "B1b_x3",
        "B1b_x4",
        "B1b_x5",
        "B1b_x6",
        "B1b_x7",
        "B1b_x8",
        "B1b_x9",
        "B1b_x10",
        "B1b_x11",
        "B1b_x12",
        "B1b_x13",
        "B1b_x14",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7c",
        "B8",
        "B9",
        "B10",
        "B11b",
        "B12b_1",
        "B12b_2",
        "B12b_3",
        "B12b_4",
        "B12b_5",
        "B12b_6",
        "V1",
        "V2",
        "V2a",
        "V3",
        "V4_1",
        "V4_6",
        "V4_3",
        "V4_4",
        "V4_5",
        "V5a_1",
        "V5a_2",
        "V5a_3",
        "V5a_4",
        "V5a_5",
        "V5a_6",
        "V5a_7",
        "V5a_8",
        "V5a_9",
        "V5b_1",
        "V5b_2",
        "V5b_3",
        "V5b_4",
        "V5b_5",
        "V5b_6",
        "V5b_7",
        "V5b_8",
        "V5b_9",
        "V5c_1",
        "V5c_2",
        "V5c_3",
        "V5c_4",
        "V5c_5",
        "V5c_6",
        "V5c_7",
        "V5c_8",
        "V5c_9",
        "V5d_1",
        "V5d_2",
        "V5d_3",
        "V5d_4",
        "V5d_5",
        "V5d_6",
        "V5d_7",
        "V5d_8",
        "V5d_9",
        "V6_1",
        "V6_2",
        "V6_3",
        "V6_4",
        "V6_5",
        "V6_6",
        "V6_7",
        "V9",
        "V10_1",
        "V10_2",
        "V10_3",
        "V10_4",
        "V10_5",
        "V10_6",
        "V10_7",
        "V10_8",
        "V10_9",
        "V10_10",
        "V11",
        "V12",
        "B13_1",
        "B13_2",
        "B13_3",
        "B13_4",
        "B13_5",
        "B13_6",
        "B13_7",
        "B14_1",
        "B14_2",
        "B14_3",
        "B14_4",
        "B14_5",
        "C13_1",
        "C13_2",
        "C13_3",
        "C13_4",
        "C13_5",
        "C13_6",
        "C0_1",
        "C0_2",
        "C0_3",
        "C0_4",
        "C0_5",
        "C0_6",
        "C1_m",
        "C2",
        "C7",
        "C8",
        "C3",
        "C5",
        "C6",
        "C14",
        "C9"
        "C9a",
        "C10",
        "C12",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6_1",
        "D6_2",
        "D6_3",
        "D7",
        "D8",
        "D9",
        "D10a",
        "D10b",
        "E3",
        "E4",
        "E6",
        "E2",
        "E5",
        "E7",
        "F1",
        "F2_1",
        "F2_2",
        "F3_au",
        "F3_de",
        "B1_Flag"
    ]

    additional_cols_v10 = [
        "A3",
        "B2b",
        "B1b_x1",
        "B1b_x2",
        "B1b_x3",
        "B1b_x4",
        "B1b_x5",
        "B1b_x6",
        "B1b_x7",
        "B1b_x8",
        "B1b_x9",
        "B1b_x10",
        "B1b_x11",
        "B1b_x12",
        "B1b_x13",
        "B1b_x14",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7c",
        "B8",
        "B9",
        "B10",
        "B11b",
        "B12b_1",
        "B12b_2",
        "B12b_3",
        "B12b_4",
        "B12b_5",
        "B12b_6",
        "V1",
        "V2",
        "V2a",
        "V3",
        "V4_1",
        "V4_6",
        "V4_3",
        "V4_4",
        "V4_5",
        "V5a_1",
        "V5a_2",
        "V5a_3",
        "V5a_4",
        "V5a_5",
        "V5a_6",
        "V5a_7",
        "V5a_8",
        "V5a_9",
        "V5a_10",
        "V5b_1",
        "V5b_2",
        "V5b_3",
        "V5b_4",
        "V5b_5",
        "V5b_6",
        "V5b_7",
        "V5b_8",
        "V5b_9",
        "V5b_10",
        "V5c_1",
        "V5c_2",
        "V5c_3",
        "V5c_4",
        "V5c_5",
        "V5c_6",
        "V5c_7",
        "V5c_8",
        "V5c_9",
        "V5c_10",
        "V5d_1",
        "V5d_2",
        "V5d_3",
        "V5d_4",
        "V5d_5",
        "V5d_6",
        "V5d_7",
        "V5d_8",
        "V5d_9",
        "V5d_10",
        "V6_1",
        "V6_2",
        "V6_3",
        "V6_4",
        "V6_5",
        "V6_6",
        "V6_7",
        "V9",
        "V10_1",
        "V10_2",
        "V10_3",
        "V10_4",
        "V10_5",
        "V10_6",
        "V10_7",
        "V10_8",
        "V10_9",
        "V10_10",
        "V11",
        "V12",
        "V13",
        "V15",
        "V16",
        "B13_1",
        "B13_2",
        "B13_3",
        "B13_4",
        "B13_5",
        "B13_6",
        "B13_7",
        "B14_1",
        "B14_2",
        "B14_3",
        "B14_4",
        "B14_5",
        "C13_1",
        "C13_2",
        "C13_3",
        "C13_4",
        "C13_5",
        "C13_6",
        "C0_1",
        "C0_2",
        "C0_3",
        "C0_4",
        "C0_5",
        "C0_6",
        "C1_m",
        "C2",
        "C7",
        "C8",
        "C3",
        "C5",
        "C6",
        "C14",
        "C9"
        "C9a",
        "C10",
        "C12",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6_1",
        "D6_2",
        "D6_3",
        "D7",
        "D8",
        "D9",
        "D10a",
        "D10b",
        "E3",
        "E4",
        "E6",
        "E2",
        "E5",
        "E7",
        "B1_Flag"
    ]

    additional_cols_v11 = [
        "A3_5",
        "A3_6",
        "A3_7",
        "B0",
        "G1",
        "G2",
        "G3",
        "H1",
        "H2",
        "H3",
        "I1",
        "I2",
        "I5",
        "I6_1",
        "I6_2",
        "I6_3",
        "I6_4",
        "I6_5",
        "I6_6",
        "I6_7",
        "I6_8",
        "I7",
        "I8",
        "I9_noneu",
        "I10_noneu_1",
        "I10_noneu_2",
        "I10_noneu_3",
        "I10_noneu_4",
        "I10_noneu_5",
        "J1",
        "J2",
        "K1",
        "V18a",
        "V18b",
        "V19",
        "B8a",
        "B12d",
        "D7a",
        "E7a",
        "E8",
        "V3a",
        "V15a",
        "V16a",
        "B2b",
        "B1b_x2",
        "B1b_x3",
        "B1b_x4",
        "B1b_x5",
        "B1b_x6",
        "B1b_x7",
        "B1b_x8",
        "B1b_x9",
        "B1b_x11",
        "B3",
        "B4",
        "B7c",
        "B8",
        "B11b",
        "B12b_1",
        "B12b_2",
        "B12b_3",
        "B12b_4",
        "B12b_5",
        "B12b_6",
        "V1",
        "V2",
        "V3",
        "V5a_1",
        "V5a_2",
        "V5a_3",
        "V5a_4",
        "V5a_5",
        "V5a_6",
        "V5a_7",
        "V5a_8",
        "V5a_9",
        "V5a_10",
        "V5b_1",
        "V5b_2",
        "V5b_3",
        "V5b_4",
        "V5b_5",
        "V5b_6",
        "V5b_7",
        "V5b_8",
        "V5b_9",
        "V5b_10",
        "V5c_1",
        "V5c_2",
        "V5c_3",
        "V5c_4",
        "V5c_5",
        "V5c_6",
        "V5c_7",
        "V5c_8",
        "V5c_9",
        "V5c_10",
        "V5d_1",
        "V5d_2",
        "V5d_3",
        "V5d_4",
        "V5d_5",
        "V5d_6",
        "V5d_7",
        "V5d_8",
        "V5d_9",
        "V5d_10",
        "V6_1",
        "V6_2",
        "V6_3",
        "V6_4",
        "V6_5",
        "V6_6",
        "V6_7",
        "V9",
        "V10_1",
        "V10_2",
        "V10_3",
        "V10_4",
        "V10_5",
        "V10_6",
        "V10_7",
        "V10_8",
        "V10_9",
        "V10_10",
        "V11",
        "V12",
        "V15",
        "V16",
        "B13_1",
        "B13_2",
        "B13_3",
        "B13_4",
        "B13_5",
        "B13_6",
        "B13_7",
        "B14_1",
        "B14_2",
        "B14_3",
        "B14_4",
        "B14_5",
        "C13_1",
        "C13_2",
        "C13_3",
        "C13_4",
        "C13_5",
        "C13_6",
        "C0_1",
        "C0_2",
        "C0_3",
        "C0_4",
        "C0_5",
        "C0_6",
        "C5",
        "C14",
        "C9"
        "C10",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D7",
        "D10a",
        "E3",
        "E4",
        "E6",
        "E2",
        "E5",
        "E7",
        "B1_Flag"
    ]

    keep_pd_dic = dict()
    if len(eu_fname) > 0 and len(neu_fname) > 0:
        keep_pd_dic[1] = merge_pd(
            eu_fname, neu_fname, additional_cols_v1, survey_version=1
        )
    if len(eu_fname2) > 0 and len(neu_fname2) > 0:
        keep_pd_dic[2] = merge_pd(
            eu_fname2, neu_fname2, additional_cols_v2, survey_version=2
        )
    if len(eu_fname3) > 0 and len(neu_fname3) > 0:
        keep_pd_dic[3] = merge_pd(
            eu_fname3, neu_fname3, additional_cols_v3_v4, survey_version=3
        )
    if len(eu_fname4) > 0 and len(neu_fname4) > 0:
        keep_pd_dic[4] = merge_pd(
            eu_fname4, neu_fname4, additional_cols_v3_v4, survey_version=4
        )
    if len(eu_fname5) > 0 and len(neu_fname5) > 0:
        keep_pd_dic[5] = merge_pd(
            eu_fname5, neu_fname5, additional_cols_v5, survey_version=5
        )
    if len(eu_fname6) > 0 and len(neu_fname6) > 0:
        keep_pd_dic[6] = merge_pd(
            eu_fname6, neu_fname6, additional_cols_v6, survey_version=6
        )
        
    if len(eu_fname6_1119) > 0 and len(neu_fname6_1119) > 0:
        keep_pd_dic['6b'] = merge_pd(
            eu_fname6_1119, neu_fname6_1119, additional_cols_v6_1119, survey_version='6b'
        )
    
    if len(eu_fname7) > 0 and len(neu_fname7) > 0:
        keep_pd_dic[7] = merge_pd(
            eu_fname7, neu_fname7, additional_cols_v7, survey_version=7
        )
    
    if len(eu_fname8) > 0 and len(neu_fname8) > 0:
        keep_pd_dic[8] = merge_pd(
            eu_fname8, neu_fname8, additional_cols_v8, survey_version=8
        )
    
    if len(eu_fname9) > 0 and len(neu_fname9) > 0:
        keep_pd_dic[9] = merge_pd(
            eu_fname9, neu_fname9, additional_cols_v9, survey_version=9
        )

    if len(eu_fname10) > 0 and len(neu_fname10) > 0:
        keep_pd_dic[10] = merge_pd(
            eu_fname10, neu_fname10, additional_cols_v10, survey_version=10
        )
    
    if len(eu_fname11) > 0 and len(neu_fname11) > 0:
        keep_pd_dic[11] = merge_pd(
            eu_fname11, neu_fname11, additional_cols_v11, survey_version=11
        )


    return keep_pd_dic
