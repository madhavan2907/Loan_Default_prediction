# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 2020 - 18:22:58
@author: Sethumadhavan Aravindakshan
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st
import pandas as pd

# from xgboost import XGBRFClassifier()

pickle_in = open("XGB_HYP.sav", "rb")
hybrid_model = pickle.load(pickle_in)


def main():
    st.title("Probability of Default prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Probability of Default predictor ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    page_bg_img = '''
    <style>
    body {
    background-image: url("https://i.pinimg.com/originals/85/6f/31/856f31d9f475501c7552c97dbe727319.jpg");
    background-size: cover;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Numerical Inputs
    loan_amnt = st.number_input("Enter the loan amount")
    funded_amnt = st.number_input("Enter the funded amount")
    funded_amnt_inv = st.number_input("Enter the The total amount committed by investors for that loan at that point in time")
    int_rate = st.number_input("Enter the interest rate %")
    installment = st.number_input("Enter the installment amount")
    dti = st.number_input("Enter the Devt to Income ratio")
    delinq_2yrs = st.number_input("Enter the number of times the user was delinquent in last 2 years")
    inq_last_6mths = st.number_input("Number of inquiries made in last 6 months")
    pub_rec = st.number_input("Number of derogatory public records")
    revol_bal = st.number_input("Enter the total credit revolving balance")
    out_prncp = st.number_input("Enter the remaining outstanding principal for total amount funded")
    out_prncp_inv = st.number_input("Enter the remaining outstanding principal for total amount funded by the investors")
    total_pymnt = st.number_input("Enter the Payments received to date for total amount funded")
    total_pymnt_inv = st.number_input("Payments received to date for portion of total amount funded by investors")
    total_rec_prncp = st.number_input("Enter the Principal received to date")
    total_rec_int = st.number_input("Enter the Interest received to date")
    total_rec_late_fee = st.number_input("Enter the Late fees received to date")
    recoveries = st.number_input("Enter the post charge off gross recovery")
    annual_inc = st.number_input("Enter the Annual income")
    collection_recovery_fee = st.number_input("Enter the post charge off collection fee")
    last_pymnt_amnt = st.number_input("Enter the last paymenet amount")
    #Categorical Inputs
    home_ownership = st.radio('Select the appropriate options',('RENT','OWN', 'MORTGAGE', 'OTHER'))
    purpose = st.radio('Select the purpose of loan enquiry',('credit_card','car','small_business','other','wedding','debt_consolidation','home_improvement','major_purchase','medical','moving','vacation','house','renewable_energy','educational'))
    title = st.radio('Select the Title of the expense',('Other_expenses','Debt_consolidations','Credit_card_expenses','Home_expenses','Medical_expenses','Major_purchases'))
    addr_state = st.radio('Select the region',('West','South','Mid_west','Nort_East'))
    earliest_cr_line = st.radio('Select the earliest credit_line time frame',('1980-99', '2000-05', '2006-10', '1960-79', '1940-59', '2011-15'))
    initial_list_status = st.radio('Select the Initial listing status',('f', 'w'))
    last_pymnt_d = st.radio('Select the last payement quarter',('Q1-2015','Q2-2013','Q2-2014','Q1-2016','Q2-2012','Q4-2012','Q3-2013','Q3-2012','Q4-2013','Q3-2015','Q4-2014','Q1-2014','Q3-2014','Q2-2015','Q1-2013','Q1-2012','Q4-2015','Q4-2011','Q3-2011','Q2-2011','Q1-2011','Q4-2010','Q3-2010','Q2-2010','Q1-2010','Q4-2009','Q3-2009','Q2-2009','Q1-2009','Q4-2008','Q3-2008','Q2-2008','Q1-2008'))
    next_pymnt_d = st.radio('Select the next payement quarter',('Unknown', 'Feb-2016', 'Jan-2016', 'Mar-2016'))
    last_credit_pull_d = st.radio('Select the last credit pull quarter',('Q1-2016','Q3-2013','Q1-2015','Q3-2015','Q4-2014','Q3-2012','Q1-2013','Q4-2015','Q4-2012','Q1-2014','Q2-2015','Q2-2014','Q2-2013','Q2-2012','Q3-2014','Q4-2013','Q1-2012','Q4-2011','Q3-2011','Q2-2011','Q1-2011','Q4-2010','Q3-2010','Q2-2010','Q1-2010','Q3-2007','Q4-2009','Q3-2009','Q2-2009','Q1-2009','Q4-2008','Q2-2008','Q3-2008','Q1-2008','Q4-2007','Q2-2007'))
    application_type = st.radio('Select the Application type',('INDIVIDUAL', 'JOINT'))
    term = st.radio('Select the term', ('36.0', '60.0'))
    grade = st.radio('Select the Grade',('A','B','C','D','E','F','G'))
    sub_grade = st.radio('Select the Sub-Grade',('A1','B1','C1','D1','E1','F1','G1','A2','B2','C2','D2','E2','F2','G2','A3','B3','C3','D3','E3','F3','G3','A4','B4','C4','D4','E4','F4','G4','A5','B5','C5','D5','E5','F5','G5'))
    emp_length = st.radio('Select the Employment length',('Unknown','1 Yr','2 Yrs','3 Yrs','4 Yrs','5 Yrs','6 Yrs','7 Yrs','8 Yrs','9 Yrs','10 & 10+ Yrs'))
    verification_status = st.radio('Select the Verification status',('Not Verified','Verified','Source Verified'))

    if term == '36.0':
        term = 36.0
    else:
        term = 60.0

    if grade == 'A':
        grade = 1
    elif grade == 'B':
        grade = 2
    elif grade == 'C':
        grade = 3
    elif grade == 'D':
        grade = 4
    elif grade == 'E':
        grade = 5
    elif grade == 'F':
        grade = 6
    elif grade == 'G':
        grade = 7

    if sub_grade == 'A1':
        sub_grade = 1.1
    elif sub_grade == 'B1':
        sub_grade = 2.1
    elif sub_grade == 'C1':
        sub_grade = 3.1
    elif sub_grade == 'D1':
        sub_grade = 4.1
    elif sub_grade == 'E1':
        sub_grade = 5.1
    elif sub_grade == 'F1':
        sub_grade = 6.1
    elif sub_grade == 'G1':
        sub_grade = 7.1
    elif sub_grade == 'A2':
        sub_grade = 1.2
    elif sub_grade == 'B2':
        sub_grade = 2.2
    elif sub_grade == 'C2':
        sub_grade = 3.2
    elif sub_grade == 'D2':
        sub_grade = 4.2
    elif sub_grade == 'E2':
        sub_grade = 5.2
    elif sub_grade == 'F2':
        sub_grade = 6.2
    elif sub_grade == 'G2':
        sub_grade = 7.2
    elif sub_grade == 'A3':
        sub_grade = 1.3
    elif sub_grade == 'B3':
        sub_grade = 2.3
    elif sub_grade == 'C3':
        sub_grade = 3.3
    elif sub_grade == 'D3':
        sub_grade = 4.3
    elif sub_grade == 'E3':
        sub_grade = 5.3
    elif sub_grade == 'F3':
        sub_grade = 6.3
    elif sub_grade == 'G3':
        sub_grade = 7.3
    elif sub_grade == 'A4':
        sub_grade = 1.4
    elif sub_grade == 'B4':
        sub_grade = 2.4
    elif sub_grade == 'C4':
        sub_grade = 3.4
    elif sub_grade == 'D4':
        sub_grade = 4.4
    elif sub_grade == 'E4':
        sub_grade = 5.4
    elif sub_grade == 'F4':
        sub_grade = 6.4
    elif sub_grade == 'G4':
        sub_grade = 7.4
    elif sub_grade == 'A5':
        sub_grade = 1.5
    elif sub_grade == 'B5':
        sub_grade = 2.5
    elif sub_grade == 'C5':
        sub_grade = 3.5
    elif sub_grade == 'D5':
        sub_grade = 4.5
    elif sub_grade == 'E5':
        sub_grade = 5.5
    elif sub_grade == 'F5':
        sub_grade = 6.5
    elif sub_grade == 'G5':
        sub_grade = 7.5

    if emp_length == 'Unknown':
        emp_length = 0
    elif emp_length == '1 Yr':
        emp_length = 1
    elif emp_length == '2 Yrs':
        emp_length = 2
    elif emp_length == '3 Yrs':
        emp_length = 3
    elif emp_length == '4 Yrs':
        emp_length = 4
    elif emp_length == '5 Yrs':
        emp_length = 5
    elif emp_length == '6 Yrs':
        emp_length = 6
    elif emp_length == '7 Yrs':
        emp_length = 7
    elif emp_length == '8 Yrs':
        emp_length = 8
    elif emp_length == '9 Yrs':
        emp_length = 9
    else:
        emp_length = 10
# verification_status
    if verification_status == 'Not Verified':
        verification_status = 0
    elif verification_status == 'Source Verified':
        verification_status = 1
    else:
        verification_status = 2
# home_ownership
    if home_ownership == 'OTHER':
        home_ownership_OTHER = 1
    else:
        home_ownership_OTHER = 1

    if home_ownership == 'OWN':
        home_ownership_OWN = 1
    else:
        home_ownership_OWN = 0

    if home_ownership == 'RENT':
        home_ownership_RENT = 1
    else:
        home_ownership_RENT = 0
    # purpose
    if purpose == 'credit_card':
        purpose_credit_card = 1
    else :
        purpose_credit_card = 0

    if purpose == 'debt_consolidation':
        purpose_debt_consolidation = 1
    else:
        purpose_debt_consolidation = 0

    if purpose == 'educational':
        purpose_educational = 1
    else:
        purpose_educational = 0


    if purpose == 'home_improvement':
        purpose_home_improvement = 1
    else:
        purpose_home_improvement = 0

    if purpose == 'house':
        purpose_house = 1
    else:
        purpose_house = 0

    if purpose == 'major_purchase':
        purpose_major_purchase = 1
    else:
        purpose_major_purchase = 0

    if purpose == 'medical':
        purpose_medical = 1
    else:
        purpose_medical = 0

    if purpose == 'moving':
        purpose_moving = 1
    else :
        purpose_moving = 0

    if purpose == 'other':
        purpose_other = 1
    else:
        purpose_other = 0

    if purpose == 'renewable_energy':
        purpose_renewable_energy = 1
    else:
        purpose_renewable_energy = 0

    if purpose == 'small_business':
        purpose_small_business = 1
    else:
        purpose_small_business  = 0

    if purpose == 'vacation':
        purpose_vacation = 1
    else:
        purpose_vacation = 0

    if purpose == 'wedding':
        purpose_wedding = 1
    else:
        purpose_wedding = 0

    # title
    if title == 'Debt_consolidations':
        title_Debt_consolidations = 1
    else:
        title_Debt_consolidations = 0

    if title == 'Home_expenses':
        title_Home_expenses = 1
    else:
        title_Home_expenses = 0

    if title == 'Major_purchases':
        title_Major_purchases = 1
    else:
        title_Major_purchases = 0

    if title == 'Medical_expenses':
        title_Medical_expenses = 1
    else :
        title_Medical_expenses = 0

    if title == 'Other_expenses':
        title_Other_expenses = 1
    else:
        title_Other_expenses = 0

    # addr_state
    if addr_state == 'Nort_East':
        addr_state_Nort_East = 1
    else :
        addr_state_Nort_East = 0

    if addr_state == 'South':
        addr_state_South = 1
    else:
        addr_state_South = 0

    if addr_state == 'West':
        addr_state_West = 1
    else:
        addr_state_West = 0

    # earliest_cr_line
    if earliest_cr_line == '1960-79':
        earliest_cr_line_1960_79 = 1
    else:
        earliest_cr_line_1960_79 =0

    if earliest_cr_line == '1980-99':
        earliest_cr_line_1980_99 = 1
    else:
        earliest_cr_line_1980_99 = 0

    if earliest_cr_line == '2000-05':
        earliest_cr_line_2000_05 = 1
    else:
        earliest_cr_line_2000_05 = 0

    if earliest_cr_line == '2006-10':
        earliest_cr_line_2006_10 = 1
    else:
        earliest_cr_line_2006_10 = 0

    if earliest_cr_line == '2011-15':
        earliest_cr_line_2011_15 = 1
    else:
        earliest_cr_line_2011_15 = 0

    #initial_list_status
    if initial_list_status == 'w':
        initial_list_status_w = 1
    else:
        initial_list_status_w = 0

    #last_pymnt_d
    if last_pymnt_d == 'Q1-2009':
        last_pymnt_d_Q1_2009 = 1
    else:
        last_pymnt_d_Q1_2009 = 0

    if last_pymnt_d == 'Q1-2010':
        last_pymnt_d_Q1_2010 = 1
    else:
        last_pymnt_d_Q1_2010 = 0

    if last_pymnt_d == 'Q1-2011':
        last_pymnt_d_Q1_2011 = 1
    else:
        last_pymnt_d_Q1_2011 = 0

    if last_pymnt_d == 'Q1-2012':
        last_pymnt_d_Q1_2012 = 1
    else:
        last_pymnt_d_Q1_2012 = 0

    if last_pymnt_d == 'Q1-2013':
        last_pymnt_d_Q1_2013=1
    else:
        last_pymnt_d_Q1_2013= 0

    if last_pymnt_d == 'Q1-2014':
        last_pymnt_d_Q1_2014 = 1
    else :
        last_pymnt_d_Q1_2014 = 0

    if last_pymnt_d == 'Q1-2015':
        last_pymnt_d_Q1_2015 = 1
    else:
        last_pymnt_d_Q1_2015 = 0

    if last_pymnt_d == 'Q1-2016':
        last_pymnt_d_Q1_2016 = 1
    else:
        last_pymnt_d_Q1_2016 = 0

    if last_pymnt_d == 'Q2-2008':
        last_pymnt_d_Q2_2008 = 1
    else:
        last_pymnt_d_Q2_2008 = 0

    if last_pymnt_d == 'Q2-2009':
        last_pymnt_d_Q2_2009 = 1
    else:
        last_pymnt_d_Q2_2009 = 0

    if last_pymnt_d == 'Q2-2010':
        last_pymnt_d_Q2_2010 = 1
    else:
        last_pymnt_d_Q2_2010 = 0

    if last_pymnt_d == 'Q2-2011':
        last_pymnt_d_Q2_2011 = 1
    else:
        last_pymnt_d_Q2_2011 = 0

    if last_pymnt_d == 'Q2-2012':
        last_pymnt_d_Q2_2012 = 1
    else:
        last_pymnt_d_Q2_2012 = 0

    if last_pymnt_d == 'Q2-2013':
        last_pymnt_d_Q2_2013 = 1
    else:
        last_pymnt_d_Q2_2013 = 0

    if last_pymnt_d == 'Q2-2014':
        last_pymnt_d_Q2_2014 = 1
    else:
        last_pymnt_d_Q2_2014 = 0

    if last_pymnt_d == 'Q2-2015':
        last_pymnt_d_Q2_2015 = 1
    else:
        last_pymnt_d_Q2_2015 = 0

    if last_pymnt_d == 'Q3-2008':
        last_pymnt_d_Q3_2008 = 1
    else:
        last_pymnt_d_Q3_2008 = 0

    if last_pymnt_d == 'Q3-2009':
        last_pymnt_d_Q3_2009 = 1
    else:
        last_pymnt_d_Q3_2009 = 0

    if last_pymnt_d == 'Q3-2010':
        last_pymnt_d_Q3_2010 = 1
    else:
        last_pymnt_d_Q3_2010 = 0


    if last_pymnt_d == 'Q3-2011':
        last_pymnt_d_Q3_2011 = 1
    else:
        last_pymnt_d_Q3_2011 = 0

    if last_pymnt_d == 'Q3-2012':
        last_pymnt_d_Q3_2012 = 1
    else:
        last_pymnt_d_Q3_2012 = 0

    if last_pymnt_d == 'Q3-2013':
        last_pymnt_d_Q3_2013 = 1
    else:
        last_pymnt_d_Q3_2013 = 0

    if last_pymnt_d == 'Q3-2014':
        last_pymnt_d_Q3_2014 = 1
    else:
        last_pymnt_d_Q3_2014 = 0

    if last_pymnt_d == 'Q3-2015':
        last_pymnt_d_Q3_2015 = 1
    else:
        last_pymnt_d_Q3_2015 = 0

    if last_pymnt_d == 'Q4-2008':
        last_pymnt_d_Q4_2008 = 1
    else:
        last_pymnt_d_Q4_2008 = 0

    if last_pymnt_d == 'Q4-2009':
        last_pymnt_d_Q4_2009 = 1
    else:
        last_pymnt_d_Q4_2009 = 0

    if last_pymnt_d == 'Q4-2010':
        last_pymnt_d_Q4_2010 = 1
    else:
        last_pymnt_d_Q4_2010 = 0

    if last_pymnt_d == 'Q4-2011':
        last_pymnt_d_Q4_2011 = 1
    else:
        last_pymnt_d_Q4_2011 = 0

    if last_pymnt_d == 'Q4-2012':
        last_pymnt_d_Q4_2012 = 1
    else:
        last_pymnt_d_Q4_2012 = 0

    if last_pymnt_d == 'Q4-2013':
        last_pymnt_d_Q4_2013 = 1
    else:
        last_pymnt_d_Q4_2013 = 0

    if last_pymnt_d == 'Q4-2014':
        last_pymnt_d_Q4_2014 = 1
    else:
        last_pymnt_d_Q4_2014 = 0

    if last_pymnt_d == 'Q4-2015':
        last_pymnt_d_Q4_2015 = 1
    else:
        last_pymnt_d_Q4_2015 = 0

    # next_pymnt_d
    if next_pymnt_d == 'Jan-2016' :
        next_pymnt_d_Jan_2016 = 1
    else:
        next_pymnt_d_Jan_2016 = 0

    if next_pymnt_d == 'Mar-2016':
        next_pymnt_d_Mar_2016 = 1
    else:
        next_pymnt_d_Mar_2016 = 0

    if next_pymnt_d == 'Unknown':
        next_pymnt_d_Unknown = 1
    else:
        next_pymnt_d_Unknown = 0

# last_credit_pull_d
    if last_credit_pull_d == 'Q1-2009':
        last_credit_pull_d_Q1_2009 = 1
    else:
        last_credit_pull_d_Q1_2009 = 0

    if last_credit_pull_d == 'Q1-2010':
        last_credit_pull_d_Q1_2010 = 1
    else:
        last_credit_pull_d_Q1_2010 = 0

    if last_credit_pull_d == 'Q1-2011':
        last_credit_pull_d_Q1_2011 = 1
    else:
        last_credit_pull_d_Q1_2011 = 0

    if last_credit_pull_d == 'Q1-2012':
        last_credit_pull_d_Q1_2012 = 1
    else:
        last_credit_pull_d_Q1_2012 = 0

    if last_credit_pull_d == 'Q1-2013':
        last_credit_pull_d_Q1_2013 = 1
    else:
        last_credit_pull_d_Q1_2013 = 0

    if last_credit_pull_d == 'Q1-2014':
        last_credit_pull_d_Q1_2014 = 1
    else:
        last_credit_pull_d_Q1_2014 = 0

    if last_credit_pull_d == 'Q1-2015':
        last_credit_pull_d_Q1_2015 = 1
    else:
        last_credit_pull_d_Q1_2015 = 0

    if last_credit_pull_d == 'Q1-2016':
        last_credit_pull_d_Q1_2016 = 1
    else:
        last_credit_pull_d_Q1_2016 = 0

    if last_credit_pull_d == 'Q2-2007':
        last_credit_pull_d_Q2_2007 = 1
    else:
        last_credit_pull_d_Q2_2007 = 0

    if last_credit_pull_d == 'Q2-2008':
        last_credit_pull_d_Q2_2008 = 1
    else:
        last_credit_pull_d_Q2_2008 = 0

    if last_credit_pull_d == 'Q2-2009':
        last_credit_pull_d_Q2_2009 = 1
    else:
        last_credit_pull_d_Q2_2009 = 0

    if last_credit_pull_d == 'Q2-2010':
        last_credit_pull_d_Q2_2010 = 1
    else:
        last_credit_pull_d_Q2_2010 = 0

    if last_credit_pull_d == 'Q2-2011':
        last_credit_pull_d_Q2_2011 = 1
    else:
        last_credit_pull_d_Q2_2011 = 0

    if last_credit_pull_d == 'Q2-2012':
        last_credit_pull_d_Q2_2012 = 1
    else:
        last_credit_pull_d_Q2_2012 = 0

    if last_credit_pull_d == 'Q2-2013':
        last_credit_pull_d_Q2_2013 = 1
    else:
        last_credit_pull_d_Q2_2013 = 0

    if last_credit_pull_d == 'Q2-2014':
        last_credit_pull_d_Q2_2014 = 1
    else:
        last_credit_pull_d_Q2_2014 = 0

    if last_credit_pull_d == 'Q2-2015':
        last_credit_pull_d_Q2_2015 = 1
    else:
        last_credit_pull_d_Q2_2015 = 0

    if last_credit_pull_d == 'Q3-2007':
        last_credit_pull_d_Q3_2007 = 1
    else:
        last_credit_pull_d_Q3_2007 = 0

    if last_credit_pull_d == 'Q3-2008':
        last_credit_pull_d_Q3_2008 = 1
    else:
        last_credit_pull_d_Q3_2008 = 0

    if last_credit_pull_d == 'Q3-2009':
        last_credit_pull_d_Q3_2009 = 1
    else:
        last_credit_pull_d_Q3_2009 = 0

    if last_credit_pull_d == 'Q3-2010':
        last_credit_pull_d_Q3_2010 = 1
    else:
        last_credit_pull_d_Q3_2010 = 0

    if last_credit_pull_d == 'Q3-2011':
        last_credit_pull_d_Q3_2011 = 1
    else:
        last_credit_pull_d_Q3_2011 = 0

    if last_credit_pull_d == 'Q3-2012':
        last_credit_pull_d_Q3_2012 = 1
    else:
        last_credit_pull_d_Q3_2012 = 0

    if last_credit_pull_d == 'Q3-2013':
        last_credit_pull_d_Q3_2013 = 1
    else:
        last_credit_pull_d_Q3_2013 = 0

    if last_credit_pull_d == 'Q3-2014':
        last_credit_pull_d_Q3_2014 = 1
    else:
        last_credit_pull_d_Q3_2014 = 0

    if last_credit_pull_d == 'Q3-2015':
        last_credit_pull_d_Q3_2015 = 1
    else:
        last_credit_pull_d_Q3_2015 = 0

    if last_credit_pull_d == 'Q4-2007':
        last_credit_pull_d_Q4_2007 = 1
    else:
        last_credit_pull_d_Q4_2007 = 0

    if last_credit_pull_d == 'Q4-2008':
        last_credit_pull_d_Q4_2008 = 1
    else:
        last_credit_pull_d_Q4_2008 = 0

    if last_credit_pull_d == 'Q4-2009':
        last_credit_pull_d_Q4_2009 = 1
    else:
        last_credit_pull_d_Q4_2009 = 0

    if last_credit_pull_d == 'Q4-2010':
        last_credit_pull_d_Q4_2010 = 1
    else:
        last_credit_pull_d_Q4_2010 = 0

    if last_credit_pull_d == 'Q4-2011':
        last_credit_pull_d_Q4_2011 = 1
    else:
        last_credit_pull_d_Q4_2011 = 0

    if last_credit_pull_d == 'Q4-2012':
        last_credit_pull_d_Q4_2012 = 1
    else:
        last_credit_pull_d_Q4_2012 = 0

    if last_credit_pull_d == 'Q4-2013':
        last_credit_pull_d_Q4_2013 = 1
    else:
        last_credit_pull_d_Q4_2013 = 0

    if last_credit_pull_d == 'Q4-2014':
        last_credit_pull_d_Q4_2014 = 1
    else:
        last_credit_pull_d_Q4_2014 = 0

    if last_credit_pull_d == 'Q4-2015':
        last_credit_pull_d_Q4_2015 = 1
    else:
        last_credit_pull_d_Q4_2015 = 0

    # application_type
    if application_type == 'JOINT':
        application_type_JOINT = 1
    else:
        application_type_JOINT = 0




    if st.button("Predict"):
        prediction = hybrid_model.predict([[
                                loan_amnt, funded_amnt, funded_amnt_inv, term, int_rate, installment, grade, sub_grade, emp_length,
                                annual_inc, verification_status, dti, delinq_2yrs, inq_last_6mths, pub_rec, revol_bal, out_prncp,
                                out_prncp_inv, total_pymnt, total_pymnt_inv, total_rec_prncp, total_rec_int, total_rec_late_fee,
                                recoveries, collection_recovery_fee, last_pymnt_amnt, home_ownership_OTHER,
                                home_ownership_OWN, home_ownership_RENT, purpose_credit_card, purpose_debt_consolidation,
                                purpose_educational, purpose_home_improvement, purpose_house, purpose_major_purchase, purpose_medical,
                                purpose_moving, purpose_other, purpose_renewable_energy, purpose_small_business, purpose_vacation,
                                purpose_wedding, title_Debt_consolidations, title_Home_expenses, title_Major_purchases,
                                title_Medical_expenses, title_Other_expenses, addr_state_Nort_East, addr_state_South, addr_state_West,
                                earliest_cr_line_1960_79, earliest_cr_line_1980_99, earliest_cr_line_2000_05,
                                earliest_cr_line_2006_10, earliest_cr_line_2011_15, initial_list_status_w, last_pymnt_d_Q1_2009,
                                last_pymnt_d_Q1_2010, last_pymnt_d_Q1_2011, last_pymnt_d_Q1_2012, last_pymnt_d_Q1_2013, last_pymnt_d_Q1_2014,
                                last_pymnt_d_Q1_2015, last_pymnt_d_Q1_2016, last_pymnt_d_Q2_2008, last_pymnt_d_Q2_2009, last_pymnt_d_Q2_2010,
                                last_pymnt_d_Q2_2011, last_pymnt_d_Q2_2012, last_pymnt_d_Q2_2013, last_pymnt_d_Q2_2014, last_pymnt_d_Q2_2015,
                                last_pymnt_d_Q3_2008, last_pymnt_d_Q3_2009, last_pymnt_d_Q3_2010, last_pymnt_d_Q3_2011, last_pymnt_d_Q3_2012,
                                last_pymnt_d_Q3_2013, last_pymnt_d_Q3_2014, last_pymnt_d_Q3_2015, last_pymnt_d_Q4_2008, last_pymnt_d_Q4_2009,
                                last_pymnt_d_Q4_2010, last_pymnt_d_Q4_2011, last_pymnt_d_Q4_2012, last_pymnt_d_Q4_2013, last_pymnt_d_Q4_2014,
                                last_pymnt_d_Q4_2015, next_pymnt_d_Jan_2016, next_pymnt_d_Mar_2016, next_pymnt_d_Unknown, last_credit_pull_d_Q1_2009,
                                last_credit_pull_d_Q1_2010, last_credit_pull_d_Q1_2011, last_credit_pull_d_Q1_2012, last_credit_pull_d_Q1_2013,
                                last_credit_pull_d_Q1_2014, last_credit_pull_d_Q1_2015, last_credit_pull_d_Q1_2016, last_credit_pull_d_Q2_2007,
                                last_credit_pull_d_Q2_2008, last_credit_pull_d_Q2_2009, last_credit_pull_d_Q2_2010, last_credit_pull_d_Q2_2011,
                                last_credit_pull_d_Q2_2012, last_credit_pull_d_Q2_2013, last_credit_pull_d_Q2_2014, last_credit_pull_d_Q2_2015,
                                last_credit_pull_d_Q3_2007, last_credit_pull_d_Q3_2008, last_credit_pull_d_Q3_2009, last_credit_pull_d_Q3_2010,
                                last_credit_pull_d_Q3_2011, last_credit_pull_d_Q3_2012, last_credit_pull_d_Q3_2013, last_credit_pull_d_Q3_2014,
                                last_credit_pull_d_Q3_2015, last_credit_pull_d_Q4_2007, last_credit_pull_d_Q4_2008, last_credit_pull_d_Q4_2009,
                                last_credit_pull_d_Q4_2010, last_credit_pull_d_Q4_2011, last_credit_pull_d_Q4_2012, last_credit_pull_d_Q4_2013,
                                last_credit_pull_d_Q4_2014, last_credit_pull_d_Q4_2015, application_type_JOINT
                                            ]])
        if prediction == 0:
            st.warning('Wont default')
        else:
            st.warning('Will default')

    mylist = [
        loan_amnt, funded_amnt, funded_amnt_inv, term, int_rate, installment, grade, sub_grade, emp_length,
        annual_inc, verification_status, dti, delinq_2yrs, inq_last_6mths, pub_rec, revol_bal, out_prncp,
        out_prncp_inv, total_pymnt, total_pymnt_inv, total_rec_prncp, total_rec_int, total_rec_late_fee,
        recoveries, collection_recovery_fee, last_pymnt_amnt, home_ownership_OTHER,
        home_ownership_OWN, home_ownership_RENT, purpose_credit_card, purpose_debt_consolidation,
        purpose_educational, purpose_home_improvement, purpose_house, purpose_major_purchase, purpose_medical,
        purpose_moving, purpose_other, purpose_renewable_energy, purpose_small_business, purpose_vacation,
        purpose_wedding, title_Debt_consolidations, title_Home_expenses, title_Major_purchases,
        title_Medical_expenses, title_Other_expenses, addr_state_Nort_East, addr_state_South, addr_state_West,
        earliest_cr_line_1960_79, earliest_cr_line_1980_99, earliest_cr_line_2000_05,
        earliest_cr_line_2006_10, earliest_cr_line_2011_15, initial_list_status_w, last_pymnt_d_Q1_2009,
        last_pymnt_d_Q1_2010, last_pymnt_d_Q1_2011, last_pymnt_d_Q1_2012, last_pymnt_d_Q1_2013, last_pymnt_d_Q1_2014,
        last_pymnt_d_Q1_2015, last_pymnt_d_Q1_2016, last_pymnt_d_Q2_2008, last_pymnt_d_Q2_2009, last_pymnt_d_Q2_2010,
        last_pymnt_d_Q2_2011, last_pymnt_d_Q2_2012, last_pymnt_d_Q2_2013, last_pymnt_d_Q2_2014, last_pymnt_d_Q2_2015,
        last_pymnt_d_Q3_2008, last_pymnt_d_Q3_2009, last_pymnt_d_Q3_2010, last_pymnt_d_Q3_2011, last_pymnt_d_Q3_2012,
        last_pymnt_d_Q3_2013, last_pymnt_d_Q3_2014, last_pymnt_d_Q3_2015, last_pymnt_d_Q4_2008, last_pymnt_d_Q4_2009,
        last_pymnt_d_Q4_2010, last_pymnt_d_Q4_2011, last_pymnt_d_Q4_2012, last_pymnt_d_Q4_2013, last_pymnt_d_Q4_2014,
        last_pymnt_d_Q4_2015, next_pymnt_d_Jan_2016, next_pymnt_d_Mar_2016, next_pymnt_d_Unknown,last_credit_pull_d_Q1_2009,
        last_credit_pull_d_Q1_2010, last_credit_pull_d_Q1_2011, last_credit_pull_d_Q1_2012, last_credit_pull_d_Q1_2013,
        last_credit_pull_d_Q1_2014, last_credit_pull_d_Q1_2015, last_credit_pull_d_Q1_2016, last_credit_pull_d_Q2_2007,
        last_credit_pull_d_Q2_2008, last_credit_pull_d_Q2_2009, last_credit_pull_d_Q2_2010, last_credit_pull_d_Q2_2011,
        last_credit_pull_d_Q2_2012, last_credit_pull_d_Q2_2013, last_credit_pull_d_Q2_2014, last_credit_pull_d_Q2_2015,
        last_credit_pull_d_Q3_2007, last_credit_pull_d_Q3_2008, last_credit_pull_d_Q3_2009, last_credit_pull_d_Q3_2010,
        last_credit_pull_d_Q3_2011, last_credit_pull_d_Q3_2012, last_credit_pull_d_Q3_2013, last_credit_pull_d_Q3_2014,
        last_credit_pull_d_Q3_2015, last_credit_pull_d_Q4_2007, last_credit_pull_d_Q4_2008, last_credit_pull_d_Q4_2009,
        last_credit_pull_d_Q4_2010, last_credit_pull_d_Q4_2011, last_credit_pull_d_Q4_2012, last_credit_pull_d_Q4_2013,
        last_credit_pull_d_Q4_2014, last_credit_pull_d_Q4_2015, application_type_JOINT
                                            ]
    print(mylist)


if __name__ == '__main__':
    main()