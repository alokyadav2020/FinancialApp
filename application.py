import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate, LLMChain
import json
import sqlite3
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Access the API key
api_key = os.getenv('hf_key')
import pandas as pd


st.title("Financial Report Generator App")

# openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")



def load_model(temp,token,repo_id,HF_key,top_p,top_k):
    
    # repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct"
    llm=HuggingFaceEndpoint(repo_id=repo_id,max_new_tokens=token,temperature=temp, huggingfacehub_api_token=HF_key,top_k=top_k,top_p=top_p)
    return llm



system_promt = """

<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are financial adviser. You have to provide financial report based on nput data. 



START OF Business Estimation Value i.e. BEV CALCULATION--

Valuation Methodologies Application:

1. Method Asset-Based Valuation: display calculate values in tabular formate and explane the What does this mean.
2. Discounted Cash Flow (DCF):  Explanation: Projects future cash flows and discounts them back to present value using a discount rate reflecting the risk-adjusted cost of capital. Example: Estimated Future Cash Flows for 5 years sum to $84,000 (assuming growth and discount rate), Discount Rate = 10%, DCF Valuation = $52,155.
3. Comparable Company Analysis (CCA): display calculate values in tabular formate and explane the What does this mean.
4. Rule of Thumb Methods:  display calculate values in tabular formate and explane the What does this mean.
5. Earnings Multiplier: display calculate values in tabular formate and explane the What does this mean.
6. Liquidation Value:display calculate values in tabular formate and explane the What does this mean.
7. Monte Carlo Simulations: Use probabilistic models to simulate a range of possible outcomes based on the variability of input factors like cash flows, discount rates, and growth projections to provide a valuation range rather than a fixed figure. Calculate the Low , Mid and High Range of monte carlo simulations. Comments what Low, Mid and High ranges could means to the given business.

END OF Business Estimation Value i.e. BEV CALCULATION--

Valuation Summary:
The business's valuation varies significantly based on the methodology applied:
- Rule of Thumb Method: $3,600,000
- Earnings Multiplier: $2,500,000

Consider how local or international political changes could impact the business environment and market stability. Economic Indicators: Include macroeconomic factors such as inflation rates, interest rates, and economic growth rates that could affect the business’s performance. Industry-Specific Trends: Look at growth rates, technological changes, and new regulations affecting the industry. Regulatory Environment: Updates on new laws or regulations that could impact business operations or costs (e.g., environmental regulations, minimum wage increases). Market Competition: Analyze market share, the number of competitors, and their relative size and performance to adjust the valuation accordingly. 




Use this Example as Reference.
Try to DISPAY result as given in example.
Do not use data of the example in extual output.

Example:

Business Estimation Valuation ( BEV ) Summary:

BEV for the Retail-CPG <Business Type> business is as follows:


Method 1. Asset-Based Valuation :

                 2023                2022           2021           Final

               1100000             1000000         100000         15000


Explanetion: He business has a net asset value of $15000, representing its liquidation value if all assets were sold and liabilities settled. This method primarily reflects the tangible asset base of the business.


Method 2. Discounted Cash Flow (DCF)

   - Future Cash Flows: Estimated to sum to $1,350,000 over the next three years, assuming growth and discounting.
   - Discount Rate: 10% 
   - Present Value: $844,155

Explanetion:
Based on the present value of projected future cash flows, the business is valued at $844,155. This approach offers a forward-looking estimate grounded in the company’s expected profitability and the time value of money.


Method 3. Comparable Company Analysis (CCA)

   
   
      2023                2022           2021          
  
    1100000             1000000         100000        
  
    200000              1000000         200000


Explanetion:
The CCA valuation is $5,000,000, reflecting the business's worth based on the average price-to-earnings ratio of similar companies. This method leverages market data from comparable firms to estimate the business's value relative to its peers.


Method 4. Rule of Thumb Methods
   

     2023                2022           2021          
  
    1100000             1000000         100000        
  
    200000              1000000         20000

   

Explanetion:
This method provides a valuation of $3,600,000 by applying an industry-specific multiplier to gross revenue. It is a quick estimate often used for benchmarking purposes in specific sectors.


Method 5. Earnings Multiplier Method
  

   
     2023                2022           2021          
  
    1100000             1000000         100000        
  
    200000              1000000         20000


  

Explanetion:
The earnings multiplier method yields a valuation of $2,500,000, reflecting the business's profitability and earnings potential.   

   
Method 6. Liquidation Value


     
       2023                2022           2021          
    
      1100000             1000000         100000        
  

Explanetion:
The liquidation value is $150,000, representing the estimated proceeds if all assets were liquidated and liabilities paid. This is the minimum valuation in a worst-case scenario.


   
Method 7. Monte Carlo Simulation Results
   - Mid-Range Valuation: $1,444,155
   - High-Range Valuation: $1,844,155

Explanetion: 
The Monte Carlo simulation suggests that the business’s valuation could range between $1,444,155 and $1,844,155. This probabilistic model captures a spectrum of potential outcomes, factoring in uncertainties around future financial performance.  

   

   Valuation Summary
The business's valuation varies significantly based on the methodology applied:

- Rule of Thumb Method: $3,600,000
- Earnings Multiplier: $2,500,000

In conclusion, the business is valued between $2,500,000 and $3,600,000, offering a range of worth derived from various approaches.

Disclaimer: 
The valuation summary shows the estimated business value using various valuation methods. The range of valuations indicates the uncertainty and variability in the business valuation based on different assumptions and methodologies. The final valuation will depend on the specific circumstances and the judgment of the financial analyst.



<|eot_id|><|start_header_id|>user<|end_header_id|>

Your task is to provide clear and consice Business Estimation Value i.e. BEV CALCULATION and SUMMERY of given financial data.
- Zip Code: {zipCode}
- Business Type: {businessType}
- PE_Ratio : {PE_Ratio}
- Industry_Multiplier: {Industry_Multiplier}
- Earnings Multiplier: {Earnings_Multiplier}

             

Display all the points i.e. point # 1 to point # 7. , Valuation Summary, and Disclaimer.

From the given financial inputs and using different valuation methods

1. Asset-Based Valuation:   Try to display for year and results in tabular format as shown in example.
 
Use this input data to display result:

  {Year_1}        {Year_2}           {Year_3}           Final

  {Result_1}      {Result_2}         {Result_3}         {Result_Final}


Explanetion:  Provide a simple explanation in one line what it means to the given business.

2. Discounted Cash Flow (DCF): Provide a simple calculation and explain in one line what it means to the given business.
Use this Input Data for calculation-- Discount Rate : {Discount_Rate}
DO not display in tabular form.
Explanetion: Provide a simple explanation in one line what it means to the given business.

3. Comparable Company Analysis (CCA): Try to display for year and results in tabular format as shown in example.
 
Use this input data to display result in tabular form:
Make it should be in only One table.
PE_Ratio : {PE_Ratio}

       {Year_1}                         {Year_2}                         {Year_3}                          
    
    {Net_Profit_Year_1}              {Net_Profit_Year_2}              {Net_Profit_Year_3}                 
    
    {Net_Profit_result_1}            {Net_Profit_result_2}            {Net_Profit_result_3}
 
Explanetion: Provide a simple explanation in one line what it means to the given business.

4. Rule of Thumb Methods:  This uses industry-specific revenue multipliers to estimate value.
Try to display for year and results in tabular format as shown in example.
Use this input data to display result in tabular form:
Make it should be in only One table.
Industry_Multiplier: {Industry_Multiplier}


     {Year_1}                           {Year_2}                        {Year_3} 
   
     {Gross_revenu_Year_1}        {Gross_revenu_Year_2}         {Gross_revenu_Year_3}
   
     {Gross_revenu_result_1}      {Gross_revenu_result_2}       {Gross_revenu_result_3}

Explanetion: Provide a simple explanation in one line what it means to the given business.

5. Earnings Multiplier:  Provide a simple calculation and explain in one line what it means to the given business.
Try to display for year and results in tabular format as shown in example.
Use this input data to display result in tabular form:
Make it should be in only One table.
Earnings Multiplier: {Earnings_Multiplier}



   
     {Year_1}                       {Year_2}               {Year_3} 
   
     {Net_Profit_Year_1}      {Net_Profit_Year_2}    {Net_Profit_Year_3}
    
    {Net_earning_result_1}   {Net_earning_result_2}  {Net_earning_result_3}


Explanetion: Provide a simple explanation in one line what it means to the given business.

6. Liquidation Value: Provide a simple calculation and explain in one line what it means to the given business.
Try to display for year and results in tabular format as shown in example.
Use this input data to display result in tabular form:
Make it should be in only One table

          {Year_1}                  {Year_2}               {Year_3}          

       {Liquidation_Value_1}    {Liquidation_Value_2}   {Liquidation_Value_3}        



Explanetion: Provide a simple explanation in one line what it means to the given business.

7. Monte Carlo Simulation Results:
Use this Input Data for calculation--
List the Mid and High Range of monte carlo simulations. 

Explanetion: Provide a simple explanation in one line what it means to the given business.


Valuation Summary: Provide consice Valuation generated by Rule of thumb method and Earnings multiplier.



Disclaimer: 
The valuation summary shows the estimated business value using various valuation methods. The range of valuations indicates the uncertainty and variability in the business valuation based on different assumptions and methodologies. The final valuation will depend on the specific circumstances and the judgment of the financial analyst.


Do not display any other section.
Use all currency notation from {currency}





<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
M_1 ='meta-llama/Meta-Llama-3.1-8B-Instruct'
M_2 ='meta-llama/Meta-Llama-3.1-70B-Instruct'
M_3 ='meta-llama/Meta-Llama-3-8B-Instruct'
M_4 ='mistralai/Mistral-7B-Instruct-v0.2'
M_5= 'google/gemma-2-9b-it'
model_list = [M_1,M_2,M_3,M_4,M_5]


def method_1(C_A_F_Y,T_A_F_Y,C_L_Y_Y,T_L_F_Y):
    rt= [(C_A_F_Y + T_A_F_Y ) - (C_L_Y_Y + T_L_F_Y) for C_A_F_Y,T_A_F_Y,C_L_Y_Y,T_L_F_Y in zip(C_A_F_Y,T_A_F_Y,C_L_Y_Y,T_L_F_Y) ]
 
    result_1 = rt[0]
    result_2  = rt[1]
    result_3  = rt[2]
    final_result = result_1*0.5 +    result_2*0.5 + result_3*0.25

    return result_1,result_2,result_3,final_result 

def method_3(revenue,expense,PE_Ratio):
    net_profits = [revenue - expense for revenue, expense in zip(revenue, expense)]
    result_net = [i*PE_Ratio for i in net_profits]

    return net_profits,result_net


def method_4(revenues,Industry_Multiplier):
    gross_rev = [i*Industry_Multiplier for i in revenues]

    return gross_rev


def method_5(revenue,expense,Earnings_Multiplier):
    
    net_profits = [revenue - expense for revenue, expense in zip(revenue, expense)]
    result_net = [i*Earnings_Multiplier for i in net_profits]

    return result_net
    

def method_6(current_asset,current_liabilites):
    net_profits = [current_asset - current_liabilites for current_asset, current_liabilites in zip(current_asset, current_liabilites)]
    return net_profits
    





def blog_outline(temp,token,model,hf_key,t_k,t_p,
                 zipCode,businessType,currency,
                 financial_year_1,financial_year_2,financial_year_3,
                 current_assets_financial_year,
                 total_assets_financial_year,
                 current_liabilities_financial_year,
                 total_liabilities_financial_year,
                 revenues,
                 expenses,
                 
                 ):
   df_dat = get_data_sql(f'{businessType}')
   print(df_dat)
   Discount_Rate =float(df_dat[1].replace("%",""))
   PE_Ratio =float(df_dat[2].replace("%",""))
   Industry_Multiplier = float(df_dat[3].replace("%",""))
   Earnings_Multiplier = float(df_dat[4].replace("%",""))
   print(Discount_Rate,PE_Ratio,Industry_Multiplier,Earnings_Multiplier)
    # Instantiate LLM model
   Result_1,Result_2,Result_3,Result_Final = method_1(current_assets_financial_year,total_assets_financial_year,current_liabilities_financial_year,total_liabilities_financial_year)
   Net_Profit_Year,Net_Profit_result = method_3(revenues,expenses,PE_Ratio)
   print(f"Method 3 :{Net_Profit_Year,Net_Profit_result}")
   Net_Profit_Year_1 =Net_Profit_Year[0]
   Net_Profit_Year_2 =Net_Profit_Year[1]
   Net_Profit_Year_3 =Net_Profit_Year[2]
   Net_Profit_result_1 = Net_Profit_result[0]
   Net_Profit_result_2 = Net_Profit_result[1]
   Net_Profit_result_3 = Net_Profit_result[2]
   Gross_revenu_result =method_4(revenues,Industry_Multiplier)
   print(f"Method 4 :{Gross_revenu_result}")
   Gross_revenu_Year_1 = revenues[0]
   Gross_revenu_Year_2 = revenues[1]
   Gross_revenu_Year_3 = revenues[2]
   Gross_revenu_result_1 = Gross_revenu_result[0]
   Gross_revenu_result_2 = Gross_revenu_result[1]
   Gross_revenu_result_3 = Gross_revenu_result[2]
   Net_earning_result= method_5(revenues,expenses,Earnings_Multiplier)
   print(f"Method 5 :{Net_earning_result}")
   Net_earning_result_1 = Net_earning_result[0]
   Net_earning_result_2 = Net_earning_result[1] 
   Net_earning_result_3 = Net_earning_result[2]  
   Liquidation_Value  = method_6(current_assets_financial_year,current_liabilities_financial_year)
   print(f"Method 6 :{Liquidation_Value}")
   Liquidation_Value_1 = Liquidation_Value[0]
   Liquidation_Value_2 = Liquidation_Value[1]
   Liquidation_Value_3 = Liquidation_Value[2]
   input_params = {
               "zipCode" : zipCode,
         "businessType" : businessType,
         "currency" : currency,
         "Year_1" : financial_year_1,
         "Year_2" : financial_year_2,
         "Year_3" : financial_year_3,
         "Result_1" : Result_1,
         "Result_2" : Result_2,
         "Result_3" : Result_3,
         "Result_Final" : Result_Final,
         "Net_Profit_Year_1" : Net_Profit_Year_1,
         "Net_Profit_Year_2" : Net_Profit_Year_2,
         "Net_Profit_Year_3" : Net_Profit_Year_3,
         "Net_Profit_result_1" : Net_Profit_result_1,
         "Net_Profit_result_2" : Net_Profit_result_2,  
         "Net_Profit_result_3" : Net_Profit_result_3,  
         "Gross_revenu_Year_1" : Gross_revenu_Year_1,
         "Gross_revenu_Year_2" : Gross_revenu_Year_2,
         "Gross_revenu_Year_3" : Gross_revenu_Year_3,
         "Gross_revenu_result_1" : Gross_revenu_result_1,
         "Gross_revenu_result_2" : Gross_revenu_result_2,
         "Gross_revenu_result_3" : Gross_revenu_result_3,
         "Net_Profit_Year_1" : Net_Profit_Year_1,
         "Net_Profit_Year_2" : Net_Profit_Year_2,
         "Net_Profit_Year_3" : Net_Profit_Year_3,
         "Net_earning_result_1" : Net_earning_result_1,
         "Net_earning_result_2" : Net_earning_result_2,
         "Net_earning_result_3" : Net_earning_result_3,
         "Liquidation_Value_1" : Liquidation_Value_1,
         "Liquidation_Value_2" : Liquidation_Value_2,
         "Liquidation_Value_3" : Liquidation_Value_3,

         
        "Discount_Rate": Discount_Rate,
        "PE_Ratio" : PE_Ratio,
        "Industry_Multiplier": Industry_Multiplier,
        "Earnings_Multiplier" :Earnings_Multiplier
    }
   llm = load_model(temp,token,model,HF_key=hf_key,top_k=t_k,top_p=t_p)
   promt= PromptTemplate.from_template(system_promt)
   llm_chain=LLMChain(llm=llm,prompt=promt)
   data = llm_chain.invoke(input_params)
   response = data['text']
   return st.info(response)


def get_data_sql( Industry_Name:str) -> pd.DataFrame:
    conn = sqlite3.connect('BEV_database.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM BEV WHERE Industry_Name = '{Industry_Name}';"
    cursor.execute(query )
    result = cursor.fetchone()

    # df_from_db = pd.read_sql(sql=query,con= conn)
    # conn.close()

    return result




with st.form("myform"):
    try:
      #   hf_key = st.sidebar.text_input("HF Key:", "",type="password")
        hf_key = api_key

        topic_text = st.text_input("Enter prompt:", "")
        
        
        submitted = st.form_submit_button("Submit")
        tokens = st.sidebar.slider(label="Tokens",min_value=500,max_value=6000,value=4000)
        value = st.sidebar.slider(label="Temperature",min_value=0.0,max_value=1.0,value=0.1)
        t_k_value = st.sidebar.slider(label="top_k",min_value=1,max_value=15,value=10)
        t_p_value = st.sidebar.slider(label="top_p",min_value=0.0,max_value=1.0,value=0.25)
        models = st.sidebar.selectbox(label="Select Model",options=model_list)
        if submitted:
            jdata = json.loads(f"""{topic_text}""")
            print(jdata)
            blog_outline(
                        temp=value,
                        token=tokens,
                        model=models,
                        hf_key=hf_key,
                        t_k=t_k_value,
                        t_p=t_p_value,
                        zipCode = jdata['zipCode'],
                        businessType = jdata['businessType'],
                        currency = jdata['currency'],
                        financial_year_1 = jdata['financialMetrics']['years'][0],
                        financial_year_2 = jdata['financialMetrics']['years'][1],
                        financial_year_3 = jdata['financialMetrics']['years'][2],
                        revenues=jdata['financialMetrics']['revenue'],
                        expenses=jdata['financialMetrics']['expenses'],
                       
                        current_assets_financial_year=jdata['financialMetrics']['assets']['current'],
                       
                        total_assets_financial_year= jdata['financialMetrics']['assets']['total'],
                       
                        current_liabilities_financial_year= jdata['financialMetrics']['liabilities']['current'],
                      
                        total_liabilities_financial_year= jdata['financialMetrics']['liabilities']['total']
                         )

    except Exception as e:
        raise e    
        