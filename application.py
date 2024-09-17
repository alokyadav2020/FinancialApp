import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate, LLMChain
import json
import sqlite3
import pandas as pd


st.title("Financial Report Generator App")

# openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")



def load_model(temp,token,repo_id,HF_key,top_p,top_k):
    
    # repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct"
    llm=HuggingFaceEndpoint(repo_id=repo_id,max_new_tokens=token,temperature=temp, huggingfacehub_api_token=HF_key,top_k=top_k,top_p=top_p)
    return llm



system_promt = """

<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are financial adviser. 



START OF Business Estimation Value i.e. BEV CALCULATION--

Valuation Methodologies Application:

1. Method Asset-Based Valuation: Explanation: Calculates the net asset value by subtracting total liabilities from total assets. Example: Total Assets = $5,000, Total Liabilities = $3,000, Asset-Based Valuation = $2,000.

2. Discounted Cash Flow (DCF): Explanation: Projects future cash flows and discounts them back to present value using a discount rate reflecting the risk-adjusted cost of capital. Example: Estimated Future Cash Flows for 5 years sum to $84,000 (assuming growth and discount rate), Discount Rate = 10%, DCF Valuation = $52,155.

3. Comparable Company Analysis (CCA): Explanation: Evaluates financial ratios and statistics from similar companies to derive a valuation multiple, consider lowest (25%) of population for multiplier. Example: Average PE Ratio of Comparable Firms = 20, Net Profit = $24,000, CCA Valuation = $480,000.

4. Rule of Thumb Methods: Explanation: Uses industry-specific formulas or averages to estimate business value. Consider lowest (25%) of population for multiplier.Example: Industry Multiplier = 3x Gross Revenue, Gross Revenue = $36,000, Valuation = $108,000.

5. Earnings Multiplier: Explanation: Multiplies the company's earnings by a factor typical for the industry or market, consider the average multiplier. Example: Earnings Multiplier = 10, Net Profit = $24,000, Valuation = $240,000.

6. Liquidation Value: Explanation: Estimates the amount that could be realized if all assets were sold and liabilities paid off today. Example: Total Asset Liquidation Value = $5,000, Liabilities = $3,000, Liquidation Value = $2,000.

7. Monte Carlo Simulations: Use probabilistic models to simulate a range of possible outcomes based on the variability of input factors like cash flows, discount rates, and growth projections to provide a valuation range rather than a fixed figure. Calculate the Low , Mid and High Range of monte carlo simulations. Comments what Low, Mid and High ranges could means to the given business.


END OF Business Estimation Value i.e. BEV CALCULATION--

Valuation Summary:
The business's valuation varies significantly based on the methodology applied:
- Rule of Thumb Method: $3,600,000
- Earnings Multiplier: $2,500,000

Consider how local or international political changes could impact the business environment and market stability. Economic Indicators: Include macroeconomic factors such as inflation rates, interest rates, and economic growth rates that could affect the business’s performance. Industry-Specific Trends: Look at growth rates, technological changes, and new regulations affecting the industry. Regulatory Environment: Updates on new laws or regulations that could impact business operations or costs (e.g., environmental regulations, minimum wage increases). Market Competition: Analyze market share, the number of competitors, and their relative size and performance to adjust the valuation accordingly. 

Use this Example as Reference. Do not use data of the example in extual output.

Example:

Business Estimation Valuation ( BEV ) Summary:

BEV for the Retail-CPG <Business Type> business is as follows:


Method 1. Asset-Based Valuation :

   - Formula: Total Assets - Total Liabilities
   - Calculation: (Current Assets + Fixed Assets) - (Current Liabilities + Long-Term Liabilities)  
   - Result: (300,000 + 200,000) - (150,000 + 50,000) = $300,000

Explanation:  
The business has a net asset value of $300,000, representing its liquidation value if all assets were sold and liabilities settled. This method primarily reflects the tangible asset base of the business.


Method 2. Discounted Cash Flow (DCF)

   - Future Cash Flows: Estimated to sum to $1,350,000 over the next three years, assuming growth and discounting.
   - Discount Rate: 10% 
   - Present Value: $844,155

Explanation: 
Based on the present value of projected future cash flows, the business is valued at $844,155. This approach offers a forward-looking estimate grounded in the company’s expected profitability and the time value of money.


Method 3. Comparable Company Analysis (CCA)

   - Average P/E Ratio of Comparable Firms: 20 
   - Net Profit: $250,000
   - Valuation: 250,000 * 20 = $5,000,000

Explanation: 
The CCA valuation is $5,000,000, reflecting the business's worth based on the average price-to-earnings ratio of similar companies. This method leverages market data from comparable firms to estimate the business's value relative to its peers.


Method 4. Rule of Thumb Methods
   - Industry Multiplier: 3x Gross Revenue
   - Gross Revenue: $1,200,000
   - Valuation: 1,200,000 * 3 = $3,600,000

Explanation:
This method provides a valuation of $3,600,000 by applying an industry-specific multiplier to gross revenue. It is a quick estimate often used for benchmarking purposes in specific sectors.


Method 5. Earnings Multiplier Method
   - Earnings Multiplier: 10
   - Net Profit: $250,000
   - Valuation: 250,000 * 10 = $2,500,000

Explanation:
The earnings multiplier method yields a valuation of $2,500,000, reflecting the business's profitability and earnings potential.   

   
Method 6. Liquidation Value
   - Total Asset Liquidation Value: $300,000
   - Liabilities: $150,000
   - Liquidation Value: $300,000 - 150,000 = $150,000

Explanation: 
The liquidation value is $150,000, representing the estimated proceeds if all assets were liquidated and liabilities paid. This is the minimum valuation in a worst-case scenario.


   
Method 7. Monte Carlo Simulation Results
   - Mid-Range Valuation: $1,444,155
   - High-Range Valuation: $1,844,155

Explanation:  
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
- Zip Code: {zip_code},
- Business Type: {business_type},
- Revenue: {revenue},
- Expenses: {expenses},
- EBITDA: {ebitda}


Display all the points i.e. point # 1 to point # 7. , Valuation Summary, and Disclaimer.

From the given financial inputs and using different valuation methods

1. Asset-Based Valuation:  Provide a simple calculation and explain in one line what it means to the given business. 
Use this Input Data for calculation--  Current Assets: {current_assets} , Fixed Assets: {fixed_assets}, Current Liabilities: {current_liabilities}, Long-Term Liabilities: {long_term_liabilities}.
Explanation: Provide a simple explanation in one line what it means to the given business.

2. Discounted Cash Flow (DCF): Provide a simple calculation and explain in one line what it means to the given business.
Use this Input Data for calculation-- Discount Rate : {Discount_Rate}
Explanation: Provide a simple explanation in one line what it means to the given business.

3. Comparable Company Analysis (CCA): Provide a simple calculation and explain in one line what it means to the given business.
Use this Input Data for calculation-- Average P/E Ratio of Comparable Firms: {PE_Ratio}
Explanation: Provide a simple explanation in one line what it means to the given business.

4. Rule of Thumb Methods:  This uses industry-specific revenue multipliers to estimate value.Provide a simple calculation and explain in one line what it means to the given business.
Use this Input Data for calculation-- Industry Multiplier:{Industry_Multiplier}
Explanation: Provide a simple explanation in one line what it means to the given business.

5. Earnings Multiplier:  Provide a simple calculation and explain in one line what it means to the given business.
Use this Input Data for calculation-- Earnings Multiplier:{Earnings_Multiplier}
Explanation: Provide a simple explanation in one line what it means to the given business.

6. Liquidation Value: Provide a simple calculation and explain in one line what it means to the given business.
Use this Input Data for calculation-- Total Asset Liquidation Value:{current_assets},  Liabilities:{current_liabilities}
Explanation: Provide a simple explanation in one line what it means to the given business.

7. Monte Carlo Simulation Results:
Use this Input Data for calculation--
List the Mid and High Range of monte carlo simulations. 
Explanation: Provide a simple explanation in one line what it means to the given business.


Valuation Summary: Provide consice Valuation generated by Rule of thumb method and Earnings multiplier.



Disclaimer: 
The valuation summary shows the estimated business value using various valuation methods. The range of valuations indicates the uncertainty and variability in the business valuation based on different assumptions and methodologies. The final valuation will depend on the specific circumstances and the judgment of the financial analyst.


Do not display any other section.

Make sure your summary should be based on above calculation only.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
M_1 ='meta-llama/Meta-Llama-3.1-8B-Instruct'
M_2 ='meta-llama/Meta-Llama-3.1-70B-Instruct'
M_3 ='meta-llama/Meta-Llama-3-8B-Instruct'
M_4 ='mistralai/Mistral-7B-Instruct-v0.2'
M_5= 'google/gemma-2-9b-it'
model_list = [M_1,M_2,M_3,M_4,M_5]





def blog_outline(temp,token,model,hf_key,t_k,t_p,zip_code,business_type,revenue,expenses,ebitda,current_assets,fixed_assets,current_liabilities,long_term_liabilities):
    # Instantiate LLM model

    df_dat = get_data_sql(f'{business_type}')

   
    input_params = {
        "zip_code": zip_code,
        "business_type": business_type,
        "revenue": revenue,
        "expenses": expenses,
        "ebitda": ebitda,
        "current_assets": current_assets,
        "fixed_assets": fixed_assets,
        "current_liabilities": current_liabilities,
        "long_term_liabilities": long_term_liabilities,
        "Discount_Rate": df_dat['Cost_of_Capital(Discount_rate)'],
        "PE_Ratio" : df_dat['Trailing_PE(PE_Ratio)'],
        "Industry_Multiplier": df_dat['PBV(Industry_multiplier)'],
        "Earnings_Multiplier" :df_dat['EV/EBITDA(Earning_multiplier)']
    }
    llm = load_model(temp,token,model,HF_key=hf_key,top_k=t_k,top_p=t_p)
    promt= PromptTemplate.from_template(system_promt)
    llm_chain=LLMChain(llm=llm,prompt=promt)
    data = llm_chain.invoke(input_params)
    response = data['text']
    return st.info(response)


def get_data_sql(Industry_Name:str) -> pd.DataFrame:
    conn = sqlite3.connect('BEV_database.db')
    query = 'SELECT * FROM BEV WHERE Industry_Name = ?'
    df_from_db = pd.read_sql(sql=query,con= conn,params=(Industry_Name,))
    conn.close()

    return df_from_db




with st.form("myform"):
    try:
        hf_key = st.sidebar.text_input("HF Key:", "",type="password")

        topic_text = st.text_input("Enter prompt:", "")
        
        
        submitted = st.form_submit_button("Submit")
        tokens = st.sidebar.slider(label="Tokens",min_value=500,max_value=2500,value=1200)
        value = st.sidebar.slider(label="Temperature",min_value=0.0,max_value=1.0,value=0.1)
        t_k_value = st.sidebar.slider(label="top_k",min_value=1,max_value=15,value=10)
        t_p_value = st.sidebar.slider(label="top_p",min_value=0.0,max_value=1.0,value=0.25)
        models = st.sidebar.selectbox(label="Select Model",options=model_list)
        if submitted:
            data_json = json.loads(f"""{topic_text}""")
            print(data_json)
            blog_outline(zip_code = data_json['zipCode'],
                        business_type = data_json['businessType'],
                        revenue = data_json['financialMetrics']['revenue'],
                        expenses = data_json['financialMetrics']['expenses'],
                        ebitda = data_json['financialMetrics']['ebitda'],
                        current_assets = data_json['financialMetrics']['assets']['current'],
                        fixed_assets = data_json['financialMetrics']['assets']['fixed'],
                        current_liabilities = data_json['financialMetrics']['liabilities']['current'],
                        long_term_liabilities = data_json['financialMetrics']['liabilities']['longTerm'],
                         temp=value,
                         token=tokens,
                         model=models,
                         hf_key=hf_key,
                         t_k=t_k_value,
                         t_p=t_p_value
                         )

    except Exception as e:
        raise e    
        