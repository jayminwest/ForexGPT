import streamlit as st
from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
from langchain import OpenAI, SerpAPIWrapper
from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish
from langchain import LLMMathChain, OpenAI, SerpAPIWrapper, SQLDatabase, SQLDatabaseChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import yfinance as yf
from datetime import datetime, timedelta

search=SerpAPIWrapper()

def get_current_ask_price(currency_pair):
    # Get the data from yfinance
    forex = yf.Ticker(currency_pair)
    # Get the current ask price
    current_ask = forex.info['ask']
    return current_ask

def get_past_X_days_data(currency_pair, num_days):
    # Get the current date and the date 3 days ago
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days)
    
    # Format the dates in the correct format for yfinance
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Get the data from yfinance
    forex = yf.Ticker(currency_pair)
    # Get historical data
    hist = forex.history(start=start_date_str, end=end_date_str)
    
    return hist

tools = [
    Tool(
        name="Search", 
        func=search.run, 
        description="Useful for finding current real world events", 
        return_direct=True
    ),
    Tool.from_function( 
        func=get_current_ask_price,
        name="get_current_ask_price",
        description="useful for finding the current ask price of currencies with the yfinance API. An example would be to pass it 'EURUSD=X'"
    ),
]

class ForexTradingAssistant:
    def __init__(self):
        self.user_input = None
        self.ai_response = None
        llm = ChatOpenAI(
                temperature=0
            )
        llm.model_name="gpt-3.5-turbo-0613" 
        self.agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True, max_iterations=5)

    def get_user_position_inputs(self):
        st.sidebar.title('Forex Trading Details')

        with st.sidebar.form(key='user_inputs_form'):
            account_balance = st.number_input("Account Balance", min_value=0.0, step=0.01)
            risk_percentage = st.number_input("Risk Percentage", min_value=0.0, max_value=100.0, step=0.1)
            stop_loss = st.number_input("Stop Loss (Pips)", min_value=0.0, step=0.01)
            currency_pair = st.text_input("Currency Pair")
            current_ask_price = st.number_input("Current Ask Price", min_value=0.0, step=0.01)

            submitted = st.form_submit_button('Submit')

        if submitted:
            # Calculate position size using the inputs
            risk_amount = account_balance * (risk_percentage / 100.0)
            position_size = risk_amount / stop_loss
            st.sidebar.markdown(f'**Position Size:** {position_size}')
            
            return account_balance, risk_percentage, stop_loss, currency_pair, current_ask_price, position_size
        else:
            return None, None, None, None, None, None 

    def get_ai_response(self, user_input):
        return f"AI Response: {self.agent.run(user_input)}"

    def display_interface(self):
        st.title('Forex Trading Assistant')

        # Call the function for sidebar
        self.get_user_position_inputs()

        # For user input and AI response
        self.user_input = st.text_input("Enter your question here")
        if st.button('Submit'):
            self.ai_response = self.get_ai_response(self.user_input)
            st.text_area("AI Response", value=self.ai_response, height=200, max_chars=None, key=None)

if __name__ == '__main__':
    ForexAssistant = ForexTradingAssistant()
    ForexAssistant.display_interface()
