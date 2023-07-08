import streamlit as st
from typing import List, Tuple, Any, Union
from langchain import LLMMathChain, OpenAI, SerpAPIWrapper, SQLDatabase, SQLDatabaseChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import yfinance as yf
from datetime import datetime, timedelta
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re

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

template = """
In your role as a professional forex analyst, your task is to analyze the available tools and information and provide a recommendation on whether to take a sell or buy position on the given currency. Your recommendation should be based on a thorough analysis of the currency's current market conditions, trends, and any other relevant factors that may impact its value.

To generate a comprehensive recommendation, please explain the reasoning behind your decision, taking into account technical analysis indicators, chart patterns, economic news, and any other relevant information. Your response should provide a clear and detailed explanation of why you believe a sell or buy position is the most appropriate choice, considering the potential risks and rewards.

Please note that your recommendation should consider key factors such as the currency's historical performance, market sentiment, and any upcoming events or news that may affect its value. Your response should be concise, well-reasoned, and demonstrate a deep understanding of forex analysis principles.

Finally, remember to make your recommendation clear and actionable, providing the necessary information for traders to execute the recommended trade effectively.

You have access to the following tools: 

{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
...(this Though/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer the original input question

Being!

Question: {input}
{agent_scratchpad}
"""

class CustomPromptTemplate(StringPromptTemplate):
    # Template to use:
    template: str
    # List of tools to use: 
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediete steps: Agent Action and the Observation Tuples
        # Format them as desired:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought:"
        # Set the agent_scratchpad variable to that value:
        kwargs["agent_scratchpad"] = thoughts
        # Create tools variable from the list of tools provided:
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided:
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])

        return self.template.format(**kwargs)
    
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction,  AgentFinish]:
        if "Final Answer" in llm_output:
            return AgentFinish(
                # Return values is always a dictionary with just a single "output" entry
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
prompt = CustomPromptTemplate(
    template=template, 
    tools=tools,
    input_variables=["input", "intermediate_steps"],
)

output_parser = CustomOutputParser()

class ForexTradingAssistant:
    def __init__(self):
        self.user_input = None
        self.ai_response = None
        llm = ChatOpenAI(
                temperature=0
            )
        llm.model_name="gpt-3.5-turbo-0613" 
        self.llm_chain = LLMChain(llm=llm, prompt=prompt)
        tool_names = [tool.name for tool in tools]
        self.agent = LLMSingleActionAgent(
            llm_chain=self.llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names,
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=self.agent, tools=tools, verbose=True)
        # self.agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True, max_iterations=5)

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
        # You will replace this function with your AI response function
        # For now, it just echoes back the user's input
        agent_response =  f"AI Response: {self.agent_executor.run(user_input)}"
        
        # Append the response to a text file
        with open('responses.txt', 'a') as f:
            f.write(agent_response + '\n')
        
        return agent_response

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
