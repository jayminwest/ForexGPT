import streamlit as st
from dataclasses import dataclass
from typing import Literal
import streamlit as st
import streamlit.components.v1 as components
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os
from langchain.callbacks import get_openai_callback
from langchain import LLMMathChain, OpenAI, SerpAPIWrapper, SQLDatabase, SQLDatabaseChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

@dataclass
class Message:
    origin: Literal["human", "ai"]
    message: str

class ChatApplication:
    def __init__(self):
        load_dotenv()

        st.title("Forex Trading Advisor:")
        st.subheader("Powered by LangChain 🦜🔗")

        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        self.chat_placeholder = st.container()
        self.prompt_placeholder = st.form("chat-form")
        self.log_placeholder = st.empty()

        self.search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERP_API_KEY"))
        self.tools = [
            Tool(
                name="Search", 
                func=self.search.run, 
                description="useful when you need to find current prices of currency pairs"
            )
        ]

        self.initialize_session_state()

    def get_user_position_inputs(self):
        # Side bar for user input
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
            risk_amount = account_balance * (risk_percentage / 1000.0)
            position_size = risk_amount / stop_loss
            st.sidebar.markdown(f'**Position Size:** {position_size}')
            
            return account_balance, risk_percentage, stop_loss, currency_pair, current_ask_price, position_size
        else:
            return None, None, None, None, None, None

    def initialize_session_state(self):
        if "history" not in st.session_state:
            st.session_state.history = []
        if "token_count" not in st.session_state:
            st.session_state.token_count = 0
        if "agent" not in st.session_state:
            llm = ChatOpenAI(
                temperature=0,
                openai_api_key=self.openai_api_key
            )
            llm.model_name="gpt-3.5-turbo-0613"
            st.session_state.agent = initialize_agent(
                self.tools,
                llm,
                agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
                verbose=True, 
                memory=ConversationBufferMemory(return_messages=True),
                handle_parsing_errors=True
            )

    def on_click_callback(self):
        with get_openai_callback() as cb:
            human_prompt = st.session_state.human_prompt
            history = st.session_state.history

            llm_response = st.session_state.agent.run(
                    input=human_prompt,
                    chat_history=history
                )

            st.session_state.history.append(
                Message("human", human_prompt)
            )
            st.session_state.history.append(
                Message("ai", llm_response)
            )

            st.session_state.token_count += cb.total_tokens
            if self.account_balance is not None:  # To ensure the message is only added if user inputs have been submitted
                st.session_state.history.append(
                    Message("system", f"Account Balance: {self.account_balance}, Risk Percentage: {self.risk_percentage}, Stop Loss: {self.stop_loss}, Currency Pair: {self.currency_pair}, Current Ask Price: {self.current_ask_price}, Position Size: {self.position_size}")
            )

    def run(self): 
        self.account_balance, self.risk_percentage, self.stop_loss, self.currency_pair, self.current_ask_price, self.position_size = self.get_user_position_inputs()
        
        with self.chat_placeholder:
            for chat in st.session_state.history:
                div = f"""
        <div class="chat-row 
            {'' if chat.origin == 'ai' else 'row-reverse'}">
            <img class="chat-icon" src="app/static/{
                'ai_icon.png' if chat.origin == 'ai'
                else 'user_icon.png'}"
                width=32 height=32>
            <div class="chat-bubble
            {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
                &#8203;{chat.message}
            </div>
        </div>
                """
                st.markdown(div, unsafe_allow_html=True)

            for _ in range(3):
                st.markdown("")

        with self.prompt_placeholder:
            st.markdown("**Chat**")
            cols = st.columns((6, 1))
            cols[0].text_input(
                "Chat",
                value="Enter Question Here",
                label_visibility="collapsed",
                key="human_prompt",
            )
            cols[1].form_submit_button(
                "Submit",
                type="primary",
                on_click=self.on_click_callback,
            )

        self.log_placeholder.caption(f"""
    Used {st.session_state.token_count} tokens \n
    Debug Langchain conversation: 
    {st.session_state.agent.memory.buffer}
    """)

app = ChatApplication()
app.run()
