import os
#import serpapi
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
#from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()
#SERPAPI_API_KEY='e0032eba753263ba5721da636562fd922e932a52f41a1c33515977e25051fc46'
ser_key=os.getenv('SERPAPI_API_KEY')
openai_key=os.getenv('OPENAI_API_KEY')
MODEL='gpt-4o'
#print(type(ser_key))
# print(openai_key)
os.environ["OPENAI_API_KEY"] = openai_key
os.environ["SERPAPI_API_KEY"] = ser_key
messages = [
    (
        "system",
        "你是個AI助手，你很理解日本流行文化，要盡可能回答我的問題",
    ),
    ("human", "你知道HOLOLIVE嗎?，那你知道湊阿夸要畢業了嗎?"),
]
llm = ChatOpenAI(model=MODEL,temperature=0,max_tokens=1024)
#json_llm=llm.bind(response_format={"type": "json_object"})
response=llm.invoke(messages)
print(response)

# tools = load_tools(["serpapi"])
# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# agent.run("What's the date today? What great events have taken place today in history?")