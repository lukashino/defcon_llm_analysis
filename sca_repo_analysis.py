from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load Env Variables
from dotenv import load_dotenv

load_dotenv()

# For BedRock
from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings

faiss_db_path = f"./vector_databases/vtm_faiss"
#faiss_db_path = f"./OWASP_Top_10_2021_faiss"

db = FAISS.load_local(
    faiss_db_path,
    BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0"),
    allow_dangerous_deserialization=True,
)

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 100},
)

system_prompt_template = """
You are an agent designed to analyze Python/Django VTM web application for the owasp top 10 web vulnerabilities that a red team should focus on
The source code is located at ./repo/

### Analysis Process
1. Initial Review:
   - Identify OWASP Top 10 issues for a red team to focus on

2. Reflection Questions:
   Consider these questions carefully:
   - Is this an OWASP Top 10 issue?
   - Is this a web vulnerability?
   - Is this a red team issue?

3. Challenge Initial Assessment:
   - Is it really insecure
   - Am I certain
   - What would an attacker try first to bypass these controls?

### **TOOLS**
You have access to a vector database to search for code-related information. Use it to understand how custom functions handle authorization.

### **Output Format**
Your final response must be in the form of a red team report.  The red team report has the following sections:

1. Executive Summary for the read team penetration test effort
2. Methodology and Goals of the red team penetration test effort
3. Recommended Attack Vectors for the red team penetration test effort
4. Recommended Attack Narratives drawn in ascii art for the red team penetration test effort to follow
5. Conclusion of red team recommendations including a summary count of found high priority vulnerabilities for the red team penetration test effort to exploit



Your final response must be in the form of a red team report.  The red team report has the following sections:

1. Executive Summary
2. Methodology and Goals
3. Recommended Attack Vectors
4. Recommended Attack Narratives drawn in ascii art
5. Conclusion  

Begin!
"""

# CORRECT/FORMAL WAY TO PERFORM PROMPTING
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        ("human", """<question>{question}</question>"""),
    ]
)

llm = ChatBedrock(
    model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
    model_kwargs={"temperature": 0.6},
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# CHANGE AS DESIRED
user_question = """
Tell me about only the highest criticality security 
vulnerabilities found in the codebase. The categories should
loosely pertain to:

- Injection
- Broken Authentication
- Sensitive Data Exposure
- Unsafe deserialization
- Remote Code Execution
- Hardcoded Secrets
- CSRF
"""

for chunk in chain.stream(user_question):
    print(chunk, end="", flush=True)
