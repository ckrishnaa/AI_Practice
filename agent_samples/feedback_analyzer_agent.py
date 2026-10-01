"""
Simple agent to read the customer feedback from CSV.
Review the feedback and provide a sentiment score, priority and response
llm -> using groq and gemini models
"""

import logging
import os
import warnings

# Suppress all standard UserWarnings (like the AFC chat session notice)
warnings.filterwarnings("ignore", category=UserWarning)

# Force the root logger to ignore anything below ERROR
logging.basicConfig(level=logging.ERROR)

# Explicitly silence every variant of the Google API loggers
for logger_name in ["google", "google_genai", "google.genai", "langchain_google_genai", "langchain", "langchain_groq", "groq"]:
    log = logging.getLogger(logger_name)
    log.setLevel(logging.ERROR)
    log.propagate = False

import random
import time
from typing import Literal, cast

import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

# Define the desired output structure using Pydantic
class SupportAnalysis(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(
        description="The emotional tone of the customer review"
    )
    priority: Literal['high', 'medium', 'low'] = Field(
        description="How urgently this review needs attention from a human agent"
    )
    drafted_reply: str = Field(
        description="A brief, professional 2-3 sentence email response addressing the customer's specific feedback"
    )

def promt_chain(llmStr: str):
    # Initialize the LangChain LLM
    if llmStr == "gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", api_key=os.environ["GEMINI_API_KEY"], temperature=0)
    elif llmStr == "groq":
        llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    else:
        raise ValueError(f"Unsupported LLM: {llmStr}")

    # Bind the Pydantic schema to force structured JSON output
    structured_llm = llm.with_structured_output(SupportAnalysis)

    # Create the prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an elite customer support triage agent. Analyze the customer's review and extract the required information structured exactly as requested."),
            ("user", "Customer Name: {customer_name}\nReview Text: {review_text}")
        ]
    )

    # Create the prompt chain
    agent_chain = prompt_template | structured_llm
    return agent_chain


# Main Workflow Execution
def main(llmStr: str):
    input_file = "customer_feedback.csv"
    output_file = "customer_feedback_analyzed.csv"

    if not os.path.exists(input_file):
        print(f"Error: Please ensure {input_file} exists in this directory.")
        return

    print("Reading customer feed via LangChain pipeline...\n")
    df = pd.read_csv(input_file)

    # Initialize the agent chain with the provided LLM string
    agent_chain = promt_chain(llmStr)

    results = []

    print("Running reviews through LangChain agent...\n")
    for count, (index, row) in enumerate(df.iterrows(), start=1):
        print(f"Processing row {count}: Review by {row['CustomerName']}...")

        # Add a mandatory 1-second pause between requests to prevent overwhelming the free tier
        time.sleep(1)

        analysis_dict = None
        max_retries = 3

        for attempt in range(max_retries):
            try:
                # Invoke the chain; LangChain automatically manages formatting and parsing
                raw_analysis = agent_chain.invoke({
                    "customer_name": row['CustomerName'],
                    "review_text": row['ReviewText']
                })

                # Explicitly cast the generic output to your Pydantic model
                analysis = cast(SupportAnalysis, raw_analysis)
                analysis_dict = analysis.model_dump()
                #analysis_dict = {
                #    "ReviewID": row['ReviewID'],
                #    "CustomerName": row['CustomerName'],
                #    "ReviewText": row['ReviewText'],
                #    "Sentiment": analysis.sentiment,
                #    "Priority": analysis.priority,
                #    "DraftedReply": analysis.drafted_reply
                #}
                #results.append(analysis_dict)
                break # Success! Exit retry block
            except ValidationError as val_err:
                print(f"Schema validation failed on row {count}: {val_err}")
                results.append({
                    "ReviewID": row['ReviewID'],
                    "CustomerName": row['CustomerName'],
                    "ReviewText": row['ReviewText'],
                    "Sentiment": "Error",
                    "Priority": "High",
                    "DraftedReply": "Failed to generate reply due to an error."
                })
                break  # Stop retrying if it's a data validation schema issue
            except RuntimeError as run_err:
                print(f" Runtime Error on row {count}: {run_err}")
                # Check if it looks like a server/overload error (like 503 or 429)
                if "503" in str(run_err) or "demand" in str(run_err).lower() or "429" in str(run_err):
                    wait_time = (attempt + 1) * 2 + random.uniform(0, 1)
                    print(f"   [Server Busy] Attempt {attempt + 1} failed. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    # If it's another runtime issue, log it and stop retrying this row
                    print(f" Runtime Error on row {count}: {run_err}")
                    analysis_dict = {
                        "sentiment": "Error",
                        "priority": "High",
                        "drafted_reply": "API execution runtime failed."
                    }
                    break # Stop retrying if it's a runtime issue
        # If all retries failed entirely
        if not analysis_dict:
            analysis_dict = {
                "sentiment": "Error",
                "priority": "High",
                "drafted_reply": "Server unavailable after maximum retries."
            }

        results.append({
            "ReviewID": row['ReviewID'],
            "CustomerName": row['CustomerName'],
            "ReviewText": row['ReviewText'],
            "Sentiment": analysis_dict["sentiment"],
            "Priority": analysis_dict["priority"],
            "DraftedReply": analysis_dict["drafted_reply"]
        })

    # Save structured results to a fresh CSV file
    processed_df = pd.DataFrame(results)
    processed_df.to_csv(output_file, index=False)
    print(f"\nCompleted! Saved structured insights to '{output_file}'")

if __name__ == "__main__":
    #main("gemini")
    main("groq")
