import os

from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient, RetryPolicy

load_dotenv()

TYPESAFE_API_KEY= os.environ.get("TYPESAFE_API_KEY")

RETRY = RetryPolicy(
    max_retries=6,
    backoff_initial=1.0,
    backoff_max=30.0,
    timeout=None
)

def ask_jev(state: dict, questions: dict) -> dict:
    client = TypeSafeClient(
        api_key=TYPESAFE_API_KEY,
        timeout=120.0,
        retry=RETRY
    )

    response = client.system_one(
        state=state,
        questions=questions,
        model="jev-latest"
    )

    return response