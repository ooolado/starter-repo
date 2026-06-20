"""Before/after demo: Bedrock Guardrail blocking a cooking prompt."""
import os, sys
from dotenv import load_dotenv

load_dotenv()

TEST_PROMPT = "Give me a step-by-step recipe to make chicken biryani."
MODEL = os.getenv("MONK_MODEL", "bedrock_converse:openai.gpt-oss-120b-1:0")

if __name__ == "__main__":
    if MODEL == "fake" or not MODEL.startswith("bedrock"):
        print("This demo requires a real Bedrock model. Set MONK_MODEL to a bedrock_converse:... model.")
        sys.exit(1)

    from langchain_aws import ChatBedrockConverse

    model_id = MODEL.split(":", 1)[1]  # strip "bedrock_converse:" prefix
    guardrail_id = os.getenv("BEDROCK_GUARDRAIL_ID", "")
    guardrail_version = os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT")

    kwargs: dict = {"model": model_id, "region_name": os.getenv("AWS_REGION", "us-east-1")}
    if guardrail_id:
        print("=== CASE 2: GUARDRAIL ON ===")
        print(f"    guardrailId={guardrail_id}  version={guardrail_version}")
        kwargs["guardrail_config"] = {
            "guardrailIdentifier": guardrail_id,
            "guardrailVersion": guardrail_version,
            "trace": "enabled",
        }
    else:
        print("=== CASE 1: NO GUARDRAIL (env not set) ===")

    llm = ChatBedrockConverse(**kwargs)
    resp = llm.invoke(TEST_PROMPT)

    print(f"\nResponse:\n{resp.content}")
    print(f"\nFull response_metadata:\n{resp.response_metadata}")
    stop_reason = resp.response_metadata.get("stopReason")

    if stop_reason == "guardrail_intervened":
        print("\nBLOCKED by guardrail ✅")
    else:
        print("\nAnswered freely (no block).")
