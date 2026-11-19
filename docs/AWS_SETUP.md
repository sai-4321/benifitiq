# AWS setup: Claude on Amazon Bedrock

## 1. Create the AWS account
1. Go to **https://aws.amazon.com/free** → *Create a free account*. Verify email, phone and a **Visa/Mastercard** card. New accounts get promotional credits (currently up to $200) that can be spent on Bedrock, including Claude.
2. When asked, choose the **Paid plan**, or upgrade from the Free plan later (**Billing → Plans**). Claude is sold through AWS Marketplace, and Free-plan or brand-new accounts sometimes can't subscribe to it. Unused credits carry over when you upgrade, so you still pay nothing until they run out.
3. Protect yourself: sign in as root once, turn on **MFA** (top-right name → *Security credentials*), then **Billing → Budgets → Create budget → Monthly cost budget** of **$10** with an email alert.

## 2. Open Bedrock in the right region
1. In the console search bar type **Bedrock** → open **Amazon Bedrock**.
2. Top-right region selector → **US East (N. Virginia) us-east-1**. Every model in this project is there. The `us.` model IDs we use are *cross-region inference profiles*: Bedrock may run your request in any US region for capacity, and you pay the same price.

## 3. Fill in Anthropic's one-time use-case form (required for Claude)
1. Left menu → **Model catalog** → filter Provider = **Anthropic** → open **Claude Sonnet 5**.
2. Click **Open in playground** (or *Request access*). The first time, a **use case details** form appears: company name (you can write "Personal portfolio project"), website (your GitHub or LinkedIn URL), intended users = *Internal*, use case = "Learning project: RAG assistant over public Medicare documents with synthetic data". Submit.
3. Access is normally immediate. Type "Hello" in the playground. If Claude answers, you're done. Do the same quick test with **Claude Haiku 4.5**.
4. **Titan Text Embeddings V2** (provider *Amazon*) needs no form.

> If you skip this step every call fails with `ResourceNotFoundException: Model use case details have not been submitted for this account`.

## 4. Create credentials - pick ONE option

*Option A - Bedrock API key (easiest, recommended for this project)*
1. Bedrock console → left menu **API keys** → **Long-term API keys** tab → **Generate long-term API keys**.
2. Expiry: **30 days** (renew when it expires). Click **Generate** and **copy the key now**. It starts with `bedrock-api-key-` and is shown only once. AWS automatically creates a small IAM user for it with the **AmazonBedrockLimitedAccess** policy.
3. Paste it into `.env` as `AWS_BEARER_TOKEN_BEDROCK=...`. (Or set it for your whole Windows user: `setx AWS_BEARER_TOKEN_BEDROCK "bedrock-api-key-..."` and open a new terminal.)
4. AWS recommends long-term keys for learning only. In production you use IAM roles or short-term (12-hour) keys. Say this in interviews.

*Option B - IAM user + AWS CLI (the classic way)*
1. Console → **IAM → Users → Create user** `benefitiq-dev` → *Attach policies directly* → **AmazonBedrockLimitedAccess** → Create.
2. Open the user → **Security credentials → Create access key** → *Command Line Interface (CLI)* → copy the Access key ID and Secret.
3. In PowerShell: `aws configure` → paste both → region `us-east-1` → output `json`. Keys are saved in `C:\Users\<you>\.aws\credentials`, and boto3 finds them automatically. Leave `AWS_BEARER_TOKEN_BEDROCK` as the placeholder in `.env`.

## 5. Check it works (1 minute, costs a fraction of a cent)
```powershell
python -c "import boto3; r=boto3.client('bedrock-runtime',region_name='us-east-1').converse(modelId='us.anthropic.claude-sonnet-5',messages=[{'role':'user','content':[{'text':'Say hi in 3 words'}]}]); print(r['output']['message']['content'][-1]['text'])"
```
(Run it after `pip install -r requirements.txt`. With Option A, set the key first in the same terminal: `$env:AWS_BEARER_TOKEN_BEDROCK="bedrock-api-key-..."`.)

**Cost (on-demand, us-east-1):** Claude Sonnet 5 about **$2 per 1M input tokens / $10 per 1M output tokens**, Claude Haiku 4.5 about **$1 / $5**, Titan Embeddings V2 about **$0.02 per 1M tokens**. Embedding all four booklets costs about one cent. One question costs roughly half a cent, and a full 44-question eval with the judge costs well under $1. The cache makes repeat questions $0. Check https://aws.amazon.com/bedrock/pricing/ for current prices.

## 6. LangSmith tracing (optional, free Developer plan)

Sign up at https://smith.langchain.com, create an API key (Settings -> API Keys), then in `.env` set
`LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY=...`. Every request then shows up as a trace with each agent
step, tokens and latency. Raw (unmasked) questions are redacted from traces.

## 7. Verify

```powershell
python scripts/check_bedrock.py
```

Expect three `OK` lines (Sonnet, Haiku, Titan). Each `FAIL` line prints the most likely fix.
