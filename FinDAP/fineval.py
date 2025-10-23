from datasets import DatasetDict, Dataset
from datasets import load_dataset, load_from_disk
import os
from datasets import DatasetDict, load_dataset, GeneratorBasedBuilder, SplitGenerator, Split

# fin_eval = load_dataset('Salesforce/FinEval', 'CFA-Challenge')
# print('fin_eval: ',fin_eval['test']['query'])

# exit()


# Get HuggingFace token from environment variable
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError("HF_TOKEN environment variable is required but not set. Please set it with: export HF_TOKEN=your_token_here") 

# repo_id = 'ZixuanKe/FinEval'
repo_id = 'Salesforce/FinEval'

fpb = load_dataset('TheFinAI/en-fpb', token=hf_token, split='test')
fiqasa = load_dataset('TheFinAI/flare-fiqasa', token=hf_token, split='test')
fomc = load_dataset('TheFinAI/flare-fomc', token=hf_token, split='test')
ner = load_dataset('ZixuanKe/flare-ner', token=hf_token, split='test')
edtsum = load_dataset('ChanceFocus/flare-edtsum', token=hf_token, split='test')




mmlu_finance = load_dataset('ZixuanKe/MMLU-finance', token=hf_token)
# Create a new DatasetDict excluding the train split
mmlu_finance = DatasetDict({
    "test": mmlu_finance["train"]  # Rename train to test, or keep test as it is
})

ectsum = load_dataset('ZixuanKe/flare-ectsum', token=hf_token, split='test')
ectsum = ectsum.rename_column("label_text", "summary")


ma = load_dataset('TheFinAI/flare-ma', token=hf_token, split='test')
mlesg = load_dataset('TheFinAI/flare-mlesg', token=hf_token, split='test')
bigdata = load_dataset('TheFinAI/flare-sm-bigdata', token=hf_token, split='test')
acl = load_dataset('TheFinAI/flare-sm-acl', token=hf_token, split='test')
cikm = load_dataset('TheFinAI/flare-sm-cikm', token=hf_token, split='test')
ccf = load_dataset('ChanceFocus/cra-ccf', token=hf_token, split='test')
ccfraud = load_dataset('TheFinAI/cra-ccfraud', token=hf_token, split='test')
german = load_dataset('ChanceFocus/flare-german', token=hf_token, split='test')
australian = load_dataset('ChanceFocus/flare-australian', token=hf_token, split='test')
lendingclub = load_dataset('ChanceFocus/cra-lendingclub', token=hf_token, split='test')
polish = load_dataset('ChanceFocus/cra-polish', token=hf_token, split='test')
taiwan = load_dataset('TheFinAI/cra-taiwan', token=hf_token, split='test')
protoseguro = load_dataset('TheFinAI/cra-portoseguro', token=hf_token, split='test')
travel_insturance = load_dataset('TheFinAI/cra-travelinsurace', token=hf_token, split='test')
tatqa = load_dataset('TheFinAI/flare-tatqa', token=hf_token, split='test')
finance_bench = load_dataset('ChanceFocus/cra-lendingclub', token=hf_token, split='test')

cfa_challenge = load_dataset('ZixuanKe/CFA-Level-III', token=hf_token)
# Create a new DatasetDict excluding the train split
cfa_challenge = DatasetDict({
    "test": cfa_challenge["train"]  # Rename train to test, or keep test as it is
})

cfa_challenge = cfa_challenge.remove_columns("explain_query")
cfa_challenge = cfa_challenge.remove_columns("gpt-query")
cfa_challenge = cfa_challenge.remove_columns("explain-query")

cfa_easy = load_dataset('TheFinAI/flare-cfa', token=hf_token, split='test')



# fpb.push_to_hub(repo_id)
# exit()

fpb.push_to_hub(repo_id, 'FPB')
fiqasa.push_to_hub(repo_id, 'FIQASA')
fomc.push_to_hub(repo_id, 'FOMC')
ner.push_to_hub(repo_id, 'NER')

edtsum.push_to_hub(repo_id, 'EDTSUM')
mmlu_finance.push_to_hub(repo_id, 'MMLU-finance')

ectsum.push_to_hub(repo_id, 'ECTSUM')
ma.push_to_hub(repo_id, 'MA')
mlesg.push_to_hub(repo_id, 'MLESG')

bigdata.push_to_hub(repo_id, 'CRA-Bigdata')
acl.push_to_hub(repo_id, 'SM-ACL')
cikm.push_to_hub(repo_id, 'SM-CIKM')

ccf.push_to_hub(repo_id, 'CRA-CCF')
ccfraud.push_to_hub(repo_id, 'CRA-CCFraud')

lendingclub.push_to_hub(repo_id, 'CRA-LendingClub')
german.push_to_hub(repo_id, 'Flare-German')
australian.push_to_hub(repo_id, 'Flare-Australian')

lendingclub.push_to_hub(repo_id, 'CRA-LendingClub')
polish.push_to_hub(repo_id, 'CRA-Polish')
taiwan.push_to_hub(repo_id, 'CRA-Taiwan')
protoseguro.push_to_hub(repo_id, 'CRA-ProtoSeguro')
travel_insturance.push_to_hub(repo_id, 'CRA-TravelInsurance')
tatqa.push_to_hub(repo_id, 'Flare-TATQA')

finance_bench.push_to_hub(repo_id, 'FinanceBench')

cfa_challenge.push_to_hub(repo_id, 'CFA-Challenge')
cfa_easy.push_to_hub(repo_id, 'CFA-Easy')
