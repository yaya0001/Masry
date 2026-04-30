# 🤖 Masry — Llama-3.2-3B LoRA Fine-tuning: Technical Troubleshooting Assistant

Fine-tuning **Meta Llama-3.2-3B-Instruct** on Stack Overflow Q&A data using **QLoRA (4-bit quantization)** via [Unsloth](https://github.com/unslothai/unsloth), then deploying the model behind a REST API on AWS. The result is a domain-adapted assistant that answers technical troubleshooting questions in a Stack Overflow style.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Phase 1 — Data Preparation](#phase-1--data-preparation)
- [Phase 2 — Fine-tuning (QLoRA)](#phase-2--fine-tuning-qlora)
- [Phase 3 — AWS Infrastructure (Terraform)](#phase-3--aws-infrastructure-terraform)
- [Phase 4 — Model Deployment & API Server](#phase-4--model-deployment--api-server)
- [Cost Summary Table](#cost-summary-table)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

| Property | Value |
|---|---|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` |
| Fine-tuning method | QLoRA (4-bit quantization + LoRA adapters) |
| Dataset | Stack Overflow Q&A — 79,901 examples |
| Training duration | ~4h 47m on NVIDIA RTX 5000 Ada (32 GB) |
| Trainable parameters | 9.2M / 3.2B (0.28%) |
| Final training loss | 2.09 → 1.76 over 2 epochs |
| Inference server | FastAPI / Streamlit on port 8501 |
| Cloud provider | AWS (us-east-1) |
| IaC tool | Terraform |

---

## Repository Structure

```
Masry/
├── finetune1_prep.py              # Phase 1 — dataset download & formatting
├── Masry_fine_tuning_code.ipynb  # Phase 2 — QLoRA training + evaluation
├── api_server.py                  # Phase 4 — model inference API server
├── main.tf                        # Phase 3 — Terraform AWS infrastructure
└── README.md
```

---

## Prerequisites

### Local Tools

| Tool | Version | Purpose |
|---|---|---|
| Python | ≥ 3.10 | All scripts |
| pip | ≥ 23.x | Package manager |
| Terraform | ≥ 1.6 | AWS infrastructure provisioning |
| AWS CLI | ≥ 2.x | Auth & resource management |
| Git | any | Clone the repo |
| CUDA toolkit | ≥ 12.1 | GPU training (local only) |
| Jupyter / JupyterLab | any | Running the training notebook |

Install Terraform:
```bash
# macOS
brew install terraform

# Ubuntu / Debian
sudo apt-get install -y gnupg software-properties-common
wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | \
  sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] \
  https://apt.releases.hashicorp.com $(lsb_release -cs) main" | \
  sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt-get update && sudo apt-get install terraform
```

Install AWS CLI:
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip awscliv2.zip && sudo ./aws/install
```

### Accounts & Access

| Account | Why you need it | Where to get it |
|---|---|---|
| AWS account | Hosts EC2, S3, VPC | [aws.amazon.com](https://aws.amazon.com) |
| Hugging Face account | Download Llama-3.2-3B (gated model) | [huggingface.co](https://huggingface.co) |
| Meta AI access request | Required to use Llama 3.2 weights | [ai.meta.com/resources/models-and-libraries/llama-downloads](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) |
| (Optional) Google account | Run notebook in Colab | [colab.research.google.com](https://colab.research.google.com) |

### AWS Region

All resources in this project target **`us-east-1` (N. Virginia)**. Do not switch regions mid-project — the S3 bucket name and AZ (`us-east-1a`) are hardcoded in `main.tf`.

### Minimum GPU Requirements

| Environment | VRAM required | Recommended instance |
|---|---|---|
| Local machine | ≥ 15 GB | Any RTX 3090 / 4090 / A-series |
| AWS EC2 | ≥ 16 GB | `g5.2xlarge` (24 GB A10G) |
| Google Colab | ≥ 15 GB | T4 GPU (free tier is sufficient) |

---

## Phase 1 — Data Preparation

This phase downloads and formats the Stack Overflow Q&A dataset into the instruction-tuning format expected by the training notebook.

### 1.1 Clone the repository

```bash
git clone https://github.com/yaya0001/Masry.git
cd Masry
```

### 1.2 Install Python dependencies

```bash
pip install datasets transformers tqdm
```

### 1.3 Authenticate with Hugging Face

```bash
pip install huggingface_hub
huggingface-cli login
# Paste your HF token when prompted (read access is sufficient for the dataset)
```

### 1.4 Run the data preparation script

```bash
python finetune1_prep.py
```

**What this script does:**
- Downloads the Stack Overflow Q&A dataset from Hugging Face
- Filters and cleans question/answer pairs
- Formats each record into the instruction-tuning template:

```json
{"text": "<s>[INST] Your question here [/INST] The answer here </s>"}
```

- Saves the processed dataset as `stackoverflow_qa_formatted.json` (or equivalent) locally

**Expected output:**
```
Downloading dataset...
Processed 79,901 examples
Saved to: ./data/stackoverflow_qa_formatted.json
```

### 1.5 (Optional) Upload dataset to S3

After Phase 3 provisions the S3 bucket, you can upload the dataset for use on EC2:

```bash
aws s3 cp ./data/stackoverflow_qa_formatted.json \
  s3://netid-25jpkj-cloud-storage-project/data/stackoverflow_qa_formatted.json
```

---

## Phase 2 — Fine-tuning (QLoRA)

The training is done inside `Masry_fine_tuning_code.ipynb`. You can run it locally, on Colab, or on the EC2 instance provisioned in Phase 3.

### Training Configuration

| Hyperparameter | Value |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Batch size (per device) | 4 |
| Gradient accumulation steps | 2 (effective batch = 8) |
| Epochs | 2 |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Warmup steps | 100 |
| Optimizer | adamw_8bit |
| Weight decay | 0.01 |
| Precision | bfloat16 |
| Max sequence length | 512 (with packing) |

### 2.1 Install training dependencies

```bash
pip install unsloth trl transformers datasets bitsandbytes peft accelerate rouge-score
```

> **Note:** On a fresh EC2 instance, also install Jupyter:
> ```bash
> pip install jupyter
> jupyter notebook --ip=0.0.0.0 --no-browser &
> ```

### 2.2 (Option A) Run on Google Colab

1. Open the notebook via the Colab badge in the repo.
2. Go to **Runtime → Change runtime type → T4 GPU**.
3. Upload `stackoverflow_qa_formatted.json` to your Google Drive.
4. Update the `DATA_PATH` variable in the *Load dataset* cell.
5. Click **Runtime → Run all**.

### 2.3 (Option B) Run locally or on EC2

```bash
# Launch Jupyter
jupyter notebook Masry_fine_tuning_code.ipynb

# Or run as a script if you convert the notebook first
jupyter nbconvert --to script Masry_fine_tuning_code.ipynb
python Masry_fine_tuning_code.py
```

Update `DATA_PATH` inside the notebook to point to your formatted JSON:

```python
DATA_PATH = "./data/stackoverflow_qa_formatted.json"   # local path
# or
DATA_PATH = "/mnt/s3/data/stackoverflow_qa_formatted.json"  # if mounted from S3
```

### 2.4 Monitor training

Training loss should decrease from ~2.09 to ~1.76 over 2 epochs (~4h 47m on 32 GB GPU). Monitor via the progress bar printed to the notebook output.

### 2.5 Evaluate perplexity

After training completes, run the evaluation block inside the notebook:

```python
eval_loss, eval_ppl = compute_perplexity(eval_dataset)
print(f"Loss: {eval_loss:.4f} | Perplexity: {eval_ppl:.2f}")
```

### 2.6 Save the fine-tuned adapters

The notebook saves LoRA adapters to `./llama3-3b-finetuned/`. Upload to S3 for use in deployment:

```bash
aws s3 sync ./llama3-3b-finetuned/ \
  s3://netid-25jpkj-cloud-storage-project/models/llama3-3b-finetuned/
```

---

## Phase 3 — AWS Infrastructure (Terraform)

`main.tf` provisions all AWS networking and storage needed to host the model API.

### Resources created

| Resource | Name | Details |
|---|---|---|
| VPC | NetID-25jpkj-vpc | CIDR 10.0.0.0/16, DNS enabled |
| Subnet | NetID-25jpkj-subnet | 10.0.1.0/24, AZ us-east-1a, public IPs |
| Internet Gateway | NetID-25jpkj-igw | Attached to VPC |
| Route Table | NetID-25jpkj-rt | Default route → IGW |
| Security Group | NetID-25jpkj-sg | Ingress: SSH (22), Streamlit (8501); Egress: all |
| S3 Bucket | netid-25jpkj-cloud-storage-project | Dataset & model artifact storage |

> **Note:** The Terraform file does not include an EC2 instance resource. You must launch the GPU instance manually (see 3.5) or add an `aws_instance` block to `main.tf`.

### 3.1 Configure AWS credentials

```bash
aws configure
# AWS Access Key ID:     <your-key-id>
# AWS Secret Access Key: <your-secret>
# Default region name:   us-east-1
# Default output format: json
```

### 3.2 Initialise Terraform

```bash
cd Masry/
terraform init
```

### 3.3 Preview the plan

```bash
terraform plan
```

Expected output: **7 resources to add**, 0 to change, 0 to destroy.

### 3.4 Apply the infrastructure

```bash
terraform apply -auto-approve
```

After ~60 seconds you should see:

```
Apply complete! Resources: 7 added, 0 changed, 0 destroyed.
```

### 3.5 Launch an EC2 GPU instance (manual)

Because the Terraform file does not define an EC2 instance, launch one manually in the provisioned subnet:

```bash
# Find the latest Deep Learning AMI (GPU PyTorch)
aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*" \
            "Name=architecture,Values=x86_64" \
  --query "sort_by(Images,&CreationDate)[-1].ImageId" \
  --output text

# Export the AMI ID returned above
export AMI_ID=ami-xxxxxxxxxxxxxxxxx

# Get the subnet ID from Terraform output
export SUBNET_ID=$(terraform output -raw subnet_id 2>/dev/null || \
  aws ec2 describe-subnets \
    --filters "Name=tag:Name,Values=NetID-25jpkj-subnet" \
    --query "Subnets[0].SubnetId" --output text)

# Get the security group ID
export SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=NetID-25jpkj-sg" \
  --query "SecurityGroups[0].GroupId" --output text)

# Launch a g5.2xlarge (24 GB A10G GPU)
aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type g5.2xlarge \
  --key-name <your-key-pair-name> \
  --subnet-id $SUBNET_ID \
  --security-group-ids $SG_ID \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=masry-inference-server}]' \
  --count 1
```

### 3.6 SSH into the instance

```bash
# Get the public IP
export EC2_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=masry-inference-server" \
            "Name=instance-state-name,Values=running" \
  --query "Reservations[0].Instances[0].PublicIpAddress" \
  --output text)

ssh -i ~/.ssh/<your-key-pair>.pem ubuntu@$EC2_IP
```

### 3.7 Pull model from S3 onto EC2

```bash
aws s3 sync s3://netid-25jpkj-cloud-storage-project/models/llama3-3b-finetuned/ \
  ~/llama3-3b-finetuned/
```

### 3.8 Tear down infrastructure when done

```bash
# Terminate EC2 instance first (manually or via CLI)
aws ec2 terminate-instances --instance-ids <instance-id>

# Then destroy Terraform-managed resources
terraform destroy -auto-approve
```

---

## Phase 4 — Model Deployment & API Server

`api_server.py` loads the fine-tuned LoRA adapters and exposes the model as a web service (Streamlit on port **8501**).

### 4.1 Install server dependencies on EC2

```bash
pip install unsloth transformers bitsandbytes peft accelerate streamlit fastapi uvicorn
```

### 4.2 Set model path

Edit `api_server.py` and set the path to your saved LoRA adapters:

```python
MODEL_PATH = "/home/ubuntu/llama3-3b-finetuned"   # update if different
```

### 4.3 Start the server

```bash
# Run in the background with nohup so it survives SSH disconnection
nohup python api_server.py > api_server.log 2>&1 &

# Or for Streamlit-based UI
nohup streamlit run api_server.py --server.port 8501 --server.address 0.0.0.0 \
  > streamlit.log 2>&1 &
```

### 4.4 Access the interface

Open your browser and navigate to:

```
http://<EC2_PUBLIC_IP>:8501
```

The security group already allows inbound traffic on port 8501.

### 4.5 Test the API (curl)

```bash
curl -X POST http://<EC2_PUBLIC_IP>:8501/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How do I fix a segmentation fault in C++?"}'
```

### 4.6 Load the model programmatically

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "/home/ubuntu/llama3-3b-finetuned",
    max_seq_length = 1024,
    dtype          = None,
    load_in_4bit   = True,
)
FastLanguageModel.for_inference(model)

inputs = tokenizer(
    "<s>[INST] How do I fix a NullPointerException in Java? [/INST]",
    return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Cost Summary Table

All prices are **approximate** and based on AWS **us-east-1** on-demand pricing (as of 2025). Actual spend depends on usage duration and data volume.

| AWS Service | Resource / Config | Unit Price | Estimated Usage | Approx. Cost |
|---|---|---|---|---|
| **EC2** | `g5.2xlarge` (A10G 24 GB) | $1.006/hr | ~6 hrs training + 8 hrs inference | **~$14.08** |
| **EC2 EBS** | gp3 100 GB root volume | $0.08/GB-month | 1 day (~0.033 month) | **~$0.26** |
| **S3** | Storage (dataset ~2 GB + model ~7 GB) | $0.023/GB-month | ~9 GB × 1 month | **~$0.21** |
| **S3** | PUT/GET requests | $0.005 per 1K PUT | ~5,000 PUT requests | **~$0.03** |
| **VPC / IGW** | Internet Gateway data transfer | $0.09/GB out | ~5 GB egress | **~$0.45** |
| **Data Transfer** | S3 ↔ EC2 (same region) | Free | — | **$0.00** |
| **Terraform / CLI** | No cost for IaC tooling | Free | — | **$0.00** |
| | | | **Total estimate** | **~$15.03** |

> **💡 Cost-saving tips:**
> - Use a **Spot Instance** for the `g5.2xlarge` to cut EC2 cost by ~70% (~$4 vs ~$14).
> - **Stop** (don't terminate) the EC2 instance between sessions to avoid re-downloading the model.
> - Enable **S3 Intelligent-Tiering** if you store artifacts for more than 30 days.
> - Use **Google Colab** (free T4) for the training phase and reserve EC2 for inference only.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA out of memory` | Batch size too large | Reduce `per_device_train_batch_size` to 2 |
| `ModuleNotFoundError: unsloth` | Missing install | `pip install unsloth` |
| `Access Denied` on S3 | Missing IAM permissions | Attach `AmazonS3FullAccess` policy to your IAM user/role |
| Port 8501 not reachable | SG rule missing or EC2 not running | Verify the security group and that the server process is up |
| Terraform `BucketAlreadyExists` | S3 bucket name is global | Rename the bucket in `main.tf` to something unique |
| `ValueError: Unrecognized model` | Wrong model path | Double-check `MODEL_PATH` in `api_server.py` |
| Hugging Face 401 on model download | Not logged in or no Meta approval | Run `huggingface-cli login` and request Llama 3.2 access |

---

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) — optimised QLoRA training that makes 3B fine-tuning feasible on a single GPU
- [Meta AI](https://ai.meta.com/) — Llama 3.2 base model
- [Stack Overflow](https://stackoverflow.com/) / Hugging Face — Q&A dataset
- [Hugging Face TRL](https://github.com/huggingface/trl) — `SFTTrainer` used for supervised fine-tuning
- [HashiCorp Terraform](https://www.terraform.io/) — infrastructure-as-code for AWS provisioning

---

*README generated from full repository analysis — `yaya0001/Masry`*
