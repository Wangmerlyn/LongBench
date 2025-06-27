source /opt/conda/etc/profile.d/conda.sh 
conda create --name cic python==3.10 -y
conda activate cic
which python
pip install numpy==1.23.5
pip install tqdm
# pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install openai
pip install tiktoken
# pip install transformers==4.44.2
pip install datasets
pip install accelerate
# pip install flash-attn==2.5.6
pip install azure-identity

pip install vllm -U
pip install flash-attn
export only_last_logits=1

# pip install vllm
# export only_last_logits=1

# # sglang
# pip install --upgrade pip
# pip install sgl-kernel --force-reinstall --no-deps
# pip install "sglang[all]>=0.4.3" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python

