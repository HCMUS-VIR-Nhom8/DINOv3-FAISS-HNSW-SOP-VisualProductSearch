#!/usr/bin/env bash
set -e

# Cach A (khuyen nghi, model.backend=hf trong configs/*.yaml): khong can script nay,
# chi can accept license tren Hugging Face roi login:
pip install -q -U "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli download facebook/dinov3-vitb16-pretrain-lvd1689m

# Cach B (model.backend=torchhub): xin quyen tai checkpoint qua form cua Meta
# (https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/), Meta se
# gui URL qua email, roi tai bang wget (KHONG dung trinh duyet):
#   wget "<URL_DUOC_GUI_QUA_EMAIL>" -O external/dinov3_vitb16_checkpoint.pth
# Xem external/README.md de biet chi tiet.
