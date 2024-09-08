```bash
# Clone repo
git clone https://github.com/rlsu9/Open-Sora-Plan.git

# Prepare the env
cd Open-Sora-Plan
conda create -n cm_distill python=3.8 -y
conda activate cm_distill
pip install -e .
pip install -e ".[train]"
pip install -e '.[dev]'
pip install peft datasets bitsandbytes

# Prepare google cloud platform, skip if it is ready
curl https://sdk.cloud.google.com | bash
source ~/.bashrc
conda activate cm_distill
gcloud init

# Download reliance
mkdir reliance
gsutil -m cp -r gs://vid_gen/runlong_temp_folder_for_pandas70m_debugging/Open-Sora-Plan-v1.2.0 ./reliance
gsutil cp gs://vid_gen/runlong_temp_folder_for_pandas70m_debugging/video_mixkit_65f_54735.jsonl ./reliance

# Launch the job, need to do wandb login before launch to see the report
wandb login
git checkout dist_version

# adjust the deepspeed config marked in following sh file for dist training and launch.
bash osp_lmcm_x.sh


