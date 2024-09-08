git clone https://github.com/rlsu9/Open-Sora-Plan.git

cd Open-Sora-Plan

conda create -n cm_distill python=3.8 -y
conda activate cm_distill
pip install -e .
pip install -e ".[train]"
pip install -e '.[dev]'

mkdir reliance
curl https://sdk.cloud.google.com | bash
source ~/.bashrc
conda activate cm_distill
gcloud init

gsutil -m cp -r gs://vid_gen/runlong_temp_folder_for_pandas70m_debugging/Open-Sora-Plan-v1.2.0 ./reliance

gsutil cp gs://vid_gen/runlong_temp_folder_for_pandas70m_debugging/video_mixkit_65f_54735.jsonl ./reliance

git checkout dist_version

pip install peft datasets bitsandbytes

bash osp_lmcm_x.sh


