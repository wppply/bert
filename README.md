# Bert

```bash
# config project
export PROJECT_NAME=wmt-mtech-search-dp-sib-dev
gcloud config set project ${PROJECT_NAME}
 
# create bucket if necessary
export STORAGE_BUCKET=gs://tpu_dataset
# gsutil mb ${STORAGE_BUCKET}
 
# create instance
export IMAGE="debian-9-tf-1-15-v20200218"
export IMAGE_PROJECT="ml-images"
export ZONE="us-central1-b"
export INSTANCE_NAME="tpu-bert"
export INSTANCE_TYPE="n1-standard-8" # 8CPU

# network-interface matters, always vpc first
gcloud beta compute --project=${PROJECT_NAME} instances create ${INSTANCE_NAME} \
    --zone=${ZONE} \
    --machine-type=${INSTANCE_TYPE} \
        --network-interface subnet=projects/shared-vpc-admin/regions/us-central1/subnetworks/prod-us-central1-01,no-address \
        --network-interface subnet=tpu-network,no-address \
        --metadata=startup-script-url=gs://gcp-wmt-managed-gce-services/gcp-wmt-custom-setup.sh \
        --can-ip-forward --maintenance-policy=TERMINATE \
        --service-account=1079351012527-compute@developer.gserviceaccount.com \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --image=${IMAGE} \
        --image-project=${IMAGE_PROJECT} \
        --boot-disk-size=200GB \
        --boot-disk-type=pd-standard \
        --labels=applicationname=${INSTANCE_NAME},costcenter=unknown
 
# gcloud compute instances start $INSTANCE_NAME --zone ${ZONE} --internal-ip
gcloud compute ssh ${INSTANCE_NAME} --zone ${ZONE} --internal-ip



export STORAGE_BUCKET=gs://tpu_dataset
sudo apt-get install nmap
# pip3 install --user tensorflow-ranking==0.1.3    
# pip3 install --user tensorflow-text==2.2  

export TPU_NAME="tf115-bert"
export PYTHONPATH="${PYTHONPATH}:/usr/share/tpu/models"
ctpu up --zone=$ZONE  --machine-type=n1-standard-8 --tf-version=1.15 --name=$TPU_NAME --gcp-network tpu-network --tpu-only


sudo route add -net 10.240.1.0 netmask 255.255.255.248 gw 10.128.0.1



export BERT_BASE_DIR=gs://tpu_dataset/pre-train/BERT/multi_cased_L-12_H-768_A-12
export GLUE_DIR=gs://tpu_dataset/GLUE/glue_data

python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=gs://s0l04qa/mrpc_output/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --do_lower_case=False


python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=1000.0 \
  --output_dir=gs://s0l04qa/mrpc_output/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --do_lower_case=False \
  --num_train_steps=100000 \
  --iterations_per_loop 100

capture_tpu_profile --tpu=$TPU_NAME --logdir=gs://s0l04qa/mrpc_output/ 



##########
# Export # 
##########
export BERT_BASE_DIR=/Users/s0l04qa/Desktop/uncased_L-2_H-128_A-2
export GLUE_DIR=gs://tpu_dataset/GLUE/glue_data

python run_classifier.py \
  --task_name=MRPC \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=gs://s0l04qa/tiny_bert_ouput\
  --do_lower_case=False \
  --do_train \
  --do_export \
  --export_dir=/Users/s0l04qa/Desktop/bert_export


```