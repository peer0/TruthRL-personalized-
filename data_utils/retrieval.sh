# Refer to Search-R1 repo for retrieval setup

DATA_LIST=(
    "nq"
    "hotpotqa"
    "musique"
)

SPLIT='train'
TOPK=50
INDEX_PATH=Search-R1/retrieval_corpus
CORPUS_PATH=Search-R1/retrieval_corpus/wiki-18.jsonl
SAVE_NAME=e5_${TOPK}_wiki18.json

for DATA_NAME in ${DATA_LIST[@]}; do
    echo "Processing $DATA_NAME"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python retrieval.py --retrieval_method e5 \
                    --retrieval_topk $TOPK \
                    --index_path $INDEX_PATH \
                    --corpus_path $CORPUS_PATH \
                    --dataset_name $DATA_NAME \
                    --data_split $SPLIT \
                    --retrieval_model_path "intfloat/e5-base-v2" \
                    --retrieval_pooling_method "mean" \
                    --retrieval_batch_size 512
done