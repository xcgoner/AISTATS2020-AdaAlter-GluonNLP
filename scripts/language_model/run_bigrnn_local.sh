export HOSTS=$1
export NGPU=$2
export OPTIMIZER=$3
export BS=$4
export LR=$5
export WARMUP=$6
export PY_VERSION="$7"
export OUTPUT_DIR=$8

if [ $PY_VERSION = 'py27' ]; then
  export PY='python27';
elif [ $PY_VERSION = 'py36' ]; then
  export PY='python36';
elif [ $PY_VERSION = 'py3' ]; then
  export PY='python3';
fi

HVD_PREFIX=" --hostfile hosts -mca pml ob1 \
             -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo --map-by ppr:4:socket \
             -x NCCL_MIN_NRINGS=16 -x NCCL_DEBUG=WARNING -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 \
             --tag-output ";

echo "killing running python processes ... ";
hudl -h $HOSTS "sudo pkill -9 $PY";

echo "creating host file ... ";

rm -f hosts;
while read line; do
  echo "$line slots=8" >> hosts
done < $HOSTS

HPARAMS=" --batch_size $BS --lr $LR --warmup-steps $WARMUP"

echo -e "\n====================== command: ====================== "
CMD=" $PY large_word_language_model_hvd_v1.py $HPARAMS --optimizer $OPTIMIZER "
echo -e "$CMD \n =====================================================\n"

echo -e "\n=================== mpirun command: ================== "
MPICMD="mpirun -np $NGPU $HVD_PREFIX -x MXNET_SAFE_ACCUMULATION=1 $CMD"
echo -e "$MPICMD \n =====================================================\n"

mpirun -np $NGPU --mca plm_rsh_agent 'ssh -q -o StrictHostKeyChecking=no' \
       $HVD_PREFIX -x MXNET_SAFE_ACCUMULATION=1 $CMD 2>&1 | tee $OUTPUT_DIR/result.log