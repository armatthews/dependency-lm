#!/usr/bin/env sh
#SBATCH --time 7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=20000 # in MB
#SBATCH --export ALL
#SBATCH --gres=gpu:1
#SBATCH --job-name "td_is_test"

# TD
model_type_dir=../td-deplm
model=$model_type_dir/td_1024_tied_l2_d0.5
flags="--hidden_dim 1024 --layers 2 --tied --minibatch_size 16 --autobatch --sent_level"
train_corpus=/home/austinma/git/DepLM/ptb/dep-dataset/new/train.02-21.sd.proj.unk.oracle2

# BU
#model_type_dir=../bu-deplm
#model=$model_type_dir/dep_1024_tied_l2_d0.5_4
#flags="--hidden_dim 1024 --layers 2 --tied --minibatch_size 16 --autobatch --sent_level"
#train_corpus=/home/austinma/tmp/train.02-21.sd.proj.unk.oracle #BU

#samples=/home/austinma/git/lstm-parser/build/parser/stuff/dev_samples_norel_t0.5_2
samples=/home/austinma/git/lstm-parser/build/parser/stuff/test_samples_norel_t0.5_2
oracle_script=$model_type_dir/oracle.py
n=1000

set -x
set -eou pipefail
#tmproot=/home/austinma/tmp/
tmproot=/projects/tir2/users/austinma/tmp/
#tmp=$(mktemp -d --tmpdir=$tmproot)
tmp=/projects/tir2/users/austinma/tmp/tmp.kmNzrltJWn
echo "Evaluating $model w.r.t. $samples" >&2
echo "Using $tmp as temporary directory." >&2
grep -v '	\|^$' $samples > $tmp/dscores
cat $samples | grep '	\|^$' | sed 's/_LRB_/-LRB-/g' | sed 's/_RRB_/-RRB-/g' > $tmp/s
#cat $tmp/s | python $oracle_script > $tmp/o
#cat $tmp/o | LC_ALL=C sort -u > $tmp/u
python $model_type_dir/test.py $model $train_corpus $tmp/u $flags > $tmp/test_out
cat $tmp/test_out | head -n -1 | cut -f 2 -d ' ' > $tmp/uniq_gscores
cat $tmp/o | python ununiq.py $tmp/uniq_gscores $tmp/u > $tmp/gscores
python is.py $tmp/gscores $tmp/dscores $n --neg > $tmp/is_score
