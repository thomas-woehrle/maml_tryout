python3 rowfollow_maml.py \
  --num_episodes=60000 \
  --meta_batch_size=4 \
  --k=4 \
  --inner_gradient_steps=1 \
  --alpha=0.4 \
  --beta=0.001 \
  --device=cpu \
  --ckpt_base_dir=checkpoints/rowfollow_maml \
  --data_dir=/Users/tomwoehrle/Documents/research_assistance/cornfield1_labeled_new