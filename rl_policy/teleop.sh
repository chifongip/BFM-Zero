
POLICY_CONFIG=./config/policy/motivo_29dof.yaml
MODEL_ONNX_PATH=./results/249M_checkpoint/exported/FBcprAuxModel.onnx
TASK=./config/exp/teleop/locomotion.yaml

python rl_policy/bfm_zero.py \
    --robot_config config/robot/g1_29dof.yaml \
    --policy_config ${POLICY_CONFIG} \
    --model_path ${MODEL_ONNX_PATH} \
    --task  ${TASK}
