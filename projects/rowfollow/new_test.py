import torch

import maml_api
import maml_eval

from rowfollow_task import RowfollowTask

run_id = 'a1c1639d073847db983e33785a509868'
episode = 9999

support_bag_path = '/Users/tomwoehrle/Documents/research_assistance/cornfield_representative_pictures/20220603_cornfield/ts_2022_06_03_02h57m53s'
support_cam_side = 'left_cam'
target_bag_path = '/Users/tomwoehrle/Documents/research_assistance/cornfield1_labeled_new/20220603_cornfield/ts_2022_06_03_02h57m53s'
target_cam_side = 'both'


def sample_task(stage: maml_api.Stage, seed: int = 0):
    return RowfollowTask(support_bag_path, 4, torch.device('cpu'),
                         support_cam_side=support_cam_side,
                         target_bag_path=target_bag_path, target_cam_side=target_cam_side,
                         seed=seed)


evaluator = maml_eval.MamlEvaluator(run_id, episode, sample_task,
                                    lib_directory='/Users/tomwoehrle/Documents/research_assistance/artifact_manager')

evaluator.finetune()

total_eval_loss = 0
for i in range(10):
    print('prediction step {}...'.format(i))
    total_eval_loss += evaluator.predict(seed=i)[1].item()
    print(total_eval_loss)

print('Total eval loss:', total_eval_loss)

# TODO visual test
