"""Source-preserving target-only current-joint expansion for native full IDM.

The original qualified67D pair continues serving Panda source. Target199D
planner/IDM start with all new columns zero, preserving their old taskspace
functions for any new input; native full-IDM adaptation can learn even slots
inactive on source. No future joint correspondence or source rewrite.
"""
from .joint import lift_joint_checkpoint


def lift_target_joint_checkpoint(payload):
    result=lift_joint_checkpoint(payload)
    for key,value in result['model_state_dict'].items():
        if key.endswith('cmd_embed.weight'):value[:,67:]=0.
    result['planner_state_dict']={k[8:]:v for k,v in result['model_state_dict'].items() if k.startswith('planner.')};result['idm_state_dict']={k[4:]:v for k,v in result['model_state_dict'].items() if k.startswith('idm.')}
    result['genedynamics_task_adapter'].update(target_only_joint_expansion=True,new_joint_columns_zero_initialization=True,source_deployment='unchangedqualified67D originalpair; target199Dmodelnot usedasPanda source replacement',target_finetune_requirement='nativefullIDM trainallIDMweights includingcurrentcommandprojection; plannerfrozen',requires_new_source_qualification=True)
    return result
