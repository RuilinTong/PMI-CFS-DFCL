# -*-coding:utf8-*-


def get_model(model_name, args, data_manager=None):
    name = model_name.lower()
    if 'zs' in name:
        from models.zeroshot_clip import Learner
    elif 'codavlpt' in name:
        from models.codavlpt import Learner
    elif 'clcoop' in name:
        from models.clcoop import Learner
    elif 'vlpt' in name:
        from models.vlpt import Learner
    elif 'vicoop' in name:
        from models.vicoop import Learner
    elif 'coop' in name:
        from models.coop import Learner
    elif 'l2p' in name:
        from models.l2p import Learner
    elif 'dualprompt' in name:
        from models.dualprompt import Learner
    elif 'codaprompt' in name:
        from models.codaprompt import Learner
    elif 'finetune' in name:
        from models.finetune import Learner
    elif 'promptsrc' in name:
        from models.promptsrc import Learner
    elif 'imgf_prcil' in name:
        from models.imgf_prcil import Learner
    elif 'textf_prcil' in name:
        from models.textf_prcil import Learner
    elif 'prcil' in name:
        from models.prcil import Learner
    elif 'proof' in name:
        from models.proof import Learner
    elif 'dap' in name:
        if 'moe_adapter' in name:
            if 'inv' in name:
                from models.moe_adapter import Learner
            if 'openset' in name:
                from models.moe_adapter_open_world import Learner
        else:
            from models.dap import Learner
    else:
        assert 0
    
    return Learner(args)
