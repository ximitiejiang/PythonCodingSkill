from .hook import Hook

class CheckpointHook(Hook):

    def __init__(self, checkpoint_config):
        self.interval = checkpoint_config['interval']
        if not checkpoint_config['save_optimizer']:
            self.save_optimizer = True
        else:
            self.save_optimizer = checkpoint_config['save_optimizer']
        self.out_dir = checkpoint_config['out_dir']

    def after_train_epoch(self, runner):
        if self.interval == -1:
            return
        elif runner._epoch % self.interval ==0:
            runner.save_checkpoint(
                self.out_dir, 
                runner.cfg.model_name, 
                save_optimizer=self.save_optimizer)
