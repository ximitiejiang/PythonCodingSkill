from .logger_hook import LoggerHook


class LoggerTextHook(LoggerHook):

    def log(self, runner):
#        if runner.mode == 'train':
        
        # 显示 epoch + lr

        log_str = 'Epoch [{}][{}/{}], '.format(
            runner._epoch + 1, runner._inner_iter + 1,
            len(runner.dataloader))
#        else:
#            log_str = 'Epoch({}) [{}][{}]\t'.format(runner.mode, runner.epoch,
#                                                    runner.inner_iter + 1)
        
        # 显示lr
        lr_str = ', '.join(
            ['{:.4f}'.format(lr) for lr in runner.current_lr()])
        log_str += ('lr: {}'.format(lr_str))
        
        # 显示loss + accuracy
        log_str += (', loss: {:.3f}, acc_top1: {:.2f}%, acc_top{}: {:.2f}%'.format(
            runner.log_buffer.average_output['loss'],
            runner.log_buffer.average_output['acc_top1'],
            runner.cfg.topk,
            runner.log_buffer.average_output['acc_topk']))
        
        # 显示iter time 和 data time
        if 'time' in runner.log_buffer.average_output:
            log_str += (
                ', iter_time: {log[time]:.3f}, data_time: {log[data_time]:.3f}, '.
                format(log=runner.log_buffer.average_output))

        print(log_str)
