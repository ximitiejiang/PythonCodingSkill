from abc import ABCMeta, abstractmethod

from .hook import Hook


class LoggerHook(Hook):
    """作为logger的抽象类，完成针对log_buffer的基本操作.
    核心变量的操作都在基类完成，其他显示工作由各子类在log函数中完成
    reset_flag作用是设定是否在显示后清除平均值(有的场合需要预留则不清空)
    
    Args:
        interval (int): Logging interval (every k iterations).
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
        reset_flag (bool): Whether to clear the output buffer after logging.
    """
    __metaclass__ = ABCMeta   
    # 通过指定__metaclass__为ABCMeta后，该类为抽象类
    # 旧版python写法是继承class xxx(ABCMeta),新版写法是定义__metaclass__=ABCMeta
    # 抽象类不能被实例化，且抽象类中抽象方法是必须被重写

    def __init__(self, config, reset_flag=True):
        # 所有子类Logger共用该init，初始化interval代表n iter
        self.interval = config['interval']
        self.ignore_last = config['ignore_last']
        self.reset_flag = reset_flag

    @abstractmethod
    def log(self, runner):
        pass

    def before_run(self, runner):  
        # 如果是LoggerHook则flag设置为需要清空
        for hook in runner.hooks[::-1]:
            if isinstance(hook, LoggerHook):  
                hook.reset_flag = True
                break

    def before_epoch(self, runner):  
        # 清空每个epoch的log_buffer
        runner.log_buffer.clear()             

    def after_train_iter(self, runner):       
        # 每个iter求平均，每个iter调用log()函数，每个iter清空output
        if not ((runner._iter+1) % self.interval):
            runner.log_buffer.average(self.interval)
            
        elif (runner._inner_iter +1) == len(runner.dataloader) and not self.ignore_last:
            # 这里是不忽略剩余不足一个interval的数据：也进行一次平均，数量少了可能精度差些，但不会断档曲线更稳定
            # not precise but more stable
            runner.log_buffer.average(self.interval)

        if runner.log_buffer.ready:
            self.log(runner)  # 显示
            if self.reset_flag:  # 如果要求清空，则每次显示完就清空
                runner.log_buffer.clear_average_output()
    """
    def after_train_epoch(self, runner):  # 每个epoch调用log()函数
        if runner.log_buffer.ready:  
            self.log(runner)
    """
    
    def after_val_epoch(self, runner):   # 
        runner.log_buffer.average()
        self.log(runner)
        if self.reset_flag:
            runner.log_buffer.clear_output()
