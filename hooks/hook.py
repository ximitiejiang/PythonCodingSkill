"""该Hook基类参考开源mmcv的Hook基类：
https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/hook.py
提供了一种awesome机制，用于深度学习的调试，且该机制不限于框架类型均可实现。
"""

class Hook(object):
    """Hook的基类，所有其他Hook类都需要基于该基类，避免在遍历钩子方法时缺失某方法报错。
    如果需要定义某一子方向的钩子，比如learning_rate, 比如grad，可遵循如下步骤：
    1. 可以先继承Hook，定义子类如LrHook(Hook), GradHook(Hook)
    2. 然后在子类中定义自己的方法，比如update_lr(), mean_grad()
    3. 在子类中重写基类的钩子函数，并在钩子函数中调用子类自己定义的方法
    """

    def before_run(self, runner):
        """核心的3组hooks: before_run"""
        pass

    def after_run(self, runner):
        """核心的3组hooks: before_run"""
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass

    def before_train_epoch(self, runner):
        """核心的3组hooks: before_run"""
        self.before_epoch(runner)

    def before_val_epoch(self, runner):
        """核心的3组hooks: before_run"""
        self.before_epoch(runner)

    def after_train_epoch(self, runner):
        """核心的3组hooks: before_run"""
        self.after_epoch(runner)

    def after_val_epoch(self, runner):
        """核心的3组hooks: before_run"""
        self.after_epoch(runner)

    def before_train_iter(self, runner):
        """核心的3组hooks: before_run"""
        self.before_iter(runner)

    def before_val_iter(self, runner):
        """核心的3组hooks: before_run"""
        self.before_iter(runner)

    def after_train_iter(self, runner):
        """核心的3组hooks: before_run"""
        self.after_iter(runner)

    def after_val_iter(self, runner):
        """核心的3组hooks: before_run"""
        self.after_iter(runner)

    def every_n_epochs(self, runner, n):
        return (runner.epoch + 1) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, runner, n):
        return (runner.inner_iter + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, runner, n):
        return (runner.iter + 1) % n == 0 if n > 0 else False

    def end_of_epoch(self, runner):
        return runner.inner_iter + 1 == len(runner.data_loader)
    
    def test_hook(self, runner):  # only for function test
        print('hello, this is father hook!')


class Test(Hook):
    import time
    def test_hook(self):
        print('hello, this is son hook!')
        

if __name__ == '__main__':
    
    def call_hook(fn_name, _hooks):
        for hook in _hooks:
            getattr(hook, fn_name)()
            
    hook1 = Hook()
    hook2 = Test()  # 由于是继承了Hook，所以test_hook()方法也被继承下来了。
    
    _hooks = [hook1, hook2]
    call_hook('test_hook', _hooks=_hooks)
        
