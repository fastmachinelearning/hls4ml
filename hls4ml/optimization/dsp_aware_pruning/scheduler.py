from abc import ABC, abstractmethod


class OptimizationScheduler(ABC):
    '''
    Baseline class handling logic regarding target sparsity and its updates at every step
    '''

    def __init__(self, initial_sparsity=0, final_sparsity=1):
        '''
        intial_sparsity and final_sparsity are between 0.0 and 1.0, NOT 0% and 100%
        '''
        if initial_sparsity < 0 or initial_sparsity > 1:
            raise Exception('intial_sparsity must be between 0.0 and 1.0')
        if final_sparsity < 0 or final_sparsity > 1:
            raise Exception('final_sparsity must be between 0.0 and 1.0')
        self.sparsity = initial_sparsity
        self.lower_bound = initial_sparsity
        self.upper_bound = final_sparsity

    @abstractmethod
    def update_step(self):
        '''
        Increments the current sparsity, according to the rule.

        Examples:
            - ConstantScheduler, sparsity = 0.5, increment = 0.05 -> sparsity = 0.55
            - BinaryScheduler, sparsity = 0.5, target = 1.0 -> sparsity = 0.75

        Returns:
            tuple containing

            - updated (boolean) - Has the sparsity changed? If not, the optimization algorithm can stop
            - sparsity (float) - Updated sparsity

        '''
        pass

    @abstractmethod
    def repair_step(self):
        '''
        Method used when the neural architecture does not meet satisfy performance requirement for a given sparsity.
        Then, the target sparsity is decreased according to the rule.

        Examples:
            - ConstantScheduler, sparsity = 0.5, increment = 0.05 -> sparsity = 0.55 [see ConstantScheduler for explanation]
            - BinaryScheduler, sparsity = 0.75, target = 1.0, previous = 0.5 -> sparsity = (0.5 + 0.75) / 2 = 0.625

        Returns:
            tuple containing

            - updated (boolean) - Has the sparsity changed? If not, the optimization algorithm can stop
            - sparsity (float) - Updated sparsity

        '''
        pass

    def get_sparsity(self):
        return self.sparsity


class ConstantScheduler(OptimizationScheduler):
    '''
    Sparsity updated by a constant term, until
        (i) sparsity target reached OR
        (ii) optimization algorithm stops requesting state updates
    '''

    def __init__(self, initial_sparsity=0, final_sparsity=1.0, update_step=0.05):
        self.increment = update_step
        super().__init__(initial_sparsity, final_sparsity)

    def update_step(self):
        if self.sparsity + self.increment <= self.upper_bound:
            self.sparsity += self.increment
            return True, self.sparsity
        else:
            return False, self.sparsity

    '''
    In certain cases, a model might underperform at the current sparsity level,
    But perform better at a higher sparsity
    In this case, constant sparsity (since it increments by a small amount every time),
    Will simply jump to the next sparsity level
    The model's performance over several sparsity levels optimization is tracked and S
    Stopped after high loss over several trials (see top level pruning/optimization function)

    '''

    def repair_step(self):
        return self.update_step()


class BinaryScheduler(OptimizationScheduler):
    '''
    Sparsity updated by binary halving the search space; constantly updates lower and upper bounds
    In the update step, sparsity is incremented, as the midpoint between previous sparsity and target sparsity (upper bound)
    In the repair step, sparsity is decrement, as the midpoint between between the lower bound and previous sparsity
    '''

    def __init__(self, initial_sparsity=0, final_sparsity=1.0, threshold=0.01):
        self.threshold = threshold
        super().__init__(initial_sparsity, final_sparsity)

    def update_step(self):
        if self.upper_bound - self.sparsity >= self.threshold:
            self.lower_bound = self.sparsity
            self.sparsity = 0.5 * (self.lower_bound + self.upper_bound)
            return True, self.sparsity
        else:
            self.lower_bound = self.sparsity
            return False, self.sparsity

    def repair_step(self):
        if self.sparsity - self.lower_bound >= self.threshold:
            self.upper_bound = self.sparsity
            self.sparsity = 0.5 * (self.lower_bound + self.upper_bound)
            return True, self.sparsity
        else:
            self.upper_bound = self.sparsity
            return False, self.sparsity


class PolynomialScheduler(OptimizationScheduler):
    '''
    Sparsity updated by at a polynomial decay, until
        (i) sparsity target reached OR
        (ii) optimization algorithm stops requesting state updates

    For more information, see Zhu & Gupta (2016) -
        'To prune, or not to prune: exploring the efficacy of pruning for model compression'

    Note, the implementation is slightly different, since TensorFlow Prune API depends on the total number of
    epochs and update frequency.

    In certain cases, a model might underperform at the current sparsity level, but perform better at a higher sparsity.
    In this case, polynomial sparsity will simply jump to the next sparsity level
    The model's performance over several sparsity levels optimization is tracked and
    toped after high loss over several trials (see top level pruning/optimization function)
    '''

    def __init__(self, maximum_steps, initial_sparsity=0, final_sparsity=1.0, decay_power=3):
        self.decay_power = decay_power
        self.current_step = 0
        self.maximum_steps = maximum_steps
        super().__init__(initial_sparsity, final_sparsity)

    def update_step(self):
        if self.current_step < self.maximum_steps:
            self.current_step += 1
            self.sparsity = self.upper_bound + (self.lower_bound - self.upper_bound) * (
                (1 - self.current_step / self.maximum_steps) ** self.decay_power
            )
            return True, self.sparsity
        else:
            return False, self.sparsity

    def repair_step(self):
        return self.update_step()
