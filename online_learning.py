
import numpy as np
import pandas as pd

class ExpWeights(object):
    
    def __init__(self, 
                 arms=[0,0.1,0.2,0.3,0.4,0.5],
                 lr = 0.2,
                 window = 20, # we don't use this yet.. 
                 epsilon = 0.2,
                 decay = 0.9):
        
        self.arms = arms
        self.l = {i:0 for i in range(len(self.arms))}
        self.arm = 0
        self.value = self.arms[self.arm]
        self.error_buffer = []
        self.window = window
        self.lr = lr
        self.epsilon = epsilon
        self.decay = decay
        
        self.choices = [self.arm]
        self.data = []
        
    def sample(self):
        
        if np.random.uniform() > self.epsilon:
            p = [np.exp(x) for x in self.l.values()]
            p /= np.sum(p) # normalize to make it a distribution
            print(p)
            self.arm = np.random.choice(range(0,len(p)), p=p)
        else:
            self.arm = int(np.random.uniform() * len(self.arms))

        self.value = self.arms[self.arm]
        self.choices.append(self.arm)
        
        return(self.value)
        
    def update_dists(self, feedback, norm=1):
        
        # Need to normalize score. 
        # Since this is non-stationary, subtract mean of previous 5. 
        
        self.error_buffer.append(feedback)
        self.error_buffer = self.error_buffer[-5:]
        
        feedback -= np.mean(self.error_buffer)
        feedback /= norm
        
        self.l[self.arm] *= self.decay
        self.l[self.arm] += self.lr * (feedback/max(np.exp(self.l[self.arm]), 0.0001))
        
        self.data.append(feedback)
       

''' 
test

b = ExpWeights()

b.error_buffer.append(0)
for _ in range(100):
    if b.arm == 2:
        b.update_dists(10)
    else:
        b.update_dists(np.random.rand())
    b.sample()

print(b.choices) 
print(b.l) 
p = [np.exp(x) for x in b.l.values()]
p /= np.sum(p)
print(p)

we should see more 2s at the end...
    
'''