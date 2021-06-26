# deep_learning_final_assignment

# 2048实验报告

18373105 童涛 18373676 黎昊轩 https://github.com/LeavesLi1015/deep_learning_final_assignment

18373528 杨凌华 https://github.com/LarryHawkingYoung/deep_learning_final_assignment

18373542 胥浩宇 https://github.com/cpxhylty/deep_learning_final_assignment



## 绪论

在进行调研之后，我们从两个方向分别展开了实验

方向一是对baseline中的DQN方法进行改进，目前能够达到的最高平均分数是**3045.63**

方向二是实现了文献[1]中的N-Tuple Network方法，目前能够达到的最高平均分数是**4482.86**

**平均分数的计算方式**

采用与baseline相同的方式进行分数计算，两个x tile合并时只加x分

平均分数avg_score首先设置为结束的第一局游戏的分数，然后每局游戏结束时按照avg_score = avg_score * 0.99 + game.score * 0.01的公式进行更新









































[TOC]







## 方法一：DQN

给出的baseline使用的是Double-DQN方法，我们在baseline的基础上实现了多个tricks，探究不同改动对性能的影响。
参数方面，该模型几个较为重要的参数为：记忆内存大小、学习率、Soft Target Update更新系数(0.99).

首先要说明的是，由于GPU计算时长有限，并且使用CPU训练时间过长（一次5小时），我们并不能将所有的组合都进行尝试，也不能无限制的增加模型的复杂度和训练的时长。

### 1 实现的tricks

#### 1.1 网络改进

在baseline中，神经网络的输出只有三个元素。这就意味着，神经网络只会给出三种可能的步骤。

~~~python
self.fc2 = M.Linear(16, 3)
~~~

我们将网络的输出改为4。即，

~~~python
self.fc2 = M.Linear(16, 4)
~~~

从结果来看，改进之后的分数并没有太多的改变。很可能是因为神经网络给出的三个种可能已经够用了（在后面的action选择中也可以看到），基本上最优选择都可以执行。
在最优选择无法执行的情况下，才会选择其他两种action。而另外两种action基本不会出现无法执行的情况。所以，单纯的增加一个选择似乎对于模型的选择没有太大的影响。
所以只输出三种可能可以降低模型的复杂度和训练时的难度。

#### 1.2 为网络增加一个全连接层

Baseline的网络中，只有两个全连接层。

~~~python
self.conv3 = M.Conv2d(256, 256, kernel_size=(1, 2), stride=1, padding=0)
self.relu3 = M.ReLU()
self.fc1 = M.Linear(1024, 16)
self.relu5 = M.ReLU()
self.fc2 = M.Linear(16, 3)
~~~

我们为网络添加了一个全连接层

~~~python
self.conv3 = M.Conv2d(256, 256, kernel_size=(1, 2), stride=1, padding=0)
self.relu3 = M.ReLU()
self.fc1 = M.Linear(1024, 128)
self.relu5 = M.ReLU()
self.fc2 = M.Linear(128, 16)
self.relu6 = M.ReLU()
self.fc3 = M.Linear(16, 4) # changed to 4 outputs
~~~

#### 1.3 Priority Steps

在baseline中，关于下一个`action`的选择规则是：选取最大可能分数最高的点，如果这个`action`无法执行（即游戏里卡住），则执行`action+1`步骤。

~~~python
‘’’Choose the action with the highest probability’’’
a = F.argmax(model(status).detach(), 1)
a = a.numpy()
action = a[k]

while (game[k].grid == pre_grid).all():
    action = (action + 1) % 4
    game[k].move(action)
~~~

根据如下情况，我们给出了一定的改进方法：
在最优选择无法执行时，选择次优的`action`，如果次优移动无法执行，则选择第三优，以此类推。
我们根据该网络的特征（每次同时执行32盘游戏），写出了对步骤的排序函数：

~~~python
def sortSteps(games_steps): # steps->(32,4), 32 games, each game with 4 possiable steps
    sortSteps = [[],[],[],[]]
    # 4 lists means 4 steps, each list with 32 games.
    # The first list is the best. The last is the worst.
    for game_steps in games_steps: # 32 games
        temp = np.array(game_steps)
        for i in range(4):
            temp_ = temp[:]
            # print(temp_)
            s = np.argmax(temp_)
            # print(s)
            sortSteps[i].append(s)
            temp_[s] = np.min(temp) - 1
            temp = temp_[:]
        
    return sortSteps[0], sortSteps[1], sortSteps[2], sortSteps[3] # best, second, third, worst
~~~

在选择下一步时，将单纯的`action+1`改成了按照优先级进行选择。

~~~python
while (game[k].grid == pre_grid).all():
    action = actions[count][k]
    game[k].move(action)
    count = (count + 1) % 4
~~~


#### 1.4 Dropout Layer

在原baseline的网络中，并没有采用Dropout层对神经元进行丢弃。

~~~python
self.conv3 = M.Conv2d(256, 256, kernel_size=(1, 2), stride=1, padding=0)
self.relu3 = M.ReLU()
self.fc1 = M.Linear(1024, 16)
self.relu5 = M.ReLU()
self.fc2 = M.Linear(16, 3)
~~~

我们尝试了在倒数第二层全联接层后加入Dropout层。加入全联接层可以在一定程度上防止神经元之间的相互依赖，可以防止模型过拟合。

~~~python
self.conv3 = M.Conv2d(256, 256, kernel_size=(1, 2), stride=1, padding=0)
self.relu3 = M.ReLU()
self.fc1 = M.Linear(1024, 16)
self.drop1 = M.Dropout(0.3)
self.relu5 = M.ReLU()
self.fc2 = M.Linear(16, 3)
~~~

但是加入了Dropout层之后效果不加，loss函数一直无法收敛到一个较小的值。推测原因：加入Dropout层后，模型欠拟合。

所以决定放弃使用Dropout层。

#### 1.5 Randomly Fill Buffer

在原模型中，填充记忆模块用的是顺序填充，即当模块被填满时，最先填充的模块会被最先覆盖。

~~~python
def append(self, obj):
    if self.size() > self.buffer_size:
        print('buffer size larger than set value, trimming...')
        self.buffer = self.buffer[(self.size() - self.buffer_size):]
    elif self.size() == self.buffer_size:
        self.buffer[self.index] = obj
        self.index += 1
        self.index %= self.buffer_size
    else:
        self.buffer.append(obj)
~~~

随机模块填充，并不是将先填充的模块先覆盖，而是随机的将位置填充。

~~~python
def append(self, obj):
    if self.size() > self.buffer_size:
        print('buffer size larger than set value, trimming...')
        self.buffer = self.buffer[(self.size() - self.buffer_size):]
    elif self.size() == self.buffer_size:
        self.buffer[self.index] = obj
        self.index = np.random.randint(0,self.buffer_size)
        # self.index += 1
        # self.index %= self.buffer_size
    else:
        self.buffer.append(obj)
~~~

这样填充模块能够使得模块中保留一些“历史久远”的信息，可以防止所有的信息都来源于近期，避免模型陷入局部最优。

#### 1.6 Epsilon Decay($\epsilon$ Decay)

Epsilon Decay是一种常用在DQN中的贪心策略。这个贪心策略作用于action的选择上，算法步骤如下：

>1)投一枚硬币，该硬币正面朝上的概率为$\epsilon$
>2)当这个硬币正面朝上时，选择纯随机策略
>3)当这个硬币反面朝上时，采用最优action策略

在baseline中，epsilon可以看作取0。因为此时所有的选择都在采用最优策略，这会使得模型训练时容易跌入局部最优。
加入Epsilon后，在一定程度上，用随机的action取代了最优的action，能够避免陷入局部最优。
而为了模型的稳定性，epsilon应该随着迭代次数的增加而减小。这样有助于模型趋于稳定，并且让所得到的模型趋向于最优解。这里我们采用了线性递减的方式。

~~~python
init_epsilon = 0.10
epsilon = init_epsilon * (1 - epoch/epochs)

'''''
when memory is enough, randomly action. when memory is full, at epsilon percentage take best action.
'''
if data.size() == max_size and np.random.random() > epsilon: # greedy action
    while (game[k].grid == pre_grid).all():
        action = actions[count][k]
        game[k].move(action)
        count = (count + 1) % 4
else:
    action = np.random.randint(0,4)
~~~

#### 1.7 AdamW优化器

baseline使用的优化器是Adam。在此基础上，我们选择了AdamW进行改进。AdamW相比于Adam增加了一个L2的惩罚值。他能够有效地防止梯度爆炸的出现，但是带来的坏处就是，相比使用Adam，模型的成长速度减缓了不少。

~~~python
opt = AdamW(model.parameters(), lr = 1e-4)
~~~

### 2 可调整参数

这里列举了几个比较重要的参数，还有许多参数没有举例。

#### 2.1 记忆规模，默认5000

~~~python
data = rpm(5000)
~~~

#### 2.2 学习率，默认0.0001；学习器（Adam）

~~~python
opt = Adam(model.parameters(), lr = 1e-4)
~~~

#### 2.3 Soft Target Update(0.99)

~~~python
loss += F.loss.square_loss(Q, pred_s1[i].detach() * 0.99 * (1 - d[i]) + reward[i])
~~~

#### 2.4 同时可开展的游戏次数（32）

~~~
for game in games: # 32 games
~~~

### 3 总结

总的来说，Baseline所实现的Double-DQN较为简单，使用的技巧并不多。
在方法一中，我们主要添加了一些在DQN训练时较为常见的几个tricks。添加的tricks所达到的效果已经记录在了训练日志当中。部分trick在添加后反而起到了副作用，原因很可能是模型本身的特点所决定的。比如这个模型比较容易陷入局部最优；不适合添加Dropout层...
但总的来说，目前所有选择实现的tricks能够帮助模型训练出更好的结果。在算力资源有限的情况下，虽然avg_score提升并不明显（最高也只有3000出头），但是可以明显的发现，模型是有提升空间的。这一点体现在Q值上：如果Q值在训练过程中无法持续增加，意味着模型选择一次移动，所获得的分数很少。而我们的模型在有限的迭代次数中，Q值不断上升，成绩也稳步提升。这也进一步证明了我们添加的tricks能够更好的完成任务。

### 4 训练日志

#### 4.1 所有Tricks开启

这应该是最有潜力的一组，因为得分稳步增长，并且Q值一直处于增长阶段。证明如果加大迭代次数，这个模型还有很大的提升空间。

~~~
epoch:   0%|          | 5/100000 [00:00<1:08:47, 24.23it/s, loss=0.00028, Q=0.03781, reward=0.56250, avg_score=0.00000]
epoch:   5%|▌         | 5004/100000 [03:23<1:05:58, 24.00it/s, loss=0.08359, Q=4.32059, reward=3.25000, avg_score=529.71293]
epoch:  10%|█         | 10004/100000 [06:49<1:02:30, 24.00it/s, loss=0.17213, Q=9.23870, reward=5.43750, avg_score=728.29572]
epoch:  15%|█▌        | 15005/100000 [10:16<57:54, 24.46it/s, loss=0.12767, Q=16.17513, reward=3.43750, avg_score=990.61025]   
epoch:  20%|██        | 20004/100000 [13:41<56:34, 23.57it/s, loss=0.32310, Q=21.93606, reward=3.12500, avg_score=1188.41061]  
epoch:  25%|██▌       | 25004/100000 [17:06<51:19, 24.35it/s, loss=0.48368, Q=25.01408, reward=4.68750, avg_score=1251.10009] 
epoch:  30%|███       | 30005/100000 [20:32<47:59, 24.31it/s, loss=1.41843, Q=25.20588, reward=4.06250, avg_score=1518.52155]  
epoch:  35%|███▌      | 35004/100000 [23:57<45:42, 23.70it/s, loss=1.07101, Q=31.44183, reward=2.00000, avg_score=1545.66837]  
epoch:  40%|████      | 40004/100000 [27:24<40:35, 24.64it/s, loss=0.39668, Q=28.92669, reward=3.43750, avg_score=1667.49770]  
epoch:  45%|████▌     | 45005/100000 [30:49<37:17, 24.57it/s, loss=2.69036, Q=33.33669, reward=2.87500, avg_score=1910.70973]  
epoch:  50%|█████     | 50004/100000 [34:14<34:57, 23.84it/s, loss=1.25572, Q=36.33077, reward=3.25000, avg_score=1891.38512]  
epoch:  55%|█████▌    | 55004/100000 [37:40<31:00, 24.18it/s, loss=0.81370, Q=36.35245, reward=5.00000, avg_score=1848.78859]  
epoch:  60%|██████    | 60005/100000 [41:05<27:04, 24.62it/s, loss=3.28809, Q=37.84313, reward=7.12500, avg_score=1918.86686]  
epoch:  65%|██████▌   | 65004/100000 [44:31<24:40, 23.64it/s, loss=1.95699, Q=39.52359, reward=7.56250, avg_score=2253.00424]  
epoch:  70%|███████   | 70004/100000 [47:58<20:26, 24.45it/s, loss=0.92885, Q=37.58751, reward=4.62500, avg_score=2278.98536]  
epoch:  75%|███████▌  | 75005/100000 [51:25<17:15, 24.13it/s, loss=1.57380, Q=41.51716, reward=6.12500, avg_score=2280.92122]  
epoch:  80%|████████  | 80004/100000 [54:50<14:02, 23.74it/s, loss=1.11217, Q=46.57273, reward=7.81250, avg_score=2260.62683]  
epoch:  85%|████████▌ | 85004/100000 [58:16<10:13, 24.45it/s, loss=1.25295, Q=41.03788, reward=6.93750, avg_score=2278.10763]  
epoch:  90%|█████████ | 90005/100000 [1:01:43<06:50, 24.33it/s, loss=0.56762, Q=47.11993, reward=6.50000, avg_score=2461.73157]  
epoch:  95%|█████████▌| 95004/100000 [1:05:10<03:32, 23.53it/s, loss=0.77186, Q=47.28543, reward=3.93750, avg_score=2581.67211]  
epoch: 100%|██████████| 100000/100000 [1:08:36<00:00, 24.29it/s, loss=3.40540, Q=56.12973, reward=6.18750, avg_score=2453.03791] 
maxscore:7760
avg_score:2453.0379128732407

+------+------+------+------+
|  2   |  4   | 1024 | 512  |
+------+------+------+------+
|  16  |  64  | 256  |  32  |
+------+------+------+------+
|  4   |  16  |  4   |  16  |
+------+------+------+------+
|  2   |  32  |  2   |  4   |
+------+------+------+------+
~~~

<img src="./image/3-1.png" alt="3-1" style="zoom:33%;" />

#### 4.2（从baseline开始）网络输出改为4

~~~
epoch: 100%|██████████| 50000/50000 [37:46<00:00, 22.06it/s, loss=0.05948, Q=5.44860, reward=12.56250, avg_score=2651.80934]
maxscore:7896
avg_score:2651.8093411440545
~~~

#### 4.3 无效移动`action+1`—>次优移动

- reward震荡严重，范围2~10
- loss震荡，0.01~0.09
- avg_score增大到2000左右开始震荡，下降到1600，开始慢慢上升到1900左右

后来发现是有一个地方没有改好，导致效果不增反降。

~~~
epoch: 100%|██████████| 50000/50000 [38:42<00:00, 21.53it/s, loss=0.05070, Q=4.89405, reward=3.81250, avg_score=1893.23744] 
maxscore:6434
avg_score:1893.2374433444083
~~~

#### 4.4 在上一步的基础上，调整SoftTargetUpdate参数为0.999

~~~
epoch:   0%|          | 4/50000 [00:00<44:43, 18.63it/s, loss=0.10281, Q=4.65261, reward=4.87500, avg_score=0.00000]  
epoch:  10%|█         | 5004/50000 [03:53<37:53, 19.79it/s, loss=0.08687, Q=8.16222, reward=5.00000, avg_score=1388.10020] 
epoch:  20%|██        | 10005/50000 [07:46<32:10, 20.72it/s, loss=0.32168, Q=12.84697, reward=5.25000, avg_score=2097.89126]
epoch:  30%|███       | 15004/50000 [11:39<29:24, 19.84it/s, loss=0.46031, Q=18.53462, reward=8.93750, avg_score=2197.32582] 
epoch:  40%|████      | 20004/50000 [15:32<25:03, 19.95it/s, loss=0.20618, Q=20.43308, reward=6.37500, avg_score=2372.96677] 
epoch:  50%|█████     | 25004/50000 [19:25<21:01, 19.81it/s, loss=0.27678, Q=22.43907, reward=4.81250, avg_score=2648.55656]  
epoch:  60%|██████    | 30004/50000 [23:19<16:17, 20.45it/s, loss=0.61412, Q=26.37881, reward=9.93750, avg_score=2901.75986]  
epoch:  70%|███████   | 35005/50000 [27:11<11:47, 21.20it/s, loss=0.33737, Q=29.13754, reward=6.56250, avg_score=2661.33176]  
epoch:  80%|████████  | 40004/50000 [31:04<08:09, 20.41it/s, loss=0.38669, Q=30.04513, reward=3.43750, avg_score=2736.06205]  
epoch:  90%|█████████ | 45004/50000 [34:57<04:06, 20.24it/s, loss=0.35807, Q=31.68990, reward=4.00000, avg_score=2924.52263]  
epoch: 100%|██████████| 50000/50000 [38:49<00:00, 21.46it/s, loss=1.38336, Q=31.83402, reward=10.37500, avg_score=3045.63207] 
maxscore:8142
avg_score:3045.6320680842
~~~

<img src="./image/3-4.png" alt="3-4" style="zoom:33%;" />

#### 4.5 采用Epsilon Decay

~~~
epoch:   0%|          | 5/50000 [00:00<39:19, 21.19it/s, loss=0.00226, Q=1.24186, reward=3.37500, avg_score=0.00000]  
epoch:  10%|█         | 5005/50000 [03:46<34:25, 21.79it/s, loss=0.03849, Q=4.79506, reward=5.87500, avg_score=982.02194]  
epoch:  20%|██        | 10004/50000 [07:32<31:22, 21.25it/s, loss=0.10363, Q=9.03690, reward=6.25000, avg_score=1396.17066]
epoch:  30%|███       | 15004/50000 [11:19<28:19, 20.59it/s, loss=0.30082, Q=10.05284, reward=5.25000, avg_score=1500.18189] 
epoch:  40%|████      | 20004/50000 [15:04<23:33, 21.22it/s, loss=0.09822, Q=10.43662, reward=1.43750, avg_score=1605.74873] 
epoch:  50%|█████     | 25005/50000 [18:50<19:17, 21.60it/s, loss=0.05739, Q=12.12671, reward=5.06250, avg_score=1596.09813] 
epoch:  60%|██████    | 30005/50000 [22:36<15:45, 21.14it/s, loss=0.10602, Q=11.82512, reward=3.75000, avg_score=1724.11383] 
epoch:  70%|███████   | 35004/50000 [26:21<11:41, 21.36it/s, loss=0.32748, Q=12.68642, reward=7.06250, avg_score=1746.96458] 
epoch:  80%|████████  | 40004/50000 [30:11<07:48, 21.34it/s, loss=0.17132, Q=13.03945, reward=4.12500, avg_score=1761.15174] 
epoch:  90%|█████████ | 45004/50000 [33:58<03:48, 21.88it/s, loss=0.16110, Q=10.59168, reward=6.06250, avg_score=1867.21660] 
epoch: 100%|██████████| 50000/50000 [37:43<00:00, 22.09it/s, loss=0.09503, Q=12.31988, reward=6.50000, avg_score=1919.71672] 
maxscore:5644
avg_score:1919.7167232776671
~~~

<img src="./image/3-5.png" alt="3-5" style="zoom:33%;" />

#### 4.6 Epsilon Decay参数调整（buff-size:5000->2000，init_epsilon:0.15-0.1）

~~~
epoch:   0%|          | 4/50000 [00:00<47:16, 17.63it/s, loss=0.00057, Q=0.08594, reward=0.25000, avg_score=0.00000]  
epoch:  10%|█         | 5005/50000 [03:47<35:18, 21.23it/s, loss=0.00279, Q=1.18189, reward=2.68750, avg_score=410.57400]
epoch:  15%|█▍        | 7270/50000 [05:31<32:26, 21.95it/s, loss=0.00828, Q=1.08324, reward=2.37500, avg_score=384.83023]
~~~

效果不好，提前终止。

#### 4.7 Epsilon Decay参数调整（buffer_size:2000->8000，epochs:10000->15000）

~~~
epoch:   0%|          | 5/75000 [00:00<57:26, 21.76it/s, loss=0.00035, Q=0.03386, reward=0.25000, avg_score=0.00000]  
epoch:   7%|▋         | 5004/75000 [03:52<59:08, 19.73it/s, loss=0.02977, Q=4.74931, reward=4.75000, avg_score=884.16725]   
epoch:  13%|█▎        | 10004/75000 [07:49<53:27, 20.26it/s, loss=0.03363, Q=8.03254, reward=6.50000, avg_score=1371.69223] 
epoch:  20%|██        | 15005/75000 [11:44<47:45, 20.94it/s, loss=0.04327, Q=9.44598, reward=2.31250, avg_score=1738.48894] 
epoch:  27%|██▋       | 20004/75000 [15:40<46:48, 19.58it/s, loss=0.06418, Q=10.42684, reward=4.00000, avg_score=1960.57912] 
epoch:  33%|███▎      | 25004/75000 [19:38<40:16, 20.69it/s, loss=0.05436, Q=11.12187, reward=4.37500, avg_score=2170.28065] 
epoch:  40%|████      | 30004/75000 [23:35<37:35, 19.95it/s, loss=0.41796, Q=10.51460, reward=6.18750, avg_score=2067.08620] 
epoch:  47%|████▋     | 35004/75000 [27:33<33:28, 19.91it/s, loss=0.11905, Q=11.30016, reward=4.00000, avg_score=1852.44310] 
epoch:  53%|█████▎    | 40005/75000 [31:31<28:56, 20.15it/s, loss=0.12435, Q=11.13156, reward=3.43750, avg_score=2020.26731] 
epoch:  57%|█████▋    | 42610/75000 [33:35<25:31, 21.15it/s, loss=0.24599, Q=9.34995, reward=6.93750, avg_score=2037.64702] 
~~~

发生震荡，提前终止。

#### 4.8 Dropout（全连接层最后加入Dropout层，概率0.2）

~~~
epoch:   0%|          | 4/50000 [00:00<51:29, 16.18it/s, loss=0.00101, Q=0.02423, reward=0.12500, avg_score=0.00000]  
epoch:  10%|█         | 5004/50000 [04:14<39:04, 19.20it/s, loss=0.19428, Q=1.38549, reward=4.81250, avg_score=847.36938] 
epoch:  20%|██        | 10004/50000 [08:33<35:59, 18.52it/s, loss=0.07778, Q=1.36181, reward=3.43750, avg_score=1125.10873]
epoch:  30%|███       | 15004/50000 [12:51<31:18, 18.63it/s, loss=0.05483, Q=1.52981, reward=4.31250, avg_score=1263.07748] 
epoch:  40%|████      | 20004/50000 [17:13<27:59, 17.86it/s, loss=0.10145, Q=1.69371, reward=8.68750, avg_score=1242.19591] 
epoch:  50%|█████     | 25004/50000 [21:37<22:45, 18.30it/s, loss=0.08980, Q=1.58534, reward=6.93750, avg_score=1267.76650] 
epoch:  60%|██████    | 30004/50000 [26:02<18:14, 18.27it/s, loss=0.01866, Q=1.82084, reward=4.43750, avg_score=1424.65073] 
epoch:  70%|███████   | 35004/50000 [30:26<13:29, 18.52it/s, loss=0.01345, Q=1.95794, reward=3.06250, avg_score=1461.52375] 
epoch:  80%|████████  | 40004/50000 [34:47<09:10, 18.17it/s, loss=0.00803, Q=2.11296, reward=3.43750, avg_score=1468.78108] 
epoch:  90%|█████████ | 45003/50000 [39:08<04:37, 18.00it/s, loss=0.02391, Q=2.50486, reward=6.62500, avg_score=1521.40200] 
epoch:  90%|█████████ | 45007/50000 [39:08<04:37, 18.00it/s, loss=0.02452, Q=2.31132, reward=2.62500, avg_score=1521.40200]
epoch: 100%|██████████| 50000/50000 [43:27<00:00, 19.18it/s, loss=0.03553, Q=2.53317, reward=3.31250, avg_score=1521.32538] 
maxscore:4076
avg_score:1521.3253815767862
~~~

<img src="./image/3-8.png" alt="3-8" style="zoom:33%;" />

#### 4.9 Memory buff-size->5000 学习率0.001

~~~
epoch:   0%|          | 5/50000 [00:00<40:57, 20.34it/s, loss=0.00143, Q=0.00167, reward=0.50000, avg_score=0.00000]  
epoch:  10%|█         | 5004/50000 [03:57<35:58, 20.84it/s, loss=0.03674, Q=0.30084, reward=6.43750, avg_score=743.03215] 
loss 0.03385 Q 0.38043 reward 5.68750 avg_score 743.03215
loss 0.00131 Q 0.27179 reward 1.62500 avg_score 743.03215
loss 0.00658 Q 0.26482 reward 3.93750 avg_score 743.03215
loss 0.00288 Q 0.26978 reward 3.37500 avg_score 743.03215
loss 0.03674 Q 0.30084 reward 6.43750 avg_score 743.03215
epoch:  20%|██        | 10004/50000 [07:44<32:45, 20.34it/s, loss=0.00180, Q=0.32779, reward=4.25000, avg_score=889.44966]
epoch:  30%|███       | 15005/50000 [11:37<27:48, 20.97it/s, loss=0.00194, Q=0.32432, reward=3.06250, avg_score=847.49260] 
epoch:  34%|███▍      | 17238/50000 [13:22<25:24, 21.49it/s, loss=0.00214, Q=0.32978, reward=1.87500, avg_score=852.40793] 
~~~

~~~
epoch:   0%|          | 5/50000 [00:00<43:39, 19.08it/s, loss=0.00568, Q=-0.00817, reward=4.00000, avg_score=0.00000]  
epoch:  10%|█         | 5004/50000 [03:57<38:02, 19.71it/s, loss=15.61639, Q=17.59051, reward=3.43750, avg_score=573.86230] 
epoch:  20%|██        | 10004/50000 [07:55<33:55, 19.65it/s, loss=16.37549, Q=21.63450, reward=3.75000, avg_score=596.82945]
epoch:  30%|███       | 15004/50000 [11:54<28:56, 20.15it/s, loss=1.99956, Q=9.41308, reward=3.06250, avg_score=778.82203]   
epoch:  40%|████      | 20005/50000 [15:51<24:26, 20.46it/s, loss=1.07725, Q=4.58631, reward=2.56250, avg_score=1058.85101] 
epoch:  50%|█████     | 25004/50000 [19:46<20:38, 20.18it/s, loss=0.26071, Q=2.99617, reward=3.43750, avg_score=1088.28699] 
epoch:  60%|██████    | 30005/50000 [23:40<16:06, 20.68it/s, loss=0.03816, Q=1.78495, reward=2.00000, avg_score=1046.60566] 
epoch:  70%|███████   | 35004/50000 [27:35<12:17, 20.34it/s, loss=0.16492, Q=1.88245, reward=2.81250, avg_score=1107.96531] 
epoch:  80%|████████  | 40005/50000 [31:28<08:13, 20.27it/s, loss=0.10913, Q=2.43369, reward=3.25000, avg_score=1114.48686] 
epoch:  90%|█████████ | 45005/50000 [35:25<04:08, 20.08it/s, loss=0.11077, Q=2.36197, reward=4.12500, avg_score=1168.30842] 
epoch: 100%|██████████| 50000/50000 [39:17<00:00, 21.21it/s, loss=0.09077, Q=2.54650, reward=2.37500, avg_score=1179.09103] 
maxscore:3808
avg_score:1179.0910284435654
~~~

#### 4.10 双Dropout，多一层全连接

~~~
epoch:   0%|          | 4/100000 [00:00<1:54:03, 14.61it/s, loss=0.00279, Q=0.07822, reward=0.43750, avg_score=0.00000]
epoch:  10%|█         | 10004/100000 [07:38<1:17:00, 19.48it/s, loss=91.05108, Q=36.70674, reward=2.43750, avg_score=1014.30940]
epoch:  15%|█▌        | 15004/100000 [11:47<1:08:40, 20.63it/s, loss=118.83199, Q=49.29102, reward=7.87500, avg_score=1107.39320] 
epoch:  20%|██        | 20005/100000 [15:43<1:04:11, 20.77it/s, loss=107.27513, Q=52.40406, reward=4.56250, avg_score=1085.97321] 
epoch:  20%|██        | 20067/100000 [15:46<1:02:49, 21.21it/s, loss=125.23623, Q=59.71001, reward=2.81250, avg_score=1095.15348]
~~~

loss太高，正则化过度。

<img src="./image/3-10.png" alt="3-10" style="zoom:33%;" />

#### 4.11 取消Dropout层

~~~
epoch:   0%|          | 4/100000 [00:00<1:31:59, 18.12it/s, loss=0.00008, Q=0.02717, reward=0.43750, avg_score=0.00000]
epoch:   5%|▌         | 5004/100000 [03:48<1:17:17, 20.48it/s, loss=0.07178, Q=3.61712, reward=4.50000, avg_score=936.56982] 
epoch:  10%|█         | 10005/100000 [07:40<1:12:50, 20.59it/s, loss=0.04467, Q=8.96950, reward=4.18750, avg_score=1447.23755]
epoch:  15%|█▌        | 15004/100000 [11:29<1:06:52, 21.18it/s, loss=0.22352, Q=11.00501, reward=8.62500, avg_score=1682.09863] 
epoch:  20%|██        | 20004/100000 [15:08<1:03:30, 20.99it/s, loss=0.19563, Q=13.07572, reward=7.81250, avg_score=1742.00251] 
epoch:  25%|██▌       | 25005/100000 [18:56<1:00:02, 20.82it/s, loss=0.20120, Q=13.88394, reward=13.18750, avg_score=1699.06435]
epoch:  25%|██▌       | 25496/100000 [19:19<56:29, 21.98it/s, loss=0.48355, Q=11.37489, reward=1.81250, avg_score=1663.97994]   
~~~

提前终止。

#### 4.12 无效移动更正

~~~
epoch:   0%|          | 4/50000 [00:00<1:21:46, 10.19it/s, loss=0.00006, Q=-0.00286, reward=0.31250, avg_score=0.00000]
epoch:  10%|█         | 5005/50000 [04:03<38:08, 19.66it/s, loss=0.02044, Q=5.44548, reward=4.18750, avg_score=565.59711] 
epoch:  20%|██        | 10004/50000 [08:07<34:04, 19.56it/s, loss=1.85931, Q=14.43054, reward=5.00000, avg_score=1010.96392]
epoch:  30%|███       | 15005/50000 [12:11<29:07, 20.03it/s, loss=1.90816, Q=24.35327, reward=4.25000, avg_score=1279.26304] 
epoch:  40%|████      | 20004/50000 [16:13<24:50, 20.12it/s, loss=7.25223, Q=32.56019, reward=3.87500, avg_score=1507.22472] 
epoch:  50%|█████     | 25005/50000 [20:15<20:32, 20.29it/s, loss=1.50292, Q=37.33854, reward=7.68750, avg_score=1751.96377]  
epoch:  60%|██████    | 30005/50000 [24:19<16:52, 19.74it/s, loss=3.57725, Q=47.99241, reward=4.43750, avg_score=1901.75718]  
epoch:  70%|███████   | 35005/50000 [28:22<12:27, 20.06it/s, loss=2.01264, Q=55.13494, reward=2.75000, avg_score=1911.88897]  
epoch:  80%|████████  | 40004/50000 [32:26<08:39, 19.23it/s, loss=5.17427, Q=64.48306, reward=7.93750, avg_score=2264.89807]  
epoch:  90%|█████████ | 45005/50000 [36:29<04:12, 19.81it/s, loss=1.54954, Q=64.72044, reward=3.68750, avg_score=2515.44872]   
epoch: 100%|██████████| 50000/50000 [40:33<00:00, 20.55it/s, loss=1.27439, Q=77.06068, reward=3.93750, avg_score=2650.08703]  
maxscore:8148
avg_score:2650.087027050852

+------+------+------+------+
|  2   | 1024 | 512  |  2   |
+------+------+------+------+
|  32  | 128  |  64  | 256  |
+------+------+------+------+
|  4   |  2   |  16  |  64  |
+------+------+------+------+
|  2   |  8   |  2   |  4   |
+------+------+------+------+
~~~

<img src="./image/3-12.png" alt="3-12" style="zoom:33%;" />

#### 4.13 将lr修改为0.002

```
epoch: 100%|██████████| 50000/50000 [34:46<00:00, 23.97it/s, loss=0.00120, Q=3.35089, reward=3.18750, avg_score=1322.43493] 
maxscore:4088
avg_score:1322.4349255205716
```

#### 4.14 将lr改为0.0005

效果不佳，提前终止。

```
epoch:  40%|████      | 20004/50000 [14:02<20:56, 23.87it/s, loss=0.00825, Q=1.76280, reward=1.93750, avg_score=433.00770]
loss 0.03105 Q 1.84764 reward 2.62500 avg_score 433.00770
loss 0.00339 Q 1.51040 reward 2.43750 avg_score 433.00770
loss 0.00949 Q 1.68409 reward 2.43750 avg_score 433.00770
loss 0.00670 Q 1.79042 reward 1.25000 avg_score 433.00770
loss 0.00825 Q 1.76280 reward 1.93750 avg_score 433.00770
```

#### 4.15 lr=0.001

```
epoch: 100%|██████████| 50000/50000 [34:49<00:00, 23.93it/s, loss=0.22029, Q=10.75897, reward=2.56250, avg_score=1358.40334] 
maxscore:4562
avg_score:1358.4033414286516
```

## 方法二：N-Tuple Temporal Difference Learning

### 1 原理阐述

#### 1.1 Markov Decision Processes 

马尔科夫决策过程

引入如下符号：

> $S:$ 游戏的局面状态集
>
> $A(s) \subseteq \{N, E, S, W\},s \in S:$ 游戏局面状态 s 的下一步可行操作集，本游戏中为上、下、左、右四个方向的滑动
>
> $R(s, a),s \in S, a \in A(s):$ 状态 s 经过操作 a 后，这一步得到的分数
>
> $N(s, a),s \in S, a \in A(s):$ 状态 s 经过操作 a 后的下一个状态 (经过随机填充1个数后的局面状态)
>
> $V(s),s \in S:$ 由状态 s 开始一直到游戏结束，得到的总分数回报的期望

其中对于任何一个状态 s ，其总分回报期望 V(s) 都可以通过搜索树来进行准确计算，但这样的算法复杂度过高，对存储空间需求过大，不切合实际。因此采用函数近似（Function Approximation）的思想，构造神经网络模型来预测 V(s)，并对参数进行学习优化，以实现较准确预测每个状态 s 的期望回报 V(s) 值。通过比较每个 $a \in A(s)$ 操作得到的总回报期望 $R(s, a) + V(N(s, a))$ ，取总回报期望最大的 a 作为当前状态的下一步操作，即：
$$
\arg\max\limits_{a \in A(s)}(R(s, a) + V(N(s, a)))
$$

#### 1.2 N-Tuple Network

采用 N-Tuple 神经网络来构造期望回报 V(s) 的函数近似，形式化表示为：
$$
V_\theta : S \rightarrow R
$$
其计算方式如下：

首先定义 m 个 n-tuples，图1中以m=2，n=3为例。游戏四宫格如图1所示标号为0~15，则每个n-tuple可以用一个n元序列来表示，例如图1(a)中，两个3-tuples可以表示为(0, 1, 2)和(0, 1, 4). 我们认为每一个格子的取值在2<sup>0</sup>到2<sup>15</sup>之间，即每个格子有16种可能的取值。为每一个n-tuple覆盖区域的每一种可能取值分配一个weight值，则一个n-tuple的weight包含16<sup>n</sup>个值。

如图1(b), 每一种盘面 s 经过旋转和翻转变换后共有8种等价的形式，计算 V(s) 的方法是对每一个 n-tuple 在每一种 s 的等价情况下的权重之和，如图1(c). 每个tuple的weight值将采用下一小节所述方式进行训练。

<img src="./image/图1.png" style="zoom:50%;" />

<center>图1 n-tuple的定义和使用</center>

#### 1.3 Temporal Difference Learning

设 $s_t$ 表示t 时刻的游戏局面状态。

机器玩家通过上文所述公式：$\arg\max\limits_{a \in A(s)}(R(s_t, a) + V(N(s_t, a)))$ 来选出下一步的a.

得到下一状态以及单步回报分别为： $s_{t+1} = N(s_t, a), r = R(s_t, a)$.

定义 TD Error为  $\triangle = r + V(s_{t+1}) - V(s_t)$    （不用加绝对值）

优化目标为降低 TD Error 的值，因此参数更新方式为：  $V^*(s_t) = V(s_t) + \alpha \triangle$

其中 $V^*(s_t)$ 为更新 weight 之后新的回报期望值，$\alpha$是学习率。

### 2 实验

#### 2.1 实验配置

采取与baseline类似的方法进行训练：首先实例化32局游戏，然后进行若干个iteration。

每个iteration中，根据weights为这32局游戏分别选取移动的方向，进行一步移动，将对应的32个五元组(移动前的状态，移动后的状态，移动方向，移动得分，游戏是否结束)放入记忆体中；然后从记忆体中进行5次随机抽样，每次抽取32个样例，对weights进行更新。

考虑到时间和效果的权衡，参考论文中的结果，我们选择了4个6-tuple组成N-tuple network，tuple的具体形态如图2所示.

<img src="./image/图2.png" />

<center>图2 本次实验中选择的4个6-tuple</center>

#### 2.2 关键代码

##### 2.2.1 Tuple类

```python
class Tuple():
    def __init__(self, dots):
        self.n = len(dots)
        self.weights = [0]*(16**self.n)
        self.dots8 = []
        self.dots8.append([(x, y)     for x, y in dots])
        self.dots8.append([(x, 3-y)   for x, y in dots])
        self.dots8.append([(3-x, y)   for x, y in dots])
        self.dots8.append([(y, 3-x)   for x, y in dots])
        self.dots8.append([(y, x)     for x, y in dots])
        self.dots8.append([(3-y, 3-x) for x, y in dots])
        self.dots8.append([(3-x, 3-y) for x, y in dots])
        self.dots8.append([(3-y, x)   for x, y in dots])
        
    def forward(self, grid):
        res = 0
        idxs = []
        for ds in self.dots8:
            idx = 0
            for x, y in ds:
                idx = idx * 16 + grid[x][y]
            res += self.weights[idx]
            idxs.append(idx)
        return res, idxs
    
    def backward(self, idxs, error):
        delta = error / 8
        for idx in idxs:
            self.weights[idx] += delta
```

Tuple类构造方法输入坐标形式表示的点列，例如[(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (2, 0)].

weights使用一个长度为16\^n的列表存储。

1.2节中提到计算V时需要对网格的8种对称形式分别计算。为了减小计算的开销，我们将点列的八种对称形式计算并存在Tuple类中，这样做的好处是只需要在实例化时计算一次。

forward方法输入4\*4数组表示的盘面，盘面上的数用0-15表示；返回V(s)和计算过程用到的weights中的值对应的下标序列。

backward方法输入下标序列和TD error，对weights进行更新。

##### 2.2.2 训练过程

```python
for _ in range(5):
    s0, s1, a, reward, d = data.sample_batch(32)

    for i in range(32):
        for t in tuples:
            v0, idxs0 = t.forward(s0[i])
            v1, _ = t.forward(s1[i])
            delta = alpha * (v1 + reward[i] - v0)
            t.backward(idxs0, delta)
```

对于记忆体中的每个样例，对每个tuple分别进行计算V(s), 计算TD error, 更新weights的过程。

#### 2.3 运行时间和空间

使用MegStudio的普通版或高级版CPU环境，在4个6-tuple的环境下，运行时间约为9 iterations每秒。对于每个tuple，其weight包含16<sup>6</sup>个值，每个值使用32位浮点数存储，4个tuple占用总空间约为256MB.

#### 2.4 结果与分析

采用与baseline相同的方式进行分数计算，两个x tile合并时加x分；平均分数avg_score首先设置为结束的第一局游戏的分数，然后每局游戏结束时按照avg_score = avg_score * 0.99 + game.score * 0.01的公式进行更新。

我们基于不同的学习率开展了两次实验。第一次实验中，学习率初始1e-1, 随后设置为1e-2, 随后设置为1e-3, 结果如图3. 每100 iterations记录一次平均分，平均分最高达到**4482.86**.

<img src="./image/图3.png" style="zoom:33%;" />

<center>图3 第一次实验的average score变化图</center>

第二次实验中，学习率初始1e-3, 随后设置为1e-4, 结果如图4. 每100 iterations记录一次平均分，平均分最高达到**4052.18**.

<img src="./image/图4.png" style="zoom:33%;" />

<center>图4 第二次实验的average score变化图</center>

按照一局游戏500步进行估算，每次实验中我们大约模拟了40000局游戏，这与文献当中数百万的游戏局数还有一定差距。因此，如果投入更多时间进行训练，这个模型仍有一定的上升空间。

## 参考文献

[1] Hasselt H V ,  Guez A ,  Silver D . Deep Reinforcement Learning with Double Q-learning[J]. Computer ence, 2015.

[2] Oka K ,  Matsuzaki K . Systematic Selection of N-Tuple Networks for 2048[C]// International Conference on Computers and Games. Springer International Publishing, 2016.

[3] Jaskowski, Wojciech. Mastering $2048$ with Delayed Temporal Coherence Learning, Multi-State Weight Promotion, Redundant Encoding and Carousel Shaping[J]. IEEE Transactions on Computational Intelligence and AI in Games, 2016, PP(99):1-1.