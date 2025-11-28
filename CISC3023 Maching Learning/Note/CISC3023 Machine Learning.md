# CISC3023 Machine Learning

# I. Naïve Bayesian Classifier 

例题：以下历史数据记录了过去14天的各种天气数据以及该天气是否进行了网球运动，结合历史数据求解在[outlook = rain, temp = mild, humidity = normal, wind = strong]条件下是否Play Tennis?

<img src="/Users/rocher/Library/Application Support/typora-user-images/image-20240422183529973.png" alt="image-20240422183529973" style="zoom:40%;" />

【解】记 A = 'Play Tennis: Yes'，则 P(A) = 9/14

 	   记 A‘ = 'Play Tennis: No'，则P(A') = 5/14

记B1 = 'rain'，在Play Tennis = Yes的9天中，有3天为rain，P(B1|A) = 3/9；在Play Tennis = No的5天中，有2天为rain，P(B1|A')  = 3/5；

记B2 = 'mild'，同理可得P(B2|A) = 4/9，P(B2|A')  = 2/5；

记B3 = 'normal'，同理可得P(B3|A) = 6/9，P(B3|A')  = 1/5；

记B4 = 'strong'，同理可得P(B4|A) = 3/9，P(B4|A')  = 3/5

目标P(A|B1B2B3B4) = P(A) * P(B1|A) * P(B2|A) * P(B3|A) * P(B4|A)  = 4/189

而P(A'|B1B2B3B4) = P(A') * P(B1|A') * P(B2|A') * P(B3|A') * P(B4|A') = 9/875

P(A|B1B2B3B4) > P(A'|B1B2B3B4) 因此Play Tennis = Yes.

# II.Decision Trees

- 熵：表示某个随机变量的不确定性。其计算公式如下：

  $H(X) = -\sum\limits_{i=1}^{N}p_i\log p_i$​

  $p_i$为随机变量取第i个值时对应的概率。例如掷六面骰子所得点数的熵为$H=-\sum\limits_{i=1}^6 \frac{1}{6}\log\frac{1}{6}=-\log\frac{1}{6}$​.

- 条件熵：事件X发生的情况下事件Y发生的熵

  $H(Y|X) = \sum\limits_{i=1}^Np_iH(Y|X=x_i)$​

- 信息增益：条件熵的减少程度，即熵减去条件熵

  $\text{Gain}(D, A)=H(D)-H(D|A)$

  > 例：如下数据集D中，特征为性别G（男/女）以及活跃度A（高/中/低），标签L为是否流失
  >
  > | G    | M    | F    | M    | F    | M    | M    | M    | F    | F    | F    | F    | M    | F    | M    | M    |
  > | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
  > | A    | 高   | 中   | 低   | 高   | 高   | 中   | 中   | 中   | 低   | 中   | 高   | 低   | 低   | 高   | 高   |
  > | L    | 0    | 0    | 1    | 0    | 0    | 0    | 1    | 0    | 1    | 0    | 0    | 1    | 1    | 0    | 0    |
  >
  > 总的熵$H(D) = -(P(L=0)\log P(L=0) + P(L=1) \log P(L=1)) = -\frac{5}{15}\log\frac{5}{15} - \frac{10}{15}\log \frac{10}{15}$ 
  >
  > 性别为男的熵：$H(D|G=M) = -(P(L=0|G=M)\log P(L=0|G=M) + P(L=1|G=M)\log P(L=1|G=M))$$=-\frac{3}{8}\log\frac{3}{8} - \frac{3}{8}\log \frac{3}{8}$​​
  >
  > 同理：
  >
  > <img src="/Users/rocher/Library/Application Support/typora-user-images/image-20240422192049592.png" alt="image-20240422192049592" style="zoom:50%;" />
  >
  > 因此：
  >
  > - 性别的信息增益
  >
  >   $\text{Gain}(D,G)=H(D)-H(D|G) = H(D)-(P(G=M)H(D|G=M)+P(G=F)H(D|G=F))$
  >
  > - 活跃度的信息增益
  >
  >   $\text{Gain}(D,A)=H(D)-H(D|A)=H(D)-$$(P(A='高')H(D|A='高')+P(A='中')H(D|A='中')+P(A='低')H(D|A='低'))$​

- ID3算法构建决策树：

  - 计算每个特征的信息增益，选取其中最大的信息增益对应的特征作为分裂节点
  - 迭代进行，直至使用了所有的特征

> 例题：以下表格记录了大米的特征，及是否能用于制作寿司
>
> ![image-20240422222827586](/Users/rocher/Library/Application Support/typora-user-images/image-20240422222827586.png)
>
> 1. 使用Find-S算法找到最具体的能够满足上述样例的假设
>
>    【解】Find-S算法：
>
>    初始化S为第一个正例：S1=<long wide light white>
>
>    遇到反例时，在S1中将与反例相同的特征设置为？：S2=<long ? light white>
>
>    遇到正例时，在S2中将与正例不同的特征设置为？：S3=<? ? light white>
>
>    同理，S4 = <? ? light white>, S5=<? ? light white>
>
> 2. 使用candidate-elimination算法给出上述样例的version space
>
>    

# III. K-Nearest Neighbors

例题：给出五个点$X_1 \sim X_5$及其对应坐标，给点$X_q=(4,8,4)$，求解以下问题：

|       | $x_1$ | $x_2$ | $x_3$ | output |
| :---: | :---: | :---: | :---: | :----: |
| $X_1$ |   7   |   5   |   3   |   2    |
| $X_2$ |   3   |   9   |   4   |   10   |
| $X_3$ |  10   |   3   |   2   |   6    |
| $X_4$ |   4   |   9   |   7   |   2    |
| $X_5$ |   5   |   2   |  10   |   8    |

1. 使用距离加权的3-最近邻算法计算$X_q$的output（output类型为class label而非实数）：

   【解】计算$X_q$与每个点之间的欧式距离$d_i$：

   $d_1 = \sqrt{19}, d_2 = \sqrt{2}, d_3=\sqrt{65}, d_4=\sqrt{10}, d_5=\sqrt{73}$​

   因此$X_q$的三个最近邻点为$X_1, X_2, X_4$，计算对应权重$w_i = 1/d_i^2$：

   $w_1=\frac{1}{19}, w_2=\frac{1}{2}, w_1=\frac{1}{10}$​

   $X_1$和$X_4$具有相同的标签2，其权重和为29/190；$X_2$的权重为1/2。因此$X_q$目标输出与$X_2$相同，为10.

2. 使用距离加权的3-最近邻算法计算$X_q$​的output（output类型为实数）：

   【解】

   $output = \frac{w_1f(x_1)+w_2f(x_2)+w_4f(x_4)}{w_1+w_2+w_4}=8.1 $​

   

# IV. Linear Regression

在线性回归中，我们的目标是最小化损失函数 $L(W, b) $，这通常是均方误差。梯度下降法用于更新权重 ( W ) 和偏置 ( b )，以减少损失。以下是求 ( W ) 和 ( b ) 的梯度的公式：

1. **损失函数** $L(W, b)$  定义为：

   $L(W, b) = \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)} - (Wx^{(i)} + b))^2$

2. **梯度计算**：

   - 对 ( W ) 的梯度：

     $\frac{\partial L(W, b)}{\partial W} = -\frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - (Wx^{(i)} + b))x^{(i)}$

   - 对 ( b ) 的梯度：

     $\frac{\partial L(W, b)}{\partial b} = -\frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - (Wx^{(i)} + b))$

3. **梯度下降更新规则**：

   - 更新 ( W )：

     $W := W - \alpha \frac{\partial L(W, b)}{\partial W}$

   - 更新 ( b )：

     $b := b - \alpha \frac{\partial L(W, b)}{\partial b}$

其中，( m ) 是样本数量，$\alpha $是学习率，$x^{(i)} $是第 ( i ) 个样本的特征，$y^{(i)} $是第 ( i ) 个样本的真实标签。通过反复应用这些更新规则，我们可以逐渐减少损失函数的值，从而找到最佳的 ( W ) 和 ( b ) 来拟合我们的数据。

# V. Logistic Regression

在Logistic回归中，我们的目标是最小化损失函数 $ L(W) $，这通常是对数似然损失。梯度下降法用于更新参数 $W $，以减少损失。以下是求参数 $W $ 的梯度的公式：

1. **Sigmoid函数** $ \sigma(z)$ 定义为：

   $\sigma(z) = \frac{1}{1 + e^{-z}}$

2. **损失函数** $ L(W) $ 定义为：

   $L(W) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\sigma(Wx^{(i)})) + (1 - y^{(i)}) \log(1 - \sigma(Wx^{(i)}))]$

3. **梯度计算**：

   - 对 ( W ) 的梯度 $\nabla_W L(W)$：

   $\nabla_W L(W) = \frac{1}{m} \sum_{i=1}^{m} (\sigma(Wx^{(i)}) - y^{(i)}) x^{(i)}$

4. **参数更新**：

   - 使用学习率 $ \alpha $ 更新 ( W )：

   $W := W - \alpha \nabla_W L(W)$

在每次迭代中，用上述梯度计算公式来更新权重 ( W )，直到损失函数 ( L(W) ) 收敛到一个最小值或达到预定的迭代次数。              

# VI. Support Vectore Machines

支持向量机（SVM）是一种监督学习算法，用于分类和回归分析。对于二维数据点的分类，SVM的目标是找到一个最优的决策边界（即超平面），使得不同类别的数据点能够被正确地分开。以下是SVM分类二维数据点的基本步骤和相关公式：

1. **选择合适的核函数**：核函数用于将原始数据映射到更高维的空间，以便在新的空间中找到线性可分的超平面。常用的核函数包括线性核、多项式核、径向基函数（RBF）核等。

2. **构建目标函数**：SVM的目标是最大化两个类别之间的间隔，同时确保数据点正确分类。目标函数和约束条件可以表示为：

   $\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2$          $\text{s.t. } y^{(i)}(\mathbf{w} \cdot \mathbf{x}^{(i)} + b) \geq 1, \forall i$

   其中，$\mathbf{w}$ 是超平面的法向量，$b$是偏置项，$y^{(i)}$是第$i $个数据点的类别标签，$\mathbf{x}^{(i)}$是第$i $个数据点的特征向量。

3. **拉格朗日乘子法**：为了解决上述优化问题，引入拉格朗日乘子$\alpha_i$并构建拉格朗日函数：

   $L(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2} \|\mathbf{w}\|^2 - \sum_{i=1}^{m} \alpha_i [y^{(i)}(\mathbf{w} \cdot \mathbf{x}^{(i)} + b) - 1]$

4. **对偶问题**：通过求解拉格朗日函数的对偶问题，可以得到最优化问题的解。对偶问题的目标函数为：

   $\max_{\boldsymbol{\alpha}} \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{m} \alpha_i \alpha_j y^{(i)} y^{(j)} (\mathbf{x}^{(i)} \cdot \mathbf{x}^{(j)})$

   $\text{s.t. } \sum_{i=1}^{m} \alpha_i y^{(i)} = 0$

5. **求解对偶问题**：使用序列最小优化（SMO）算法或其他优化方法求解 $\boldsymbol{\alpha} $。

6. **计算 $\mathbf{w}$和 $b$**：根据求得的 $\boldsymbol{\alpha}$值，计算超平面的法向量 $\mathbf{w}$ 和偏置项$b$：

   $\mathbf{w} = \sum_{i=1}^{m} \alpha_i y^{(i)} \mathbf{x}^{(i)}$

   $b = y^{(k)} - \mathbf{w} \cdot \mathbf{x}^{(k)}$

   其中，$k$是任意一个满足 $0 < \alpha_k < C $ 的索引。

7. **分类新数据点**：使用学习到的模型对新的数据点进行分类：

   $f(\mathbf{x}) = \text{sign}(\mathbf{w} \cdot \mathbf{x} + b)$

以上就是使用SVM对一系列二维数据点进行分类的过程及公式。在实际应用中，还需要考虑如何处理数据不完全线性可分的情况，这时可以引入软间隔和松弛变量来允许一定程度的分类错误。

> 例题：求解下列问题的SVM模型
>
> |  y   |  x1  |  x2  |
> | :--: | :--: | :--: |
> |  1   |  2   |  1   |
> |  -1  |  1   |  2   |
> |  -1  |  3   |  3   |
>
> Object function: $\min\limits_{w,b} = \frac{1}{2}\|w\|^2, \quad s.t. y_i(wx_i+b)\ge 1$
>
> Lagrange Form: $L(w, b, \alpha)=\frac{1}{2}\|w\|^2-\sum\limits_{i=1}^3\alpha_i[y_i(wx_i+b)-1]$​
>
> Dual Problem: $\max\limits_{\alpha} \sum\limits_{i=1}^3\alpha_i-\frac{1}{2}\sum\limits_{i,j=1}^3\alpha_i\alpha_jy_iy_j(x_ix_j), \quad \sum\limits_{i=1}^3\alpha_iy_i = 0$
>
> i.e., 



​              

# VI. Neural Networks and Backpropogation

# VII. K-Means for Clustering



