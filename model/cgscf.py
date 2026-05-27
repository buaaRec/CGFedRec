import torch

class CGSCF(torch.nn.Module):
    def __init__(self, config):
        super(CGSCF, self).__init__()
        self.config = config
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.temperature = self.config['cl_t']
        self.base_temperature = self.config['base_t']
        self.reg = self.config['reg']

        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, item_indices, labels):#[1,2,5],[1,1,0]
        item_embedding = self.embedding_item(item_indices)
        logits = self.affine_output(item_embedding)
        rating = self.logistic(logits)
        if labels.shape[0] == 2:
            supcon = torch.tensor(0)
        else:
            embedding = torch.nn.functional.normalize(self.embedding_item.weight)
            supcon = self.sc_loss(embedding,labels)
        return rating, supcon
    
    def forward_test(self, user_id, item_indices):#[1,2,5],[1,1,0]
        item_embedding = self.embedding_item(item_indices)
        # print("item_embedding",item_embedding)
        # print("affine_output_weight",self.affine_output.weight.data)
        logits = self.affine_output(item_embedding)
        # print("logits",logits)
        rating = self.logistic(logits)
        return rating

    def sc_loss(self, features, labels=None, mask=None):
        """
        监督对比学习损失 (SupCon)。

        Args:
            features: 归一化后的特征向量，shape [N, D]。
            labels:   类别标签，shape [N]。与 mask 二选一。
            mask:     正样本二值掩码，shape [N, N]。与 labels 二选一。
                      mask[i, j] = 1 表示 i 与 j 属于同一类（正对）。
        Returns:
            标量损失值。
        """
        device = features.device
        N = features.shape[0]

        # ── 1. 构建正样本掩码 ─────────────────────────────────────────
        if labels is not None and mask is not None:
            raise ValueError("labels 和 mask 不能同时指定")

        if labels is None and mask is None:
            # 自监督：仅自身为正样本（SimCLR 风格需要多视角，此处退化为对角）
            mask = torch.eye(N, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.view(-1, 1)  # [N, 1]
            if labels.shape[0] != N:
                raise ValueError("labels 数量与 features 不一致")
            mask = torch.eq(labels, labels.T).float().to(device)  # [N, N]
        else:
            mask = mask.float().to(device)

        # ── 2. 计算相似度 logits（余弦相似度已在 features 归一化时隐含） ──
        sim = torch.matmul(features, features.T) / self.temperature  # [N, N]

        # 数值稳定：减去每行最大值（等价于 log-sum-exp trick）
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        # ── 3. 排除自身对（对角线置 0）──────────────────────────────────
        self_mask = torch.ones(N, N, device=device)
        self_mask.fill_diagonal_(0)  # 比 scatter 更直观

        mask = mask * self_mask  # 正样本掩码中也排除自身

        # ── 4. 计算对数概率 ──────────────────────────────────────────
        # 分母：对所有非自身样本求 exp 之和
        exp_sim = torch.exp(sim) * self_mask  # [N, N]
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)

        # ── 5. 对每个 anchor 的正样本取均值 ──────────────────────────
        # 若某行无正样本（孤立类），分母加 eps 防止除零
        pos_count = mask.sum(dim=1).clamp(min=1e-6)  # [N]
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos_count

        # ── 6. 最终损失（加温度缩放，取负号转为最小化） ─────────────────
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()
