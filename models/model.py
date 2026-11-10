import torch
import torch.nn as nn
import torch.nn.functional as F
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize


class VirtualPatch(nn.Module):
    """Learnable virtual patches (Eq. 9): global semantic anchors appended to the patch features."""

    def __init__(self, num_virtual=32, dim=512):
        super(VirtualPatch, self).__init__()
        # m learnable d-dimensional tokens; the same instance is shared by the 5x and 20x branches
        self.virtual_patches = nn.Parameter(torch.empty(1, num_virtual, dim))
        nn.init.normal_(self.virtual_patches, std=0.02)

    def forward(self, x):
        # x: [1, N, d] -> [1, N + m, d], virtual patches appended after the real ones
        return torch.cat((x, self.virtual_patches), dim=1)


class GatedABMIL(nn.Module):
    """Gated attention pooling from ABMIL (Eq. 6): softmax over the token axis, then weighted sum."""

    def __init__(self, input_dim, hidden_dim=128):
        super(GatedABMIL, self).__init__()
        self.attention_V = nn.Linear(input_dim, hidden_dim)
        self.attention_U = nn.Linear(input_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B, L, input_dim]; V and U are the two gating branches, `*` is element-wise (not matmul)
        Vx = torch.tanh(self.attention_V(x))
        Ux = torch.sigmoid(self.attention_U(x))
        A = self.attention_w(Vx * Ux).squeeze(-1)   # [B, L]
        A = F.softmax(A, dim=1)                     # attention weight of every token

        A = A.unsqueeze(-1)
        out = (A * x).sum(dim=1)                    # [B, input_dim] pooled representation

        return out


class TextEncoder(nn.Module):
    """CONCH text transformer + token-level compression (TSC, Eq. 6).

    The CONCH text encoder is reused here and trained together with the proposed modules
    (Sec. 3.3.2); the visual encoder is frozen and its features are pre-extracted, so it does not
    appear in this model.
    """

    def __init__(self, conch_model):
        super().__init__()
        self.transformer = conch_model.text.transformer
        self.positional_embedding = conch_model.text.positional_embedding
        self.ln_final = conch_model.text.ln_final
        self.text_projection = conch_model.text.text_projection
        self.dtype = next(conch_model.parameters()).dtype
        self.mil = GatedABMIL(input_dim=768, hidden_dim=128)   # token-level compression, Eq. (6a)

    def forward(self, prompts, name_lens):
        # prompts: [n_prompts, 128, 768], one prompt per class per scale
        x = prompts + self.positional_embedding.type(self.dtype)

        # CONCH's transformer expects [seq_len, batch, dim]
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        # keep the real tokens of each prompt (drop the padding), then pool them (Eq. 6a)
        x = [x[i, :name_lens[i], :] for i in range(len(name_lens))]
        class_feats = []
        for class_feat in x:
            class_feats.append(self.mil(class_feat.unsqueeze(0)))
        class_feats = torch.cat(class_feats, dim=0)     # [n_prompts, 768]
        x = class_feats @ self.text_projection          # W_b: 768 -> 512

        return x


class AttentionPatch(nn.Module):
    """Patch-level classifier and importance weights (Eq. 2-3), shared by the two scales."""

    def __init__(self, keep_patch_threshold=True, top_patch_num=None, n_class=2):
        super(AttentionPatch, self).__init__()
        self.keep_patch_threshold = keep_patch_threshold
        self.top_patch_num = top_patch_num
        self.n_class = n_class
        self.cls = nn.Linear(512, n_class)      # W_e: patch feature -> patch logits

    # x: 1 * patch_num * 512
    def forward(self, x):
        gamma = x.shape[1]                      # number of patches N
        out_alpha = self.cls(x)                 # [1, N, C] patch logits O

        # importance = L2 norm of the patch logits, normalised over all patches (Eq. 3a)
        out = torch.sqrt((out_alpha ** 2).sum(dim=2))
        alpha = out / out.sum(dim=1, keepdim=True).expand_as(out)

        # remove trivial patches below beta / N (beta = 0.1) and re-normalise (Eq. 3b)
        if self.keep_patch_threshold:
            alpha = F.relu(alpha - .1 / float(gamma))
        alpha = alpha / alpha.sum(dim=1, keepdim=True).expand_as(alpha)

        return out_alpha.squeeze(dim=0), alpha.squeeze(dim=0)


class PromptLearner(nn.Module):
    """Learnable prompt context (CoOp style).

    Each prompt is [SOS] + n_ctx learnable context tokens + the class description tokens: the first
    n_ctx tokens after [SOS] are replaced by the learnable context, the remaining tokens keep the
    class-specific description. The context is shared by all classes and by both scales; `name_lens`
    counts the non-padding tokens of each prompt.
    """

    def __init__(self, classnames, conch_model, n_ctx=16):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = n_ctx
        ctx_init = ""
        dtype = next(conch_model.parameters()).dtype
        ctx_dim = conch_model.text.ln_final.weight.shape[0]

        self.tokenizer = get_tokenizer()

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(self.tokenizer, [ctx_init])
            with torch.no_grad():
                embedding = conch_model.text.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
        else:
            # random initialisation of the learnable context tokens
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [name for name in classnames]

        tokenized_prompts = tokenize(self.tokenizer, prompts)

        with torch.no_grad():
            embedding = conch_model.text.token_embedding(tokenized_prompts).type(dtype)

        # [SOS] token and the description tokens after the first n_ctx positions
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        # number of real (non-padding) tokens per prompt, used to trim the encoded sequence
        self.name_lens = [len(self.tokenizer.encode(name,
                                                    max_length=127,
                                                    truncation=True))
                          for name in classnames]
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        # [SOS] + learnable context + class description  ->  [n_cls, 128, 768]
        prompts = torch.cat(
            [
                prefix,
                ctx,
                suffix,
            ],
            dim=1,
        )

        return prompts, self.name_lens



class MMF(nn.Module):
    """Multi-scale multi-modal framework: TSC (text) + PDCA (fusion) + patch attention.

    The two scales share every trainable module: `*_s` is the 5x branch (conch_2048 features) and
    `*_l` the 20x branch (conch features).
    """

    def __init__(self, config, num_classes=2):
        super(MMF, self).__init__()
        self.loss_ce = nn.CrossEntropyLoss()            # WSI-level loss, Eq. (13)
        self.num_classes = num_classes
        self.gate_attn = GatedABMIL(512, 128)           # f_gate of Eq. (11), an independent instance

        self.L = config.input_size
        self.D = 512

        conch_model_cfg = 'conch_ViT-B-16'
        conch_model, _ = create_model_from_pretrained(conch_model_cfg, config.conch_checkpoint_path)

        self.feature_dim = conch_model.text.text_projection.shape[1]  # 512

        # text branch: learnable prompt + CONCH text encoder (TSC)
        self.prompt_learner = PromptLearner(config.text_prompt, conch_model.float())
        self.text_encoder = TextEncoder(conch_model.float())

        # patch projection shared by both scales (W_c in Eq. 9; implemented as Linear + LayerNorm + ReLU + Dropout)
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.LayerNorm(self.D),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        # one multi-head cross-attention, shared by both scales (Eq. 10)
        num_heads = 8
        self.head_dim = self.feature_dim // num_heads

        self.q_proj_l = nn.Linear(self.feature_dim, self.feature_dim)   # W_q
        self.k_proj_l = nn.Linear(self.feature_dim, self.feature_dim)   # W_k
        self.v_proj_l = nn.Linear(self.feature_dim, self.feature_dim)   # W_v
        self.o_proj_l = nn.Linear(self.feature_dim, self.feature_dim)   # output projection
        self.num_heads = num_heads

        self.classifier = nn.Linear(self.feature_dim, num_classes)      # W_o of Eq. (12)
        self.virtual_patch = VirtualPatch(num_virtual=32, dim=self.D)   # m = 32 virtual patches

        self.lossfunc = nn.CrossEntropyLoss(reduction='none')           # per-patch loss, Eq. (4)
        self.patch_attn = AttentionPatch(keep_patch_threshold=True, n_class=num_classes)

    def cross_attention(self, queries, keys, values, attention_mask=None):
        """Multi-head text-to-visual cross-attention (Eq. 10).

        queries are the class-level text embeddings (one row per class), keys/values are the
        augmented patch tokens (real patches + virtual patches).
        """
        bsz, q_len, _ = queries.size()
        _, kv_len, _ = keys.size()

        query_states = self.q_proj_l(queries)
        key_states = self.k_proj_l(keys)
        value_states = self.v_proj_l(values)

        # [B, L, dim] -> [B, heads, L, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        # softmax(Q K^T / sqrt(d_h)) V
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attention_mask
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.feature_dim)
        attn_output = self.o_proj_l(attn_output)

        return attn_output

    def auxiliary_cosine_distance_loss(self, text_features, margin=0.5):
        """Margin-based inter-class similarity loss (Eq. 7-8).

        Penalises class pairs whose cosine similarity exceeds `margin` (delta = 0.5 in the paper),
        which pushes the class-level text embeddings apart. Averaged over all class pairs.
        """
        if text_features.shape[1] == 2:
            feat1 = F.normalize(text_features[:, 0, :], dim=1)
            feat2 = F.normalize(text_features[:, 1, :], dim=1)
            cosine_sim = (feat1 * feat2).sum(dim=1)
            loss = torch.clamp(cosine_sim - margin, min=0.0)
            loss = loss.pow(2).mean()
        else:
            B, N, D = text_features.shape
            feats = F.normalize(text_features, dim=-1)
            sim_matrix = torch.bmm(feats, feats.transpose(1, 2))
            mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()   # upper triangle, i < j
            cosine_sims = sim_matrix[mask]
            loss = F.relu(cosine_sims - margin).pow(2).mean()
        return loss

    def weight_cross_entropy(self, y, label, w, reduction='sum', loss=None):
        """Attention-weighted patch-level cross-entropy (Eq. 4-5): sum_i gamma_i * CE(patch_i)."""
        if loss is None:
            l = F.cross_entropy(y, label, reduction='none') * w
        else:
            l = loss(y, label) * w

        return l.mean() if reduction != 'sum' else l.sum()

    def forward(self, x_s, x_l, label, camelyon_patch=False):
        # ---- text branch: one text embedding per class per scale (first C = 5x, last C = 20x)
        prompts, name_lens = self.prompt_learner()
        text_features = self.text_encoder(prompts, name_lens)
        text_features_l = text_features[self.num_classes:]
        text_features_s = text_features[:self.num_classes]

        text_features_l = text_features_l.unsqueeze(0)      # [1, C, 512]
        text_features_s = text_features_s.unsqueeze(0)

        # ---- image branch: project the pre-extracted CONCH features, then add virtual patches
        features_l = self.feature_encoder(x_l.float())
        features_l = features_l.unsqueeze(0)                # [1, N_l, 512]

        features_s = self.feature_encoder(x_s.float())
        features_s = features_s.unsqueeze(0)                # [1, N_s, 512]

        augmented_features_l = self.virtual_patch(features_l)   # [1, N_l + 32, 512]
        augmented_features_s = self.virtual_patch(features_s)

        # ---- fusion branch (PDCA): text queries attend to each scale separately (Eq. 10)
        attended_features_l = self.cross_attention(
            text_features_l,
            augmented_features_l,
            augmented_features_l,
        )

        attended_features_s = self.cross_attention(
            text_features_s,
            augmented_features_s,
            augmented_features_s,
        )

        # concatenate the C class tokens of both scales and fuse them with gated attention (Eq. 11)
        attended_features = torch.cat([attended_features_l, attended_features_s], dim=1)   # [1, 2C, 512]
        final_features = self.gate_attn(attended_features)                                 # [1, 512]
        logits = self.classifier(final_features)                                           # [1, C]

        # ---- losses: L = L_WSI + lambda1 * L_sim + lambda2 * L_patch (lambda1 = lambda2 = 1)
        loss = self.loss_ce(logits, label) + 1 * (self.auxiliary_cosine_distance_loss(text_features_l) +
                                                  self.auxiliary_cosine_distance_loss(text_features_s))

        # patch-level supervision for both scales, each patch weighted by its importance gamma_i
        out_alpha_l, alpha_l = self.patch_attn(features_l)
        out_alpha_s, alpha_s = self.patch_attn(features_s)
        loss_patch_l = self.weight_cross_entropy(out_alpha_l, label.repeat(alpha_l.shape[0]), alpha_l,
                                                 loss=self.lossfunc)
        loss_patch_s = self.weight_cross_entropy(out_alpha_s, label.repeat(alpha_s.shape[0]), alpha_s,
                                                 loss=self.lossfunc)
        loss += 1 * (loss_patch_l + loss_patch_s)

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
        if camelyon_patch:
            # patch-level output of the 5x branch (`out_alpha_s`); use `out_alpha_l` for the 20x one
            patch_prob = F.softmax(out_alpha_s, dim=1)
            patch_hat = (patch_prob[:, 1] > 0.5).long()
            return Y_prob, Y_hat, loss, patch_prob, patch_hat
        else:
            return Y_prob, Y_hat, loss
