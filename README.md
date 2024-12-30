# QSA for GPT-1: Structure, Training Details

**Note:** During all experiments with GPT-1:
- **2 epochs**, **1024 batch size**
- **6 stacked multihead self-attentions (MHSA)**, each MHSA has only **1 head**.
- **CSA** has a classical structure: a single `q/k/v` layer for all tokens.

---

## QSA Original Version: Each Token Has Its Own `q/k/v` Layers

**Number of parameters per a single self-attention layer:**

| **CSA Parameters**                          | **QSA Parameters**                                      |
|---------------------------------------------|---------------------------------------------------------|
| `3 × embedding_size × hidden_size`          | `3 × ⌈log₂(embedding_size)⌉ × context_size`             |

**Training Setup:**
- **Context size** = 16
- **Embedding size** = 4

![Cross-entropy comparison: QSA vs CSA with embedding size = 4](https://github.com/user-attachments/assets/60e86311-bb46-4fff-b151-05c7a9c8ce49)

**Training Setup:**
- **Context size** = 16
- **Embedding size** = 16

![Cross-entropy comparison: QSA vs CSA with embedding size = 16](https://github.com/user-attachments/assets/2aadbdb1-a0c7-49f3-adec-6bfd52da27f3)


---

## QSA Fixed Version: A Single `q/k/v` Layer for All Tokens

**Number of parameters per a single self-attention layer:**

| **CSA Parameters**                          | **Original QSA Parameters**                             | **Fixed QSA Parameters**                     |
|---------------------------------------------|---------------------------------------------------------|----------------------------------------------|
| `3 × embedding_size × hidden_size`          | `3 × ⌈log₂(embedding_size)⌉ × context_size`             | `3 × ⌈log₂(embedding_size)⌉`                 |

**Training Setup:**
- **Context size** = 16
- **Embedding size** = 4

![Cross-entropy comparison: Original QSA vs Fixed QSA vs CSA with embedding size = 4](https://github.com/user-attachments/assets/46031a83-8881-481f-907c-9b986d77c90b)


**Training Setup:**
- **1 epoch**
- **Context size** = 16
- **Embedding size** = 16

![Cross-entropy comparison: Original QSA vs Fixed QSA vs CSA with embedding size = 16](https://github.com/user-attachments/assets/b5908d27-ae87-4ee5-a138-71b02e2536fc)

## Changing the QK normalization
The origianl paper proposed the plain averaging:
![image](https://github.com/user-attachments/assets/06adef59-0773-44f2-9168-2c849f906c2a)


Changed to the softmaxing, become slightly worse:

**Training Setup:**
- **Context size** = 16
- **Embedding size** = 4

![image](https://github.com/user-attachments/assets/ba845d79-e38a-47d6-8e53-bb1304725086)

