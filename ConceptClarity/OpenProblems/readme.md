## Open Research Problems That Interest Me
1. Invariance property
    - Currently, NNs can be made invariant to transformations such as (in the context of CV) illumination, rotation, scale, deformation and so on using data augmentation.
    - However, data augmentation requires training (fine-tuning). How can we introduce such invariance to pre-trained networks?
    - How do we make a pre-trained model robust without additional parameter tuning?
2. Knowledge Distillation
    - KD in a traditional setting does not distil robustness from one model to another (inductive bias even within the same family of models)
    - Transferring knowledge from a big model to a small model *_efficiently and effectively_* is the need of the hour in LLM
    - From my experience, I believe that the activation functions play an important role.  
