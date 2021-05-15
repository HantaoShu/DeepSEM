# Demo data and output of Demo data

We provide demo data for three major function of DeepSEM.
- GRN Inference
    - input\
     including 500_ChIP-seq_hESC (cell type specific) and 500_STRING_hESC(cell type non-specific)
    - output\
    including predicting result of random seed 0. Hyperparameters are determined by cross-validation. \
    alpha=0.1,beta=0.01,epoch=150 for 500_ChIP-seq_hESC and alpha=100,beta=1,epoch=90 for 500_STIRNG_hESC
    
  Experiment can be finish within a few minutes.
  
- Embedding 
    - input\
    including Zeisel dataset
    - output\
    including directly output pickle file and the h5ad file which transformed from the pickle file.
    
  Experiment can be finish within a few minutes.
   
- Simulation
   - input\
   including CD19+B cells
   - output\
   including directly output h5ad file which is transfromed from the output pickle file.
      
  Experiment can be finish within a few minutes.