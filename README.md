# NeurogenomicsEngine

**Brain Cell-Type Transcriptomics and Neurodegeneration Analysis**

A pure-Python pipeline for brain transcriptomics analysis including cell-type deconvolution, synaptic scoring, and neurodegeneration risk.

## Features
- Cell-type deconvolution (7 brain cell types: neurons, astrocytes, oligodendrocytes, microglia, OPC, endothelial, pericytes)
- Synaptic gene module scoring (pre/post-synaptic, AMPA/NMDA receptors)
- Neurodegeneration risk score (AD/PD/ALS gene sets)
- Brain region-specific expression analysis (5 regions)
- Neuron-glia communication network (20 ligand-receptor pairs)

## Results
- 150 brain samples × 10,000 genes, 5 regions, AD vs control
- Deconvolution mean r: 0.998
- AD risk score: AD=2.54 vs Ctrl=0.58 (p=2.26e-52)
- Synaptic score: AD=3.87 vs Ctrl=4.54
- Region-specific genes: 18/200

## Usage
```bash
pip install numpy scipy matplotlib
python neurogenomics_engine.py
```

## Tags
`neurogenomics` `brain-transcriptomics` `neurodegenerative` `synaptic` `cell-type-specific` `alzheimers`
