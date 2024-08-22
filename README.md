Finite Volume Features, Global Geometry Representations, and Residual Training for Deep Learning-based CFD Simulation
----
GNNs are one of the state-of-the-art surrogates for numerical CFD simulations. In this work, we propose two novel geometric representations, Shortest Vector (SV) and Directional Integrated Distance (DID), that provide a global geometry perspective to the nodes in GNNs. We also introduce Finite Volume Features (FVF) in the graph convolutions as node and edge attributes, enabling GNNs message-passing operations to adjust to different nodes. Experiments show that the proposed techniques help boost SOTA GNN-based methods performance by up to 41%.

![](fvf.png)

**Datasets**
- Raw Coarse AirfRANS data: https://zenodo.org/records/11366835?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImZhMzZjODUyLTdkMGYtNGFiZS1iZGU0LTI1MjNlY2NlZDEwNSIsImRhdGEiOnt9LCJyYW5kb20iOiJjNDY2MDQ1MWYxM2I3MDg1YTM2MmRlNzBjOTYzYTg5OSJ9.O3McNOr9MCtyi2tHvKAjweTRewer3N6Wx4DTGfLlsv7-_a9fVXXRLuoNMdppyG3kHByiF0EN-s0mMy3eaX5SQw 

**Instructions**
- Running MeshgraphNet w/ FVF w/ Geo: 
  - `python mainNEW.py -e fvmgn_geom --model fvmgn --batch-size 1 --saf 1 --dsdf 1 --hidden-size 128 -FV --gpus 0`
- Running BSMSGNN w/ FVF w/ Geo: 
  - ``
- Running ChenGCNN w/ FVF w/ Geo: 
  - with IVE conv => Please run the cell under `model3: CHEN-GCNN W/ FVF W/ GEO` in the notebook: *ChenGCNN/GCNN_Chen-Cleaned.ipynb*
- Running GraphUNet w/ FVF w/ Geo: 
  - with SAGE conv => `python mainNEW_wip_rebuttal.py GNetFVnewGraphSAGE_FV_SAF_dSDF -t scarce -n 1 -s 1 -p half -cuda cuda:0`
  - with GCN => `python mainNEW_wip_rebuttal.py GNetFVnewGCN_FVnew_SAF_dSDF -t scarce -n 1 -s 1 -p half -cuda cuda:0`
- Running CFDGCN w/ Res w/ FVF w/ Geo: 
    - with SAGE conv => `mpirun -np $((BATCH_SIZE+1)) --oversubscribe python mainNEWRebut.py --batch-size $BATCH_SIZE --gpus 1 -dw 2 --su2-config coarse.cfg --model cfd_fvnewgsage --hidden-size 315 --num-layers 6 --num-end-convs 3 --optim adam -lr 5e-5 --saf 1 --dsdf 1 --FV True --residual True --A_pow 1 -e cfd_fvgsage_GEO_FV_RES > /dev/null`
    - with GCN => `mpirun -np $((BATCH_SIZE+1)) --oversubscribe python mainNEWRebut.py --batch-size $BATCH_SIZE --gpus 1 -dw 2 --su2-config coarse.cfg --model cfd_fvnewgcn --hidden-size 284 --num-layers 6 --num-end-convs 3 --optim adam -lr 5e-5 --saf 1 --dsdf 1 --FV True --residual True --A_pow 1 -e cfd_fvgcn_GEO_FV_RES > /dev/null`

Citation:
---
```
@inproceedings{
jessica2024finite,
title={Finite Volume Features, Global Geometry Representations, and Residual Training for Deep Learning-based {CFD} Simulation},
author={Loh Sher En Jessica and Naheed Anjum Arafat and Wei Xian Lim and Wai Lee Chan and Adams Wai-Kin Kong},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=WzD4a5ufN8}
}
```

Disclaimer: 
----
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
