Finite Volume Features, Global Geometry Representations, and Residual Training for Deep Learning-based CFD Simulation
----
GNNs are one of the state-of-the-art surrogates for numerical CFD simulations. In this work, we propose two novel geometric representations, Shortest Vector (SV) and Directional Integrated Distance (DID), that provide a global geometry perspective to the nodes in GNNs. We also introduce Finite Volume Features (FVF) in the graph convolutions as node and edge attributes, enabling GNNs message-passing operations to adjust to different nodes. Experiments show that the proposed techniques help boost SOTA GNN-based methods performance by up to 41%.

![](fvf.png)

**Datasets**
- Raw Coarse AirfRANS data: https://zenodo.org/records/11366835?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImZhMzZjODUyLTdkMGYtNGFiZS1iZGU0LTI1MjNlY2NlZDEwNSIsImRhdGEiOnt9LCJyYW5kb20iOiJjNDY2MDQ1MWYxM2I3MDg1YTM2MmRlNzBjOTYzYTg5OSJ9.O3McNOr9MCtyi2tHvKAjweTRewer3N6Wx4DTGfLlsv7-_a9fVXXRLuoNMdppyG3kHByiF0EN-s0mMy3eaX5SQw 

**Instructions**
- Running MeshgraphNet w/ FVF w/ Geo: 
- Running BSMSGNN w/ FVF w/ Geo: 
- Running ChenGCNN w/ FVF w/ Geo: 
- Running GraphUNet w/ FVF w/ Geo: 
- Running CFDGCN w/ Res w/ FVF w/ Geo: 


Disclaimer: 
----
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
