# TellusFM — A Pretrained Transferable Multitask Model for Material Fracture Prediction


## Overview

TellusFM is a Python package accompanying the manuscript on pretrained transferable transformer models for 2D material fracture prediction. The repository provides model code, example configurations, preprocessing utilities, pretrained weights, and small test datasets for reproducing the fracture-prediction workflows described in the paper.

---

## Quick Start

Clone the repository

```bash
git clone git@github.com:lanl/TellusFM.git
cd TellusFM
```

Install a new environment
(for other options such as `conda` or `uv`, see [DETAILED_GUIDE.md](DETAILED_GUIDE.md)) 

```bash
python3 -m venv tellusfm-env
source tellusfm-env/bin/activate
```

Install the required packages

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e .
```

(Optional) Validate the paths used in the examples

```bash
python example/validate_config.py --show-paths
```

Download the pretrained fracture checkpoint from Huggingface 

```bash
mkdir -p models
cd models
hf download smartFRACs/material_fracturing_model material_fracturing_model.ckpt --local-dir .
cd ..
```

Run a test using the test datasets included in the git repository

```bash
python example/run_all_phase_tests.py --pattern "ortho|curved"
```


Results are saved in `example/test_runs/<timestamp>/`:
- `summary.csv` — metrics for each test
- `summary.json` — detailed results in JSON format
- `<test_name>/` — per-test directory with logs and outputs

For more information on where to find the other training, validation and test sets, see [DETAILED_GUIDE.md](DETAILED_GUIDE.md)) or contact Agnese Marcato ([agnese.marcato@polito.it](mailto:agnese.marcato@polito.it)) or Aleksandra Pachalieva ([apachalieva@lanl.gov](mailto:apachalieva@lanl.gov)).



### Notice of Copyright Assertion (O4924):

*This program is Open-Source under the BSD-3 License.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:*
- *Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.*
- *Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.*
- *Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.*

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.




