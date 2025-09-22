# Hybrid MambaVision and Transformer-Based Architecture for Robust 3D Lane Detection

This repository is hosting the implementation of our “Hybrid MambaVision and Transformer-Based Architecture for Robust 3D Lane Detection” model.

## Environments
For package installation instructions, please see the [installation guide](./docs/install.md).

## Data
To download the desired dataset, follow the steps in [data preparation](./docs/data_preparation.md).

## Evaluation
For evaluation details, refer to the [evaluation guide](./docs/train_eval.md#evaluation).

## Training
To train the model, follow the instructions in [training](./docs/train_eval.md#train).
Pretrained models will be available soon.

## Benchmark

### Apollo

Here are the results for the balanced scenes. For the rest of the scenes please check our work.

<table>
    <tr>
        <td>Model</td>
        <td>F1 (%)</td>
        <td>X error near (m)</td>
        <td>X error far (m)</td>
        <td>Z error near (m)</td>
        <td>Z error far (m)</td>
    </tr>
    <tr>
        <td>3D-LaneNet <sup>[1]</sup></td>
        <td>86.4</td>
        <td>0.068</td>
        <td>0.477</td>
        <td>0.015</td>
        <td><b>0.202</b></td>
    </tr>
    <tr>
        <td>Gen-LaneNet <sup>[2]</sup></td>
        <td>88.1</td>
        <td>0.061</td>
        <td>0.469</td>
        <td>0.012</td>
        <td>0.214</td>
    </tr>
    <tr>
        <td>CLGo <sup>[3]</sup></td>
        <td>91.9</td>
        <td>0.061</td>
        <td>0.361</td>
        <td>0.029</td>
        <td>0.250</td>
    </tr>
    <tr>
        <td>PersFormer <sup>[4]</sup></td>
        <td>92.9</td>
        <td>0.054</td>
        <td>0.356</td>
        <td>0.010</td>
        <td>0.234</td>
    </tr>
    <tr>
        <td>Anchor3DLane <sup>[5]</sup></td>
        <td>95.6</td>
        <td>0.052</td>
        <td>0.306</td>
        <td>0.015</td>
        <td>0.233</td>
    </tr>
    <tr>
        <td>CurveFormer <sup>[6]</sup></td>
        <td>95.8</td>
        <td>0.078</td>
        <td>0.326</td>
        <td>0.018</td>
        <td>0.219</td>
    </tr>
    <tr>
        <td>LATR <sup>[7]</sup></td>
        <td>96.8</td>
        <td>0.022</td>
        <td>0.253</td>
        <td><b>0.007</b></td>
        <td><b>0.202</b></td>
    </tr>
    <tr>
        <td>BEV-LaneDet <sup>[8]</sup></td>
        <td>96.9</td>
        <td><b>0.016</b></td>
        <td><b>0.242</b></td>
        <td>0.020</td>
        <td>0.216</td>
    </tr>
    <tr>
        <td>LaneCPP <sup>[9]</sup></td>
        <td><b>97.4</b></td>
        <td>0.030</td>
        <td>0.277</td>
        <td>0.011</td>
        <td>0.216</td>
    </tr>
    <tr>
        <td><b>Ours</b></td>
        <td>97.0</td>
        <td>0.024</td>
        <td>0.255</td>
        <td>0.009</td>
        <td>0.204</td>
    </tr>
</table>


### ONCE
Our method achieves state-of-the-art (SOTA) results on the ONCE dataset, outperforming previous approaches.
<table>
    <tr>
        <td>Model</td>
        <td>F1 (%)</td>
        <td>Precision (%)</td>
        <td>Recall (%)</td>
        <td>CD error (m)</td>
    </tr>
    <tr>
        <td>3D-LaneNet <sup>[1]</sup></td>
        <td>44.73</td>
        <td>61.46</td>
        <td>35.16</td>
        <td>0.127</td>
    </tr>
    <tr>
        <td>Gen-LaneNet <sup>[2]</sup></td>
        <td>45.59</td>
        <td>63.95</td>
        <td>35.42</td>
        <td>0.121</td>
    </tr>
    <tr>
        <td>SALAD <sup>[3]</sup></td>
        <td>64.07</td>
        <td>75.90</td>
        <td>55.42</td>
        <td>0.098</td>
    </tr>
    <tr>
        <td>PersFormer <sup>[4]</sup></td>
        <td>74.33</td>
        <td>80.30</td>
        <td>69.18</td>
        <td>0.074</td>
    </tr>
    <tr>
        <td>Anchor3DLane <sup>[5]</sup></td>
        <td>74.87</td>
        <td>80.85</td>
        <td>69.71</td>
        <td>0.060</td>
    </tr>
    <tr>
        <td>WS-3D-Lane <sup>[6]</sup></td>
        <td>77.02</td>
        <td>84.51</td>
        <td>70.75</td>
        <td>0.058</td>
    </tr>
    <tr>
        <td>LATR <sup>[7]</sup></td>
        <td>80.59</td>
        <td><b>86.12</b></td>
        <td>75.73</td>
        <td><b>0.052</b></td>
    </tr>
    <tr>
        <td><b>Ours</b></td>
        <td><b>82.39</b></td>
        <td>85.09</td>
        <td><b>79.85</b></td>
        <td>0.055</td>
    </tr>
</table>

## Inference Speed and FLOPs

You can evaluate the inference speed (FPS) and computational complexity (FLOPs) of the model using the provided scripts in `tools/`.

### Calculate FPS

Run the following command to measure FPS:

```bash
python tools/calc_fps.py --ckpt <path_to_checkpoint.pth.tar> --config <path_to_config.py> --cuda
```
- `--ckpt`: Path to the model checkpoint file.
- `--config`: Path to the config file.
- `--cuda`: Use GPU for inference (recommended).

### Calculate FLOPs

Run the following command to measure FLOPs:

```bash
python tools/calc_flops.py --ckpt <path_to_checkpoint.pth.tar> --config <path_to_config.py> --cuda
```
- `--ckpt`: Path to the model checkpoint file.
- `--config`: Path to the config file.
- `--cuda`: Use GPU for inference (recommended).

Both scripts require a valid config and checkpoint. For more options, use `--help` with each script.


## Acknowledgment

This project builds upon the contributions of [LATR](https://github.com/JMoonr/LATR), [OpenLane](https://github.com/OpenDriveLab/PersFormer_3DLane), [GenLaneNet](https://github.com/yuliangguo/Pytorch_Generalized_3D_Lane_Detection), [mmdetection3d](https://github.com/open-mmlab/mmdetection3d), [SparseInst](https://github.com/hustvl/SparseInst), [ONCE](https://github.com/once-3dlanes/once_3dlanes_benchmark), and many other related works. We gratefully acknowledge their efforts in sharing code and datasets.


## Citation
If you find our research helpful for your own work, please cite our paper as follows:

```tex
@article{cap2025hybrid,
  title={Hybrid MambaVision and Transformer-Based Architecture for 3D Lane Detection},
  author={Cap, Raul-Mihai and Popa, C{\u{a}}lin-Adrian},
  journal={Sensors},
  volume={25},
  number={18},
  pages={5729},
  year={2025},
  publisher={MDPI}
}
```
