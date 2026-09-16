# L0-L4 waveguide lengths

Nominal geometric waveguide centerline length; includes straight and curved sections.
Labels follow the five supplied pictures in order. Values recalculate on every run.

| Path | Short stage | Long stage | Length (µm) | Length (mm) |
|---|---|---|---:|---:|
| L0 | bypass | bypass | 6375.631 | 6.375631 |
| L1 | lower | lower | 9534.934 | 9.534934 |
| L2 | upper | lower | 13934.934 | 13.934934 |
| L3 | lower | upper | 14934.934 | 14.934934 |
| L4 | upper | upper | 19334.934 | 19.334934 |

## Reference planes

Start: Downstream end of splitter output coupling straight (lower branch for L0; upper for L1-L4).

End: Wide input of right-edge taper: edge_174 (reference) for L0; edge_161 (lower SW3 output) for L1-L4. The upper SW3 output edge_186 has equal geometric length to edge_161.

Excluded: Splitter internal arms/output coupling straight; all edge-coupler tapers and tips. No bottom combiner remains.

Output reference planes changed after deleting the bottom combiner; totals are not directly comparable with reports ending at its coupling entrance.

Analytic Euler arclength, quadrature for fitted S bends, and grid-snapped straight dimensions. This is not refractive-index-weighted optical path length or group delay.

Each switch contributes one arm and both coupling straights. SW3 output transition: L_PAIR. The two arms have equal length; no transverse distance is assigned to directional-coupler transfer.

## Component contributions

### L0: Bottom reference path

| Component | Length (µm) |
|---|---:|
| splitter_lower_output_lead | 187.997 |
| bottom_reference_bus | 6187.634 |
| **Total** | **6375.631** |

### L1: Lower short-stage / lower long-stage route

| Component | Length (µm) |
|---|---:|
| splitter_upper_output_lead | 187.997 |
| splitter_to_switch_1 | 600.000 |
| SW1_complete_MZI | 1608.104 |
| short_stage_lower | 1675.994 |
| SW2_complete_MZI | 1608.104 |
| long_stage_lower | 1675.994 |
| SW3_complete_MZI | 1796.101 |
| switch_3_lower_output | 382.639 |
| **Total** | **9534.934** |

### L2: Upper short-stage / lower long-stage route

| Component | Length (µm) |
|---|---:|
| splitter_upper_output_lead | 187.997 |
| splitter_to_switch_1 | 600.000 |
| SW1_complete_MZI | 1608.104 |
| short_stage_upper | 6075.994 |
| SW2_complete_MZI | 1608.104 |
| long_stage_lower | 1675.994 |
| SW3_complete_MZI | 1796.101 |
| switch_3_lower_output | 382.639 |
| **Total** | **13934.934** |

### L3: Lower short-stage / upper long-stage route

| Component | Length (µm) |
|---|---:|
| splitter_upper_output_lead | 187.997 |
| splitter_to_switch_1 | 600.000 |
| SW1_complete_MZI | 1608.104 |
| short_stage_lower | 1675.994 |
| SW2_complete_MZI | 1608.104 |
| long_stage_upper | 7075.994 |
| SW3_complete_MZI | 1796.101 |
| switch_3_lower_output | 382.639 |
| **Total** | **14934.934** |

### L4: Upper short-stage / upper long-stage route

| Component | Length (µm) |
|---|---:|
| splitter_upper_output_lead | 187.997 |
| splitter_to_switch_1 | 600.000 |
| SW1_complete_MZI | 1608.104 |
| short_stage_upper | 6075.994 |
| SW2_complete_MZI | 1608.104 |
| long_stage_upper | 7075.994 |
| SW3_complete_MZI | 1796.101 |
| switch_3_lower_output | 382.639 |
| **Total** | **19334.934** |

