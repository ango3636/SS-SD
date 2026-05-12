"""JIGSAWS Suturing kinematics: **76 values per timestep** (headerless rows).

Layout follows the public JIGSAWS description: four 19-column tool blocks
(**master left**, **master right**, **slave left**, **slave right**). Each
block is: tip position xyz (3), rotation ``R`` (9), translational velocity
(3), rotational velocity (3), gripper angle (1).

All indices below are **0-based** (Python). For **1-based** columns as in
dataset READMEs, add 1 to each index.

This module is the canonical reference for column ranges used in config and
narration.
"""

from __future__ import annotations

# --- Master left (1-based 1–19 → 0-based 0–18) --------------------------------
MASTER_LEFT_XYZ = slice(0, 3)
MASTER_LEFT_TRANS_VEL = slice(12, 15)  # 1-based 13–15
MASTER_LEFT_ROT_VEL = slice(15, 18)  # 1-based 16–18
MASTER_LEFT_GRIPPER = 18  # 1-based 19

# --- Master right (1-based 20–38 → 0-based 19–37) -------------------------------
MASTER_RIGHT_XYZ = slice(19, 22)
MASTER_RIGHT_TRANS_VEL = slice(31, 34)  # 1-based 32–34
MASTER_RIGHT_ROT_VEL = slice(34, 37)  # 1-based 35–37
MASTER_RIGHT_GRIPPER = 37  # 1-based 38

# --- Slave left (1-based 39–57 → 0-based 38–56) ---------------------------------
SLAVE_LEFT_XYZ = slice(38, 41)
SLAVE_LEFT_TRANS_VEL = slice(50, 53)  # 1-based 51–53
SLAVE_LEFT_ROT_VEL = slice(53, 56)  # 1-based 54–56
SLAVE_LEFT_GRIPPER = 56  # 1-based 57

# --- Slave right (1-based 58–76 → 0-based 57–75) -------------------------------
SLAVE_RIGHT_XYZ = slice(57, 60)
SLAVE_RIGHT_TRANS_VEL = slice(69, 72)  # 1-based 70–72
SLAVE_RIGHT_ROT_VEL = slice(72, 75)  # 1-based 73–75
SLAVE_RIGHT_GRIPPER = 75  # 1-based 76

# Default columns for ``compute_kinematic_features`` (magnitude of master-left
# translational velocity); must match ``configs/base.yaml`` ``kinematics.*``.
TRANSLATIONAL_VELOCITY_COL_INDICES: list[int] = [12, 13, 14]
ROTATIONAL_VELOCITY_COL_INDICES: list[int] = [15, 16, 17]
