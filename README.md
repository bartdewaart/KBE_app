# KBE Propulsion System Design Tool

Automated design and optimisation of UAV propulsion systems using Blade Element Momentum Theory (BEMT), XFOIL aerodynamic analysis, and structural integrity checks.

---

## Quick Start (pre-built ZIP)

1. Download **`KBE_PropDesign_dist.zip`** from [GitHub Releases](../../releases)
2. Extract the ZIP anywhere on your Windows machine
3. Edit `data\input\mission.xlsx` with your mission requirements
4. Double-click **`run.bat`** — no Python or package installation needed

---

## Quick Start (from source)

> Requires [UV](https://docs.astral.sh/uv/) and ParaPy credentials.

1. Clone or download this repository
2. Log in to the ParaPy package index once:
   ```
   uv auth login pypi.parapy.nl --username <username> --password <password>
   ```
3. Install dependencies:
   ```
   uv sync
   ```
4. Edit `data\input\mission.xlsx` with your mission requirements
5. Double-click **`run.bat`** (or run `python main.py` in a terminal)

---

## Mission Parameters (`data\input\mission.xlsx`)

| Parameter | Unit | Description |
|-----------|------|-------------|
| payload_mass | kg | Total payload the UAV must carry |
| n_rotors | - | Number of rotors (e.g. 4 for quadrotor) |
| safety_margin | - | Thrust safety factor (1.5 = 50% overhead) |
| max_diameter | m | Maximum allowed propeller diameter |
| propeller_material | - | Carbon Fibre / Fibreglass / Aluminium / PLA |
| battery_energy_density | Wh/kg | Battery specific energy |
| battery_efficiency | - | Battery discharge efficiency (0-1) |
| min_endurance_min | min | Minimum required flight time |
| w_power | - | Objective weight for shaft power (weights must sum to 1) |
| w_mass | - | Objective weight for rotor mass |
| w_endurance | - | Objective weight for endurance |

---

## GUI Actions

Right-click the `PropulsionSystem` node in the GUI tree to access:

| Action | Description |
|--------|-------------|
| Re-run optimization | Re-optimise after editing inputs in the GUI tree |
| Plot spanwise distribution | Chord, pitch, and thrust along the blade span |
| Plot structural analysis | Stress, factor of safety, and tip deflection |
| Plot motor curve | Motor torque vs RPM at the operating point |
| Export design PDF | Full 6-page design report |
| Export STEP files | 3D geometry files for CAD tools |
| Export design CSV | Tabular spanwise data |

---

## Creating a Shareable ZIP

To share the app with others (no installation required for recipients):

1. Ensure `.venv` is built and up to date (`uv sync`)
2. Double-click **`make_dist.bat`**
3. Upload the generated `KBE_PropDesign_dist.zip` to GitHub Releases
