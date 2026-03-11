# SoftZoo Assets

Assets (meshes, textures) are stored in `data/softzoo/assets/`.

## Directory Structure

```
data/softzoo/assets/
├── meshes/
│   ├── pcd/     # Annotated PCD files (download required)
│   └── stl/     # STL meshes (shipped with vendor)
├── textures/    # Terrain textures (shipped with vendor)
└── skybox/      # Skybox (optional)
```

## Download

1. Open [Google Drive](https://drive.google.com/drive/folders/1AYeZsr2ZMb1DkeOndQM0nBlNfx7dorUL)
2. Download `meshes/pcd/*.pcd` (Caterpillar, Panda, etc.)
3. Place in `data/softzoo/assets/meshes/pcd/`

## Setup Script

```bash
./scripts/setup/setup_softzoo.sh
```

This creates the directory structure and copies vendor assets (textures, stl) from `third_party/environments/softzoo/softzoo/assets/`.

## Environment Variables

- `SOFTZOO_ASSETS_ROOT`: Override assets path (default: `data/softzoo/assets`)
- `SOFTZOO_ROOT`: Override code root (default: `third_party/environments/softzoo`)
