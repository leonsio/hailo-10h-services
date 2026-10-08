# Virtual assistants

`ha/` and `frigate/` implement independent virtual-model preparation. They share primitives from `shared/` and configured resident backends, but never import each other or share request catalogues.
