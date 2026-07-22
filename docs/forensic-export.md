# Forensic export and verification

The result player can create a portable, tamper-evident evidence package. This is a
prototype integrity control, not a claim that the package has automatically been admitted
as legal evidence. Deployment policy, access control, trusted time, key custody, and officer
procedures still matter.

## Package layout

```text
case-export/
├── original_or_source_clip.mp4  byte-for-byte copy of indexed camera media
├── selected_clip.mp4            derived, padded review clip
├── annotated_frame.jpg          derived full frame with tracklet box
├── manifest.json                provenance and artifact roles
├── report.pdf                   human-readable case summary
├── SHA256SUMS                   hashes every file above plus public_key.pem
├── signature.sig                Ed25519 signature over exact SHA256SUMS bytes
└── public_key.pem               portable verification material
```

`original_or_source_clip.mp4` is never annotated or re-encoded during export. It is the
indexed media referenced by the tracklet's `video_ref`; that indexed media may itself have
been normalized from NVR footage during ingestion. The manifest states this scope so the
export does not overclaim that a transcoded ingest artifact is the camera-original bitstream.

The manifest cannot contain its own hash. Instead, `SHA256SUMS` hashes `manifest.json` and
all other artifacts. `signature.sig` signs `SHA256SUMS`, avoiding a circular self-hash.

## Apply the additive schema migration

On an existing local database, run from the repository root:

```sh
docker compose -f infra/docker-compose.yml exec -T db \
  psql -U cctv -d cctv -f /docker-entrypoint-initdb.d/10_schema.sql
```

`forensic_exports` records the export ID, case, officer, tracklet, source and manifest
hashes, package path, signer fingerprint, timestamp, and the emitted manifest. A database
trigger rejects updates and deletes, making receipts append-only.

## Key custody

The backend lazily generates:

```text
apps/backend_py/.forensic_keys/ed25519-private.pem  mode 0600
apps/backend_py/.forensic_keys/ed25519-public.pem   mode 0644
```

Both the key directory and generated exports are ignored by Git. For a real deployment,
set `FORENSIC_KEY_DIR` to protected, backed-up key storage and restrict filesystem access to
the backend service account. Never silently replace the key: the SHA-256 fingerprint of the
public key is the signing-key ID stored in manifests and database receipts.

## Verification

The dashboard's **Verify export** button uploads a ZIP to the local backend. It recomputes
every hash and verifies the Ed25519 signature against the backend's trusted public key.

The portable CLI can either pin that key or verify only against the embedded key:

```sh
cd apps/backend_py

# Trusted deployment verification — preferred
uv run python -m app.forensics verify path/to/export.zip \
  --public-key .forensic_keys/ed25519-public.pem

# Portable integrity check; warns that signer identity was not pinned
uv run python -m app.forensics verify path/to/export.zip
```

Exit status is `0` for `VALID` and `1` for `TAMPERED`.

## Safe tamper demonstration

Never modify the evidence package used for a case. The demo command refuses to overwrite
the input and creates a separate ZIP with extra bytes appended to `selected_clip.mp4`:

```sh
uv run python -m app.forensics tamper-demo valid-export.zip /tmp/tampered-demo.zip
uv run python -m app.forensics verify /tmp/tampered-demo.zip \
  --public-key .forensic_keys/ed25519-public.pem
```

The second command exits non-zero and reports:

```text
TAMPERED
checksum mismatch: selected_clip.mp4
```

An attacker cannot repair that mismatch by rewriting `SHA256SUMS`: the pinned Ed25519
signature would then fail.

## Manifest provenance

The signed manifest records:

- case ID, export ID, officer/user, and UTC creation time;
- original and rewritten query, search timestamp, and explicit filters;
- scene, camera/location, scene-clock and video-local timestamps;
- tracklet ID, every stored crop ID, and selected crop;
- entity metadata and the annotated-frame bounding box;
- detector, tracker, semantic model, model revision configuration, and VLM state;
- retrieval mode, crop aggregation, composition, prompts, result score, and component scores;
- source and artifact SHA-256 values, roles, sizes, algorithm, and signer fingerprint.

Query context originates from the search response selected in the dashboard; tracklet,
camera, timestamps, crop rows, plate, media reference, and source bytes are reloaded and
validated by the backend at export time.
