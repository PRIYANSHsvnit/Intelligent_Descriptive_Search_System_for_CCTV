# Audited investigation workflow

## Operator flow

1. Enter a case ID using letters, numbers, dots, underscores, or dashes.
2. Enter the investigator name or badge ID. The prototype deliberately does not claim this
   is authenticated identity; a production deployment must supply it from SSO/RBAC.
3. Run a text, plate, reference-image, or similar-object search.
4. Open **Why this matched** to inspect the detector fact, location/time constraints, raw
   component similarity, and the stored crop that best supports each component.
5. Pin useful sightings or exclude false positives with a reason.
6. Add notes and review the immutable timeline in **Case board**.
7. Select **Export pinned** to download a signed case bundle.

Match-strength words are relative labels within the returned candidate set. They are not
probabilities or calibrated confidence values.

## Data model

- `cases` stores current case identity and display metadata.
- `case_items` stores current pinned/excluded state, notes, and the result/search snapshot.
- `search_events` is append-only. A database trigger rejects `UPDATE` and `DELETE` and records
  searches, pins, exclusions, note changes, individual exports, and case-bundle exports.
- `forensic_exports` remains the append-only receipt table for every child evidence package.

Apply the additive schema:

```sh
docker compose -f infra/docker-compose.yml exec -T db \
  psql -U cctv -d cctv -v ON_ERROR_STOP=1 -f /dev/stdin < infra/schema.sql
```

## API summary

- `POST /cases`
- `GET /cases/{case_id}`
- `GET /cases/{case_id}/timeline`
- `PUT /cases/{case_id}/items/{tracklet_id}`
- `POST /cases/{case_id}/export`

Search endpoints accept `case_id` and `officer` together. The frontend always supplies both;
the API rejects a partially supplied audit identity.

## Signed case bundles

`Export pinned` creates:

```text
case-bundle/
├── evidence-001-<tracklet>.zip
├── evidence-002-<tracklet>.zip
├── manifest.json
├── public_key.pem
├── SHA256SUMS
└── signature.sig
```

The bundle manifest records its selection policy and child export IDs. `SHA256SUMS` binds the
manifest, public key, and every child package; Ed25519 signs that checksum file. Each child is
the complete single-result forensic export documented in `forensic-export.md` and retains its
own signature, checksums, source media, derived clip, annotated frame, and report.

The existing UI and CLI verifier auto-detect both package schemas. Bundle verification checks
the parent signature and hashes, validates the manifest inventory, then recursively verifies
every child against the same trusted deployment public key.
