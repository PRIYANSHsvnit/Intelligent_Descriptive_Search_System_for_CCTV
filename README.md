# ER_GlacierNova Monorepo

Turborepo monorepo with:

- `apps/frontend`: Next.js frontend app (package name: `web`)
- `apps/backend_py`: FastAPI backend app (Python + `uv`)
- `packages/ui`: shared UI components
- `packages/eslint-config`: shared ESLint configs
- `packages/typescript-config`: shared TypeScript configs

## Prerequisites

- Node.js `>=18`
- `pnpm` (repo uses `pnpm@9`)
- Python `>=3.11` (for backend)
- [`uv`](https://docs.astral.sh/uv/) (for backend Python dependency management and running)

## Quick Start (Full Repo)

From the repository root:

```sh
pnpm install
pnpm dev
```

This starts all `dev` tasks in Turborepo (frontend and backend).

Expected local URLs:

- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`

## What's Inside?

### Apps and Packages

- `apps/frontend` (`web`): [Next.js](https://nextjs.org/) app
- `apps/backend_py` (`backend_py`): [FastAPI](https://fastapi.tiangolo.com/) backend service
- `@repo/ui`: shared React component library
- `@repo/eslint-config`: shared ESLint setup
- `@repo/typescript-config`: shared TypeScript setup

### Utilities

- [TypeScript](https://www.typescriptlang.org/) for static type checking
- [ESLint](https://eslint.org/) for linting
- [Prettier](https://prettier.io) for formatting
- [Turborepo](https://turborepo.com/) for orchestration and caching

## App Setup Guides

### Frontend Setup (`apps/frontend`)

Run from repo root:

```sh
pnpm install
pnpm --filter=web dev
```

Or run from the frontend app folder:

```sh
cd apps/frontend
pnpm dev
```

Frontend scripts:

- `pnpm --filter=web dev`
- `pnpm --filter=web build`
- `pnpm --filter=web start`
- `pnpm --filter=web lint`
- `pnpm --filter=web check-types`

### Backend Setup (`apps/backend_py`)

The backend uses `uv` with dependencies defined in `apps/backend_py/pyproject.toml`.

### 1) Create and use virtual environment

From repo root:

```sh
cd apps/backend_py
uv venv
```

Activate it (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2) Sync/install dependencies

```sh
uv sync
```

This installs everything from `pyproject.toml` and `uv.lock`.

### 3) Add a new backend library

Inside `apps/backend_py`:

```sh
uv add <package-name>
uv sync
```

Example:

```sh
uv add httpx
uv sync
```

### 4) Run backend server

From repo root:

```sh
pnpm --filter=backend_py dev
```

Or from `apps/backend_py` directly:

```sh
pnpm dev
```

Backend serves on `http://localhost:8000`.

Useful backend endpoints:

- Health/root: `http://localhost:8000/`
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Retrieval architecture

The production search path uses individual SigLIP2 vectors for each retained tracklet
view, broad HNSW candidate retrieval, exact crop reranking, and deterministic component
prompts for descriptions such as “yellow shirt and black cap.” Explicit camera, time,
entity, and vehicle-colour filters are never silently removed.

See [`improvement-plan.md`](./improvement-plan.md) for the implemented design, evaluation
protocol, measured latency, and remaining human-labelling gate.

## Forensic evidence export

Open a search result, choose **Export evidence**, and enter a case ID plus officer/badge ID.
The backend creates a ZIP with the unannotated indexed source recording, a selected clip,
an annotated frame, a PDF report, a JSON manifest, SHA-256 checksums, and an Ed25519
signature. **Verify export** in the top bar checks an uploaded package against this
deployment's trusted signing key and reports `VALID` or `TAMPERED`.

Generated packages and private keys are ignored by Git. Back up
`apps/backend_py/.forensic_keys/ed25519-private.pem` securely: losing or replacing it
breaks continuity of signer identity. See
[`docs/forensic-export.md`](./docs/forensic-export.md) for the package format, CLI verifier,
trust model, database migration, and safe tamper demo.

### Backend environment variables (`.env`)

If you need environment variables for the API:

1. Create `apps/backend_py/.env`
2. Add keys, for example:

```env
APP_ENV=development
API_KEY=replace-me
```

3. Load them in FastAPI code using your preferred approach (for example `pydantic-settings` + `python-dotenv` if you adopt those).

## Turborepo Commands

### Build

Build all apps and packages:

```sh
pnpm build
```

Build a specific app/package:

```sh
pnpm exec turbo build --filter=web
pnpm exec turbo build --filter=backend_py
```

### Develop

Develop all apps:

```sh
pnpm dev
```

Develop only one app:

```sh
pnpm exec turbo dev --filter=web
pnpm exec turbo dev --filter=backend_py
```

### Lint / Typecheck / Format

```sh
pnpm lint
pnpm check-types
pnpm format
```

### Remote Caching

Turborepo can use [Remote Caching](https://turborepo.dev/docs/core-concepts/remote-caching) (for example with Vercel).

```sh
pnpm exec turbo login
pnpm exec turbo link
```

## Useful Links

- [Turborepo Tasks](https://turborepo.dev/docs/crafting-your-repository/running-tasks)
- [Turborepo Caching](https://turborepo.dev/docs/crafting-your-repository/caching)
- [Turborepo Remote Caching](https://turborepo.dev/docs/core-concepts/remote-caching)
- [Turborepo Filtering](https://turborepo.dev/docs/crafting-your-repository/running-tasks#using-filters)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Next.js Docs](https://nextjs.org/docs)
