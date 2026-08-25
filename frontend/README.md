# AIC 2026 Retrieval Workspace (frontend)

Desktop-first React/Vite workspace over the AIC2026 retrieval HTTP API.

## Run

Backend (separate terminal, repo root):

```bash
uvicorn aic2026.server:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
npm install
npm run dev        # http://localhost:5173
```

The FastAPI backend does not send CORS headers, so development requests stay
same-origin: the Vite dev server proxies `/api` to
`http://127.0.0.1:8000` (override with the `BACKEND_PROXY` environment
variable). To call a backend that explicitly allows cross-origin requests,
set `VITE_API_URL` to its absolute origin instead.

## Test / build / lint

```bash
npx vitest run     # unit + component tests
npm run build      # typecheck + production build
npm run lint
```

## Structure

- `src/api/` — typed contracts (`types.ts`) and the fetch client (`client.ts`)
- `src/hooks/queries.ts` — TanStack Query hooks (health polling, search,
  video/frame browsing)
- `src/state/uiStore.ts` — local UI state (task, selections, QA evidence,
  TRAKE picks); no server data lives here
- `src/components/` — shared workspace shell components
- `src/features/{kis,qa,trake}/` — task-specific action panels sharing one
  results grid + inspector

## Keyboard

| Key | Action |
| --- | --- |
| `/` | focus search input |
| `←` `→` | previous/next temporal keyframe |
| `s` | select inspected frame for the active task |
| `Esc` | clear inspector |

## Notes

- Result ordering is server-defined and rendered as-is (no client-side
  re-ranking or grouping).
- Thumbnails are browser-lazy; the full-size image loads only for the
  inspected frame.
- VLM answer generation is not exposed by the API; the QA panel renders it
  as unavailable and supports manual answers only.
