/**
 * Server-state hooks (TanStack Query) over the typed API client.
 *
 * Query keys are stable per parameter set; search results are deterministic
 * on the backend so completed queries are kept and never silently refetched.
 */
import { useQuery } from "@tanstack/react-query";
import { ApiError, getFrames, getSystem, getVideo, health, search } from "../api/client";
import type { FramesAroundParams } from "../api/types";
import { backendTask, useUiStore } from "../state/uiStore";

export type BackendStatus = "loading" | "ready" | "starting" | "unavailable";

/** Poll /api/health until READY; expose a coarse UI status. */
export function useBackendStatus(): {
  status: BackendStatus;
  retry: () => void;
} {
  const query = useQuery({
    queryKey: ["health"],
    queryFn: ({ signal }) => health(signal),
    refetchInterval: (q) =>
      q.state.data?.status === "ready" ? false : 2_000,
    retry: false,
  });

  // health() already turns the documented 503-starting flat body into data,
  // so any surviving error means the backend is genuinely unreachable
  // (down, or the dev proxy cannot connect).
  const data = query.data;
  let status: BackendStatus;
  if (query.isPending && query.error === null) status = "loading";
  else if (data?.status === "ready") status = "ready";
  else if (data?.status === "starting") status = "starting";
  else status = "unavailable";

  return { status, retry: () => void query.refetch() };
}

/** GET /api/system — enabled only once the backend is ready. */
export function useSystem(enabled: boolean) {
  return useQuery({
    queryKey: ["system"],
    queryFn: ({ signal }) => getSystem(signal),
    enabled,
    staleTime: Infinity,
    retry: 1,
  });
}

export interface SearchState {
  pending: boolean;
  error: ApiError | null;
  /** Undefined until the first successful search for the current parameters. */
  data: Awaited<ReturnType<typeof search>> | undefined;
}

/**
 * POST /api/search for the active task + last submitted query.
 * Disabled until submitSearch() has been called; repeated identical
 * searches reuse the cached response instead of hitting the backend.
 */
export function useSearch(enabled: boolean): SearchState {
  const task = useUiStore((s) =>
    s.activeQueryId !== null ? s.queries[s.activeQueryId]?.task : undefined,
  );
  const lastSearch = useUiStore((s) =>
    s.activeQueryId !== null ? s.queries[s.activeQueryId]?.lastSearch : undefined,
  );
  const queryTask = backendTask(task ?? "kis");

  const query = useQuery({
    queryKey: ["search", queryTask, lastSearch?.query ?? "", lastSearch?.limit ?? 0],
    queryFn: ({ signal }) =>
      search(
        {
          query: lastSearch?.query ?? "",
          task: queryTask,
          limit: lastSearch?.limit ?? 100,
        },
        signal,
      ),
    // NB: strict null/undefined check — the selector yields undefined when
    // no query workspace is active, and `undefined !== null` is true.
    enabled: enabled && lastSearch != null,
    staleTime: Infinity,
    gcTime: 30 * 60_000,
    retry: false,
  });

  // `isLoading` (= isPending && isFetching) reflects an actually in-flight
  // request. Raw `isPending` would also stay true for the disabled
  // not-yet-submitted state, permanently disabling the search button.
  return { pending: query.isLoading, error: (query.error as ApiError) ?? null, data: query.data };
}

/** GET /api/videos/{video_id} */
export function useVideo(videoId: string | null) {
  return useQuery({
    queryKey: ["video", videoId],
    queryFn: ({ signal }) => getVideo(videoId as string, signal),
    enabled: videoId !== null,
    staleTime: Infinity,
  });
}

/** GET /api/videos/{video_id}/frames around the inspected keyframe. */
export function useTemporalFrames(
  videoId: string | null,
  params: FramesAroundParams | null,
) {
  return useQuery({
    queryKey: [
      "frames",
      videoId,
      params?.center ?? null,
      params?.radius ?? null,
      params?.limit ?? null,
    ],
    queryFn: ({ signal }) => getFrames(videoId as string, params as FramesAroundParams, signal),
    enabled: videoId !== null && params !== null,
    staleTime: Infinity,
    placeholderData: (previous) => previous,
  });
}
